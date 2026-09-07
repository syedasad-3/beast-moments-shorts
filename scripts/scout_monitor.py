"""
Scout + Monitor — runs daily on GitHub Actions (schedule/workflow_dispatch only).

What it does:
  1. Checks whether a video is already queued (emailed, but not yet
     downloaded/uploaded by the operator). If so, that's the ONLY video
     this run will talk about — Scout reminds the operator about it
     instead of searching for and queuing a different one. This keeps
     exactly one video pending at a time, so the queue can never grow
     faster than the operator can act on it.
  2. Only if nothing is currently queued does Scout look for a video to
     queue: it prefers MrBeast's most recent long-form (non-Short)
     upload if unprocessed; if there's no new upload, it falls back to
     scanning older uploads (newest-to-oldest) for the most recent one
     that still hasn't been processed — so the pipeline always has
     something to work on instead of sitting idle waiting for a brand
     new video.
  3. Either way, if there's a video to talk about, sends an email with
     the link and instructs the operator to download it and upload it to
     the Drive "incoming" folder.
  4. Also checks state/pipeline_errors.json for errors reported by other
     pipeline stages and includes them in the same email, regardless of
     which video path was taken above.
  5. Commits state changes back to the repo (GitHub Actions runners are
     ephemeral — state must be persisted in git, not left on disk).

Design notes (why it's built this way):
  - Only emails when there's something new to report (a video OR an
    error) — no daily spam when nothing changed.
  - Does NOT mark a video as "queued" in state until the email has
    actually sent successfully. If SMTP fails, the same video will be
    picked up again next run instead of being silently skipped forever.
  - A reminder about an already-queued video never touches
    processed_videos.json or the YouTube API at all — there's nothing
    new to record, and no need to spend API quota searching while a
    video is already waiting on the operator.
  - Shorts (<3 min) are filtered out — this pipeline repurposes long-form
    videos, not the channel's own Shorts.
  - Backlog fallback deliberately needs no separate "how far we've
    scanned" cursor: processed_videos.json already records every video
    ever queued (new-upload or backlog), so each search naturally skips
    everything already handled and lands on the next
    most-recent-but-untouched video, moving steadily from newest toward
    oldest over time. A stored page-token cursor would be fragile here —
    it would drift out of sync every time a genuinely new video gets
    added at the top of the channel, shifting every older video's
    position by one.
  - The backlog scan is capped at MAX_VIDEOS_TO_SCAN per run purely as a
    safety limit on API quota/runtime — not expected to be hit in normal
    operation, but without it, a channel that's been fully processed
    would cause every run to page through the entire upload history.
"""

import json
import os
import smtplib
import sys
import time
from datetime import datetime, timezone
from email.mime.text import MIMEText

try:
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError
except ImportError:
    print("ERR_MISSING_DEPENDENCY: google-api-python-client not installed.")
    sys.exit(1)

# --- Config ---------------------------------------------------------------

MRBEAST_CHANNEL_ID = "UCX6OQ3DkcsbYNE6H8uQQuVA"  # public, stable identifier
MIN_VIDEO_DURATION_SECONDS = 180  # filters out Shorts / trailers
MAX_VIDEOS_TO_SCAN = 500  # safety cap on how deep one run will page into the back catalog
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATE_DIR = os.path.join(REPO_ROOT, "state")
PROCESSED_VIDEOS_FILE = os.path.join(STATE_DIR, "processed_videos.json")
PIPELINE_ERRORS_FILE = os.path.join(STATE_DIR, "pipeline_errors.json")

YT_API_KEY = os.environ.get("YT_API_KEY")
SMTP_EMAIL = os.environ.get("SMTP_EMAIL")
SMTP_APP_PASSWORD = os.environ.get("SMTP_APP_PASSWORD")
ALERT_EMAIL = os.environ.get("ALERT_EMAIL")
DRIVE_INCOMING_FOLDER_ID = os.environ.get("DRIVE_INCOMING_FOLDER_ID")

MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = [5, 15, 30]


# --- Helpers ---------------------------------------------------------------

def load_json(path, default):
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        # Corrupt state file is a serious problem — don't silently reset it,
        # since that could cause the pipeline to re-download/re-post old
        # videos. Fail loudly instead.
        print(f"ERR_CORRUPT_STATE_FILE: {path} is not valid JSON: {e}")
        sys.exit(1)


def save_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def with_retries(fn, error_code, *args, **kwargs):
    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            return fn(*args, **kwargs)
        except HttpError as e:
            last_error = e
            status = e.resp.status if hasattr(e, "resp") else None
            if status == 403 and "quotaExceeded" in str(e):
                print(f"ERR_QUOTA_EXCEEDED: YouTube Data API daily quota exceeded. {e}")
                sys.exit(1)  # no point retrying a quota error
            print(f"{error_code}: attempt {attempt + 1}/{MAX_RETRIES} failed: {e}")
        except Exception as e:
            last_error = e
            print(f"{error_code}: attempt {attempt + 1}/{MAX_RETRIES} failed: {e}")
        if attempt < MAX_RETRIES - 1:
            time.sleep(RETRY_BACKOFF_SECONDS[attempt])
    print(f"{error_code}: all {MAX_RETRIES} attempts failed. Last error: {last_error}")
    sys.exit(1)


def find_pending_queued_video(processed_videos):
    """Returns the oldest still-queued video (the one that's been waiting
    longest for the operator to download/upload), or None if nothing is
    currently queued. Scout reminds about this video instead of searching
    for a new one whenever one exists, so the queue never grows past a
    single pending video."""
    queued = [
        {"video_id": vid, **info}
        for vid, info in processed_videos.items()
        if info.get("status") == "queued"
    ]
    if not queued:
        return None
    queued.sort(key=lambda v: v.get("queued_at", ""))  # oldest first
    oldest = queued[0]
    return {
        "video_id": oldest["video_id"],
        "title": oldest["title"],
        "url": oldest["url"],
        "duration_seconds": oldest.get("duration_seconds", 0),  # older entries may predate this field
    }


def get_uploads_playlist_id(youtube):
    def _call():
        resp = youtube.channels().list(part="contentDetails", id=MRBEAST_CHANNEL_ID).execute()
        items = resp.get("items", [])
        if not items:
            raise RuntimeError(f"No channel found for ID {MRBEAST_CHANNEL_ID}")
        return items[0]["contentDetails"]["relatedPlaylists"]["uploads"]

    return with_retries(_call, "ERR_YT_CHANNEL_LOOKUP_FAIL")


def get_video_candidates_page(youtube, uploads_playlist_id, page_token=None):
    """Fetches one page (up to 50, the API max) of the uploads playlist,
    in the API's default order — newest upload first. Returns the raw
    response dict (items + nextPageToken) so the caller can page through
    as many videos as it needs."""

    def _call():
        return youtube.playlistItems().list(
            part="contentDetails,snippet",
            playlistId=uploads_playlist_id,
            maxResults=50,
            pageToken=page_token,
        ).execute()

    return with_retries(_call, "ERR_YT_PLAYLIST_FETCH_FAIL")


def get_video_durations(youtube, video_ids):
    def _call():
        resp = youtube.videos().list(part="contentDetails", id=",".join(video_ids)).execute()
        return {item["id"]: item["contentDetails"]["duration"] for item in resp.get("items", [])}

    return with_retries(_call, "ERR_YT_VIDEO_DETAILS_FAIL")


def parse_iso8601_duration_to_seconds(duration):
    # Minimal ISO 8601 duration parser (PT#H#M#S) — YouTube durations
    # never include days/weeks, so this scope is sufficient.
    import re
    match = re.match(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", duration)
    if not match:
        return 0
    hours, minutes, seconds = (int(x) if x else 0 for x in match.groups())
    return hours * 3600 + minutes * 60 + seconds


def find_next_unprocessed_video(youtube, processed_videos):
    """Returns the most recent not-yet-processed, non-Short MrBeast video —
    checking the newest upload first and paging progressively further back
    into the channel's history if needed. A genuinely brand-new upload is
    just the special case where the very first candidate checked happens
    to be unprocessed; if it isn't (already done, or filtered out), this
    keeps going back in time through the back catalog rather than giving
    up, so the pipeline is never left with nothing to do just because
    MrBeast hasn't posted today.

    Only called when nothing is currently queued (see
    find_pending_queued_video) — so every candidate this sees that isn't
    in processed_videos is genuinely available to queue.

    Returns None if nothing usable was found within MAX_VIDEOS_TO_SCAN
    (either the channel's entire history was scanned, or the safety cap
    was hit first — both are logged so it's clear which happened)."""
    uploads_playlist_id = get_uploads_playlist_id(youtube)

    scanned = 0
    page_token = None

    while scanned < MAX_VIDEOS_TO_SCAN:
        resp = get_video_candidates_page(youtube, uploads_playlist_id, page_token)
        items = resp.get("items", [])
        if not items:
            print(f"Reached the end of the channel's upload history after scanning {scanned} video(s).")
            return None

        video_ids = [c["contentDetails"]["videoId"] for c in items]
        durations = get_video_durations(youtube, video_ids)

        for c in items:
            vid = c["contentDetails"]["videoId"]
            scanned += 1
            if vid in processed_videos:
                continue
            duration_str = durations.get(vid, "PT0S")
            duration_seconds = parse_iso8601_duration_to_seconds(duration_str)
            if duration_seconds < MIN_VIDEO_DURATION_SECONDS:
                continue  # skip Shorts
            return {
                "video_id": vid,
                "title": c["snippet"]["title"],
                "url": f"https://www.youtube.com/watch?v={vid}",
                "duration_seconds": duration_seconds,
            }

        page_token = resp.get("nextPageToken")
        if not page_token:
            print(f"Reached the end of the channel's upload history after scanning {scanned} video(s).")
            return None

    print(
        f"Scanned {scanned} videos (the MAX_VIDEOS_TO_SCAN safety limit) without finding an "
        f"unprocessed one. If this happens repeatedly, raise MAX_VIDEOS_TO_SCAN in scout_monitor.py."
    )
    return None


def send_email(subject, body):
    if not (SMTP_EMAIL and SMTP_APP_PASSWORD and ALERT_EMAIL):
        print("ERR_MISSING_SMTP_CONFIG: SMTP_EMAIL, SMTP_APP_PASSWORD, or ALERT_EMAIL not set.")
        sys.exit(1)

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = SMTP_EMAIL
    msg["To"] = ALERT_EMAIL

    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            with smtplib.SMTP("smtp.gmail.com", 587, timeout=30) as server:
                server.starttls()
                server.login(SMTP_EMAIL, SMTP_APP_PASSWORD)
                server.send_message(msg)
            return True
        except Exception as e:
            last_error = e
            print(f"ERR_SMTP_SEND_FAIL: attempt {attempt + 1}/{MAX_RETRIES} failed: {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_BACKOFF_SECONDS[attempt])

    print(f"ERR_SMTP_SEND_FAIL: all attempts failed. Last error: {last_error}")
    return False


def build_email_body(video, errors, is_reminder):
    lines = []
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    lines.append(f"Beast Moments Shorts — Daily Update ({today})")
    lines.append("")

    if video:
        header = "STILL WAITING — PLEASE DOWNLOAD & UPLOAD THIS VIDEO:" if is_reminder else "NEW VIDEO TO DOWNLOAD:"
        lines.append(header)
        lines.append(f"  Title: {video['title']}")
        if video.get("duration_seconds"):
            mins = video["duration_seconds"] // 60
            lines.append(f"  Length: ~{mins} min")
        lines.append(f"  Link: {video['url']}")
        lines.append("")
        lines.append(f"IMPORTANT: after downloading, rename the file to exactly: {video['video_id']}.mp4")
        lines.append("(This lets the pipeline identify which video it is, no matter what order you upload things in.)")
        lines.append("")
        lines.append("Then upload it to the 'incoming' Drive folder:")
        lines.append(f"  https://drive.google.com/drive/folders/{DRIVE_INCOMING_FOLDER_ID}")
        lines.append("")
        if is_reminder:
            lines.append("(No new video will be queued until this one is uploaded and processed.)")
            lines.append("")
    else:
        lines.append("No video available to queue today (checked latest uploads and back catalog).")
        lines.append("")

    if errors:
        lines.append(f"PIPELINE ALERTS ({len(errors)}):")
        for err in errors:
            lines.append(f"  [{err.get('error_code', 'UNKNOWN')}] {err.get('stage', '?')}: {err.get('message', '')}")
            lines.append(f"    at {err.get('timestamp', '?')}")
        lines.append("")

    return "\n".join(lines)


def main():
    processed_videos = load_json(PROCESSED_VIDEOS_FILE, {})
    pipeline_errors = load_json(PIPELINE_ERRORS_FILE, [])

    pending = find_pending_queued_video(processed_videos)
    video = pending
    is_new_video = False

    if pending:
        print(f"A video is already queued and waiting to be uploaded ('{pending['title']}') — "
              f"reminding instead of searching for a new one.")
    else:
        if not YT_API_KEY:
            print("ERR_MISSING_CONFIG: YT_API_KEY environment variable not set.")
            sys.exit(1)
        youtube = build("youtube", "v3", developerKey=YT_API_KEY)
        video = find_next_unprocessed_video(youtube, processed_videos)
        is_new_video = video is not None

    if not video and not pipeline_errors:
        print("Nothing new to report today. No email sent.")
        return

    subject = "Beast Moments Shorts — Daily Update"
    if video and pipeline_errors:
        subject += " (reminder + alerts)" if pending else " (new video + alerts)"
    elif video:
        subject += " (reminder)" if pending else " (new video)"
    else:
        subject += " (alerts only)"

    body = build_email_body(video, pipeline_errors, is_reminder=bool(pending))
    sent = send_email(subject, body)

    if not sent:
        # Don't update state if the email failed — we want this video
        # (and these errors) to be reported again next run, not lost.
        sys.exit(1)

    state_changed = False

    if is_new_video:
        processed_videos[video["video_id"]] = {
            "status": "queued",
            "title": video["title"],
            "url": video["url"],
            "duration_seconds": video["duration_seconds"],
            "queued_at": datetime.now(timezone.utc).isoformat(),
        }
        save_json(PROCESSED_VIDEOS_FILE, processed_videos)
        state_changed = True
        print(f"Emailed and queued: {video['title']}")

    if pipeline_errors:
        # Errors have now been reported via email — clear them so they
        # don't get re-sent every day forever.
        save_json(PIPELINE_ERRORS_FILE, [])
        state_changed = True
        print(f"Reported and cleared {len(pipeline_errors)} pipeline error(s).")

    if state_changed:
        # Signal to the workflow that there's something to commit.
        print("STATE_CHANGED=true")


if __name__ == "__main__":
    main()
