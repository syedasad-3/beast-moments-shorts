"""
Scout + Monitor — runs daily on GitHub Actions (schedule/workflow_dispatch only).

What it does:
  1. Checks MrBeast's channel for the most recent long-form (non-Short) video.
  2. Compares against state/processed_videos.json to avoid re-notifying
     the same video.
  3. If a new video is found, sends an email with the link and instructs
     the operator to download it and upload it to the Drive "incoming" folder.
  4. Also checks state/pipeline_errors.json for errors reported by other
     pipeline stages (which don't exist yet in early phases — this file
     will just be empty/absent for now) and includes them in the same email.
  5. Commits state changes back to the repo (GitHub Actions runners are
     ephemeral — state must be persisted in git, not left on disk).

Design notes (why it's built this way):
  - Only emails when there's something new to report (a new video OR an
    error) — no daily spam when nothing changed.
  - Does NOT mark a video as "queued" in state until the email has
    actually sent successfully. If SMTP fails, the same video will be
    picked up again next run instead of being silently skipped forever.
  - Shorts (<3 min) are filtered out — this pipeline repurposes long-form
    videos, not the channel's own Shorts.
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


def get_uploads_playlist_id(youtube):
    def _call():
        resp = youtube.channels().list(part="contentDetails", id=MRBEAST_CHANNEL_ID).execute()
        items = resp.get("items", [])
        if not items:
            raise RuntimeError(f"No channel found for ID {MRBEAST_CHANNEL_ID}")
        return items[0]["contentDetails"]["relatedPlaylists"]["uploads"]

    return with_retries(_call, "ERR_YT_CHANNEL_LOOKUP_FAIL")


def get_recent_video_candidates(youtube, uploads_playlist_id, max_results=10):
    def _call():
        resp = youtube.playlistItems().list(
            part="contentDetails,snippet",
            playlistId=uploads_playlist_id,
            maxResults=max_results,
        ).execute()
        return resp.get("items", [])

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
    uploads_playlist_id = get_uploads_playlist_id(youtube)
    candidates = get_recent_video_candidates(youtube, uploads_playlist_id)

    if not candidates:
        return None

    video_ids = [c["contentDetails"]["videoId"] for c in candidates]
    durations = get_video_durations(youtube, video_ids)

    # Candidates are already newest-first from the uploads playlist.
    for c in candidates:
        vid = c["contentDetails"]["videoId"]
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

    return None  # nothing new to process


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


def build_email_body(new_video, errors):
    lines = []
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    lines.append(f"Beast Moments Shorts — Daily Update ({today})")
    lines.append("")

    if new_video:
        mins = new_video["duration_seconds"] // 60
        lines.append("NEW VIDEO TO DOWNLOAD:")
        lines.append(f"  Title: {new_video['title']}")
        lines.append(f"  Length: ~{mins} min")
        lines.append(f"  Link: {new_video['url']}")
        lines.append("")
        lines.append("Please download this video and upload it to the 'incoming' Drive folder:")
        lines.append(f"  https://drive.google.com/drive/folders/{DRIVE_INCOMING_FOLDER_ID}")
        lines.append("")
    else:
        lines.append("No new video to download today.")
        lines.append("")

    if errors:
        lines.append(f"PIPELINE ALERTS ({len(errors)}):")
        for err in errors:
            lines.append(f"  [{err.get('error_code', 'UNKNOWN')}] {err.get('stage', '?')}: {err.get('message', '')}")
            lines.append(f"    at {err.get('timestamp', '?')}")
        lines.append("")

    return "\n".join(lines)


def main():
    if not YT_API_KEY:
        print("ERR_MISSING_CONFIG: YT_API_KEY environment variable not set.")
        sys.exit(1)

    processed_videos = load_json(PROCESSED_VIDEOS_FILE, {})
    pipeline_errors = load_json(PIPELINE_ERRORS_FILE, [])

    youtube = build("youtube", "v3", developerKey=YT_API_KEY)
    new_video = find_next_unprocessed_video(youtube, processed_videos)

    if not new_video and not pipeline_errors:
        print("Nothing new to report today. No email sent.")
        return

    subject = "Beast Moments Shorts — Daily Update"
    if new_video and pipeline_errors:
        subject += " (new video + alerts)"
    elif new_video:
        subject += " (new video)"
    else:
        subject += " (alerts only)"

    body = build_email_body(new_video, pipeline_errors)
    sent = send_email(subject, body)

    if not sent:
        # Don't update state if the email failed — we want this video
        # (and these errors) to be reported again next run, not lost.
        sys.exit(1)

    state_changed = False

    if new_video:
        processed_videos[new_video["video_id"]] = {
            "status": "queued",
            "title": new_video["title"],
            "url": new_video["url"],
            "queued_at": datetime.now(timezone.utc).isoformat(),
        }
        save_json(PROCESSED_VIDEOS_FILE, processed_videos)
        state_changed = True
        print(f"Emailed and queued: {new_video['title']}")

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
