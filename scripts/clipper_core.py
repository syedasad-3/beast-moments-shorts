"""
Clipper Core — runs on GitHub Actions on a schedule.

Pulls the next raw video from Drive's "incoming" folder, transcribes it,
picks the best short-form moments, renders vertical face-tracked clips,
attaches MrBeast's official Hindi dub track where available, and delivers
the finished clips to Drive's "ready_to_upload" folder (with a suggested
title/description per clip) for the operator to manually download and
publish to YouTube themselves. The raw source file is then moved to
"processed" in Drive. This pipeline never calls the YouTube API — YouTube
publishing stays a fully manual, human step by design.

Pairing raw Drive files with queued videos:
  Scout writes an entry with status "queued" to state/processed_videos.json
  when it emails a video link. The operator manually downloads and uploads
  that video to Drive. This script pairs Drive files in "incoming" with
  queued entries in upload order (oldest Drive file <-> oldest queued
  entry). This assumes the operator uploads videos in the order Scout
  queued them, which holds for normal single-operator use. If the counts
  don't match, it's logged as a warning (not fatal) and only the videos
  that can be safely paired are processed; the rest wait for next run.
"""

import io
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATE_DIR = os.path.join(REPO_ROOT, "state")
PROCESSED_VIDEOS_FILE = os.path.join(STATE_DIR, "processed_videos.json")
PIPELINE_ERRORS_FILE = os.path.join(STATE_DIR, "pipeline_errors.json")
WORK_DIR = os.environ.get("CLIPPER_WORK_DIR", "/tmp/clipper_work")

MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = [5, 15, 30]

# How many short moments to try to extract per source video.
NUM_MOMENTS = 3
MOMENT_MIN_SECONDS = 25
MOMENT_MAX_SECONDS = 59  # keep under YouTube Shorts' 60s cutoff with margin

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
DRIVE_INCOMING_FOLDER_ID = os.environ.get("DRIVE_INCOMING_FOLDER_ID")
DRIVE_PROCESSED_FOLDER_ID = os.environ.get("DRIVE_PROCESSED_FOLDER_ID")
DRIVE_OUTPUT_FOLDER_ID = os.environ.get("DRIVE_OUTPUT_FOLDER_ID")


# ---------------------------------------------------------------------------
# State + error logging (shared with scout_monitor.py's format)
# ---------------------------------------------------------------------------

def load_json(path, default):
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print(f"ERR_CORRUPT_STATE_FILE: {path} is not valid JSON: {e}")
        sys.exit(1)


def save_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def log_pipeline_error(error_code, stage, message):
    """Append an error to the shared error log that Scout+Monitor will
    email out on its next run. Never raises — logging a failure should
    never itself crash the pipeline."""
    try:
        errors = load_json(PIPELINE_ERRORS_FILE, [])
        errors.append({
            "error_code": error_code,
            "stage": stage,
            "message": str(message)[:500],  # cap length, keep emails readable
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        save_json(PIPELINE_ERRORS_FILE, errors)
    except Exception as e:
        print(f"ERR_FAILED_TO_LOG_ERROR: could not write to error log: {e}")


def with_retries(fn, error_code, stage, *args, **kwargs):
    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            last_error = e
            print(f"{error_code}: attempt {attempt + 1}/{MAX_RETRIES} failed: {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_BACKOFF_SECONDS[attempt])
    log_pipeline_error(error_code, stage, f"All {MAX_RETRIES} attempts failed: {last_error}")
    return None


# ---------------------------------------------------------------------------
# Transcription (Groq-hosted Whisper)
# ---------------------------------------------------------------------------

def transcribe_audio(audio_path):
    """Returns Groq's verbose_json transcript (segments with timestamps),
    or None on failure (already logged)."""
    if not GROQ_API_KEY:
        log_pipeline_error("ERR_MISSING_CONFIG", "transcribe", "GROQ_API_KEY not set.")
        return None

    def _call():
        import requests
        with open(audio_path, "rb") as f:
            resp = requests.post(
                "https://api.groq.com/openai/v1/audio/transcriptions",
                headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                files={"file": (os.path.basename(audio_path), f, "audio/mp4")},
                data={
                    "model": "whisper-large-v3",
                    "response_format": "verbose_json",
                    "timestamp_granularities[]": "segment",
                },
                timeout=300,
            )
        if resp.status_code == 429:
            raise RuntimeError(f"ERR_QUOTA_EXCEEDED: Groq rate limit hit: {resp.text[:200]}")
        resp.raise_for_status()
        return resp.json()

    return with_retries(_call, "ERR_TRANSCRIBE_FAIL", "transcribe")


# ---------------------------------------------------------------------------
# Moment picking (Groq LLM chooses the best segments, re-anchored to
# sentence boundaries so clips never start/end mid-sentence)
# ---------------------------------------------------------------------------

def merge_segments_into_blocks(segments, max_blocks=50, max_words_per_block=18):
    """Merges Whisper's many small segments (typically 3-7s each) into
    larger blocks before sending to the LLM. A 20-40 minute video can
    produce 300+ raw segments — sending all of them (with full text) as
    separate numbered lines blows past Groq's free-tier token-per-minute
    limit in a single request (openai/gpt-oss-120b's free tier is only
    8,000 TPM — confirmed in production as a recurring 413 error on
    longer videos). Two things keep the prompt small regardless of video
    length: (1) grouping into a bounded number of time-blocks, and
    (2) truncating each block's text to a short preview, since the LLM
    only needs enough of the dialogue to judge whether a moment is
    exciting — not a verbatim transcript.

    Block start/end times are still exact original Whisper segment
    boundaries (real pauses), so cut-point accuracy is unaffected by the
    text truncation — only how much of the spoken content the LLM sees
    when deciding which blocks to pick."""
    if not segments:
        return []

    total_duration = segments[-1]["end"] - segments[0]["start"]
    target_block_seconds = max(15, total_duration / max_blocks)

    blocks = []
    current_start, current_end, current_texts = None, None, []
    for seg in segments:
        if current_start is None:
            current_start = seg["start"]
        current_end = seg["end"]
        current_texts.append(seg["text"].strip())
        if current_end - current_start >= target_block_seconds:
            full_text = " ".join(current_texts)
            words = full_text.split()
            preview = " ".join(words[:max_words_per_block])
            if len(words) > max_words_per_block:
                preview += " ..."
            blocks.append({"start": current_start, "end": current_end, "text": preview})
            current_start, current_end, current_texts = None, None, []

    if current_texts:  # leftover partial block at the end
        full_text = " ".join(current_texts)
        words = full_text.split()
        preview = " ".join(words[:max_words_per_block])
        if len(words) > max_words_per_block:
            preview += " ..."
        blocks.append({"start": current_start, "end": current_end, "text": preview})

    return blocks


def build_transcript_text_with_ids(segments):
    """Turns Whisper segments into a numbered list the LLM can reference
    by index, so we can map its picks back to exact timestamps without
    relying on it to reproduce numbers accurately."""
    lines = []
    for i, seg in enumerate(segments):
        lines.append(f"[{i}] ({seg['start']:.1f}-{seg['end']:.1f}) {seg['text'].strip()}")
    return "\n".join(lines)


def pick_moments_with_llm(raw_segments):
    """Asks Groq's LLM to pick NUM_MOMENTS engaging, self-contained moments
    from the transcript, referencing block indices (not free-text
    timestamps) to avoid the LLM inventing timestamps that don't exist.
    Segments are merged into larger blocks first to stay within Groq's
    free-tier per-request token limits on long videos.
    Returns a list of {start, end, reason} or None."""
    if not GROQ_API_KEY:
        log_pipeline_error("ERR_MISSING_CONFIG", "pick_moments", "GROQ_API_KEY not set.")
        return None
    if not raw_segments:
        log_pipeline_error("ERR_EMPTY_TRANSCRIPT", "pick_moments", "No transcript segments to pick from.")
        return None

    segments = merge_segments_into_blocks(raw_segments)

    transcript_text = build_transcript_text_with_ids(segments)
    prompt = f"""You are selecting clips for a YouTube Shorts channel from a MrBeast video transcript.
The transcript below is numbered by segment. Pick {NUM_MOMENTS} distinct, exciting, self-contained
moments that would work as standalone {MOMENT_MIN_SECONDS}-{MOMENT_MAX_SECONDS} second short-form clips.
Each moment must start and end on a segment boundary (a full sentence/thought) — never pick a
start or end segment that cuts a sentence in half.
Moments must not overlap. Prefer moments with a clear hook, twist, or payoff.

Transcript:
{transcript_text}

Respond ONLY with valid JSON, no other text, in this exact format:
{{"moments": [{{"start_segment": <int>, "end_segment": <int>, "reason": "<short reason>"}}]}}
"""

    def _call():
        import requests
        resp = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            json={
                "model": "openai/gpt-oss-120b",  # llama-3.3-70b-versatile was deprecated by Groq (June 2026)
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
                "response_format": {"type": "json_object"},
            },
            timeout=60,
        )
        if resp.status_code == 429:
            raise RuntimeError(f"ERR_QUOTA_EXCEEDED: Groq rate limit hit: {resp.text[:200]}")
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        return json.loads(content)

    result = with_retries(_call, "ERR_MOMENT_PICK_FAIL", "pick_moments")
    if result is None:
        return None

    moments = []
    for m in result.get("moments", []):
        try:
            start_idx = int(m["start_segment"])
            end_idx = int(m["end_segment"])
        except (KeyError, ValueError, TypeError):
            continue  # skip malformed entries rather than crash the whole run
        if not (0 <= start_idx < len(segments)) or not (0 <= end_idx < len(segments)):
            continue  # LLM hallucinated an out-of-range index — skip it
        if end_idx < start_idx:
            continue
        start_time = segments[start_idx]["start"]
        end_time = segments[end_idx]["end"]
        duration = end_time - start_time
        if duration < MOMENT_MIN_SECONDS - 5 or duration > MOMENT_MAX_SECONDS + 10:
            # LLM picked something wildly outside our target length — skip
            # rather than publish an unusably short/long clip.
            continue
        moments.append({
            "start": start_time,
            "end": end_time,
            "reason": m.get("reason", ""),
        })

    if not moments:
        log_pipeline_error(
            "ERR_NO_VALID_MOMENTS",
            "pick_moments",
            "LLM returned no usable moments after validation (all filtered out).",
        )
        return None

    return moments


# ---------------------------------------------------------------------------
# Face detection + vertical crop window (MediaPipe)
# ---------------------------------------------------------------------------
#
# Design decision: we compute ONE static horizontal crop offset per moment
# (not frame-by-frame dynamic tracking). We sample faces across several
# frames within the moment and take the median center, which is far more
# robust to a single bad detection than the mean, and avoids the jitter/
# complexity/render cost of continuously re-cropping every frame. This is
# a reasonable trade-off for fast-cut MrBeast-style footage where the
# subject is usually roughly centered within any given shot.

def get_video_dimensions(video_path):
    def _call():
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height", "-of", "json", video_path],
            capture_output=True, text=True, timeout=30, check=True,
        )
        info = json.loads(result.stdout)["streams"][0]
        return info["width"], info["height"]

    return with_retries(_call, "ERR_FFPROBE_FAIL", "crop_detection")


FACE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite"
FACE_MODEL_PATH = os.path.join(WORK_DIR, "blaze_face_short_range.tflite")


def ensure_face_model_downloaded():
    """Downloads MediaPipe's face detection model on first use. Newer
    MediaPipe versions (Tasks API) require this model file explicitly —
    it's no longer bundled like the older `mp.solutions` API. Returns the
    local path, or None if the download failed (caller falls back to
    center-crop rather than crashing the whole pipeline over this)."""
    if os.path.exists(FACE_MODEL_PATH) and os.path.getsize(FACE_MODEL_PATH) > 1000:
        return FACE_MODEL_PATH

    def _call():
        import requests
        os.makedirs(WORK_DIR, exist_ok=True)
        resp = requests.get(FACE_MODEL_URL, timeout=60)
        resp.raise_for_status()
        if len(resp.content) < 1000:
            raise RuntimeError(f"Downloaded model file suspiciously small ({len(resp.content)} bytes) — likely blocked or wrong URL.")
        with open(FACE_MODEL_PATH, "wb") as f:
            f.write(resp.content)
        return FACE_MODEL_PATH

    return with_retries(_call, "ERR_FACE_MODEL_DOWNLOAD_FAIL", "crop_detection")


def sample_frame_face_centers(video_path, start, end, width, height, num_samples=6):
    """Extracts a handful of frames within [start, end] and runs MediaPipe
    face detection on each. Returns a list of normalized x-centers (0-1)
    of detected faces. Never raises — a detection failure just means fewer
    samples, handled by the caller falling back to center-crop."""
    model_path = ensure_face_model_downloaded()
    if not model_path:
        return []  # already logged; caller falls back to center-crop

    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    try:
        import cv2
    except ImportError:
        log_pipeline_error("ERR_MISSING_DEPENDENCY", "crop_detection", "opencv-python not installed.")
        return []

    centers = []
    duration = end - start
    sample_times = [start + duration * (i + 0.5) / num_samples for i in range(num_samples)]

    base_options = mp_python.BaseOptions(model_asset_path=model_path)
    options = mp_vision.FaceDetectorOptions(base_options=base_options, min_detection_confidence=0.5)

    try:
        detector = mp_vision.FaceDetector.create_from_options(options)
    except Exception as e:
        log_pipeline_error("ERR_FACE_DETECTOR_INIT_FAIL", "crop_detection", str(e))
        return []

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        log_pipeline_error("ERR_VIDEO_OPEN_FAIL", "crop_detection", f"Could not open {video_path}")
        detector.close()
        return []

    try:
        for t in sample_times:
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            try:
                result = detector.detect(mp_image)
            except Exception as e:
                print(f"Face detection failed on one frame (continuing): {e}")
                continue
            if result.detections:
                # Use the largest detected face (most likely the main subject).
                best = max(
                    result.detections,
                    key=lambda d: d.bounding_box.width * d.bounding_box.height,
                )
                box = best.bounding_box
                center_x_norm = (box.origin_x + box.width / 2) / width
                centers.append(center_x_norm)
    finally:
        cap.release()
        detector.close()

    return centers


def compute_target_crop_width(src_width, src_height):
    """9:16 crop width for the given source height, rounded down to an
    even number — libx264 requires even dimensions (odd widths either
    fail outright or get silently rounded by ffmpeg, which would make
    our offset math inconsistent with what actually gets rendered)."""
    crop_width = int(src_height * 9 / 16)
    crop_width -= crop_width % 2
    return min(crop_width, src_width - (src_width % 2))


def compute_crop_x_offset(video_path, start, end, src_width, src_height):
    """Returns the pixel x-offset for a 9:16 crop window (full source
    height, width = height*9/16, rounded to even), clamped to stay within
    the frame. Falls back to a centered crop if no faces were detected."""
    crop_width = compute_target_crop_width(src_width, src_height)
    max_offset = src_width - crop_width

    centers = sample_frame_face_centers(video_path, start, end, src_width, src_height)
    if not centers:
        print("No faces detected in sampled frames — falling back to center crop.")
        return max_offset // 2

    import statistics
    median_center_norm = statistics.median(centers)  # 0-1, normalized
    target_center_px = median_center_norm * src_width
    offset = int(target_center_px - crop_width / 2)
    offset = max(0, min(offset, max_offset))  # clamp to valid range
    return offset


# ---------------------------------------------------------------------------
# Rendering (FFmpeg: crop to vertical 9:16, cut to the moment, mux audio)
# ---------------------------------------------------------------------------

def render_vertical_clip(video_path, start, end, crop_x_offset, src_width, src_height,
                          output_path, audio_path=None):
    """Cuts [start, end] from video_path, crops to a 9:16 vertical window
    at crop_x_offset, and renders to output_path. If audio_path is given,
    that audio (e.g. the Hindi dub track) replaces the video's own audio;
    otherwise the video's original audio is kept.
    Returns True on success (already logged on failure)."""
    crop_width = compute_target_crop_width(src_width, src_height)
    duration = end - start

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start), "-i", video_path,
    ]
    if audio_path:
        cmd += ["-ss", str(start), "-i", audio_path]

    cmd += [
        "-t", str(duration),
        "-vf", f"crop={crop_width}:{src_height}:{crop_x_offset}:0",
    ]

    if audio_path:
        # Input 0 = video (take video stream only), input 1 = replacement audio.
        cmd += ["-map", "0:v:0", "-map", "1:a:0"]
    else:
        cmd += ["-map", "0:v:0", "-map", "0:a:0?"]  # ? = audio optional, don't fail if silent source

    cmd += [
        "-c:v", "libx264", "-preset", "medium", "-crf", "20",
        "-c:a", "aac", "-b:a", "128k",
        "-movflags", "+faststart",
        output_path,
    ]

    def _call():
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg exited {result.returncode}: {result.stderr[-1000:]}")
        if not os.path.exists(output_path) or os.path.getsize(output_path) < 1000:
            raise RuntimeError(f"Output file missing or suspiciously small: {output_path}")
        return True

    result = with_retries(_call, "ERR_RENDER_FAIL", "render")
    return result is True


# ---------------------------------------------------------------------------
# Hindi audio track detection
# ---------------------------------------------------------------------------

def find_hindi_audio_track_index(video_path):
    """Checks the downloaded video's audio streams for a Hindi-tagged
    track (MrBeast's official dub, present as an alternate audio track on
    the same video file). Returns the ffmpeg stream index (e.g. 'a:1') or
    None if no Hindi track is present — that's an expected, non-error
    condition (not every video has a dub yet), so this only logs at info
    level, not as a pipeline error."""

    def _call():
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "stream=index:stream_tags=language",
             "-select_streams", "a", "-of", "json", video_path],
            capture_output=True, text=True, timeout=30, check=True,
        )
        return json.loads(result.stdout).get("streams", [])

    streams = with_retries(_call, "ERR_FFPROBE_FAIL", "hindi_track_check")
    if not streams:
        return None

    audio_streams = [s for s in streams]
    for i, s in enumerate(audio_streams):
        lang = s.get("tags", {}).get("language", "").lower()
        if lang in ("hi", "hin", "hindi"):
            return i  # position among audio streams, for ffmpeg's a:N selector

    print("No Hindi audio track found on this video — English-only short will be published.")
    return None


def extract_audio_track(video_path, audio_stream_index, output_path):
    """Extracts a single audio stream (by its position among audio
    streams, e.g. 0 for a:0) to its own file, for use as a replacement
    track in render_vertical_clip."""
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-map", f"0:a:{audio_stream_index}",
        "-c:a", "aac", "-b:a", "128k",
        output_path,
    ]

    def _call():
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg exited {result.returncode}: {result.stderr[-500:]}")
        return True

    result = with_retries(_call, "ERR_AUDIO_EXTRACT_FAIL", "hindi_extract")
    return result is True


# ---------------------------------------------------------------------------
# Google Drive (download from "incoming", move to "processed")
# ---------------------------------------------------------------------------

def get_drive_credentials():
    """Uses a Service Account (not user OAuth) for Drive access. This
    avoids the drive.file scope's fundamental limitation — it can only
    see files the app itself created, never files a human manually
    uploaded via the Drive website. Service accounts are granted access
    by explicitly sharing folders with them, which sidesteps that
    restriction entirely and needs no OAuth consent-screen verification."""
    from google.oauth2 import service_account

    service_account_json = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON")
    if not service_account_json:
        log_pipeline_error("ERR_MISSING_CONFIG", "drive_auth", "GOOGLE_SERVICE_ACCOUNT_JSON not set.")
        return None

    try:
        info = json.loads(service_account_json)
    except json.JSONDecodeError as e:
        log_pipeline_error("ERR_INVALID_CONFIG", "drive_auth", f"GOOGLE_SERVICE_ACCOUNT_JSON is not valid JSON: {e}")
        return None

    try:
        creds = service_account.Credentials.from_service_account_info(
            info, scopes=["https://www.googleapis.com/auth/drive"],
        )
        return creds
    except Exception as e:
        log_pipeline_error("ERR_DRIVE_AUTH_FAIL", "drive_auth", str(e))
        return None


def list_incoming_files(drive_service):
    def _call():
        resp = drive_service.files().list(
            q=f"'{DRIVE_INCOMING_FOLDER_ID}' in parents and trashed = false",
            fields="files(id, name, createdTime)",
            orderBy="createdTime",
        ).execute()
        return resp.get("files", [])

    result = with_retries(_call, "ERR_DRIVE_LIST_FAIL", "drive_list")
    return result if result is not None else []


def download_drive_file(drive_service, file_id, dest_path):
    from googleapiclient.http import MediaIoBaseDownload

    def _call():
        request = drive_service.files().get_media(fileId=file_id)
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        with io.FileIO(dest_path, "wb") as fh:
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while not done:
                status, done = downloader.next_chunk()
        if not os.path.exists(dest_path) or os.path.getsize(dest_path) < 1000:
            raise RuntimeError(f"Downloaded file missing or too small: {dest_path}")
        return True

    result = with_retries(_call, "ERR_DRIVE_DOWNLOAD_FAIL", "drive_download")
    return result is True


def move_drive_file_to_processed(drive_service, file_id):
    def _call():
        drive_service.files().update(
            fileId=file_id,
            addParents=DRIVE_PROCESSED_FOLDER_ID,
            removeParents=DRIVE_INCOMING_FOLDER_ID,
            fields="id, parents",
        ).execute()
        return True

    result = with_retries(_call, "ERR_DRIVE_MOVE_FAIL", "drive_move")
    return result is True


# ---------------------------------------------------------------------------
# Output delivery — uploads finished shorts to a Drive folder for the
# operator to manually review, download, and publish to YouTube themselves.
# This pipeline intentionally never touches the YouTube upload API — the
# operator wanted YouTube publishing to stay a fully manual, human step.
# ---------------------------------------------------------------------------

def upload_file_to_drive(drive_service, local_path, filename, folder_id, description=None):
    """Uploads a local file into the given Drive folder. Returns the new
    file's Drive ID, or None on failure (already logged)."""
    from googleapiclient.http import MediaFileUpload

    mimetype = "video/mp4" if local_path.endswith(".mp4") else "text/plain"
    body = {"name": filename, "parents": [folder_id]}
    if description:
        body["description"] = description[:4000]  # Drive description field has a length cap

    def _call():
        media = MediaFileUpload(local_path, mimetype=mimetype, resumable=True)
        request = drive_service.files().create(body=body, media_body=media, fields="id")
        response = None
        while response is None:
            status, response = request.next_chunk()
        return response.get("id")

    result = with_retries(_call, "ERR_DRIVE_OUTPUT_UPLOAD_FAIL", "drive_output_upload")
    return result


def deliver_clip_to_drive(drive_service, video_path, base_filename, title, description, folder_id):
    """Uploads the rendered clip plus a companion .txt file containing the
    suggested title/description, so the operator has everything needed to
    publish it on YouTube without retyping anything."""
    video_uploaded = upload_file_to_drive(
        drive_service, video_path, f"{base_filename}.mp4", folder_id,
    )
    if not video_uploaded:
        return None  # already logged

    notes_path = video_path + ".notes.txt"
    with open(notes_path, "w", encoding="utf-8") as f:
        f.write(f"Suggested title:\n{title}\n\nSuggested description:\n{description}\n")
    upload_file_to_drive(drive_service, notes_path, f"{base_filename}.txt", folder_id)
    os.remove(notes_path)

    return video_uploaded


# ---------------------------------------------------------------------------
# Pairing Drive files with queued state entries (see module docstring)
# ---------------------------------------------------------------------------

def pair_incoming_files_with_queued_videos(incoming_files, processed_videos):
    queued = [
        (vid, info) for vid, info in processed_videos.items()
        if info.get("status") == "queued"
    ]
    queued.sort(key=lambda x: x[1].get("queued_at", ""))

    if len(incoming_files) != len(queued):
        log_pipeline_error(
            "ERR_INCOMING_QUEUE_MISMATCH",
            "pairing",
            f"{len(incoming_files)} file(s) in 'incoming' but {len(queued)} 'queued' entries in state. "
            f"Processing what can be safely paired; the rest will be picked up next run.",
        )

    pairs = []
    for i in range(min(len(incoming_files), len(queued))):
        pairs.append((incoming_files[i], queued[i]))
    return pairs


# ---------------------------------------------------------------------------
# Main pipeline for a single video
# ---------------------------------------------------------------------------

def process_one_video(drive_service, drive_file, queued_entry):
    video_id, video_info = queued_entry
    run_dir = os.path.join(WORK_DIR, video_id)
    os.makedirs(run_dir, exist_ok=True)
    raw_path = os.path.join(run_dir, "raw_source.mp4")
    audio_path = os.path.join(run_dir, "audio_for_transcription.m4a")

    try:
        print(f"Processing '{video_info['title']}' (Drive file: {drive_file['name']})")

        if not download_drive_file(drive_service, drive_file["id"], raw_path):
            return False  # error already logged

        # Extract audio for transcription (smaller upload to Groq than the full video).
        # 32kbps mono is plenty for speech transcription (not the final output
        # audio quality) and keeps even long videos safely under Groq's free-tier
        # 25MB per-file limit: a 60-minute video at 32kbps mono is ~14MB.
        extract_cmd = ["ffmpeg", "-y", "-i", raw_path, "-map", "0:a:0",
                        "-ac", "1", "-c:a", "aac", "-b:a", "32k", audio_path]
        result = subprocess.run(extract_cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0 or not os.path.exists(audio_path):
            log_pipeline_error("ERR_AUDIO_EXTRACT_FAIL", "process_video", result.stderr[-500:])
            return False

        audio_size_mb = os.path.getsize(audio_path) / (1024 * 1024)
        if audio_size_mb > 24:  # stay safely under Groq's 25MB free-tier limit
            log_pipeline_error(
                "ERR_AUDIO_TOO_LARGE_FOR_TRANSCRIPTION", "process_video",
                f"Extracted audio is {audio_size_mb:.1f}MB, exceeds Groq's 25MB free-tier limit. "
                f"Video is likely too long for single-request transcription; chunking not yet implemented.",
            )
            return False

        transcript = transcribe_audio(audio_path)
        if not transcript:
            return False  # error already logged

        segments = transcript.get("segments", [])
        moments = pick_moments_with_llm(segments)
        if not moments:
            return False  # error already logged

        src_width, src_height = get_video_dimensions(raw_path)
        if not src_width:
            return False  # error already logged

        hindi_track_idx = find_hindi_audio_track_index(raw_path)
        hindi_audio_full_path = None
        if hindi_track_idx is not None:
            hindi_audio_full_path = os.path.join(run_dir, "hindi_full.aac")
            if not extract_audio_track(raw_path, hindi_track_idx, hindi_audio_full_path):
                hindi_audio_full_path = None  # extraction failed — proceed English-only, already logged

        delivered_clips = []
        video_slug = re.sub(r"[^a-zA-Z0-9]+", "_", video_info["title"])[:40].strip("_")

        for i, moment in enumerate(moments):
            crop_offset = compute_crop_x_offset(raw_path, moment["start"], moment["end"], src_width, src_height)

            # English version
            en_output = os.path.join(run_dir, f"moment_{i}_en.mp4")
            if render_vertical_clip(raw_path, moment["start"], moment["end"], crop_offset,
                                     src_width, src_height, en_output):
                title = f"{video_info['title'][:80]} #Shorts"
                description = f"Clip from: {video_info['url']}\n\n{moment.get('reason', '')}"
                base_name = f"{video_slug}_moment{i}_EN"
                file_id = deliver_clip_to_drive(drive_service, en_output, base_name, title, description,
                                                 DRIVE_OUTPUT_FOLDER_ID)
                if file_id:
                    delivered_clips.append(("en", base_name))
                os.remove(en_output)  # immediate cleanup — runner disk is limited

            # Hindi version (only if a Hindi track exists for this source video)
            if hindi_audio_full_path:
                hi_segment_audio = os.path.join(run_dir, f"moment_{i}_hi_audio.aac")
                # Cut the matching time segment out of the full Hindi audio track.
                cut_cmd = ["ffmpeg", "-y", "-ss", str(moment["start"]), "-i", hindi_audio_full_path,
                           "-t", str(moment["end"] - moment["start"]), "-c:a", "aac", hi_segment_audio]
                cut_result = subprocess.run(cut_cmd, capture_output=True, text=True, timeout=120)
                if cut_result.returncode == 0:
                    hi_output = os.path.join(run_dir, f"moment_{i}_hi.mp4")
                    if render_vertical_clip(raw_path, moment["start"], moment["end"], crop_offset,
                                             src_width, src_height, hi_output, audio_path=hi_segment_audio):
                        title_hi = f"{video_info['title'][:75]} (Hindi) #Shorts"
                        description = f"Clip from: {video_info['url']}\n\n{moment.get('reason', '')}"
                        base_name = f"{video_slug}_moment{i}_HI"
                        file_id = deliver_clip_to_drive(drive_service, hi_output, base_name, title_hi, description,
                                                         DRIVE_OUTPUT_FOLDER_ID)
                        if file_id:
                            delivered_clips.append(("hi", base_name))
                        os.remove(hi_output)
                    if os.path.exists(hi_segment_audio):
                        os.remove(hi_segment_audio)
                else:
                    log_pipeline_error("ERR_HINDI_SEGMENT_CUT_FAIL", "process_video", cut_result.stderr[-300:])

        if not delivered_clips:
            log_pipeline_error("ERR_NO_CLIPS_DELIVERED", "process_video",
                                f"All render/upload attempts failed for '{video_info['title']}'.")
            return False

        if not move_drive_file_to_processed(drive_service, drive_file["id"]):
            # Non-fatal: clips are delivered, but the raw file wasn't moved.
            # Logged already; leave state as "queued" so it's investigated,
            # not silently marked done while sitting in the wrong folder.
            return False

        video_info["status"] = "done"
        video_info["delivered_clips"] = delivered_clips
        video_info["processed_at"] = datetime.now(timezone.utc).isoformat()
        print(f"Done: {len(delivered_clips)} clip(s) delivered to 'ready_to_upload' for '{video_info['title']}'.")
        return True

    finally:
        # Always clean up local files for this video, success or failure —
        # runner disk (~14GB) must not fill up across multiple videos in one run.
        if os.path.exists(run_dir):
            import shutil
            shutil.rmtree(run_dir, ignore_errors=True)


def main():
    if not (DRIVE_INCOMING_FOLDER_ID and DRIVE_PROCESSED_FOLDER_ID and DRIVE_OUTPUT_FOLDER_ID):
        print("ERR_MISSING_CONFIG: DRIVE_INCOMING_FOLDER_ID, DRIVE_PROCESSED_FOLDER_ID, or DRIVE_OUTPUT_FOLDER_ID not set.")
        sys.exit(1)

    os.makedirs(WORK_DIR, exist_ok=True)
    processed_videos = load_json(PROCESSED_VIDEOS_FILE, {})

    drive_creds = get_drive_credentials()
    if not drive_creds:
        sys.exit(1)  # already logged

    from googleapiclient.discovery import build
    drive_service = build("drive", "v3", credentials=drive_creds)

    incoming_files = list_incoming_files(drive_service)
    if not incoming_files:
        print("No files in 'incoming' folder. Nothing to do.")
        return

    pairs = pair_incoming_files_with_queued_videos(incoming_files, processed_videos)
    if not pairs:
        print("No safely pairable (file, queued-video) combinations this run.")
        save_json(PROCESSED_VIDEOS_FILE, processed_videos)  # in case pairing logged an error
        return

    any_state_changed = False
    for drive_file, queued_entry in pairs:
        success = process_one_video(drive_service, drive_file, queued_entry)
        if success:
            any_state_changed = True
        # Persist state after every video, not just at the end — if a later
        # video in the same run crashes, earlier successes aren't lost.
        save_json(PROCESSED_VIDEOS_FILE, processed_videos)

    if any_state_changed:
        print("STATE_CHANGED=true")


if __name__ == "__main__":
    main()
