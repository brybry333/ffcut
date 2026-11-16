"""Media inspection helpers built on ``ffprobe``/``ffmpeg``.

Everything here shells out with argv lists (never ``shell=True``) and is safe to
call from a worker thread.  Nothing in this module imports Qt.
"""

from __future__ import annotations

import concurrent.futures
import json
import os
import shutil
import subprocess
import threading
from collections.abc import Sequence
from dataclasses import dataclass

__all__ = [
    "AudioFileInfo",
    "AudioTrack",
    "MediaInfo",
    "ProbeError",
    "cancel_all",
    "make_proxy",
    "probe",
    "probe_audio",
    "resume",
    "thumbnails_at",
    "waveform_png",
]

FFPROBE = shutil.which("ffprobe") or "ffprobe"
FFMPEG = shutil.which("ffmpeg") or "ffmpeg"

#: Seconds allowed for a single ``ffprobe`` invocation.
PROBE_TIMEOUT = 30.0
#: Seconds allowed for a single thumbnail extraction before it is killed.
THUMB_TIMEOUT = 20.0
#: Seconds allowed for one ``showwavespic`` pass.  Measured on the 46-minute test
#: file: 1.7 s for the whole thing (audio-only decode alone is 1.5 s), so this is
#: ten times the worst case seen.
WAVEFORM_TIMEOUT = 20.0


class ProbeError(Exception):
    """Raised when a file cannot be inspected (missing, unreadable, no video)."""


@dataclass(frozen=True)
class AudioTrack:
    """One audio stream, addressable as ``0:a:{index}`` in an ffmpeg command."""

    index: int
    codec: str
    channels: int
    sample_rate: int
    language: str | None
    title: str | None

    @property
    def label(self) -> str:
        """Short human label, e.g. ``"2: aac 2ch eng (Game)"``."""
        bits = [f"{self.index + 1}:", self.codec, f"{self.channels}ch"]
        if self.language:
            bits.append(self.language)
        if self.title:
            bits.append(f"({self.title})")
        return " ".join(bits)


@dataclass(frozen=True)
class MediaInfo:
    """Everything the renderer and the GUI need to know about an input file."""

    path: str
    duration: float
    width: int
    height: int
    fps: float
    vcodec: str
    audio: list[AudioTrack]


@dataclass(frozen=True)
class AudioFileInfo:
    """A standalone audio file dropped onto the timeline as an extra lane.

    Distinct from :class:`AudioTrack`, which addresses a stream *inside* the
    loaded video.  An external file becomes its own ffmpeg input, so what
    matters here is the path (not a stream index) and the duration — the
    renderer pads a short one out to the video's length so every keep-segment
    still has audio to concat.
    """

    path: str
    duration: float
    codec: str
    channels: int
    sample_rate: int

    @property
    def label(self) -> str:
        """Short human label for the lane header, e.g. ``"bgm.mp3 · mp3 2ch"``."""
        name = os.path.basename(self.path) or self.path
        return f"{name} · {self.codec} {self.channels}ch"


# --------------------------------------------------------------------------- #
# Cancellation
# --------------------------------------------------------------------------- #
# ``subprocess.run`` cannot be interrupted from another thread, so a GUI closing
# mid-probe had no way to do anything but wait the child out -- up to the
# timeouts above, with the GUI thread joined against it.  Every child spawned
# here is therefore registered while it lives, so :func:`cancel_all` can kill the
# whole set at once and the worker falls out of ``communicate`` in milliseconds.
_LOCK = threading.Lock()
_LIVE: set[subprocess.Popen] = set()
_CANCELLED = False


def cancel_all() -> None:
    """Kill every live ffprobe/ffmpeg child and refuse to start new ones.

    Called from the GUI thread during teardown.  Idempotent.  Afterwards the
    helpers in this module raise :class:`ProbeError` instead of doing work,
    until :func:`resume` clears the flag.
    """
    global _CANCELLED
    with _LOCK:
        _CANCELLED = True
        live = list(_LIVE)
    for proc in live:
        try:
            proc.kill()  # SIGKILL: ffmpeg can sit on SIGTERM while it flushes
        except OSError:
            pass  # already reaped by the thread that owns it


def resume() -> None:
    """Clear the :func:`cancel_all` flag (tests, or a second window per run)."""
    global _CANCELLED
    with _LOCK:
        _CANCELLED = False


def _spawn(
    argv: list[str],
    timeout: float,
    *,
    stdout: int = subprocess.PIPE,
    stderr: int = subprocess.PIPE,
) -> subprocess.CompletedProcess[str]:
    """``subprocess.run``, but the child is killable by :func:`cancel_all`.

    Raises :class:`ProbeError` if cancellation is already in effect, and lets
    ``FileNotFoundError``/``TimeoutExpired`` through for callers to translate.
    """
    with _LOCK:
        if _CANCELLED:
            raise ProbeError("cancelled")
    proc = subprocess.Popen(
        argv, stdout=stdout, stderr=stderr, stdin=subprocess.DEVNULL, text=True
    )
    with _LOCK:
        raced = _CANCELLED  # cancel_all ran between the check and the spawn
        if not raced:
            _LIVE.add(proc)
    if raced:
        proc.kill()
        proc.communicate()
        raise ProbeError("cancelled")
    try:
        out, err = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.communicate()
        raise
    finally:
        with _LOCK:
            _LIVE.discard(proc)
    return subprocess.CompletedProcess(argv, proc.returncode, out, err)


def _run(argv: list[str], timeout: float) -> subprocess.CompletedProcess[str]:
    """Run ``argv`` capturing text output, translating failures to ProbeError."""
    try:
        return _spawn(argv, timeout)
    except FileNotFoundError as exc:  # ffprobe/ffmpeg not installed
        raise ProbeError(f"{argv[0]} not found on PATH") from exc
    except subprocess.TimeoutExpired as exc:
        raise ProbeError(f"{argv[0]} timed out after {timeout:g}s") from exc


def _parse_rate(value: str | None) -> float:
    """Parse an ffprobe rational such as ``"120/1"`` into a float (0.0 if bad)."""
    if not value:
        return 0.0
    try:
        if "/" in value:
            num, _, den = value.partition("/")
            d = float(den)
            return float(num) / d if d else 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _as_float(value: object, default: float = 0.0) -> float:
    try:
        f = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    return default if f != f else f  # reject NaN


def _as_int(value: object, default: int = 0) -> int:
    try:
        return int(float(value))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


#: MP4/MKV handler names that carry no information worth showing a user.
_BORING_HANDLERS = {
    "soundhandler",
    "videohandler",
    "subtitlehandler",
    "sound media handler",
    "video media handler",
    "gpac iso audio handler",
    "gpac iso video handler",
    "mainconcept mp4 sound media handler",
    "mainconcept mp4 video media handler",
    "bento4 sound handler",
    "core media audio",
    "core media video",
}


def _useful_title(tags: dict[str, str]) -> str | None:
    """Prefer a real ``title`` tag; fall back to a non-boilerplate handler name.

    MP4 is the awkward one: a per-track title (``-metadata:s:a:N title=Mic``,
    or a track name typed into OBS) is stored in the track's ``name`` atom, and
    ffprobe reports that back as ``tag:name`` — not ``title``.  MKV keeps it as
    ``title``.  Check both, then the handler name.
    """
    for key in ("title", "name"):
        title = (tags.get(key) or "").strip()
        if title:
            return title
    handler = (tags.get("handler_name") or "").strip()
    if handler and handler.lower() not in _BORING_HANDLERS:
        return handler
    return None


def _duration(fmt: dict, stream: dict) -> float:
    """The container's duration, else the stream's own; 0.0 if neither."""
    duration = _as_float(fmt.get("duration"))
    return duration if duration > 0 else _as_float(stream.get("duration"))


def _ffprobe_json(path: str) -> dict:
    """Run ffprobe over ``path`` and hand back its parsed ``-show_*`` output."""
    if not path:
        raise ProbeError("no path given")
    if not os.path.isfile(path):
        raise ProbeError(f"not a file: {path}")

    proc = _run(
        [
            FFPROBE,
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            path,
        ],
        PROBE_TIMEOUT,
    )
    if proc.returncode != 0:
        tail = (proc.stderr or "").strip().splitlines()
        raise ProbeError(tail[-1] if tail else f"ffprobe exited {proc.returncode}")

    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise ProbeError(f"could not parse ffprobe output: {exc}") from exc
    return data if isinstance(data, dict) else {}


def _is_real_video(stream: dict) -> bool:
    """True for an actual moving-picture stream, False for cover art.

    An mp3 with embedded album art carries a ``mjpeg``/``png`` video stream
    flagged ``attached_pic``.  Treating that as video would make every tagged
    music file look like a one-frame movie to :func:`probe`.
    """
    if stream.get("codec_type") != "video":
        return False
    disposition = stream.get("disposition") or {}
    if _as_int(disposition.get("attached_pic"), 0):
        return False
    return str(stream.get("codec_name") or "").lower() not in ("mjpeg", "png", "bmp", "gif")


def probe_audio(path: str) -> AudioFileInfo:
    """Inspect a standalone audio file (mp3/wav/flac/opus/…).

    Uses the same ffprobe machinery — and therefore the same cancellation
    discipline — as :func:`probe`, but requires an *audio* stream instead of a
    video one, so it accepts files :func:`probe` rejects.

    Raises:
        ProbeError: file missing, ffprobe failed, no audio stream, or no
            usable duration (an audio lane with no length cannot be laid out).
    """
    return _audio_from(_ffprobe_json(path), path)


def _audio_from(data: dict, path: str) -> AudioFileInfo:
    """Build an :class:`AudioFileInfo` from parsed ffprobe output."""
    streams = data.get("streams") or []
    fmt = data.get("format") or {}

    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    if audio is None:
        raise ProbeError(f"no audio stream in {os.path.basename(path)}")

    duration = _duration(fmt, audio)
    if duration <= 0:
        # Some CBR mp3s only carry a bit count; derive from samples if we can.
        samples = _as_float(audio.get("duration_ts"))
        rate = _as_float(audio.get("sample_rate"))
        if samples > 0 and rate > 0:
            duration = samples / rate
    if duration <= 0:
        raise ProbeError(f"could not determine duration of {os.path.basename(path)}")

    return AudioFileInfo(
        path=path,
        duration=duration,
        codec=str(audio.get("codec_name") or "?"),
        channels=_as_int(audio.get("channels"), 0),
        sample_rate=_as_int(audio.get("sample_rate"), 0),
    )


def probe(path: str) -> MediaInfo:
    """Inspect ``path`` with ffprobe.

    Args:
        path: Path to a media file.

    Returns:
        A :class:`MediaInfo` describing the first video stream plus every audio
        stream, in container order.

    Raises:
        ProbeError: file missing, ffprobe failed, or the file has no video.
    """
    return _media_from(_ffprobe_json(path), path)


def _media_from(data: dict, path: str) -> MediaInfo:
    """Build a :class:`MediaInfo` from parsed ffprobe output."""
    streams = data.get("streams") or []
    fmt = data.get("format") or {}

    video = next((s for s in streams if _is_real_video(s)), None)
    if video is None:
        raise ProbeError(f"no video stream in {os.path.basename(path)}")

    fps = _parse_rate(video.get("r_frame_rate")) or _parse_rate(
        video.get("avg_frame_rate")
    )

    duration = _duration(fmt, video)
    if duration <= 0:
        # Last resort: nb_frames / fps.
        frames = _as_float(video.get("nb_frames"))
        if frames > 0 and fps > 0:
            duration = frames / fps
    if duration <= 0:
        raise ProbeError(f"could not determine duration of {os.path.basename(path)}")

    audio: list[AudioTrack] = []
    for i, s in enumerate(st for st in streams if st.get("codec_type") == "audio"):
        tags = s.get("tags") or {}
        lang = (tags.get("language") or "").strip()
        audio.append(
            AudioTrack(
                index=i,
                codec=str(s.get("codec_name") or "?"),
                channels=_as_int(s.get("channels"), 0),
                sample_rate=_as_int(s.get("sample_rate"), 0),
                language=None if lang.lower() in ("", "und", "unknown") else lang,
                title=_useful_title(tags),
            )
        )

    return MediaInfo(
        path=path,
        duration=duration,
        width=_as_int(video.get("width")),
        height=_as_int(video.get("height")),
        fps=fps,
        vcodec=str(video.get("codec_name") or "?"),
        audio=audio,
    )


def _thumb_argv(path: str, ts: float, width: int, out: str, *, fast: bool) -> list[str]:
    """argv for a single input-seek thumbnail grab.

    ``fast`` drops accurate seeking: the grab lands on the nearest keyframe, which
    is plenty for a strip and avoids decoding forward from it.
    """
    seek = ["-noaccurate_seek"] if fast else []
    return [
        FFMPEG,
        "-hide_banner",
        "-nostdin",
        "-loglevel",
        "error",
        *seek,
        "-ss",
        f"{ts:.3f}",  # BEFORE -i => input seek, no linear decode
        "-i",
        path,
        "-map",
        "0:v:0",
        "-an",
        "-sn",
        "-dn",
        "-frames:v",
        "1",
        "-vf",
        f"scale={width}:-2:flags=fast_bilinear",
        "-q:v",
        "4",
        "-threads",
        "1",
        "-y",
        out,
    ]


def _try_grab(path: str, ts: float, width: int, out: str, *, fast: bool) -> bool:
    """Run one extraction; True if a non-empty file landed at ``out``."""
    try:
        proc = _spawn(
            _thumb_argv(path, ts, width, out, fast=fast),
            THUMB_TIMEOUT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.TimeoutExpired, ProbeError):
        return False
    if proc.returncode != 0:
        return False
    try:
        return os.path.getsize(out) > 0
    except OSError:
        return False


def _grab(path: str, ts: float, width: int, out: str) -> str | None:
    """Extract one thumbnail, returning its path or None.

    Falls back to an accurate seek if the fast keyframe seek came up empty, which
    happens near the tail of some files.  The GUI spaces thumbnails evenly by list
    position, so a dropped entry shifts the strip — worth one retry to avoid.
    """
    if _try_grab(path, ts, width, out, fast=True):
        return out
    if _try_grab(path, ts, width, out, fast=False):
        return out
    return None


def make_proxy(
    path: str, out: str, height: int, device: str, timeout: float
) -> str:
    """Write a light preview copy of ``path`` to ``out``: ``height`` px, at most
    60 fps, a keyframe every 30 frames, every audio track copied untouched.

    The preview plays this instead of a 1440p120 high-bitrate source: measured
    on a 172 Mbps replay, playback went from ~100 % of a core to ~14 % (720p).
    Same timestamps and the same audio streams, so seeks and lane tracks map
    1:1.  The GPU (VA-API on ``device``) does decode, scale and encode; when
    that fails the CPU does it.  Written to a ``.part`` first and renamed, so a
    killed job never leaves a half file that would later be taken for done.
    """
    part = f"{out}.part.mkv"
    tail = ["-map", "0:v:0", "-map", "0:a?", "-fpsmax", "60", "-g", "30", "-c:a", "copy", part]
    base = [FFMPEG, "-hide_banner", "-nostdin", "-loglevel", "error", "-y"]
    gpu = base + [
        "-hwaccel", "vaapi", "-hwaccel_device", device, "-hwaccel_output_format", "vaapi",
        "-i", path, "-vf", f"scale_vaapi=w=-2:h={int(height)}:format=nv12",
        "-c:v", "h264_vaapi", "-qp", "30", *tail,
    ]
    cpu = base + [
        "-i", path, "-vf", f"scale=-2:{int(height)}",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "28", "-pix_fmt", "yuv420p", *tail,
    ]
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    proc = None
    for argv in (gpu, cpu):
        proc = _run(argv, timeout)
        if proc.returncode == 0 and os.path.getsize(part) > 0:
            os.replace(part, out)
            return out
    try:
        os.remove(part)
    except OSError:
        pass
    tail_lines = ((proc.stderr if proc else "") or "").strip().splitlines()
    raise ProbeError(tail_lines[-1] if tail_lines else "preview copy failed")


def thumbnails_at(
    path: str,
    out_dir: str,
    times: Sequence[float],
    width: int = 160,
    fill_gaps: bool = False,
    workers: int = 8,
) -> list[str]:
    """Extract one JPEG per timestamp in ``times`` (source seconds, any order).

    The caller picks the timestamps: evenly spaced points on an *edited-time*
    timeline map to unevenly spaced points in the source as soon as a cut
    exists, so only the widget can say where the thumbs belong.  Each grab is
    one short-lived ffmpeg with ``-ss`` *before* ``-i`` (input seek) in a small
    thread pool; the file is never decoded linearly, so a 46-minute source costs
    about the same as a 46-second one.

    Args:
        path: Source media file.
        out_dir: Directory to write JPEGs into (created if missing).
        times: Source timestamps, one per wanted thumbnail.
        width: Thumbnail width in pixels.
        workers: At most this many ffmpeg grabs at once (Preferences ▸
            Background jobs): each one decodes a full-size frame.
        fill_gaps: Repeat the nearest successful grab in place of a failed one,
            so the result is exactly ``len(times)`` long.  Callers that space
            thumbnails by list position (the timeline lane does) need this —
            otherwise one failed grab shifts every frame after it.

    Returns:
        Paths in the same order as ``times``.  Without ``fill_gaps`` failures are
        dropped, so the list may be shorter (never longer, never reordered);
        with it, the list is short only if *every* grab failed.
    """
    if not times:
        return []
    if width <= 0:
        width = 160
    try:
        os.makedirs(out_dir, exist_ok=True)
    except OSError as exc:
        raise ProbeError(f"cannot create {out_dir}: {exc}") from exc

    jobs = [
        (i, max(0.0, float(ts)), os.path.join(out_dir, f"thumb_{i:04d}.jpg"))
        for i, ts in enumerate(times)
    ]
    workers = max(1, min(int(workers), len(jobs), (os.cpu_count() or 4)))
    results: list[str | None] = [None] * len(jobs)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_grab, path, ts, width, out): i for i, ts, out in jobs}
        for fut in concurrent.futures.as_completed(futures):
            results[futures[fut]] = fut.result()
    if not fill_gaps:
        return [p for p in results if p]
    return _fill_gaps(results)


def _fill_gaps(results: list[str | None]) -> list[str]:
    """Replace each failed grab with the nearest successful one, keeping length."""
    if not any(results):
        return []
    filled: list[str] = []
    last: str | None = None
    for item in results:
        if item:
            last = item
        filled.append(item or last or "")
    # Anything before the first success still has no stand-in: back-fill it.
    first = next(p for p in results if p)
    return [p or first for p in filled]


def waveform_png(
    path: str,
    track_index: int,
    out_path: str,
    width: int,
    height: int,
    start: float = 0.0,
    end: float | None = None,
    color: str = "#4aa3df",
    scale: str = "sqrt",
) -> str:
    """Render one audio track as a ``showwavespic`` PNG.

    ``aformat`` forces mono first so a stereo track produces a single centred
    trace instead of showwavespic's stacked per-channel bands (which do not
    survive being squashed into a short lane).

    ``scale`` defaults to ``sqrt``, not ``lin``.  Linear amplitude over a whole
    46-minute file draws game audio as a flat hairline — you cannot tell which
    track carries speech, which was the point of having lanes.  ``sqrt`` lifts
    a -20 dBFS peak from 10% of the lane height to 32% while keeping the peaks
    ordered, so the shape is readable at full-file zoom.  The lane applies a
    further per-lane peak normalisation on top (see ``timeline._wave_crop``).

    ``start``/``end`` are source seconds; the seek is an *input* seek so a ranged
    request for a zoomed lane costs a fraction of the whole file.  Whole-file on
    the 46-minute test source measures 1.7 s, audio-only decode 1.5 s, so even
    the unranged call is cheap.  Returns ``out_path``; raises ProbeError when
    ffmpeg failed, timed out, or wrote nothing.
    """
    width = max(16, int(width))
    height = max(8, int(height))
    lo = max(0.0, float(start))
    argv = [
        FFMPEG,
        "-hide_banner",
        "-nostdin",
        "-loglevel",
        "error",
        "-ss",
        f"{lo:.3f}",  # BEFORE -i: input seek, no linear decode
    ]
    if end is not None and float(end) > lo:
        argv += ["-t", f"{float(end) - lo:.3f}"]
    argv += [
        "-i",
        path,
        # The track is selected *inside* the graph ([0:a:N]); an outer -map would
        # also hand the raw audio stream to the PNG muxer ("Encoder not found").
        "-filter_complex",
        f"[0:a:{int(track_index)}]aformat=channel_layouts=mono,"
        f"showwavespic=s={width}x{height}:colors={color}:scale={scale}[wave]",
        "-map",
        "[wave]",
        "-an",
        "-frames:v",
        "1",
        "-y",
        out_path,
    ]
    proc = _run(argv, WAVEFORM_TIMEOUT)
    if proc.returncode != 0 or not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
        tail = (proc.stderr or "").strip().splitlines()
        raise ProbeError(
            tail[-1] if tail else f"waveform for 0:a:{track_index} produced nothing"
        )
    return out_path
