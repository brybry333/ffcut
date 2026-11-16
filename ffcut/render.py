"""Command building and background rendering.

:func:`build_command` is a pure function: give it a :class:`~ffcut.probe.MediaInfo`
and a :class:`RenderSettings` and it hands back a complete argv list, with no
subprocess involved.  :class:`RenderJob` runs such an argv in a worker thread and
reports progress through plain callbacks (no Qt here — the GUI marshals them onto
the main thread itself).

The filter graph mirrors the user's ``ffcut`` shell function: ``tmix`` blends at
the *source* framerate, then ``fps`` decimates, then ``scale`` resizes.

A render is described by a :class:`TimelinePlan`: video lanes (a priority stack
— lane 0 over lane 1) and audio lanes (all mixed), each an ordered list of
clips, plus the timeline's own duration.  ``TimelineWidget.render_plan()`` is
the one producer; there is exactly one graph builder.

A clip may carry ``fade_in``/``fade_out`` seconds.  Video fades ride inside each
piece's chain between ``trim`` and ``setpts`` in the clip's own SOURCE time, so
the ramp continues across the piece boundaries ``concat`` joins instead of
restarting at each one (:func:`_fade_filters`); audio fades go at the head of
the clip's chain in clip-local time (:func:`_audio_fades`).  With no fades set
the graph is byte-for-byte what it was before they existed.
"""

from __future__ import annotations

import collections
import functools
import math
import os
import shutil
import subprocess
import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

from .probe import MediaInfo

__all__ = [
    "AudioClip",
    "AudioLane",
    "CONTAINERS",
    "ENCODERS",
    "Progress",
    "RenderJob",
    "RenderSettings",
    "Segment",
    "TextClip",
    "TimelinePlan",
    "VideoClip",
    "VideoLane",
    "build_command",
    "build_filter_graph",
    "default_font_file",
    "default_vaapi_device",
    "encoder_flags",
    "hole_enable",
    "input_seeks",
    "is_crf_encoder",
    "is_vaapi_encoder",
    "plan_inputs",
    "total_output_seconds",
    "visible_spans",
]

FFMPEG = shutil.which("ffmpeg") or "ffmpeg"

#: How many stderr lines to keep for diagnostics.
STDERR_TAIL_LINES = 50
#: Seconds between SIGTERM and SIGKILL when cancelling.
CANCEL_GRACE = 3.0
#: How far before a video input's first kept frame its ``-ss`` lands (see
#: :func:`input_seeks`).  The ``trim`` still makes the exact cut; the margin
#: only keeps that cut clear of whatever the seek itself does at its edge.
SEEK_MARGIN = 2.0

#: stderr fragments that mean "the GPU encoder itself is what went wrong".
#: ``[h264_vaapi @`` / ``[av1_vaapi @`` are the strongest signal: at the default
#: log level the VA-API encoders print nothing unless they are complaining.  The
#: rest cover the device never opening (``-vaapi_device``) and ``hwupload``.
_VAAPI_FAILURE_MARKERS = (
    "[h264_vaapi @",
    "[av1_vaapi @",
    "failed to initialise vaapi",
    "no va display",
    "device creation failed",
    "vaapi_device",
    "vaapi encode",
    "no usable encoding entrypoint",
    "[hwupload @",
    "failed to upload frame",
)

_VAAPI_HINT = (
    "hint: the VA-API (GPU) encoder failed (h264_vaapi/av1_vaapi). This is "
    "usually the driver or the render node, not a problem with the cut. Re-run "
    "with the libx264 (CPU) encoder."
)

#: The GPU encoders: VA-API on the RX 9070 XT (RDNA4), which encodes H.264 and AV1
#: in hardware.  Both are bitrate-driven like x264 (no CRF), need a VA-API device
#: and take hardware frames, so the graph ends in ``hwupload`` for them.
_VAAPI_ENCODERS = frozenset({"h264_vaapi", "av1_vaapi"})

#: Where VA-API goes when no amdgpu render node is found: libva's own default.
_FALLBACK_VAAPI_DEVICE = "/dev/dri/renderD128"

#: Selectable video encoders, as ``(label, value)``.
#:
#: AV1 comes twice: ``av1_vaapi`` is the GPU, bitrate-driven like H.264;
#: ``libsvtav1`` is the CPU, quality-targeted (CRF) and the smallest files
#: (measured on the old NVENC box at ~2x slower than the GPU for ~4.8x smaller
#: output).
ENCODERS: tuple[tuple[str, str], ...] = (
    ("H.264 · VA-API (GPU)", "h264_vaapi"),
    ("H.264 · x264 (CPU)", "libx264"),
    ("AV1 · VA-API (GPU)", "av1_vaapi"),
    ("AV1 · SVT-AV1 (CPU)", "libsvtav1"),
)

#: Selectable containers, as ``(label, value)``.
CONTAINERS: tuple[tuple[str, str], ...] = (
    ("MP4", "mp4"),
    ("MKV", "mkv"),
)

def is_crf_encoder(encoder: str) -> bool:
    """True when ``encoder`` takes ``-crf`` and must NOT get ``-b:v``/``-maxrate``."""
    return encoder == "libsvtav1"


def is_vaapi_encoder(encoder: str) -> bool:
    """True when ``encoder`` runs on the GPU through VA-API (needs a device + hwupload)."""
    return encoder in _VAAPI_ENCODERS


@functools.lru_cache(maxsize=None)
def default_vaapi_device(sys_drm: str = "/sys/class/drm") -> str:
    """The render node to hand VA-API: the amdgpu card, the boot VGA one first.

    libva's own default is ``renderD128``, which on this machine is the Intel
    iGPU (the RX 9070 XT comes up as ``renderD129``), and the numbering is not
    promised to stay put across boots — so look the node up by driver instead
    of hard-coding it.  Falls back to ``renderD128`` when no amdgpu node exists.
    Cached: the GPU does not change while the app is running.
    """
    found: list[tuple[bool, int, str]] = []
    try:
        names = os.listdir(sys_drm)
    except OSError:
        return _FALLBACK_VAAPI_DEVICE
    for name in names:
        if not name.startswith("renderD") or not name[7:].isdigit():
            continue
        device = os.path.join(sys_drm, name, "device")
        driver = os.path.basename(os.path.realpath(os.path.join(device, "driver")))
        if driver != "amdgpu":
            continue
        try:
            with open(os.path.join(device, "boot_vga"), encoding="ascii") as fh:
                boot = fh.read().strip() == "1"
        except OSError:
            boot = False
        found.append((not boot, int(name[7:]), name))
    if not found:
        return _FALLBACK_VAAPI_DEVICE
    return f"/dev/dri/{min(found)[2]}"


@functools.lru_cache(maxsize=None)
def default_font_file() -> str:
    """The bold sans fontconfig picks (what Qt's preview uses too); "" if none."""
    try:
        out = subprocess.run(
            ["fc-match", "-f", "%{file}", "sans-serif:bold"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return out if os.path.isfile(out) else ""


@dataclass
class Segment:
    """A keep-range in seconds (half-open: ``start`` inclusive, ``end`` exclusive)."""

    start: float
    end: float


Hole = tuple[float, float]
"""A LIFTed SOURCE-time range: removed from one lane, leaving the rest in sync."""

EPS = 1e-6
"""Tolerance for span arithmetic: anything shorter is not a span at all."""


@dataclass
class VideoClip:
    """One piece of video footage placed on a video lane.

    ``start`` is TIMELINE seconds (where the clip begins on the edit's uncut
    axis); ``src_in``/``src_out`` are seconds *inside* ``path``.  A clip that
    covers the whole file from the top of the timeline is ``start=0,
    src_in=0``, which is the shape every v3 render had.

    ``fade_in``/``fade_out`` are seconds of ramp from black at the clip's own
    two ends, 0.0 for none.  They are applied in the clip's SOURCE time — see
    :func:`_fade_filters` for why that is the only placement that survives the
    piece/concat graph.
    """

    path: str
    start: float = 0.0
    src_in: float = 0.0
    src_out: float = 0.0
    fade_in: float = 0.0
    fade_out: float = 0.0

    @property
    def length(self) -> float:
        return max(0.0, self.src_out - self.src_in)

    @property
    def end(self) -> float:
        return self.start + self.length

    def source_at(self, t: float) -> float:
        """Timestamp inside ``path`` for a TIMELINE time within this clip."""
        return self.src_in + (float(t) - self.start)


@dataclass
class VideoLane:
    """An ordered, non-overlapping list of video clips plus its lifted holes.

    Lane order is PRIORITY order: lane 0 is the main video and paints on top,
    lane 1 is "Video track 2" *underneath* it.  Where a higher lane has a clip,
    that clip wins; where it has a hole (a lift) or simply no clip, the next
    lane down shows through, and black shows through when no lane has anything.
    That is the whole reason a second video lane is useful — do not "simplify"
    the priority walk in :func:`visible_spans` into a plain concatenation.
    """

    clips: list[VideoClip] = field(default_factory=list)
    holes: list[Hole] = field(default_factory=list)


@dataclass
class AudioClip:
    """One piece of audio placed on an audio lane.

    ``stream`` selects the audio stream inside ``path`` (``0`` for a standalone
    music file, ``K`` for the K-th embedded track of a video).  ``src_out <=
    src_in`` means "to the end of the file", which is what a dropped music bed
    of unknown length is: it is padded to the timeline instead of trimmed.

    ``fade_in``/``fade_out`` are seconds of ramp at the clip's own two ends,
    applied in CLIP-LOCAL time at the head of its chain (see
    :func:`_audio_heads`).
    """

    path: str
    stream: int = 0
    start: float = 0.0
    src_in: float = 0.0
    src_out: float = 0.0
    fade_in: float = 0.0
    fade_out: float = 0.0

    @property
    def bounded(self) -> bool:
        """False when the clip runs to the end of its file (length unknown)."""
        return self.src_out - self.src_in > EPS

    def end(self, total: float) -> float:
        """Timeline time this clip stops at; an unbounded clip runs to ``total``."""
        return self.start + (self.src_out - self.src_in) if self.bounded else float(total)


@dataclass
class AudioLane:
    """One audio lane: its clips, its mix controls and its lifted holes.

    Audio lanes do not have priority — they all mix together — so their order
    only fixes the ffmpeg input numbering.  A ``muted`` lane is left out of
    ``amix`` altogether rather than mixed at ``volume=0``: a muted input would
    still drag the mix duration around and waste a decode.  ``holes`` are
    SOURCE-time ranges this lane alone is silent over (the lift), applied as a
    second ``volume=0:enable=…`` at the head of each clip's chain.
    """

    clips: list[AudioClip] = field(default_factory=list)
    gain_db: float = 0.0
    muted: bool = False
    holes: list[Hole] = field(default_factory=list)


@dataclass
class TextClip:
    """A title: ``text`` drawn over the finished picture from ``start`` to ``end``
    (TIMELINE seconds).  ``x``/``y`` are its centre and ``size`` its font height,
    all as fractions of the output frame, so it lands in the same place at any
    render resolution and in the preview."""

    start: float
    end: float
    text: str
    x: float = 0.5
    y: float = 0.85
    size: float = 0.08
    color: str = "#ffffff"


@dataclass
class TimelinePlan:
    """The whole multi-lane timeline, in TIMELINE (uncut) seconds.

    ``duration`` is the length of the longest lane, not the video's: an audio
    bed longer than the footage extends the timeline and the video is padded
    with black to match (see :func:`build_filter_graph`).
    """

    duration: float
    video: list[VideoLane] = field(default_factory=list)
    audio: list[AudioLane] = field(default_factory=list)
    texts: list[TextClip] = field(default_factory=list)


@dataclass
class RenderSettings:
    """Everything the user chose in the render dialog, plus the timeline."""

    segments: list[Segment]
    plan: TimelinePlan
    fps: int
    tmix_frames: int
    width: int | None
    height: int | None
    out_path: str
    encoder: str = "h264_vaapi"  # h264_vaapi | libx264 | av1_vaapi | libsvtav1
    container: str = "mp4"  # mp4 | mkv
    mbps: int = 60  # bitrate encoders only (everything but libsvtav1)
    crf: int = 30  # libsvtav1 only
    av1_preset: int = 8  # libsvtav1 only
    vaapi_device: str | None = None  # VA-API only; None = default_vaapi_device()


@dataclass
class Progress:
    """One progress sample parsed from ffmpeg's ``-progress`` stream."""

    out_time: float
    total: float
    fraction: float
    fps: float
    speed: float
    eta_seconds: float | None


def total_output_seconds(segments: Iterable[Segment]) -> float:
    """Sum of the keep-ranges — the expected duration of the rendered file."""
    return sum(max(0.0, s.end - s.start) for s in segments)


def _fmt(value: float) -> str:
    """Format a timestamp for a filter argument without exponent notation."""
    text = f"{float(value):.6f}".rstrip("0").rstrip(".")
    return text or "0"


def _quote(value: str) -> str:
    """Escape a filter option value for BOTH levels ffmpeg parses: the option
    string (``\\ ' :``) and then the graph, by single-quoting it."""
    level1 = value.replace("\\", "\\\\").replace("'", "\\'").replace(":", "\\:")
    return "'" + level1.replace("'", "'\\''") + "'"


def _output_ranges(start: float, end: float, segs: list[Segment]) -> list[tuple[float, float]]:
    """A TIMELINE range as OUTPUT-time ranges: the keep-segments are laid end to end."""
    out: list[tuple[float, float]] = []
    acc = 0.0
    for seg in segs:
        a, b = max(start, seg.start), min(end, seg.end)
        if b - a > EPS:
            out.append((acc + a - seg.start, acc + b - seg.start))
        acc += seg.end - seg.start
    return out


def _text_filters(texts: list[TextClip], segs: list[Segment]) -> str:
    """``,drawtext=…`` per title, after scale so it is drawn at output size.

    Each one is enabled only over the output ranges its block survives in, so a
    cut through a title removes that part of it like any other footage.
    """
    font = default_font_file()
    parts: list[str] = []
    for t in texts:
        ranges = _output_ranges(t.start, t.end, segs)
        if not ranges or not t.text.strip():
            continue
        enable = "+".join(f"between(t,{_fmt(a)},{_fmt(b)})" for a, b in ranges)
        parts.append(
            "drawtext="
            + (f"fontfile={_quote(font)}:" if font else "")
            + f"text={_quote(t.text)}:expansion=none:text_align=C"
            + f":fontcolor={_quote(t.color)}:fontsize=h*{_fmt(t.size)}"
            + f":x=w*{_fmt(t.x)}-text_w/2:y=h*{_fmt(t.y)}-text_h/2"
            + ":shadowx=2:shadowy=2:shadowcolor=black@0.6"
            + f":enable='{enable}'"
        )
    return "".join("," + part for part in parts)


def _video_chain(s: RenderSettings) -> str:
    """The post-cut video chain: tmix (source fps) -> fps -> scale."""
    chain: list[str] = []
    if s.tmix_frames and s.tmix_frames > 0:
        chain.append(f"tmix=frames={int(s.tmix_frames)}")
    chain.append(f"fps={int(s.fps)}")
    if s.width is not None or s.height is not None:
        w = int(s.width) if s.width is not None else -2
        h = int(s.height) if s.height is not None else -2
        chain.append(f"scale={w}:{h}")
    return ",".join(chain)


def _clean_holes(holes: Iterable[Hole] | None) -> list[Hole]:
    """Drop empty/reversed hole ranges and put what is left in time order."""
    out: list[Hole] = []
    for start, end in holes or ():
        lo, hi = (float(start), float(end))
        if hi < lo:
            lo, hi = hi, lo
        if hi - lo > 1e-6:
            out.append((lo, hi))
    out.sort()
    return out


def hole_enable(holes: Iterable[Hole]) -> str:
    """``enable=`` expression that is true inside any of ``holes``.

    The ranges are disjoint (the timeline normalises them), so summing the
    ``between()`` terms is a correct OR: at most one term is ever 1.
    """
    return "+".join(f"between(t,{_fmt(a)},{_fmt(b)})" for a, b in holes)


def _fade_filters(clip: VideoClip, shift: float = 0) -> str:
    """``,fade=…`` for one video clip, expressed in the clip's SOURCE time.

    ``shift`` is the ``-ss`` its input was opened with (:func:`input_seeks`),
    which moves every timestamp on that input down by the same amount.

    Returns a leading-comma fragment to splice into a piece's chain BETWEEN its
    ``trim`` and its ``setpts=PTS-STARTPTS`` — the one place the fade actually
    survives this graph, for three reasons:

    * after ``trim`` and before ``setpts`` the frames still carry their
      ORIGINAL source timestamps, so ``st=`` can name a position inside the
      file and mean it;
    * ``_piece_video_parts`` cuts a clip into several pieces at cut and
      visibility boundaries and ``concat``s them.  Every piece of one clip gets
      the IDENTICAL filter string, and because ffmpeg's ``fade`` computes its
      factor from the frame's own timestamp against ``st``, the ramp continues
      across a piece boundary instead of restarting.  Measured: a clip cut into
      0-1 s and 1-12 s pieces produced luma identical, frame for frame, to the
      same fade on one unsplit trim;
    * a piece that lies entirely PAST the fade-in window comes out at full
      brightness (the filter is already done fading), and one entirely BEFORE a
      fade-out window is untouched — which is what makes it safe to hang the
      same fragment on every piece without knowing which piece it is.

    Putting it at the head of ``[N:v]`` instead would be wrong in a way that
    exits 0: ``fade=t=in`` blacks out EVERYTHING before ``st``, so a second
    clip from the same file (or a second piece of the same clip) would come out
    black with no error anywhere.
    """
    parts: list[str] = []
    fade_in = min(max(0.0, clip.fade_in), clip.length)
    fade_out = min(max(0.0, clip.fade_out), clip.length)
    if fade_in > EPS:
        parts.append(f"fade=t=in:st={_fmt(clip.src_in - shift)}:d={_fmt(fade_in)}")
    if fade_out > EPS:
        st = clip.src_out - fade_out - shift
        parts.append(f"fade=t=out:st={_fmt(st)}:d={_fmt(fade_out)}")
    return "".join("," + part for part in parts)


def _audio_fades(clip: AudioClip, length: float) -> list[str]:
    """``afade`` entries for one audio clip, in CLIP-LOCAL time.

    They go at the head of the clip's chain — after the ``atrim``/``asetpts``
    that put it in clip-local time and BEFORE the ``adelay`` that puts it on
    the timeline — so ``st=0`` is the clip's own first sample whether or not
    the trim was needed.  Before ``apad`` too: padding first would make a
    fade-out ramp the silence rather than the material.

    ``length`` is the clip's own length; for an UNBOUNDED clip (a dropped file
    whose duration the plan does not know) it is the timeline-implied length,
    so a fade-out on one is only as accurate as that assumption.
    """
    out: list[str] = []
    fade_in = min(max(0.0, clip.fade_in), length)
    fade_out = min(max(0.0, clip.fade_out), length)
    if fade_in > EPS:
        out.append(f"afade=t=in:st=0:d={_fmt(fade_in)}")
    if fade_out > EPS:
        out.append(f"afade=t=out:st={_fmt(length - fade_out)}:d={_fmt(fade_out)}")
    return out


# --------------------------------------------------------------------------- #
# plan resolution
# --------------------------------------------------------------------------- #


def _clean_plan(info: MediaInfo, plan: TimelinePlan) -> TimelinePlan:
    """Validate a plan and sort every lane into time order.

    A stream index past what the loaded file has would reach ffmpeg as a
    ``[1:a:K]`` that matches nothing; refuse it here, where the message can
    say which track.
    """
    video: list[VideoLane] = []
    for lane in plan.video:
        clips = sorted(
            (c for c in lane.clips if c.length > EPS), key=lambda c: c.start
        )
        for clip in clips:
            if not clip.path:
                raise ValueError("video clip has no path")
        video.append(VideoLane(clips=clips, holes=_clean_holes(lane.holes)))
    audio: list[AudioLane] = []
    for lane in plan.audio:
        clips = sorted(lane.clips, key=lambda c: c.start)
        for clip in clips:
            if not clip.path:
                raise ValueError("audio clip has no path")
            if clip.path == info.path and not 0 <= int(clip.stream) < len(info.audio):
                raise ValueError(
                    f"audio track {clip.stream} out of range "
                    f"(file has {len(info.audio)} audio streams)"
                )
        audio.append(
            AudioLane(
                clips=clips,
                gain_db=float(lane.gain_db),
                muted=bool(lane.muted),
                holes=_clean_holes(lane.holes),
            )
        )
    duration = max(0.0, float(plan.duration))
    if duration <= 0.0:
        raise ValueError("the timeline has no duration")
    return TimelinePlan(duration=duration, video=video, audio=audio)


def plan_inputs(plan: TimelinePlan) -> tuple[list[str], dict[str, int], dict[str, int]]:
    """``(-i paths, video path -> index, audio path -> index)``.

    Video and audio NEVER share an input handle, even for the same file: the
    source is opened once for video and again for audio (``-i f -i f``).  This
    is not cosmetic.  With both on one input the audio chain races far ahead —
    ``atrim`` hits EOF within a second or two while the video decoder is still
    grinding forward to the trim start — and ffmpeg tears the run down, exiting
    0 with correct audio and **zero video frames**, silently.  Measured on the
    46-minute test file: a 3 s cut at t=400 rendered fine, the same cut at
    t=1200 and t=2000 produced audio-only files.  The second handle costs
    nothing measurable (52.3 s vs 57.3 s video-only at t=1200).

    Video paths are numbered first, in lane/time order, then audio paths —
    which keeps the single-video case at ``[0:v]``, ``[1:a:K]`` and externals
    from ``[2:a:0]``.  This is the ONE place input numbering is decided.
    """
    paths: list[str] = []
    video_index: dict[str, int] = {}
    audio_index: dict[str, int] = {}
    for lane in plan.video:
        for clip in lane.clips:
            if clip.path not in video_index:
                video_index[clip.path] = len(paths)
                paths.append(clip.path)
    for lane in plan.audio:
        if lane.muted:
            continue
        for clip in lane.clips:
            if clip.path not in audio_index:
                audio_index[clip.path] = len(paths)
                paths.append(clip.path)
    return paths, video_index, audio_index


Span = tuple[float, float, "VideoClip | None"]


def visible_spans(plan: TimelinePlan) -> list[Span]:
    """Cut the timeline into spans and say which clip is visible in each.

    Video lanes are a PRIORITY stack, not a sequence: at every instant the
    topmost lane that has a clip there *and* no hole there wins, and the next
    lane down shows through wherever the one above it is missing or lifted.
    A span with ``None`` is black — nothing covers it on any lane.

    Returns spans covering ``[0, plan.duration]`` in order, with neighbouring
    spans merged when they are the same clip running continuously.
    """
    total = plan.duration
    edges = {0.0, total}
    for lane in plan.video:
        for clip in lane.clips:
            edges.add(max(0.0, min(total, clip.start)))
            edges.add(max(0.0, min(total, clip.end)))
        for lo, hi in lane.holes:
            edges.add(max(0.0, min(total, lo)))
            edges.add(max(0.0, min(total, hi)))
    marks = sorted(edges)
    spans: list[Span] = []
    for lo, hi in zip(marks, marks[1:]):
        if hi - lo <= EPS:
            continue
        mid = (lo + hi) / 2.0
        winner: VideoClip | None = None
        for lane in plan.video:
            if any(a <= mid <= b for a, b in lane.holes):
                continue  # lifted here: let the lane below show through
            hit = next(
                (c for c in lane.clips if c.start - EPS <= mid <= c.end + EPS), None
            )
            if hit is not None:
                winner = hit
                break
        if spans and spans[-1][2] is winner and abs(spans[-1][1] - lo) <= EPS:
            spans[-1] = (spans[-1][0], hi, winner)
        else:
            spans.append((lo, hi, winner))
    return spans


def _pieces(segments: list[Segment], spans: list[Span]) -> list[Span]:
    """Keep-segments intersected with the visible spans, in output order.

    Both the video chains and the audio atrims are cut on these boundaries, so
    ``concat`` still gets one video and one audio input per piece.  With a
    single clip covering the timeline there is exactly one span, so the pieces
    ARE the keep-segments and the graph is the one v3 emitted.
    """
    out: list[Span] = []
    for seg in segments:
        for lo, hi, clip in spans:
            a = max(seg.start, lo)
            b = min(seg.end, hi)
            if b - a > EPS:
                out.append((a, b, clip))
    return out


def input_seeks(
    plan: TimelinePlan, segs: list[Segment], info: MediaInfo | None = None
) -> dict[int, int]:
    """``{input index: seconds}`` to seek each VIDEO input by before decoding.

    ``trim`` alone decodes linearly from t=0 and throws away everything before
    its start, so a kept range 1:41 into a 1440p120 replay cost 26 s of pure
    decode (measured 2026-10-30) and the progress bar sat frozen at the piece
    boundary while it ran.  ``-ss X`` before ``-i`` jumps there instead: ffmpeg
    seeks to the keyframe at or before X, drops the frames before it, and
    shifts that input's timestamps down by exactly X, so every source-time
    number the graph uses on that input is written ``X`` smaller and selects
    the very same frames.

    * X is the earliest source time the graph reads from the input, minus
      :data:`SEEK_MARGIN`, floored to whole seconds (an integer shift is exact
      in any stream time base).  Under one second it is not worth a seek.
    * A clip's video fade ``st=`` counts as a read: ``fade`` rejects a negative
      start, so the shift must never take one below zero.
    * The single-clip graph's keep-segments may sit in the ``tpad`` tail past
      the footage; they count as the clip's end, so the seek never lands past
      the last real frame ``tpad`` needs to repeat.
    * Audio inputs are never seeked.  Audio decode is nearly free, and seeking
      a VBR MP3 is approximate: its timestamps are then estimated from the byte
      position, which would slide the music against the picture.
    """
    _, video_index, _ = plan_inputs(plan)
    earliest: dict[int, float] = {}

    def read(index: int, t: float) -> None:
        earliest[index] = min(earliest.get(index, t), t)

    def read_fades(index: int, clip: VideoClip) -> None:
        if clip.fade_in > EPS:
            read(index, clip.src_in)
        if clip.fade_out > EPS:
            read(index, clip.src_out - min(clip.fade_out, clip.length))

    if _is_simple_video(plan, info):
        clip = plan.video[0].clips[0]
        index = video_index[clip.path]
        for seg in segs:
            read(index, min(seg.start, clip.src_out))
        read_fades(index, clip)
    else:
        for lo, _hi, clip in _pieces(segs, visible_spans(plan)):
            if clip is not None:
                index = video_index[clip.path]
                read(index, clip.source_at(lo))
                read_fades(index, clip)
    seeks: dict[int, int] = {}
    for index, t in earliest.items():
        x = math.floor(t - SEEK_MARGIN)
        if x >= 1:
            seeks[index] = x
    return seeks


def _is_simple_video(plan: TimelinePlan, info: MediaInfo | None = None) -> bool:
    """True for the one-lane, one-clip-from-the-top timeline v3 always had.

    That shape keeps the original ``[0:v] -> drawbox -> split -> trim`` graph
    (plus ``tpad`` when the timeline outlives the footage), because it is the
    shape every measured fact in SPEC/HANDOFF was established on.

    The clip must also run to the end of the loaded file: that graph trims the
    whole input and ``tpad`` pads after the FILE's end, so a clip whose right
    edge was trimmed in would play its trimmed-off footage there instead of black.
    """
    if len(plan.video) != 1 or len(plan.video[0].clips) != 1:
        return False
    clip = plan.video[0].clips[0]
    if info is not None and (clip.path != info.path or clip.src_out < info.duration - 1e-3):
        return False
    return abs(clip.start) <= EPS and abs(clip.src_in) <= EPS


def _normalise(info: MediaInfo, s: RenderSettings) -> str:
    """Filters that make one video piece concat-compatible with the others.

    Pieces come from different decoders (and from a ``color`` source for the
    black gaps).  ``scale``/``pad``/``setsar`` are what ``concat`` actually
    demands — measured on ffmpeg 9.0.1, feeding it a 640x480 piece next to a
    320x180 one is a hard failure ("Input link in0:v0 parameters … do not
    match", exit 1), not a silent one.  ``fps`` it does not demand, but
    ``tmix`` downstream blends a fixed number of FRAMES: without one rate
    across the concat, the same ``tmix=frames=N`` would smear a different
    number of milliseconds per piece.  The rate is the SOURCE rate, so ``tmix``
    still blends at source framerate and ``fps`` still decimates afterwards,
    exactly as the user's shell function does.
    """
    rate = _fmt(info.fps if info.fps and info.fps > 0 else s.fps)
    return (
        f"scale={int(info.width)}:{int(info.height)}"
        f":force_original_aspect_ratio=decrease,"
        f"pad={int(info.width)}:{int(info.height)}:-1:-1:color=black,"
        f"setsar=1,fps={rate},format=yuv420p"
    )


def build_filter_graph(info: MediaInfo, s: RenderSettings) -> str:
    """Build the ``-filter_complex`` string for ``s``.

    Exposed separately from :func:`build_command` so it can be inspected and
    tested on its own.  See the module docstring for the shape of the graph.

    Two video shapes, one audio shape:

    * a single clip filling the timeline keeps the v3 graph — ``[0:v]`` ->
      ``tpad`` (only when the timeline outlives the footage) -> ``drawbox``
      holes -> ``split`` -> per-segment ``trim`` -> ``concat``;
    * anything else (an appended clip, a second video lane) is cut into
      *pieces* by :func:`visible_spans`, each piece normalised and trimmed from
      whichever clip is visible there, black filling the gaps.

    Either way the audio is mixed first and then ``asplit``/``atrim``-ed on the
    SAME piece boundaries, so ``concat=…:v=1:a=1`` still gets one video and one
    audio input per piece.

    Lifts (per-lane holes) are applied at the head of each chain in timeline
    time: ``drawbox`` on a single video lane, transparency in the span walk
    when there is a lane underneath to show through, and a second
    ``volume=0:enable=…`` on audio.
    """
    segs = list(s.segments)
    if len(segs) < 1:
        raise ValueError("at least one keep-segment is required")

    plan = _clean_plan(info, s.plan)
    _, video_index, audio_index = plan_inputs(plan)
    seeks = input_seeks(plan, segs, info)
    total = plan.duration
    parts: list[str] = []

    if _is_simple_video(plan, info):
        pieces: list[Span] = [(seg.start, seg.end, None) for seg in segs]
        parts += _simple_video_parts(plan, s, segs, seeks.get(0, 0))
    else:
        pieces = _pieces(segs, visible_spans(plan))
        if not pieces:
            raise ValueError("the keep-segments cover no footage")
        parts += _piece_video_parts(info, s, pieces, video_index, seeks)

    n = len(pieces)
    heads = _audio_heads(plan, audio_index)
    if heads:
        if len(heads) > 1:
            parts += [chain + label for label, chain in heads]
            labels = "".join(label for label, _ in heads)
            parts.append(f"{labels}amix=inputs={len(heads)}:duration=longest:normalize=0[am]")
        else:
            parts.append(heads[0][1] + "[am]")

        if n > 1:
            outs = "".join(f"[m{i}]" for i in range(n))
            parts.append(f"[am]asplit={n}{outs}")
            for i, (lo, hi, _clip) in enumerate(pieces):
                parts.append(
                    f"[m{i}]atrim=start={_fmt(lo)}:end={_fmt(hi)},"
                    f"asetpts=PTS-STARTPTS[u{i}]"
                )
        else:
            lo, hi, _clip = pieces[0]
            parts.append(
                f"[am]atrim=start={_fmt(lo)}:end={_fmt(hi)},"
                f"asetpts=PTS-STARTPTS[ac]"
            )

    # ---- concat (only when there is more than one piece) --------------------
    if n > 1:
        if heads:
            ins = "".join(f"[t{i}][u{i}]" for i in range(n))
            parts.append(f"{ins}concat=n={n}:v=1:a=1[vc][ac]")
        else:
            ins = "".join(f"[t{i}]" for i in range(n))
            parts.append(f"{ins}concat=n={n}:v=1:a=0[vc]")

    # ---- tmix / fps / scale, then titles over the finished picture ---------
    parts.append(f"[vc]{_video_chain(s)}{_text_filters(s.plan.texts, segs)}[vout]")
    return ";".join(parts)


def _simple_video_parts(
    plan: TimelinePlan, s: RenderSettings, segs: list[Segment], shift: float = 0
) -> list[str]:
    """The v3 video graph: tpad, drawbox holes, split, per-segment trim.

    ``tpad=stop_mode=add:color=black`` is what lets a keep-segment sit past the
    end of the footage.  Without it ``trim`` yields no frames there, ``concat``
    starves and ffmpeg exits 0 having written a short (or video-less) file —
    the same class of silent failure the external-audio ``apad`` fixes.  It
    runs BEFORE ``drawbox`` so a lift landing in the padded tail still has
    frames to blank; ``tpad`` appends frames without rewriting timestamps, so
    the ``enable=`` expressions still read source time either way.

    ``shift`` is input 0's ``-ss`` (:func:`input_seeks`): every source time
    written here — trims, holes, fades — is that much smaller.
    """
    parts: list[str] = []
    clip = plan.video[0].clips[0]
    holes = [(a - shift, b - shift) for a, b in _clean_holes(plan.video[0].holes)]
    # Per SEGMENT, between trim and setpts, exactly as the piece graph does it
    # (see _fade_filters).  With no fade this is the empty string and the graph
    # is byte-for-byte the v3 one -- the shape every measured fact rests on.
    fades = _fade_filters(clip, shift)
    src = "[0:v]"
    extra = plan.duration - clip.end
    if extra > 1e-3:
        parts.append(
            f"[0:v]tpad=stop_mode=add:color=black:stop_duration={_fmt(extra)}[vp]"
        )
        src = "[vp]"
    if holes:
        parts.append(
            f"{src}drawbox=x=0:y=0:w=iw:h=ih:color=black@1:t=fill:"
            f"enable='{hole_enable(holes)}'[vh]"
        )
        src = "[vh]"
    n = len(segs)
    if n > 1:
        outs = "".join(f"[v{i}]" for i in range(n))
        parts.append(f"{src}split={n}{outs}")
        for i, seg in enumerate(segs):
            parts.append(
                f"[v{i}]trim=start={_fmt(seg.start - shift)}:end={_fmt(seg.end - shift)}"
                f"{fades},"
                f"setpts=PTS-STARTPTS[t{i}]"
            )
    else:
        seg = segs[0]
        parts.append(
            f"{src}trim=start={_fmt(seg.start - shift)}:end={_fmt(seg.end - shift)}{fades},"
            f"setpts=PTS-STARTPTS[vc]"
        )
    return parts


def _piece_video_parts(
    info: MediaInfo,
    s: RenderSettings,
    pieces: list[Span],
    video_index: dict[str, int],
    seeks: dict[int, int] | None = None,
) -> list[str]:
    """One chain per visible piece, normalised so ``concat`` will take them.

    A piece with no clip is a hole in every lane (or past the end of all of
    them) and becomes a generated black ``color`` source of the right length —
    that is how a second video lane shows through, and how the timeline keeps
    running after the footage stops.

    ``seeks`` holds each input's ``-ss`` (:func:`input_seeks`); a piece's trim
    and fades are written that much earlier in its input's time.
    """
    seeks = seeks or {}
    parts: list[str] = []
    norm = _normalise(info, s)
    # A source feeding more than one piece has to be split first.
    uses: dict[int, int] = collections.Counter(
        video_index[clip.path] for _, _, clip in pieces if clip is not None
    )
    cursor: dict[int, int] = {}
    for index, count in uses.items():
        if count > 1:
            outs = "".join(f"[s{index}_{j}]" for j in range(count))
            parts.append(f"[{index}:v]split={count}{outs}")
        cursor[index] = 0
    rate = _fmt(info.fps if info.fps and info.fps > 0 else s.fps)
    n = len(pieces)
    for i, (lo, hi, clip) in enumerate(pieces):
        label = f"[t{i}]" if n > 1 else "[vc]"
        if clip is None:
            parts.append(
                f"color=black:s={int(info.width)}x{int(info.height)}:r={rate}"
                f":d={_fmt(hi - lo)},{norm}{label}"
            )
            continue
        index = video_index[clip.path]
        if uses[index] > 1:
            src = f"[s{index}_{cursor[index]}]"
            cursor[index] += 1
        else:
            src = f"[{index}:v]"
        shift = seeks.get(index, 0)
        parts.append(
            f"{src}trim=start={_fmt(clip.source_at(lo) - shift)}"
            f":end={_fmt(clip.source_at(hi) - shift)}"
            f"{_fade_filters(clip, shift)},setpts=PTS-STARTPTS,{norm}{label}"
        )
    return parts


def _audio_heads(
    plan: TimelinePlan, audio_index: dict[str, int]
) -> list[tuple[str, str]]:
    """``(label, chain)`` for every audio clip that reaches the mix.

    One head per CLIP, not per lane: clips on a lane never overlap, so summing
    them through ``amix`` lays them end to end.  Each head is put into timeline
    time first (``atrim``/``asetpts``/``adelay``, all omitted when the clip
    already starts at 0 and runs to the end), then gain, then the lane's lift,
    then ``apad`` — the pad last so a lift's ``enable=`` window is never
    shifted by silence.

    Labels keep their v3 spelling — ``[gK]`` for the K-th embedded track of the
    loaded video, ``[xN]`` for the N-th clip from any other file — with a
    ``_1``/``_2`` suffix only where a stream genuinely appears twice.  Stable
    labels keep graph diffs (and the tests that pin them) readable.
    """
    heads: list[tuple[str, str]] = []
    total = plan.duration
    primary = plan.video[0].clips[0].path if plan.video and plan.video[0].clips else None
    used: set[str] = set()
    external = 0
    for lane in plan.audio:
        if lane.muted:
            continue
        holes = _clean_holes(lane.holes)
        for clip in lane.clips:
            index = audio_index[clip.path]
            source = f"[{index}:a:{int(clip.stream)}]"
            chain: list[str] = []
            end = clip.end(total)
            # Trim only when the clip does not simply run to the end of the
            # timeline: otherwise the per-piece atrims already bound its tail.
            if clip.src_in > EPS or (clip.bounded and end < total - EPS):
                bound = (
                    f":end={_fmt(clip.src_out)}"
                    if clip.bounded
                    else ""
                )
                chain.append(f"atrim=start={_fmt(clip.src_in)}{bound}")
                chain.append("asetpts=PTS-STARTPTS")
            # Fades next: clip-local time, before the delay puts the clip on the
            # timeline and before the pad puts silence after it.
            chain += _audio_fades(clip, end - clip.start)
            if clip.start > EPS:
                chain.append(f"adelay={_fmt(clip.start * 1000.0)}:all=1")
                # ``adelay`` fed by an atrim/asetpts pair does NOT materialise
                # its silence: it emits one 2048-sample frame and then jumps the
                # first real frame's PTS to the delay, leaving a hole with no
                # samples in it.  Fed straight off an input it fills the gap
                # properly, which is why this only shows up on a clip that was
                # trimmed AND moved.  The per-piece ``atrim`` downstream then
                # asks for a window inside that hole and gets 0.046 s of audio,
                # and ``concat`` — which needs one audio stream per piece —
                # fails the whole render with "Invalid data found when
                # processing input" (exit 183).  Measured on ffmpeg 9.0.1: a
                # single clip at 6-12 s of a 12 s timeline, which is what
                # deleting the FIRST block leaves behind, or trimming the only
                # clip's left edge inward.  ``aresample=async=1`` turns the hole
                # into real silence; the piece then measures 6.000 s and the
                # render comes out 12 s with the tone in the right place.
                chain.append("aresample=async=1")
            chain.append(f"volume={_fmt(lane.gain_db)}dB")
            if holes:
                chain.append(f"volume=0:enable='{hole_enable(holes)}'")
            # Pad when the clip stops before the timeline does, or when its
            # length is unknown (a dropped file).  Without it a keep-segment
            # past the clip's end starves concat and ffmpeg exits 0 with a
            # short audio stream — measured 2.0 s instead of 5.0 s.
            if not clip.bounded or end < total - EPS:
                chain.append(f"apad=whole_dur={_fmt(total)}")
            if clip.path == primary:
                base = f"g{int(clip.stream)}"
            else:
                base = f"x{external}"
                external += 1
            label, bump = base, 0
            while label in used:
                bump += 1
                label = f"{base}_{bump}"
            used.add(label)
            heads.append((f"[{label}]", source + ",".join(chain)))
    return heads


def encoder_flags(
    encoder: str, mbps: int = 60, *, crf: int = 30, av1_preset: int = 8
) -> list[str]:
    """Video-encoder flags.

    The supported encoders need genuinely different flags:

    * ``libx264``    — exactly the user's shell script: ``-b:v/-maxrate/-bufsize``.
    * ``h264_vaapi`` / ``av1_vaapi`` — ``-rc_mode CBR`` + the same rate trio
      (``-profile:v high`` on H.264), and **no** ``-pix_fmt``: they take hardware
      frames, which :func:`build_command` uploads at the end of the graph.
    * ``libsvtav1``  — quality-targeted: ``-crf`` + ``-preset``, and **no**
      bitrate flags at all (they would put it in a different rate-control mode).

    Any other encoder name falls back to the generic x264-style flag set.
    """
    if is_crf_encoder(encoder):
        return [
            "-c:v",
            encoder,
            "-crf",
            str(int(crf)),
            "-preset",
            str(int(av1_preset)),
            "-pix_fmt",
            "yuv420p",
        ]
    mbps = int(mbps)
    rate = [
        "-b:v",
        f"{mbps}M",
        "-maxrate",
        f"{mbps}M",
        "-bufsize",
        f"{mbps * 2}M",
    ]
    if is_vaapi_encoder(encoder):
        profile = ["-profile:v", "high"] if encoder == "h264_vaapi" else []
        return ["-c:v", encoder, "-rc_mode", "CBR"] + rate + profile
    return ["-c:v", encoder or "libx264"] + rate + ["-pix_fmt", "yuv420p"]


def build_command(info: MediaInfo, s: RenderSettings) -> list[str]:
    """Build the full ffmpeg argv for a render.

    Pure: performs no I/O and starts no process.

    Args:
        info: Probe result for the source file.
        s: The render settings (segments, audio selection, encode options).

    Returns:
        argv suitable for :class:`subprocess.Popen` (never a shell string).

    Raises:
        ValueError: no segments, a non-positive segment, an out-of-range audio
            track index, an unknown container, or a non-positive fps/bitrate/crf.

    Note:
        If every audio lane is muted (or nothing has audio at all) the command
        is video-only: no ``-map [ac]`` and no audio encoder.  Every other file
        in the plan adds one ``-i`` after the two handles on the source file.
    """
    if not s.segments:
        raise ValueError("at least one keep-segment is required")
    for seg in s.segments:
        if seg.end <= seg.start:
            raise ValueError(f"empty or reversed segment: {seg.start} -> {seg.end}")
    if int(s.fps) <= 0:
        raise ValueError("fps must be positive")
    container = (s.container or "mp4").lower()
    if container not in {value for _, value in CONTAINERS}:
        raise ValueError(f"unsupported container: {s.container!r}")
    encoder = s.encoder or "libx264"
    if is_crf_encoder(encoder):
        if not 0 <= int(s.crf) <= 63:
            raise ValueError("crf must be between 0 and 63")
    elif int(s.mbps) <= 0:
        raise ValueError("bitrate (mbps) must be positive")
    if not s.out_path:
        raise ValueError("out_path is required")

    plan = _clean_plan(info, s.plan)
    paths, _, audio_index = plan_inputs(plan)
    seeks = input_seeks(plan, list(s.segments), info)
    has_audio = bool(_audio_heads(plan, audio_index))
    graph = build_filter_graph(info, s)

    argv: list[str] = [FFMPEG, "-hide_banner", "-nostdin", "-y"]
    if is_vaapi_encoder(encoder):
        # The graph itself is encoder-independent; only the very end changes:
        # the finished software frames go up to the GPU for the VA-API encoder.
        graph = graph[: -len("[vout]")] + ",format=nv12,hwupload[vout]"
        argv += ["-vaapi_device", s.vaapi_device or default_vaapi_device()]
    for index, path in enumerate(paths):
        # The source appears twice when its audio is in the mix: a second
        # handle purely for audio.  See plan_inputs for why that is not cosmetic.
        if index in seeks:
            # Jump to the first kept frame instead of decoding up to it; the
            # graph's times on this input are already written relative to it.
            argv += ["-ss", str(seeks[index])]
        argv += ["-i", path]
    argv += ["-filter_complex", graph, "-map", "[vout]"]
    if has_audio:
        argv += ["-map", "[ac]"]
    argv += encoder_flags(
        encoder, int(s.mbps), crf=int(s.crf), av1_preset=int(s.av1_preset)
    )
    if has_audio:
        argv += ["-c:a", "aac", "-b:a", "192k"]
    if container == "mp4":
        # Invalid for matroska — ffmpeg errors out on it, so mp4 only.
        argv += ["-movflags", "+faststart"]
    argv += ["-progress", "pipe:1", "-nostats", s.out_path]
    return argv


# --------------------------------------------------------------------------- #
# progress parsing
# --------------------------------------------------------------------------- #


def _parse_number(raw: str) -> float | None:
    """Parse an ffmpeg progress value, tolerating ``N/A`` and unit suffixes."""
    raw = raw.strip()
    if not raw or raw.upper().startswith("N/A"):
        return None
    if raw.endswith("x"):  # speed=1.23x
        raw = raw[:-1]
    try:
        value = float(raw)
    except ValueError:
        return None
    return None if value != value else value  # reject NaN


def _parse_timecode(raw: str) -> float | None:
    """Parse ``HH:MM:SS.micro`` (ffmpeg's ``out_time``) into seconds."""
    raw = raw.strip()
    if not raw or raw.upper().startswith("N/A"):
        return None
    sign = -1.0 if raw.startswith("-") else 1.0
    raw = raw.lstrip("-+")
    bits = raw.split(":")
    try:
        seconds = 0.0
        for bit in bits:
            seconds = seconds * 60.0 + float(bit)
    except ValueError:
        return None
    return sign * seconds


#: How many recent samples the ETA rate is averaged over.
_ETA_WINDOW = 8
#: Minimum wall-clock span before the windowed rate is trusted over ffmpeg's speed.
_ETA_MIN_SPAN = 0.4


@dataclass
class _ProgressState:
    """Accumulates key=value lines until a ``progress=`` line closes the block.

    ETA deliberately does *not* use ffmpeg's ``speed``, which is a cumulative
    average over the whole run.  On a deep cut ffmpeg spends a long time decoding
    forward to the trim start before emitting anything, so its first reported
    speed is ~0.01x and an ETA built on it is off by two orders of magnitude.
    A rate measured across the last few samples tracks reality instead; ffmpeg's
    ``speed`` is still reported verbatim in :attr:`Progress.speed`, and is used as
    the ETA fallback until enough samples exist.
    """

    total: float
    out_time: float = 0.0
    fps: float = 0.0
    speed: float = 0.0
    pending: dict[str, str] = field(default_factory=dict)
    clock: Callable[[], float] = time.monotonic
    _samples: collections.deque[tuple[float, float]] = field(
        default_factory=lambda: collections.deque(maxlen=_ETA_WINDOW)
    )

    def _rate(self) -> float:
        """Output-seconds produced per wall-second, over the recent window."""
        if len(self._samples) < 2:
            return 0.0
        (t0, o0), (t1, o1) = self._samples[0], self._samples[-1]
        span = t1 - t0
        if span < _ETA_MIN_SPAN:
            return 0.0
        produced = o1 - o0
        return produced / span if produced > 0 else 0.0

    def feed(self, line: str) -> Progress | None:
        """Consume one ``key=value`` line; return a Progress at block end."""
        key, sep, value = line.partition("=")
        if not sep:
            return None
        key = key.strip()
        value = value.strip()

        if key == "progress":
            return self._emit(finished=value == "end")

        self.pending[key] = value
        return None

    def _emit(self, *, finished: bool) -> Progress:
        p = self.pending

        out_time: float | None = None
        if "out_time_us" in p:
            us = _parse_number(p["out_time_us"])
            out_time = us / 1_000_000.0 if us is not None else None
        if out_time is None and "out_time_ms" in p:
            # ffmpeg's out_time_ms is microseconds despite the name.
            us = _parse_number(p["out_time_ms"])
            out_time = us / 1_000_000.0 if us is not None else None
        if out_time is None and "out_time" in p:
            out_time = _parse_timecode(p["out_time"])
        if out_time is not None and out_time >= 0:
            self.out_time = out_time

        fps = _parse_number(p.get("fps", ""))
        if fps is not None and fps >= 0:
            self.fps = fps
        speed = _parse_number(p.get("speed", ""))
        if speed is not None and speed > 0:
            self.speed = speed

        self.pending = {}
        self._samples.append((self.clock(), self.out_time))

        total = self.total if self.total > 0 else 0.0
        if finished:
            fraction = 1.0
        elif total > 0:
            fraction = min(1.0, max(0.0, self.out_time / total))
        else:
            fraction = 0.0

        eta: float | None = None
        if finished:
            eta = 0.0
        elif total > 0:
            rate = self._rate() or self.speed
            if rate > 0:
                eta = max(0.0, (total - self.out_time) / rate)

        return Progress(
            out_time=self.out_time,
            total=total,
            fraction=fraction,
            fps=self.fps,
            speed=self.speed,
            eta_seconds=eta,
        )


# --------------------------------------------------------------------------- #
# the job
# --------------------------------------------------------------------------- #


class RenderJob:
    """Runs one ffmpeg render in a background thread.

    ``on_progress`` is called from the worker thread every time ffmpeg closes a
    ``-progress`` block (a few times per second); ``on_done`` is called exactly
    once — on success, on failure, on cancel and on an internal exception.  Qt
    callers must hop to the GUI thread inside the callbacks (signal emission is
    thread-safe and is the obvious way).
    """

    def __init__(self, argv: list[str], total_out_seconds: float) -> None:
        self.argv: list[str] = list(argv)
        self.total_out_seconds: float = float(total_out_seconds)
        self.cancelled: bool = False
        self.returncode: int | None = None

        self._proc: subprocess.Popen[str] | None = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._done_fired = False
        self._stderr_tail: collections.deque[str] = collections.deque(
            maxlen=STDERR_TAIL_LINES
        )

    # -- public API ---------------------------------------------------------

    def start(
        self,
        on_progress: Callable[[Progress], None],
        on_done: Callable[[int, str], None],
    ) -> None:
        """Launch ffmpeg and begin reporting. Returns immediately."""
        with self._lock:
            if self._thread is not None:
                raise RuntimeError("RenderJob already started")
            self._thread = threading.Thread(
                target=self._run,
                args=(on_progress, on_done),
                name="ffcut-render",
                daemon=True,
            )
        self._thread.start()

    def cancel(self) -> None:
        """Ask the render to stop: SIGTERM now, SIGKILL after a grace period.

        Safe to call before :meth:`start`, after completion, or twice.
        """
        self.cancelled = True
        with self._lock:
            proc = self._proc
        if proc is None or proc.poll() is not None:
            return
        try:
            proc.terminate()
        except (OSError, ValueError):
            return
        threading.Thread(
            target=self._kill_after_grace, args=(proc,), daemon=True
        ).start()

    def is_running(self) -> bool:
        """True while the ffmpeg process is alive."""
        with self._lock:
            proc = self._proc
        return proc is not None and proc.poll() is None

    def wait(self, timeout: float | None = None) -> None:
        """Block until the worker thread finishes (used by tests/CLI)."""
        thread = self._thread
        if thread is not None:
            thread.join(timeout)

    # -- internals ----------------------------------------------------------

    @staticmethod
    def _kill_after_grace(proc: subprocess.Popen[str]) -> None:
        try:
            proc.wait(timeout=CANCEL_GRACE)
        except subprocess.TimeoutExpired:
            try:
                proc.kill()
            except (OSError, ValueError):
                pass
        except (OSError, ValueError):
            pass

    def _drain_stderr(self, stream: object) -> None:
        """Keep the last STDERR_TAIL_LINES lines so failures stay diagnosable."""
        try:
            for line in stream:  # type: ignore[attr-defined]
                text = line.rstrip("\r\n")
                if text:
                    self._stderr_tail.append(text)
        except (OSError, ValueError):
            pass

    def _encoder_hint(self) -> str:
        """Explain a VA-API failure so the GUI can offer the libx264 fallback.

        Never auto-falls-back: the user must know which encoder actually ran.
        """
        if not any(is_vaapi_encoder(arg) for arg in self.argv):
            return ""
        blob = "\n".join(self._stderr_tail).lower()
        if any(marker in blob for marker in _VAAPI_FAILURE_MARKERS):
            return _VAAPI_HINT
        return ""

    def _fire_done(
        self, on_done: Callable[[int, str], None], code: int, extra: str = ""
    ) -> None:
        """Invoke on_done at most once, ever."""
        with self._lock:
            if self._done_fired:
                return
            self._done_fired = True
        self.returncode = code
        tail = "\n".join(self._stderr_tail)
        if extra:
            tail = f"{tail}\n{extra}" if tail else extra
        try:
            on_done(code, tail)
        except Exception:  # a broken callback must not crash the worker
            pass

    def _run(
        self,
        on_progress: Callable[[Progress], None],
        on_done: Callable[[int, str], None],
    ) -> None:
        state = _ProgressState(total=self.total_out_seconds)
        try:
            if self.cancelled:  # cancelled between construction and start
                self._fire_done(on_done, -1, "cancelled before launch")
                return
            try:
                proc = subprocess.Popen(
                    self.argv,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    stdin=subprocess.DEVNULL,
                    text=True,
                    bufsize=1,
                )
            except OSError as exc:
                self._fire_done(on_done, -1, f"failed to launch ffmpeg: {exc}")
                return

            with self._lock:
                self._proc = proc
            if self.cancelled:  # cancel raced the launch
                self.cancel()

            err_thread = threading.Thread(
                target=self._drain_stderr,
                args=(proc.stderr,),
                name="ffcut-stderr",
                daemon=True,
            )
            err_thread.start()

            stdout = proc.stdout
            assert stdout is not None
            while True:
                line = stdout.readline()
                if not line:
                    break
                progress = state.feed(line)
                if progress is not None:
                    try:
                        on_progress(progress)
                    except Exception:  # never let the GUI kill the reader
                        pass

            code = proc.wait()
            err_thread.join(timeout=5.0)
            try:
                stdout.close()
            except OSError:
                pass
            if proc.stderr is not None:
                try:
                    proc.stderr.close()
                except OSError:
                    pass

            if self.cancelled:
                extra = "render cancelled"
            elif code != 0:
                extra = self._encoder_hint()
            else:
                extra = ""
            self._fire_done(on_done, code, extra)
        except Exception as exc:  # pragma: no cover - defensive
            self._fire_done(on_done, -1, f"internal error: {exc!r}")
