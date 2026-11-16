"""Main window, menus, dialogs and wiring for ffcut.

Widget tree
-----------
MainWindow
 ├─ QMenuBar              File | Edit | Render
 ├─ QSplitter (vertical)  drag the player against the timeline; sizes persist
 │   ├─ player pane       QVideoWidget stage + the WMP12 control panel:
 │   │                    thin seek line over prev/back/(play)/forward/next,
 │   │                    with volume + Render on the right
 │   └─ timeline pane     TimelineWidget: video lane + one lane per audio track
 ├─ warning strip         shown only when the edit is empty
 └─ QStatusBar            play state · selection readout · keep/options summary

The timeline is the single source of truth for the edit: it owns the cut list,
the playhead (in EDITED time), the selection and the per-track mute/gain; this
module only consumes its public API.

Preview video
-------------
The timeline hands out ``preview_plan().video``: the picture as a list of RUNS,
each one contiguous stretch of one file, already in EDITED time.  Cuts, lifts,
appended clips and a second video lane all arrive as more runs, so the preview
has exactly one rule — *be inside the run the playhead is in* — and
:meth:`MainWindow._enter_run` is the only place that swaps the main player's
source, seeks it, or hands over to the gap clock.

Edited time is read back out of the run the player is in
(``run.edited_start + (position - run.src_in)``), never out of a global
source→edited mapping: with two files on the timeline the player's position is
only meaningful against the clip it is decoding.

Where NO run covers the playhead — a lift on the only video lane, a gap between
two blocks, an audio bed running past the end of the footage — the picture is
black and there is no player to take the time from, so ``_gap_timer`` advances
edited time itself until the next run starts.  That is also what keeps the
timecode moving after the video ends.

Preview audio
-------------
What you hear is what renders: every unmuted lane plays, at its lane gain.
``QMediaPlayer`` decodes exactly ONE audio track at a time, so the main player
carries the video plus the first unmuted embedded track and every *other*
unmuted lane — a second embedded track, a dropped mp3 — gets an audio-only
``_AuxPlayer`` of its own.  Those are created lazily: the common case (one
track, nothing dropped) spins up none, because each extra QMediaPlayer is
another chance at the stop() wedge documented in HANDOFF §1(c).  Qt caps an
output at unity, so the mix is normalised: the loudest lane plays at 1.0 and
the rest keep their distance below it — the render's balance, not its level.

The main player only carries audio while it is ALSO the whole picture —
``_carry_audio``, one file, no gaps.  The moment it has to swap sources or park
for a gap, its audio would swap and park with it, so every lane moves to an
``_AuxPlayer`` and the main player goes video-only.

Long work never runs on the GUI thread: probing, thumbnail extraction and
waveform rendering run in ``_Worker`` threads whose results come back as queued
signals, and ``render.RenderJob`` runs in its own thread with its callbacks
marshalled through ``_RenderBridge``.
"""

from __future__ import annotations

import atexit
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from functools import partial
from pathlib import Path
from typing import NamedTuple

from PyQt6.QtCore import (
    QEvent,
    QMimeData,
    QObject,
    QPointF,
    QRectF,
    QSize,
    Qt,
    QThread,
    QTimer,
    QUrl,
    pyqtSignal,
)
from PyQt6.QtGui import (
    QAction,
    QCloseEvent,
    QColor,
    QDesktopServices,
    QDragEnterEvent,
    QDropEvent,
    QFont,
    QFontMetricsF,
    QImage,
    QKeySequence,
    QLinearGradient,
    QMouseEvent,
    QPainter,
    QPaintEvent,
    QPen,
)
from PyQt6.QtMultimedia import QAudioOutput, QMediaPlayer
from PyQt6.QtMultimediaWidgets import QVideoWidget
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMenu,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollBar,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QSplitter,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from . import theme
from .probe import MediaInfo, cancel_all as cancel_probes, probe, resume as resume_probes
from .probe import AudioFileInfo, make_proxy, probe_audio, thumbnails_at, waveform_png
from .render import (
    CANCEL_GRACE,
    CONTAINERS,
    ENCODERS,
    Progress,
    RenderJob,
    RenderSettings,
    Segment,
    build_command,
    default_vaapi_device,
    is_crf_encoder,
    is_vaapi_encoder,
    total_output_seconds,
)
from .timeline import (
    WAVE_COLOR,
    PreviewPlan,
    PreviewVideo,
    TimelineWidget,
    format_tc,
)

#: Seconds without a progress callback before the dialog says ffmpeg is busy
#: decoding cut footage rather than encoding.
_STALL_AFTER: float = 2.0

CONFIG_DIR = Path.home() / ".config" / "ffcut"
CONFIG_PATH = CONFIG_DIR / "config.json"
#: Preview copies (Preferences ▸ Preview quality), kept across sessions.
PROXY_DIR = Path.home() / ".cache" / "ffcut" / "proxies"
#: Unused preview copies older than this are deleted at startup.
PROXY_MAX_AGE_DAYS = 14
PREVIEW_QUALITIES: list[tuple[str, str]] = [
    ("Full (the original file)", "full"),
    ("720p preview copy", "720"),
    ("540p preview copy", "540"),
]

VIDEO_SUFFIXES = {
    ".mp4", ".mkv", ".mov", ".webm", ".avi", ".m4v", ".ts", ".mts", ".flv", ".wmv",
}
VIDEO_FILTER = (
    f"Video files ({' '.join('*' + s for s in sorted(VIDEO_SUFFIXES))});;All files (*)"
)

#: Dropping one of these ADDS an audio lane under the existing ones (aligned to
#: 0:00) instead of replacing the loaded video.  See MainWindow.dropEvent.
AUDIO_SUFFIXES = {".mp3", ".wav", ".flac", ".opus", ".m4a", ".aac", ".ogg"}

RESOLUTIONS: list[tuple[str, str]] = [
    ("Source", "source"),
    ("2560 × 1440", "2560x1440"),
    ("1920 × 1080", "1920x1080"),
    ("Custom…", "custom"),
]

#: Most recent files remembered in the File ▸ Recent menu.
MAX_RECENT = 8
#: Starting split (player pane, timeline pane) in pixels, before the user has
#: ever dragged the handle.  Qt scales the pair to the real height and clamps it
#: to what each pane will accept, so these are proportions with units on.
DEFAULT_SPLIT: tuple[int, int] = (500, 300)
#: Thumbnail width handed to ffmpeg; the lane scales it to fit.
THUMB_W = 160
#: Waveform PNG height.  Much taller than the lane on purpose: the lane crops to
#: the trace's own bounding box and stretches that to full height (per-lane peak
#: normalisation, see ``timeline._wave_crop``), and showwavespic squashes a long
#: file into a few percent of whatever height it is given — so the PNG needs
#: spare vertical resolution for the stretch to stay smooth.
WAVE_H = 240
#: After a user-driven seek, ignore the player's own position reports for this
#: long so a scrub is not fought by stale positions coming back.
SEEK_GUARD_S = 0.25

#: How far an auxiliary audio player may drift from the main one before it is
#: re-seeked.  Seeks are explicit on every user seek and cut-skip; this is only
#: the guard against slow drift during free-running playback.  Re-seeking on
#: every position tick instead would stutter the audio, which is worse than the
#: few tens of milliseconds this lets through.
AUX_DRIFT_S = 0.15

#: How often the gap clock ticks while the picture is black and no player is
#: driving edited time.  ~30 Hz: fast enough that the playhead glides, cheap
#: enough that it costs nothing when it is running for a whole music bed.
GAP_TICK_MS = 33

#: Milliseconds a run must have actually PLAYED before an end-of-file signal is
#: believed to be about it — and only for a run that an end-of-file advance
#: ITSELF entered (``_run_from_eof``).  Qt announces the end of a file twice,
#: StoppedState and EndOfMedia, and the second arrives with the player sitting
#: exactly where the first one's advance just put it, having decoded nothing;
#: believing it skips the run that was just entered.
#:
#: Two narrower rules were tried and are both WRONG on their own.  A wall-clock
#: debounce swallows a genuine end-of-file landing just after a POSITION-driven
#: advance (a short run that ends at its file's end), and progress alone
#: swallows the genuine one whenever a run is entered within this window of
#: where its file really ends — a seek to 20 ms before the end, say.  Measured,
#: both freeze the transport on "▶ Playing", which is worse than the skip.
#: Together — was this entry made BY an end-of-file, and has it played? — only
#: the genuinely indistinguishable case is left.
EOF_PROGRESS_MS = 40

#: A seek closer than this to where the player already is would not move the
#: picture, so it is not armed at all -- an armed seek silences position
#: reports until it lands, and a no-op one would silence them forever.
SEEK_SLOP_MS = 2

#: Tolerance on a run boundary, in seconds.  A player reporting a position this
#: close to the end of its run has finished it; anything tighter and a demuxer
#: that stops a frame early leaves the preview parked forever.
RUN_EPS = 0.03

#: Total seconds closeEvent will spend joining asset workers, across all of
#: them.  With the ffmpeg children killed first this is never approached; it
#: only bounds a pathological case.
WORKER_JOIN_S = 5.0

#: Seconds the player teardown gets before the process is forced out.  Qt's
#: QMediaPlayer.stop() deadlocks against the QFFmpeg audio renderer roughly one
#: run in 60 (notes/hang-evidence/); everything durable is already flushed by
#: the time we get there, so a forced exit costs nothing and beats a frozen
#: window the user has to kill by hand.
PLAYER_TEARDOWN_S = 5.0


def _dir_key(path: str) -> str:
    """A short, stable, filesystem-safe directory name for one media file.

    Thumbnails are fetched per file now, so two files must not write into one
    directory — and a file's own name is neither unique across directories nor
    safe to paste into a path.
    """
    return hashlib.sha1(path.encode("utf-8", "surrogateescape")).hexdigest()[:16]


def _force_exit() -> None:
    """Last resort when the player teardown wedges (see PLAYER_TEARDOWN_S).

    Announced on stderr first: a silent ``os._exit(0)`` from a timer thread
    would look exactly like a clean exit, including under a test runner.
    """
    print("ffcut: player teardown wedged, forcing exit", file=sys.stderr, flush=True)
    os._exit(0)

#: Workers that outlived the join budget, detached from the window so their
#: destruction cannot abort the process.  Leaked deliberately, and only ever on
#: the way out.
_ORPHANS: list[_Worker] = []


def _fmt_hms(seconds: float) -> str:
    """``1:02:03`` / ``02:03`` for elapsed & ETA readouts."""
    total = max(0, int(round(seconds)))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


# --------------------------------------------------------------------------- #
# Settings
# --------------------------------------------------------------------------- #
@dataclass
class RenderOptions:
    """Everything on the render options panel, minus the output directory."""

    mbps: int = 60
    fps: int = 60
    tmix_frames: int = 7
    resolution: str = "2560x1440"  # "source" | "2560x1440" | "1920x1080" | "custom"
    custom_w: int = 2560
    custom_h: int = 1440
    encoder: str = "h264_vaapi"
    container: str = "mp4"
    crf: int = 30
    av1_preset: int = 8

    def dimensions(self) -> tuple[int | None, int | None]:
        if self.resolution == "source":
            return None, None
        if self.resolution == "custom":
            return self.custom_w, self.custom_h
        width, _, height = self.resolution.partition("x")
        try:
            return int(width), int(height)
        except ValueError:
            return None, None

    @property
    def suffix(self) -> str:
        return f".{self.container}"

    def summary(self) -> str:
        width, height = self.dimensions()
        res = "source size" if width is None else f"{width}×{height}"
        tmix = "tmix off" if self.tmix_frames <= 0 else f"tmix {self.tmix_frames}"
        rate = (
            f"CRF {self.crf} · preset {self.av1_preset}"
            if is_crf_encoder(self.encoder)
            else f"{self.mbps} Mbps"
        )
        label = next((lbl for lbl, value in ENCODERS if value == self.encoder), self.encoder)
        return f"{rate} · {self.fps} fps · {res} · {tmix} · {label} · {self.container.upper()}"


@dataclass
class Prefs:
    """Edit ▸ Preferences."""

    theme: str = "system"  # system | dark | light
    skip_seconds: int = 5  # transport skip button step
    preview: str = "full"  # full | 720 | 540 -- see PREVIEW_QUALITIES
    jobs: int = 4  # thumbnail ffmpeg grabs at once


@dataclass
class Config:
    last_open_dir: str = str(Path.home())
    last_save_dir: str = str(Path.home())
    recent: list[str] = field(default_factory=list)
    render: RenderOptions = field(default_factory=RenderOptions)
    prefs: Prefs = field(default_factory=Prefs)
    #: [player, timeline] heights in pixels.  Empty = never dragged, so the
    #: built-in default split applies.
    splitter: list[int] = field(default_factory=list)

    @staticmethod
    def _fill(target: object, stored: object) -> None:
        """Copy same-typed fields from a JSON dict onto a dataclass instance."""
        if not isinstance(stored, dict):
            return
        for name, value in stored.items():
            current = getattr(target, name, None)
            if current is None or not hasattr(target, name):
                continue
            if isinstance(value, type(current)) and not isinstance(value, bool):
                if isinstance(value, int):  # Qt spin boxes take a C int
                    value = max(-(2**31), min(2**31 - 1, value))
                setattr(target, name, value)

    @staticmethod
    def load() -> "Config":
        cfg = Config()
        try:
            raw = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return cfg
        if not isinstance(raw, dict):
            return cfg
        for key in ("last_open_dir", "last_save_dir"):
            if isinstance(raw.get(key), str):
                setattr(cfg, key, raw[key])
        if isinstance(raw.get("recent"), list):
            cfg.recent = [p for p in raw["recent"] if isinstance(p, str)][:MAX_RECENT]
        if isinstance(raw.get("splitter"), list):
            # Same discipline as _fill: wrong types are dropped, not coerced,
            # and a degenerate pair (zero/negative) falls back to the default.
            sizes = [
                int(v)
                for v in raw["splitter"]
                if isinstance(v, (int, float)) and not isinstance(v, bool)
                and math.isfinite(v) and abs(v) < 100_000
            ]
            if len(sizes) == 2 and all(v > 0 for v in sizes):
                cfg.splitter = sizes
        Config._fill(cfg.render, raw.get("render"))
        if cfg.render.encoder not in {value for _, value in ENCODERS}:
            # Includes "h264_nvenc" from before the RX 9070 XT swap → h264_vaapi.
            cfg.render.encoder = RenderOptions().encoder
        if cfg.render.container not in {value for _, value in CONTAINERS}:
            cfg.render.container = RenderOptions().container
        Config._fill(cfg.prefs, raw.get("prefs"))
        if cfg.prefs.theme not in theme.MODES:
            cfg.prefs.theme = "system"
        cfg.prefs.skip_seconds = max(1, min(120, int(cfg.prefs.skip_seconds)))
        if cfg.prefs.preview not in {value for _, value in PREVIEW_QUALITIES}:
            cfg.prefs.preview = "full"
        cfg.prefs.jobs = max(1, min(8, int(cfg.prefs.jobs)))
        return cfg

    def remember(self, path: str) -> None:
        """Push ``path`` to the front of the recent list."""
        self.recent = [path] + [p for p in self.recent if p != path]
        del self.recent[MAX_RECENT:]

    def save(self) -> None:
        try:
            CONFIG_DIR.mkdir(parents=True, exist_ok=True)
            payload = {
                "last_open_dir": self.last_open_dir,
                "last_save_dir": self.last_save_dir,
                "recent": list(self.recent),
                "render": asdict(self.render),
                "prefs": asdict(self.prefs),
                "splitter": list(self.splitter),
            }
            # Write then rename: a crash mid-write must not leave an empty config.
            tmp = CONFIG_PATH.with_name(CONFIG_PATH.name + ".tmp")
            tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
            os.replace(tmp, CONFIG_PATH)
        except OSError:
            pass  # a read-only HOME must never take the app down


# --------------------------------------------------------------------------- #
# Workers
# --------------------------------------------------------------------------- #
class _Worker(QThread):
    """Run one callable off the GUI thread.

    ``done``/``failed`` are emitted from the worker thread and therefore arrive
    queued on the GUI thread, which is the only place they are consumed.
    """

    done = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, fn: Callable[[], object], parent: QObject) -> None:
        super().__init__(parent)
        self._fn = fn

    def run(self) -> None:  # noqa: D102
        try:
            result = self._fn()
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")
        else:
            self.done.emit(result)


@dataclass
class _Slot:
    """Latest-request-wins queue in front of at most one running worker.

    Zoom and pan fire asset requests in bursts.  Only the newest matters, and
    running them all would fan out into dozens of ffmpeg processes, so a slot
    holds one pending request and starts it when the current worker finishes.
    """

    pending: tuple | list | None = None
    worker: _Worker | None = None


class _RenderBridge(QObject):
    """Turns ``RenderJob``'s worker-thread callbacks into queued Qt signals."""

    progressed = pyqtSignal(object)  # Progress
    finished_ = pyqtSignal(int, str)

    def on_progress(self, progress: Progress) -> None:
        self.progressed.emit(progress)

    def on_done(self, code: int, message: str) -> None:
        self.finished_.emit(int(code), str(message))


# --------------------------------------------------------------------------- #
# Preview audio
# --------------------------------------------------------------------------- #
def _gain_linear(db: float) -> float:
    """dB -> linear amplitude: ffmpeg's ``volume=NdB`` and ``QAudioOutput``'s
    volume are both amplitude multipliers, so the two agree.  Qt caps its
    output at 1.0; ``MainWindow._apply_preview_volumes`` normalises the whole
    mix under that cap rather than clamping one lane."""
    return 10.0 ** (float(db) / 20.0)


class _PreviewSource(NamedTuple):
    """One thing the preview has to play, as the timeline describes it.

    ``spans`` are ``(edited_start, src_in, src_out)`` triples: where this
    source's audio sits on the EDITED timeline, and which part of the file
    fills it.  One triple per contiguous run — a clip, minus whatever a cut, a
    lift or a lane above takes out of it — so :meth:`source_at` is the whole
    contract and nothing here has to know how the timeline is modelled.
    """

    path: str
    stream: int
    lane: int  # the lane's stable key: two lanes on one file are two players
    gain_db: float
    muted: bool
    spans: tuple[tuple[float, float, float], ...]

    @property
    def key(self) -> tuple[str, int, int]:
        """Identity across re-plans.  Deliberately excludes gain and mute, so
        turning a lane down does not tear its player down and build a new one."""
        return (self.path, self.stream, self.lane)

    def source_at(self, t_edited: float) -> float | None:
        """Position in THIS file for edited time ``t_edited``.

        ``None`` when the source has nothing there — past the end of a short
        music bed, or between its clips — which is the cue to pause it rather
        than let it free-run against the video.
        """
        for edited_start, src_in, src_out in self.spans:
            length = src_out - src_in
            if length <= 0:
                continue
            offset = t_edited - edited_start
            if -1e-6 <= offset < length:
                return src_in + max(0.0, offset)
        return None


class TitleStage(QWidget):
    """The paused frame as a still, with the text blocks on it to drag and size.

    Shown instead of the video while paused over a text block: QVideoWidget
    draws through a native child window, which ordinary widgets cannot be
    painted over, and a QGraphicsVideoItem preview would convert every frame on
    the CPU (the cost the preview copies exist to avoid).  Drag a title to move
    it, drag its corner square to resize it, double-click to edit its text.
    Positions are fractions of the frame, exactly what the render's drawtext uses.
    """

    picked = pyqtSignal(int)  # cid pressed
    moved = pyqtSignal(int, float, float, float)  # cid, x, y, size
    released = pyqtSignal()
    editRequested = pyqtSignal(int)

    HANDLE = 10.0

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.frame: QImage | None = None
        self.frame_size = QSize(16, 9)
        self.titles: list[object] = []  # timeline Clips carrying .title
        self.selected: frozenset[int] = frozenset()
        self._drag: tuple[str, int, QPointF, float, float, float] | None = None
        self.setMouseTracking(True)

    def frame_rect(self) -> QRectF:
        """Where the frame sits: fitted and centred, letterboxed like the video."""
        fw, fh = max(1, self.frame_size.width()), max(1, self.frame_size.height())
        scale = min(self.width() / fw, self.height() / fh)
        w, h = fw * scale, fh * scale
        return QRectF((self.width() - w) / 2, (self.height() - h) / 2, w, h)

    def _font(self, size: float) -> QFont:
        font = QFont("Sans")
        font.setBold(True)
        font.setPixelSize(max(1, round(size * self.frame_rect().height())))
        return font

    def text_rect(self, title: object) -> QRectF:
        t = title.title  # type: ignore[attr-defined]
        frame = self.frame_rect()
        box = QFontMetricsF(self._font(t.size)).boundingRect(
            QRectF(0, 0, 1e6, 1e6), int(Qt.AlignmentFlag.AlignHCenter), t.text
        )
        cx, cy = frame.left() + t.x * frame.width(), frame.top() + t.y * frame.height()
        return QRectF(cx - box.width() / 2, cy - box.height() / 2, box.width(), box.height())

    def _handle(self, rect: QRectF) -> QRectF:
        return QRectF(rect.right() - self.HANDLE / 2, rect.bottom() - self.HANDLE / 2,
                      self.HANDLE, self.HANDLE)

    def paintEvent(self, _event: QPaintEvent) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("black"))
        frame = self.frame_rect()
        if self.frame is not None and not self.frame.isNull():
            painter.drawImage(frame, self.frame)
        for clip in self.titles:
            t = clip.title  # type: ignore[attr-defined]
            rect = self.text_rect(clip)
            painter.setFont(self._font(t.size))
            flags = int(Qt.AlignmentFlag.AlignHCenter)
            painter.setPen(QColor(0, 0, 0, 153))  # the render's shadowcolor=black@0.6
            painter.drawText(rect.translated(2, 2), flags, t.text)
            painter.setPen(QColor(t.color))
            painter.drawText(rect, flags, t.text)
            if clip.cid in self.selected:  # type: ignore[attr-defined]
                painter.setPen(QPen(QColor("#4aa3df"), 1.0, Qt.PenStyle.DashLine))
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawRect(rect)
                painter.fillRect(self._handle(rect), QColor("#4aa3df"))
        painter.end()

    def _hit(self, pos: QPointF) -> tuple[str, object] | None:
        for clip in reversed(self.titles):  # topmost (drawn last) first
            rect = self.text_rect(clip)
            if self._handle(rect).contains(pos):
                return "size", clip
            if rect.adjusted(-4, -4, 4, 4).contains(pos):
                return "move", clip
        return None

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        hit = self._hit(event.position())
        if hit is None or event.button() != Qt.MouseButton.LeftButton:
            return
        mode, clip = hit
        t = clip.title  # type: ignore[attr-defined]
        self._drag = (mode, clip.cid, event.position(), t.x, t.y, t.size)  # type: ignore[attr-defined]
        self.picked.emit(clip.cid)  # type: ignore[attr-defined]

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if self._drag is None:
            hit = self._hit(event.position())
            self.setCursor(
                Qt.CursorShape.SizeFDiagCursor if hit and hit[0] == "size"
                else Qt.CursorShape.SizeAllCursor if hit else Qt.CursorShape.ArrowCursor
            )
            return
        mode, cid, origin, x, y, size = self._drag
        frame = self.frame_rect()
        dx = (event.position().x() - origin.x()) / max(1.0, frame.width())
        dy = (event.position().y() - origin.y()) / max(1.0, frame.height())
        if mode == "move":
            x, y = min(1.0, max(0.0, x + dx)), min(1.0, max(0.0, y + dy))
        else:  # the corner: dragging down/right grows the text
            size = min(0.5, max(0.01, size + dy))
        self.moved.emit(cid, x, y, size)

    def mouseReleaseEvent(self, _event: QMouseEvent) -> None:  # noqa: N802
        if self._drag is not None:
            self._drag = None
            self.released.emit()

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        hit = self._hit(event.position())
        if hit is not None:
            self.editRequested.emit(hit[1].cid)  # type: ignore[attr-defined]


class _AuxPlayer:
    """An extra audio-only player for a source the main player cannot decode.

    ``QMediaPlayer`` decodes one audio track at a time, so a second embedded
    track or a dropped file needs its own player.  It carries no video output:
    it exists purely to put that lane's audio on the same sink at the same
    position as the picture.

    Lifetime is owned by :class:`MainWindow`; ``close`` is called from inside
    ``closeEvent``'s watchdog-protected step and latches ``_closing`` first, for
    the same reason ``MainWindow._on_media_status`` checks it — Qt delivers
    ``mediaStatusChanged`` synchronously from ``stop()``, and re-entering the
    player from that callback is the teardown deadlock of HANDOFF §1(a).
    """

    def __init__(self, source: _PreviewSource, parent: QObject) -> None:
        self.source = source  # re-assigned on every re-plan (same ``key``)
        self._closing = False
        self._starved = False
        self.output = QAudioOutput(parent)
        self.output.setVolume(0.0)
        self.player = QMediaPlayer(parent)
        self.player.setAudioOutput(self.output)
        self.player.mediaStatusChanged.connect(self._on_status)
        self.player.setSource(QUrl.fromLocalFile(source.path))

    # -- state ---------------------------------------------------------------
    def apply_volume(self, scale: float) -> None:
        """Lane gain times the window's mix scale; a muted lane is silent."""
        self.output.setMuted(self.source.muted)
        self.output.setVolume(
            0.0 if self.source.muted else scale * _gain_linear(self.source.gain_db)
        )

    # -- sync ----------------------------------------------------------------
    def follow(self, t_edited: float, playing: bool, *, force: bool) -> None:
        """Put this player where edited time ``t_edited`` says it should be."""
        if self._closing:
            return
        target = self.source.source_at(t_edited)
        self._starved = target is None
        if target is None:
            # Out of material: park it rather than let it run on under a video
            # it no longer has audio for.
            if self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
                self.player.pause()
            return
        target_ms = max(0, int(target * 1000))
        if force or abs(self.player.position() - target_ms) > int(AUX_DRIFT_S * 1000):
            self.player.setPosition(target_ms)
        self.set_playing(playing)

    def set_playing(self, playing: bool) -> None:
        if self._closing:
            return
        want = playing and not self._starved and not self.source.muted
        state = self.player.playbackState()
        if want and state != QMediaPlayer.PlaybackState.PlayingState:
            self.player.play()
        elif not want and state == QMediaPlayer.PlaybackState.PlayingState:
            self.player.pause()

    # -- lifecycle -----------------------------------------------------------
    def _on_status(self, status: QMediaPlayer.MediaStatus) -> None:
        if self._closing:
            return
        if status == QMediaPlayer.MediaStatus.LoadedMedia:
            self._apply_track()
        elif status == QMediaPlayer.MediaStatus.EndOfMedia:
            self._starved = True

    def _apply_track(self) -> None:
        """Select this source's stream, once the demuxer has listed them.

        Only meaningful for an *embedded* source (an aux player opened on the
        video file for its second audio track); a dropped file has one stream
        and index 0 is already right.
        """
        tracks = self.player.audioTracks()
        if not tracks:
            return
        want = max(0, min(len(tracks) - 1, int(self.source.stream)))
        if self.player.activeAudioTrack() != want:
            self.player.setActiveAudioTrack(want)

    def close(self) -> None:
        """Tear down.  Latch first: ``stop()`` delivers status changes inline."""
        self._closing = True
        self.output.setMuted(True)
        self.player.stop()
        self.player.setSource(QUrl())
        self.player.deleteLater()
        self.output.deleteLater()


class _SubtreeFilter(QObject):
    """One event filter over a widget and everything it ever grows.

    Two jobs, picked per root:

    * ``drops`` hands drag & drop to the window.  ``QVideoWidget`` does not
      accept drops, but the render surface it keeps inside itself does — and
      then ignores them.  Qt's ``findDnDTarget`` walks UP from the widget under
      the cursor to the first one accepting drops, finds that child and stops,
      so a file dropped on the preview never reached ``MainWindow``: the big
      black box was the one place in the window that refused a drop.  The
      widget is passed along because ``event.position()`` is relative to it.
    * ``menu`` routes a right-click anywhere in the timeline to the lane menu.
      The lane headers are child widgets, so their context-menu events never
      reach the timeline's own handler.  The lane scrollbar keeps its own.

    ``QVideoWidget`` builds its surface lazily, so a child that appears after
    the filter was installed is attached the moment it is added.
    """

    def __init__(
        self, window: "MainWindow", root: QWidget, *, drops: bool = False, menu: bool = False
    ) -> None:
        super().__init__(window)
        self._window, self._drops, self._menu = window, drops, menu
        self.attach(root)

    def attach(self, widget: QWidget) -> None:
        for each in (widget, *widget.findChildren(QWidget)):
            if self._drops:
                each.setAcceptDrops(True)
            each.installEventFilter(self)

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:  # noqa: N802
        kind = event.type()
        if self._drops and kind in (QEvent.Type.DragEnter, QEvent.Type.DragMove):
            self._window.handle_drag(event)  # type: ignore[arg-type]
            return event.isAccepted()
        if self._drops and kind == QEvent.Type.Drop:
            self._window.handle_drop(event, obj)  # type: ignore[arg-type]
            return event.isAccepted()
        if self._menu and kind == QEvent.Type.ContextMenu and not isinstance(obj, QScrollBar):
            self._window.show_lane_menu(event.globalPos())  # type: ignore[attr-defined]
            event.accept()
            return True
        if kind == QEvent.Type.ChildAdded and isinstance(obj, QWidget):
            child = event.child()  # type: ignore[attr-defined]
            if isinstance(child, QWidget):
                self.attach(child)
        return False


# --------------------------------------------------------------------------- #
# Transport
# --------------------------------------------------------------------------- #
class SeekBar(QWidget):
    """The thin WMP12 seek line that runs above the transport buttons.

    Painted, not a QSlider: WMP12's scrubber is a 4px inset groove with a blue
    fill and a small round knob, and getting that out of ``QSlider`` subcontrols
    costs more QSS than it costs QPainter.

    Times here are EDITED seconds — the same coordinate everything else in the
    app speaks.  The conversion to source time happens once, in the window's
    ``_goto_edited``.  This is the *player's* scrubber; the timeline below is a
    separate widget with its own playhead, and both are driven from the same
    edited-time position.
    """

    seeked = pyqtSignal(float)  # edited seconds

    #: Side inset, so the knob at 0 % / 100 % still lands inside the widget.
    MARGIN = 6

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("seekBar")
        self.setFixedHeight(theme.METRICS["seek"])
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMouseTracking(True)
        self._total = 0.0
        self._position = 0.0
        self._dragging = False
        self._hover = False

    # -- state ---------------------------------------------------------------
    def set_position(self, seconds: float, total: float) -> None:
        """Where the playhead is.  Ignored mid-drag: the pointer wins, exactly
        like the timeline's own scrub guard."""
        if self._dragging:
            return
        total = max(0.0, float(total))
        seconds = max(0.0, min(total, float(seconds)))
        if (seconds, total) != (self._position, self._total):
            self._position, self._total = seconds, total
            self.update()

    def position(self) -> float:
        return self._position

    def fraction(self) -> float:
        return self._position / self._total if self._total > 0 else 0.0

    # -- geometry ------------------------------------------------------------
    def _groove(self) -> QRectF:
        return QRectF(
            float(self.MARGIN),
            (self.height() - 4.0) / 2.0,
            max(1.0, self.width() - 2.0 * self.MARGIN),
            4.0,
        )

    def _time_at(self, x: float) -> float:
        groove = self._groove()
        fraction = (x - groove.left()) / groove.width()
        return max(0.0, min(1.0, fraction)) * self._total

    # -- input ---------------------------------------------------------------
    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() != Qt.MouseButton.LeftButton or self._total <= 0:
            event.ignore()
            return
        self._dragging = True
        self._scrub(event.position().x())

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if self._dragging:
            self._scrub(event.position().x())

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if self._dragging:
            self._dragging = False
            self._scrub(event.position().x())

    def enterEvent(self, event: QEvent) -> None:  # noqa: N802
        self._hover = True
        self.update()

    def leaveEvent(self, event: QEvent) -> None:  # noqa: N802
        self._hover = False
        self.update()

    def _scrub(self, x: float) -> None:
        self._position = self._time_at(float(x))
        self.update()
        self.seeked.emit(self._position)

    # -- paint ---------------------------------------------------------------
    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802
        t = theme.TOKENS[theme.variant(QApplication.instance())]
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        groove = self._groove()
        live = self._total > 0 and self.isEnabled()

        # Inset groove: dark at the top, lighter at the foot — a channel cut
        # into the glass rather than a bar laid on top of it.
        track = QLinearGradient(QPointF(0, groove.top()), QPointF(0, groove.bottom()))
        track.setColorAt(0.0, QColor(t["seek_track"]).darker(118))
        track.setColorAt(1.0, QColor(t["seek_track"]).lighter(112))
        painter.setPen(QPen(QColor(t["gloss_edge"]), 1.0))
        painter.setBrush(track)
        painter.drawRoundedRect(groove.adjusted(0.5, 0.5, -0.5, -0.5), 2.0, 2.0)

        if live:
            width = groove.width() * self.fraction()
            if width > 1.0:
                filled = QRectF(groove.left(), groove.top(), width, groove.height())
                fill = QLinearGradient(
                    QPointF(0, filled.top()), QPointF(0, filled.bottom())
                )
                base = QColor(t["seek_fill"])
                fill.setColorAt(0.0, base.lighter(145))
                fill.setColorAt(0.5, base)
                fill.setColorAt(1.0, base.darker(118))
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(fill)
                painter.drawRoundedRect(filled, 2.0, 2.0)

            # The knob: a small glass dome, same family as the play button.
            diameter = float(self.height() - 4)
            centre = QPointF(
                groove.left() + groove.width() * self.fraction(), groove.center().y()
            )
            knob = QRectF(
                centre.x() - diameter / 2.0,
                centre.y() - diameter / 2.0,
                diameter,
                diameter,
            )
            face = QLinearGradient(QPointF(0, knob.top()), QPointF(0, knob.bottom()))
            face.setColorAt(0.0, QColor(t["play_face1"]))
            face.setColorAt(1.0, QColor(t["play_face2"]))
            rim = QColor(t["accent"] if (self._hover or self._dragging) else t["play_rim"])
            painter.setPen(QPen(rim, 1.0))
            painter.setBrush(face)
            painter.drawEllipse(knob)
        painter.end()


class TransportBar(QFrame):
    """The Windows-7-Media-Player control panel.

    A thin seek line across the top; under it the button row, centred on the big
    round glass play button with flat skip pills either side and cut-to-cut
    jumps on the outside.  The timecode sits at the left, and callers park the
    volume slider and the Render button on the right via :meth:`add_right`.

    Metrics come from :data:`theme.METRICS`.  The glyphs are *painted* (see
    ``theme.media_icon``) in tokens from the active theme, so the play triangle
    can be WMP blue on its white dome while the skip glyphs stay slate — and all
    of them are repainted whenever the application palette changes.
    """

    playPause = pyqtSignal()
    skip = pyqtSignal(int)  # -1 / +1 (multiplied by the skip interval)
    jumpCut = pyqtSignal(int)  # -1 / +1 splice point
    seeked = pyqtSignal(float)  # edited seconds, from the seek line

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("transportBar")
        self.setFrameShape(QFrame.Shape.NoFrame)
        self._playing = False
        pad = theme.METRICS["bar_pad"]

        column = QVBoxLayout(self)
        column.setContentsMargins(pad + 2, pad, pad + 2, pad)
        column.setSpacing(2)

        self.seek = SeekBar(self)
        self.seek.seeked.connect(self.seeked)
        column.addWidget(self.seek)

        self.prev_cut = self._button("prev", "Previous cut point")
        self.back = self._button("back", "Skip back")
        self.play = self._button("play", "Play / pause (Space)", play=True)
        self.forward = self._button("forward", "Skip forward")
        self.next_cut = self._button("next", "Next cut point")
        self.buttons = (self.prev_cut, self.back, self.play, self.forward, self.next_cut)

        self.time_label = QLabel("00:00.00 / 00:00.00")
        self.time_label.setObjectName("timecode")

        self._left = self._cell(self.time_label, stretch="trailing")
        self._right = self._cell(stretch="leading")
        centre = self._cell()
        middle = centre.layout()
        assert isinstance(middle, QHBoxLayout)
        middle.setSpacing(4)
        for button in self.buttons:
            if button is self.play:
                middle.addSpacing(4)
            middle.addWidget(button, 0, Qt.AlignmentFlag.AlignVCenter)
            if button is self.play:
                middle.addSpacing(4)

        row = QGridLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setHorizontalSpacing(6)
        row.addWidget(self._left, 0, 0)
        row.addWidget(centre, 0, 1, Qt.AlignmentFlag.AlignHCenter)
        row.addWidget(self._right, 0, 2)
        row.setColumnStretch(0, 1)
        row.setColumnStretch(2, 1)
        column.addLayout(row)

        self.prev_cut.clicked.connect(lambda: self.jumpCut.emit(-1))
        self.next_cut.clicked.connect(lambda: self.jumpCut.emit(1))
        self.back.clicked.connect(lambda: self.skip.emit(-1))
        self.forward.clicked.connect(lambda: self.skip.emit(1))
        self.play.clicked.connect(self.playPause.emit)
        self.refresh_icons()

    @staticmethod
    def _cell(*widgets: QWidget, stretch: str = "") -> QWidget:
        """One column of the button row; ``stretch`` pins its contents to the
        left ("trailing") or the right ("leading") edge of the column."""
        host = QWidget()
        layout = QHBoxLayout(host)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        if stretch == "leading":
            layout.addStretch(1)
        for widget in widgets:
            layout.addWidget(widget)
        if stretch == "trailing":
            layout.addStretch(1)
        return host

    def add_right(self, *widgets: QWidget) -> None:
        """Park ``widgets`` at the right end of the button row."""
        layout = self._right.layout()
        assert isinstance(layout, QHBoxLayout)
        for widget in widgets:
            layout.addWidget(widget)
        self._balance()

    def _balance(self) -> None:
        """Equal minimum width on the outer columns, so the dome sits dead
        centre no matter how wide the right-hand cluster happens to be."""
        want = max(self._left.sizeHint().width(), self._right.sizeHint().width())
        if self._left.minimumWidth() != want or self._right.minimumWidth() != want:
            self._left.setMinimumWidth(want)
            self._right.setMinimumWidth(want)

    def event(self, event: QEvent) -> bool:  # noqa: D102
        # A child's size hint changed — re-centre before the layout runs.
        if event.type() == QEvent.Type.LayoutRequest:
            self._balance()
        return super().event(event)

    def _button(self, glyph: str, tip: str, *, play: bool = False) -> QToolButton:
        button = QToolButton(self)
        button.setToolTip(tip)
        button.setAutoRaise(False)
        button.setCursor(Qt.CursorShape.PointingHandCursor)
        if play:
            button.setObjectName("playButton")
            size = theme.METRICS["icon_play"]
        else:
            button.setProperty("transport", "skip")
            size = theme.METRICS["icon"]
        button.setIconSize(QSize(size, size))
        button._glyph = glyph  # type: ignore[attr-defined]
        return button

    def refresh_icons(self) -> None:
        """Repaint every glyph in the ink the active theme asks for."""
        tokens = theme.TOKENS[theme.variant(QApplication.instance())]
        for button in self.buttons:
            glyph = button._glyph  # type: ignore[attr-defined]
            colour = tokens["glyph_skip"]
            if button is self.play:
                glyph = "pause" if self._playing else "play"
                colour = tokens["glyph"]  # WMP blue, on the white dome
            button.setIcon(theme.media_icon(glyph, colour, button.iconSize().width()))

    def changeEvent(self, event: QEvent) -> None:  # noqa: N802
        if event.type() in (QEvent.Type.PaletteChange, QEvent.Type.ApplicationPaletteChange):
            self.refresh_icons()
            self.seek.update()
        super().changeEvent(event)

    def set_playing(self, playing: bool) -> None:
        self._playing = playing
        self.refresh_icons()

    def set_skip_seconds(self, seconds: int) -> None:
        self.back.setToolTip(f"Skip back {seconds}s")
        self.forward.setToolTip(f"Skip forward {seconds}s")

    def set_enabled_all(self, enabled: bool) -> None:
        for button in self.buttons:
            button.setEnabled(enabled)
        self.seek.setEnabled(enabled)


# --------------------------------------------------------------------------- #
# Preferences
# --------------------------------------------------------------------------- #
def _spin(
    low: int, high: int, value: int, *, step: int = 1,
    suffix: str = "", prefix: str = "", tip: str = "",
) -> QSpinBox:
    """One configured QSpinBox — the options form needs seven of them."""
    box = QSpinBox()
    box.setRange(low, high)
    box.setSingleStep(step)
    box.setValue(value)
    box.setSuffix(suffix)
    box.setPrefix(prefix)
    box.setToolTip(tip)
    return box


def _combo(items: "tuple[tuple[str, str], ...] | list[tuple[str, str]]", tip: str = "") -> QComboBox:
    """A QComboBox of ``(label, data)`` pairs."""
    combo = QComboBox()
    for label, value in items:
        combo.addItem(label, value)
    combo.setToolTip(tip)
    return combo


def _row(*widgets: QWidget) -> QWidget:
    """Widgets side by side in a form field, left aligned."""
    host = QWidget()
    layout = QHBoxLayout(host)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(6)
    for widget in widgets:
        layout.addWidget(widget)
    layout.addStretch(1)
    return host


class PreferencesDialog(QDialog):
    """Theme + skip interval.  The theme is applied *live* as the combo changes;
    Cancel puts the previous theme back."""

    def __init__(self, prefs: Prefs, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Preferences")
        self._original_theme = prefs.theme

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(10)

        form = QFormLayout()
        form.setSpacing(8)
        self.theme_box = _combo(
            [("System", "system"), ("Dark", "dark"), ("Light", "light")],
            "System follows the desktop palette; Dark and "
            "Light override it. Applied immediately.",
        )
        self.theme_box.setCurrentIndex(max(0, self.theme_box.findData(prefs.theme)))
        form.addRow("Theme", self.theme_box)

        self.skip = _spin(
            1, 120, prefs.skip_seconds, suffix=" s",
            tip="Step used by the transport skip-back / skip-forward buttons.",
        )
        form.addRow("Skip interval", self.skip)

        self.preview = _combo(
            PREVIEW_QUALITIES,
            "What the preview plays. A preview copy is built once per file in the "
            "background (on the GPU) and cached; until it is ready the original "
            "plays. Renders always use the original.",
        )
        self.preview.setCurrentIndex(max(0, self.preview.findData(prefs.preview)))
        form.addRow("Preview quality", self.preview)
        self.jobs = _spin(
            1, 8, prefs.jobs,
            tip="How many thumbnail extractions may run at once. Lower = lighter "
            "on the CPU while a file loads, slower thumbnails.",
        )
        form.addRow("Background jobs", self.jobs)
        layout.addLayout(form)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.theme_box.currentIndexChanged.connect(
            lambda: theme.apply(QApplication.instance(), str(self.theme_box.currentData()))
        )

    def prefs(self) -> Prefs:
        return Prefs(
            str(self.theme_box.currentData()), int(self.skip.value()),
            str(self.preview.currentData()), int(self.jobs.value()),
        )

    def reject(self) -> None:  # noqa: D102 - undo the live preview
        theme.apply(QApplication.instance(), self._original_theme)
        super().reject()


# --------------------------------------------------------------------------- #
# Render options
# --------------------------------------------------------------------------- #
class RenderOptionsWidget(QWidget):
    """Container | Encoder | Bitrate *or* CRF | FPS | Resolution | TMIX | Name.

    The rate row is *swapped*, not greyed out: libsvtav1 is quality-targeted and
    must not be given ``-b:v``/``-maxrate``/``-bufsize`` at all, so showing a
    bitrate box next to it would be a lie.
    """

    changed = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        form = QFormLayout(self)
        form.setContentsMargins(0, 0, 0, 0)
        form.setSpacing(7)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        self._form = form

        self.container = _combo(
            CONTAINERS,
            "MP4 gets -movflags +faststart; MKV cannot take it and does not get it.",
        )
        form.addRow("Container", self.container)

        self.encoder = _combo(
            ENCODERS,
            "VA-API encodes H.264 or AV1 on the GPU (fast, bitrate-driven).\n"
            "x264 is the CPU fallback when VA-API fails.\n"
            "SVT-AV1 is AV1 on the CPU: CRF quality, slower, smallest files.",
        )
        form.addRow("Encoder", self.encoder)

        self.mbps = _spin(1, 500, 60, suffix=" Mbps")
        form.addRow("Bitrate", self.mbps)

        self.crf = _spin(
            0, 63, 30, prefix="CRF ",
            tip="Lower is better quality and a bigger file. 30 is a good start.",
        )
        self.av1_preset = _spin(
            0, 13, 8, prefix="preset ",
            tip="SVT-AV1 speed: 0 is slowest/best, 13 fastest/worst.",
        )
        self.quality_row = _row(self.crf, self.av1_preset)
        form.addRow("Quality", self.quality_row)

        self.fps = _spin(1, 480, 60, suffix=" fps")
        form.addRow("Frame rate", self.fps)

        self.resolution = _combo(RESOLUTIONS)
        self.resolution.setCurrentIndex(1)
        form.addRow("Resolution", self.resolution)

        self.custom_w = _spin(16, 16384, 2560, step=2)
        self.custom_h = _spin(16, 16384, 1440, step=2)
        self.custom_row = _row(self.custom_w, QLabel("×"), self.custom_h)
        form.addRow("Custom size", self.custom_row)

        self.tmix = _spin(
            0, 60, 7, suffix=" frames",
            tip="Motion blend: averages N source frames before the fps drop.\n"
                "0 disables it.",
        )
        self.tmix.setSpecialValueText("off")
        form.addRow("TMIX", self.tmix)

        self.out_name = QLineEdit()
        self.out_name.setPlaceholderText("output.mp4")
        form.addRow("Output name", self.out_name)

        self.note = QLabel(
            "SVT-AV1 is AV1 on the CPU, quality-targeted (CRF): slower than the "
            "GPU encoders for much smaller files. For AV1 on the GPU pick "
            "AV1 · VA-API, which takes a bitrate like H.264."
        )
        self.note.setObjectName("hintLabel")
        self.note.setWordWrap(True)
        form.addRow("", self.note)

        self.resolution.currentIndexChanged.connect(self._sync_custom)
        self.encoder.currentIndexChanged.connect(self._sync_encoder)
        self.container.currentIndexChanged.connect(self._sync_container)
        for widget in (
            self.mbps, self.fps, self.tmix, self.custom_w, self.custom_h,
            self.crf, self.av1_preset,
        ):
            widget.valueChanged.connect(self.changed)
        for combo in (self.resolution, self.encoder, self.container):
            combo.currentIndexChanged.connect(self.changed)
        self.out_name.textChanged.connect(self.changed)
        self._sync_custom()
        self._sync_encoder()

    # -- row visibility ------------------------------------------------------
    def _sync_custom(self) -> None:
        self._form.setRowVisible(self.custom_row, self.resolution.currentData() == "custom")

    def _sync_encoder(self) -> None:
        av1 = is_crf_encoder(str(self.encoder.currentData()))
        self._form.setRowVisible(self.mbps, not av1)
        self._form.setRowVisible(self.quality_row, av1)
        self._form.setRowVisible(self.note, av1)

    def _sync_container(self) -> None:
        """Keep the output name's extension in step with the container."""
        name = self.out_name.text().strip()
        suffix = f".{self.container.currentData()}"
        if name and Path(name).suffix.lower() in {f".{v}" for _, v in CONTAINERS}:
            self.out_name.setText(str(Path(name).with_suffix(suffix)))

    # -- value in/out --------------------------------------------------------
    def options(self) -> RenderOptions:
        return RenderOptions(
            mbps=self.mbps.value(),
            fps=self.fps.value(),
            tmix_frames=self.tmix.value(),
            resolution=str(self.resolution.currentData()),
            custom_w=self.custom_w.value() & ~1,  # encoders need even sizes
            custom_h=self.custom_h.value() & ~1,
            encoder=str(self.encoder.currentData()),
            container=str(self.container.currentData()),
            crf=self.crf.value(),
            av1_preset=self.av1_preset.value(),
        )

    def set_options(self, options: RenderOptions) -> None:
        self.mbps.setValue(options.mbps)
        self.fps.setValue(options.fps)
        self.tmix.setValue(options.tmix_frames)
        index = self.resolution.findData(options.resolution)
        self.resolution.setCurrentIndex(index if index >= 0 else 1)
        self.custom_w.setValue(options.custom_w)
        self.custom_h.setValue(options.custom_h)
        self.crf.setValue(options.crf)
        self.av1_preset.setValue(options.av1_preset)
        self.encoder.setCurrentIndex(max(0, self.encoder.findData(options.encoder)))
        self.container.setCurrentIndex(max(0, self.container.findData(options.container)))
        self._sync_custom()
        self._sync_encoder()

    def output_name(self) -> str:
        return self.out_name.text().strip()

    def set_output_name(self, name: str) -> None:
        """Set the name, adopting its container if it names one of ours."""
        index = self.container.findData(Path(name).suffix.lower().lstrip("."))
        if index >= 0:
            self.container.setCurrentIndex(index)
        self.out_name.setText(name)


class RenderOptionsDialog(QDialog):
    """Options panel shown after the save dialog has picked a directory."""

    def __init__(
        self,
        out_dir: Path,
        initial_name: str,
        options: RenderOptions,
        summary: str,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Render options")
        self.setMinimumWidth(430)
        self._out_dir = out_dir

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(10)

        head = QLabel(summary)
        head.setObjectName("sectionLabel")
        head.setWordWrap(True)
        layout.addWidget(head)

        self.options_widget = RenderOptionsWidget()
        self.options_widget.set_options(options)
        self.options_widget.set_output_name(initial_name)
        layout.addWidget(self.options_widget)

        self.path_label = QLabel()
        self.path_label.setObjectName("hintLabel")
        self.path_label.setWordWrap(True)
        layout.addWidget(self.path_label)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        ok = buttons.button(QDialogButtonBox.StandardButton.Ok)
        assert ok is not None
        ok.setText("Render")
        ok.setDefault(True)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.options_widget.changed.connect(self._sync_path)
        self._sync_path()

    def _sync_path(self) -> None:
        self.path_label.setText(f"Writes to:  {self.out_path()}")

    def out_path(self) -> Path:
        """Destination, with the extension forced to match the container."""
        options = self.options_widget.options()
        name = self.options_widget.output_name() or f"output{options.suffix}"
        if Path(name).suffix.lower() != options.suffix:
            name = str(Path(name).with_suffix(options.suffix))
        return self._out_dir / name

    def options(self) -> RenderOptions:
        return self.options_widget.options()


# --------------------------------------------------------------------------- #
# Progress
# --------------------------------------------------------------------------- #
def _prune_proxies() -> None:
    """Delete preview copies nobody has opened for PROXY_MAX_AGE_DAYS."""
    cutoff = time.time() - PROXY_MAX_AGE_DAYS * 86400
    try:
        entries = list(PROXY_DIR.iterdir())
    except OSError:
        return
    for entry in entries:
        try:
            if entry.stat().st_atime < cutoff and entry.stat().st_mtime < cutoff:
                entry.unlink()
        except OSError:
            pass


def _same_file(a: Path | str, b: Path | str) -> bool:
    """True when both name one file on disk (symlinks, hard links, relative paths)."""
    try:
        return os.path.samefile(a, b)
    except OSError:
        return False


def part_path(out_path: Path) -> Path:
    """Where ffmpeg writes until the render succeeds (keeps the extension for the muxer)."""
    return out_path.with_name(f"{out_path.stem}.part{out_path.suffix}")


class RenderProgressDialog(QDialog):
    """Live progress for one ``RenderJob``; owns the cancel path.

    ffmpeg reports *nothing* on its progress pipe until the first video frame
    reaches the muxer.  Because the cut is done with ``trim`` filters, that means
    a linear decode from t=0 up to the first kept range — measured at ~52 s of
    total silence for a first cut at t=1200 on the 46-minute test file.  So until
    the first callback arrives the dialog runs an indeterminate bar and says what
    it is waiting for; elapsed time comes from a local ``QTimer``, never from the
    engine, and the only watchdog is the one armed by Cancel.

    ffmpeg writes to :func:`part_path`; only a clean exit renames it onto
    ``out_path``, so a failed or cancelled render never touches an older file.
    """

    def __init__(
        self,
        job: RenderJob,
        out_path: Path,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Rendering")
        self.setModal(True)
        self.setMinimumWidth(430)
        self._job = job
        self._out_path = out_path
        self._started = time.monotonic()
        self._cancelled = False
        self._finished = False
        self._got_progress = False
        self._last_progress_at = self._started
        self._encoded_text = ""
        #: How long Cancel waits for the engine's on_done before giving up.
        #: Must exceed the engine's SIGTERM->SIGKILL grace, or the dialog would
        #: close while ffmpeg is still writing.
        self.cancel_timeout_s: float = CANCEL_GRACE + 5.0
        self.exit_code: int = -1
        self.message: str = ""

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(10)

        self._title = QLabel(f"Encoding → {out_path.name}")
        self._title.setObjectName("sectionLabel")
        self._title.setWordWrap(True)
        layout.addWidget(self._title)

        self._bar = QProgressBar()
        # Indeterminate ("busy") until ffmpeg's first progress block arrives;
        # _on_progress switches it to 0..1000.
        self._bar.setRange(0, 0)
        self._bar.setTextVisible(False)
        self._bar.setMinimumHeight(16)
        layout.addWidget(self._bar)

        stats = QGridLayout()
        stats.setHorizontalSpacing(16)
        stats.setVerticalSpacing(3)
        self._pct = QLabel("0.0 %")
        self._elapsed = QLabel("00:00")
        self._eta = QLabel("—")
        self._fps = QLabel("—")
        self._speed = QLabel("—")
        for col, (name, widget) in enumerate(
            (
                ("Progress", self._pct),
                ("Elapsed", self._elapsed),
                ("ETA", self._eta),
                ("FPS", self._fps),
                ("Speed", self._speed),
            )
        ):
            caption = QLabel(name)
            caption.setObjectName("hintLabel")
            stats.addWidget(caption, 0, col)
            stats.addWidget(widget, 1, col)
        layout.addLayout(stats)

        self._status = QLabel("Preparing… waiting for ffmpeg's first progress report.")
        self._status.setObjectName("hintLabel")
        self._status.setWordWrap(True)
        layout.addWidget(self._status)

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        self._cancel = QPushButton("Cancel")
        self._cancel.clicked.connect(self._request_cancel)
        buttons.addWidget(self._cancel)
        layout.addLayout(buttons)

        self._bridge = _RenderBridge(self)
        self._bridge.progressed.connect(self._on_progress)
        self._bridge.finished_.connect(self._on_done)

        self._ticker = QTimer(self)
        self._ticker.setInterval(250)
        self._ticker.timeout.connect(self._tick)
        self._ticker.start()

        self._watchdog = QTimer(self)
        self._watchdog.setSingleShot(True)
        self._watchdog.timeout.connect(self._force_close)

        QTimer.singleShot(0, self._start_job)

    # -- lifecycle -----------------------------------------------------------
    def _start_job(self) -> None:
        try:
            self._job.start(self._bridge.on_progress, self._bridge.on_done)
        except Exception as exc:  # engine failed to even spawn ffmpeg
            self._on_done(-1, f"{type(exc).__name__}: {exc}")

    def _tick(self) -> None:
        # Elapsed is wall-clock from this dialog's own timer, so it keeps moving
        # through the silent seek window when no callback arrives at all.
        now = time.monotonic()
        self._elapsed.setText(_fmt_hms(now - self._started))
        # The same silence recurs between kept ranges (ffmpeg decodes the cut
        # footage linearly), so a frozen "Encoded 0:06 of 0:09" is misleading;
        # say what is happening once nothing has arrived for a couple of seconds.
        if (
            self._got_progress
            and not self._cancelled
            and now - self._last_progress_at > _STALL_AFTER
        ):
            self._status.setText(
                f"{self._encoded_text} — decoding ahead to the next kept range… "
                f"({_fmt_hms(now - self._last_progress_at)} without output)"
            )

    def _on_progress(self, progress: Progress) -> None:
        if self._finished:
            return
        self._last_progress_at = time.monotonic()
        if not self._got_progress:
            self._got_progress = True
            self._bar.setRange(0, 1000)
        self._bar.setValue(int(progress.fraction * 1000))
        self._pct.setText(f"{progress.fraction * 100:.1f} %")
        self._fps.setText(f"{progress.fps:.0f}" if progress.fps else "—")
        self._speed.setText(f"{progress.speed:.2f}×" if progress.speed else "—")
        eta = progress.eta_seconds
        self._eta.setText(_fmt_hms(eta) if eta is not None else "—")
        if not self._cancelled:
            self._encoded_text = (
                f"Encoded {format_tc(progress.out_time, decimals=0)} "
                f"of {format_tc(progress.total, decimals=0)}"
            )
            self._status.setText(self._encoded_text)

    def _on_done(self, code: int, message: str) -> None:
        if self._finished:
            return
        self._finished = True
        self._ticker.stop()
        self._watchdog.stop()
        part = part_path(self._out_path)
        if code == 0:
            # A Cancel that lost the race to a clean exit still keeps the file.
            try:
                os.replace(part, self._out_path)
                self._cancelled = False
            except OSError as exc:
                code, message = -1, f"{message}\ncould not move {part.name} into place: {exc}"
        else:
            # Failed or cancelled: the half-written .part goes; the old file stays.
            try:
                part.unlink(missing_ok=True)
            except OSError:
                pass
        self.exit_code = code
        self.message = message
        if code != 0:
            self.reject()
        else:
            self._bar.setRange(0, 1000)
            self._bar.setValue(1000)
            self._pct.setText("100.0 %")
            self.accept()

    def _request_cancel(self) -> None:
        if self._finished or self._cancelled:
            return
        self._cancelled = True
        self._cancel.setEnabled(False)
        self._status.setText("Cancelling — waiting for ffmpeg to stop…")
        try:
            self._job.cancel()
        except Exception as exc:
            self._on_done(-1, f"cancel failed: {exc}")
            return
        # The engine SIGTERMs now and SIGKILLs after its grace period; only give
        # up after that has certainly happened (see cancel_timeout_s).
        self._watchdog.start(int(self.cancel_timeout_s * 1000))

    def _force_close(self) -> None:
        """The engine never called back after cancel; don't strand the dialog."""
        if not self._finished:
            self._on_done(-1, "cancelled")

    @property
    def was_cancelled(self) -> bool:
        return self._cancelled

    def reject(self) -> None:  # noqa: D102 - Esc / Cancel both land here
        if not self._finished:
            self._request_cancel()
            return
        super().reject()

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
        if not self._finished:
            self._request_cancel()
            event.ignore()
            return
        super().closeEvent(event)


# --------------------------------------------------------------------------- #
# Main window
# --------------------------------------------------------------------------- #
class MainWindow(QMainWindow):
    """The whole GUI."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("ffcut")
        self.resize(1180, 820)
        self.setMinimumSize(880, 600)
        self.setAcceptDrops(True)

        self.config = Config.load()
        self.options: RenderOptions = self.config.render
        self.info: MediaInfo | None = None
        self._asset_dir: Path | None = None
        self._proxy_queue: list[str] = []
        self._proxy_busy: str | None = None
        self._proxy_failed: set[str] = set()
        _prune_proxies()
        self._workers: set[_Worker] = set()
        self._closing = False
        resume_probes()  # in case an earlier closeEvent bailed before unlatching
        self._probe_gen = 0
        #: Bumped per loaded file; tags asset jobs so a late result from the
        #: previous file cannot land on this one.
        self._media_gen = 0
        #: One thumbnail job per FILE on the timeline — a second video lane's
        #: clip gets real frames, not a labelled block.
        self._thumbs: dict[str, _Slot] = {}
        self._waves: dict[int, _Slot] = {}
        self._pending_seek_ms: int | None = None
        self._seek_guard_until = 0.0
        #: Extra audio-only players, by ``_PreviewSource.key``.  Empty in the
        #: common case (one audio track, nothing dropped) — see _sync_preview.
        self._aux: dict[tuple[str, int, int], _AuxPlayer] = {}
        #: The embedded track the MAIN player is decoding, if any.
        self._primary: _PreviewSource | None = None

        # -- the picture, as runs (see the module docstring) ------------------
        #: ``preview_plan().video``: contiguous stretches of one file each, in
        #: edited time, in order.  Rebuilt by ``_sync_preview``.
        self._runs: list[PreviewVideo] = []
        #: Index into ``_runs`` of the run on screen; -1 = a black gap.
        self._run = -1
        #: What the main player's source is right now.  Only ``_enter_run``
        #: writes it, and only after a successful ``setSource``.
        self._video_path = ""
        #: True while the main player may also carry a lane's audio: one file,
        #: no gaps, so its clock and its audio are the edit's.
        self._carry_audio = True
        #: The transport's state, which is NOT the player's: in a black gap the
        #: player is parked and the gap clock is what is running.
        self._playing = False
        #: Edited position while the gap clock owns the time.
        self._gap_edited = 0.0
        self._gap_mark = 0.0
        #: Latched across a source swap so the Stopped that ``setSource``
        #: delivers is not mistaken for the file running out.
        self._in_swap = False
        #: Applied once a newly set source reports LoadedMedia — setPosition
        #: before that is dropped on the floor by the backend.
        self._load_seek_ms: int | None = None
        #: Where the player was put when the current run was entered, and
        #: whether an end-of-file advance is what put it there (see
        #: EOF_PROGRESS_MS).
        self._run_entered_ms = 0
        self._run_from_eof = False
        self._entering_from_eof = False

        theme.apply(QApplication.instance(), self.config.prefs.theme)
        self._build_ui()
        self._build_menus()
        self._build_player()
        self._update_actions()

    # ------------------------------------------------------------------- build
    def _build_ui(self) -> None:
        central = QWidget()
        central.setObjectName("central")
        root = QVBoxLayout(central)
        root.setContentsMargins(7, 6, 7, 4)
        root.setSpacing(5)

        # -- the split: player above, timeline below -------------------------
        # The preview used to take ~75 % of the window on a tool whose job is
        # the timeline.  The handle is the user's to move, and where they leave
        # it is remembered in Config.splitter.
        self.splitter = QSplitter(Qt.Orientation.Vertical)
        self.splitter.setObjectName("mainSplitter")
        self.splitter.setHandleWidth(7)
        self.splitter.setChildrenCollapsible(False)
        root.addWidget(self.splitter, 1)

        # -- player pane -----------------------------------------------------
        # One glass panel holding the video stage with its control bar welded
        # to the bottom of it, the way a media player looks — not a preview and
        # a toolbar that happen to be near each other.
        player = QFrame()
        player.setObjectName("playerPanel")
        player_column = QVBoxLayout(player)
        player_column.setContentsMargins(1, 1, 1, 1)
        player_column.setSpacing(0)

        self.video_frame = QFrame()
        self.video_frame.setObjectName("videoFrame")
        frame_layout = QVBoxLayout(self.video_frame)
        frame_layout.setContentsMargins(0, 0, 0, 0)
        self.video = QVideoWidget()
        self.video.setMinimumHeight(120)
        self.video.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        frame_layout.addWidget(self.video)
        # The preview is the most obvious place to drop a file on, and it was
        # the one place that refused: QVideoWidget takes the drop itself.
        self._video_drops = _SubtreeFilter(self, self.video_frame, drops=True)
        # Over the frame, not in its layout: it replaces the picture only while
        # paused on a text block (see TitleStage and _sync_stage).
        self.stage = TitleStage(self.video_frame)
        self.stage.hide()
        self.stage.picked.connect(self._on_stage_picked)
        self.stage.moved.connect(self._on_stage_moved)
        self.stage.released.connect(self._sync_stage)
        self.stage.editRequested.connect(self._edit_text)
        self.video_frame.installEventFilter(self)
        self.video.videoSink().videoFrameChanged.connect(self._on_video_frame)
        player_column.addWidget(self.video_frame, 1)

        self.transport = TransportBar()
        self.transport.playPause.connect(self.toggle_play)
        self.transport.skip.connect(self._skip)
        self.transport.jumpCut.connect(self._jump_cut)
        self.transport.seeked.connect(self._goto_edited)
        self.transport.set_skip_seconds(self.config.prefs.skip_seconds)
        player_column.addWidget(self.transport)
        self.splitter.addWidget(player)

        # No audition combo: the preview plays every unmuted lane at its lane
        # gain, so the M buttons on the timeline are the only mix control there
        # is and a second, preview-only one would just be a way to disagree
        # with the render.  This slider is the master over that mix.
        self.volume = QSlider(Qt.Orientation.Horizontal)
        self.volume.setObjectName("volumeSlider")
        self.volume.setRange(0, 100)
        self.volume.setValue(80)
        self.volume.setFixedWidth(84)
        self.volume.setToolTip(
            "Preview volume. Mute and gain per track live on the timeline lanes."
        )

        self.render_button = QPushButton("Render…")
        self.render_button.setObjectName("primary")
        self.render_button.setToolTip("Render the whole edit (Ctrl+R)")
        self.render_button.clicked.connect(self.render_full)
        self.transport.add_right(self.volume, self.render_button)

        # -- timeline: the other half of the split ---------------------------
        # It expands, so a taller pane means taller lanes (see
        # TimelineWidget._lane_heights), and its minimum height floors the
        # handle at whatever the lane count needs.
        self.timeline = TimelineWidget()
        self.timeline.playheadMoved.connect(self._on_playhead_moved)
        self.timeline.selectionChanged.connect(lambda *_: self._refresh_selection())
        self.timeline.armedChanged.connect(lambda *_: self._refresh_selection())
        self.timeline.blockSelectionChanged.connect(self._on_blocks_changed)
        self.timeline.blockHint.connect(lambda text: self._status(text, 6000))
        self.timeline.cutsChanged.connect(self._on_cuts_changed)
        self.timeline.playPauseRequested.connect(self.toggle_play)
        # Mute and gain are the render mix AND the preview mix — one control.
        self.timeline.trackMuteChanged.connect(self._on_track_mix_changed)
        self.timeline.trackGainChanged.connect(self._on_track_mix_changed)
        self.timeline.thumbsNeeded.connect(self._on_thumbs_needed)
        self.timeline.waveformNeeded.connect(self._on_waveform_needed)
        self._lane_menu_filter = _SubtreeFilter(self, self.timeline, menu=True)
        self.splitter.addWidget(self.timeline)
        self.splitter.setStretchFactor(0, 3)
        self.splitter.setStretchFactor(1, 2)
        self.splitter.setSizes(self.config.splitter or list(DEFAULT_SPLIT))
        # Only a *drag* counts as a preference worth persisting.
        self.splitter.splitterMoved.connect(lambda *_: self._remember_splitter())

        self.warning_label = QLabel()
        self.warning_label.setObjectName("warnLabel")
        self.warning_label.setVisible(False)
        self.warning_label.setWordWrap(True)
        root.addWidget(self.warning_label)

        self.setCentralWidget(central)
        self._build_status_strip()
        self._status("Drop a video on the window, or File ▸ Open…")

    def _build_status_strip(self) -> None:
        """The thin bottom strip: play state · selection · render summary.

        These three readouts used to be a loose row of grey labels above an
        empty status bar; folding them into the strip itself is both the WMP12
        shape and one row of window back for the timeline.
        """
        bar = self.statusBar()
        assert bar is not None
        bar.setSizeGripEnabled(False)
        self.state_label = QLabel("■  Stopped")
        self.state_label.setObjectName("statusState")
        self.selection_label = QLabel("No selection")
        self.selection_label.setObjectName("hintLabel")
        self.summary_label = QLabel(self.options.summary())
        self.summary_label.setObjectName("hintLabel")
        bar.addPermanentWidget(self.state_label)
        bar.addPermanentWidget(self._status_separator())
        bar.addPermanentWidget(self.selection_label)
        bar.addPermanentWidget(self._status_separator())
        bar.addPermanentWidget(self.summary_label)

    @staticmethod
    def _status_separator() -> QFrame:
        line = QFrame()
        line.setObjectName("statusSep")
        line.setFixedSize(1, 12)
        return line

    def _remember_splitter(self) -> None:
        """The user dragged the handle: that is the preference we persist."""
        sizes = [int(s) for s in self.splitter.sizes()]
        if all(s > 0 for s in sizes):
            self.config.splitter = sizes

    def _status(self, text: str, ms: int = 0) -> None:
        self.statusBar().showMessage(text, ms)  # type: ignore[union-attr]

    def _on_playback_state(self, state: QMediaPlayer.PlaybackState) -> None:
        """The MAIN player changed state.  The transport does not always agree.

        Over a black gap the player is parked on purpose and the gap clock is
        what is running, so the transport keeps its own ``_playing`` and this
        only reflects a state the player reached by itself: the user hitting
        play, or the file ending.  A ``setSource`` stops the player too, which
        is what ``_in_swap`` is there to tell apart.
        """
        if self._closing:
            return
        if state == QMediaPlayer.PlaybackState.PlayingState:
            self._playing = True
        elif self._in_swap or self._run < 0:
            pass  # a setSource, or the gap clock's own parked player
        elif state == QMediaPlayer.PlaybackState.PausedState:
            self._playing = False
        elif state == QMediaPlayer.PlaybackState.StoppedState and self._playing:
            # The file ran out; the edit may not have.  Routed through
            # _end_of_file because EndOfMedia says the same thing in the same
            # turn, and acting on both advances twice.  Kept as a path rather
            # than deferred to EndOfMedia alone: a backend that stops without
            # one would leave the transport reading Playing forever.
            self._end_of_file()
        self._sync_gap_timer()
        self._refresh_transport()

    def _act(
        self, menu: object, text: str, slot: object, key: object = None, tip: str = ""
    ) -> QAction:
        """One menu action, kept on ``self`` by the caller for enable/disable."""
        action = QAction(text, self)
        if key is not None:
            action.setShortcut(QKeySequence(key))  # type: ignore[arg-type]
        action.setToolTip(tip)
        action.triggered.connect(slot)  # type: ignore[arg-type]
        menu.addAction(action)  # type: ignore[attr-defined]
        return action

    def _build_menus(self) -> None:
        bar = self.menuBar()
        assert bar is not None
        std = QKeySequence.StandardKey

        # -- File ------------------------------------------------------------
        file_menu = bar.addMenu("&File")
        assert file_menu is not None
        self.act_open = self._act(file_menu, "&Open…", self.open_dialog, std.Open)
        self.act_import = self._act(
            file_menu, "&Import…", self.import_dialog, "Ctrl+I",
            "Add a video to the end of video track 1, or audio on a lane of its own.",
        )
        self.recent_menu = file_menu.addMenu("&Recent")
        assert self.recent_menu is not None
        self._rebuild_recent()
        file_menu.addSeparator()
        self._act(file_menu, "&Quit", self.close, std.Quit)

        # -- Edit ------------------------------------------------------------
        edit_menu = bar.addMenu("&Edit")
        assert edit_menu is not None
        self.act_undo = self._act(edit_menu, "&Undo", self.undo, std.Undo)
        self.act_redo = self._act(edit_menu, "&Redo", self.redo, std.Redo)
        edit_menu.addSeparator()
        self.act_select_all = self._act(
            edit_menu, "Select &All", self.timeline.select_all, std.SelectAll
        )
        self.act_mark_in = self._act(
            edit_menu, "Mark &In", self.timeline.set_mark_in, "I"
        )
        self.act_mark_out = self._act(
            edit_menu, "Mark &Out", self.timeline.set_mark_out, "O"
        )
        self.act_clear_sel = self._act(
            edit_menu, "&Clear Selection", self.timeline.clear_selection
        )
        # Ctrl+S is the user's habit for "snip"; Delete is handled by the
        # timeline widget itself so it stays out of text fields.
        self.act_cut = self._act(
            edit_menu, "Cu&t Selection", self.cut_selection, "Ctrl+S"
        )

        # -- blocks ----------------------------------------------------------
        # The same verbs as the right-click menu on a block, with the shortcuts
        # attached here so they work wherever the focus is.  Delete is NOT one
        # of them: the timeline widget handles it, so it stays out of text
        # fields (the render dialog's filename box, for one).
        edit_menu.addSeparator()
        self.act_split = self._act(
            edit_menu, "&Split Block at Playhead", self.split_blocks, "S",
            "Cut the block(s) under the playhead in two, so each can be moved "
            "on its own.  Alt+drag moves a block.",
        )
        self.act_group = self._act(
            edit_menu, "&Group Blocks", self.group_blocks, "Ctrl+G",
            "Selected blocks then select, move and delete as one.",
        )
        self.act_ungroup = self._act(
            edit_menu, "&Ungroup Blocks", self.ungroup_blocks, "Ctrl+Shift+G"
        )
        self.act_copy_blocks = self._act(
            edit_menu, "&Copy Blocks", self.copy_blocks, std.Copy
        )
        self.act_paste_blocks = self._act(
            edit_menu, "&Paste Blocks", self.paste_blocks, std.Paste,
            "Paste at the playhead, on the tracks they were copied from.",
        )
        self.act_delete_blocks = self._act(
            edit_menu, "&Delete Blocks", self.delete_blocks,
            tip="Remove the selected blocks, leaving a gap where they were.",
        )
        self.act_add_text = self._act(
            edit_menu, "Add &Text…", self.add_text, "Ctrl+T",
            "A text block at the playhead on its own video track. Pause on it to "
            "drag and resize it on the preview.",
        )
        edit_menu.addSeparator()
        self.act_prefs = self._act(
            edit_menu, "&Preferences…", self.open_preferences, "Ctrl+,"
        )

        # -- Render ----------------------------------------------------------
        render_menu = bar.addMenu("&Render")
        assert render_menu is not None
        self.act_render_full = self._act(
            render_menu, "Render &Full…", self.render_full, "Ctrl+R"
        )
        self.act_render_sel = self._act(
            render_menu, "Render &Selection…", self.render_selection, "Ctrl+Shift+R",
            "Render only the marked range (cuts inside it are still honoured).",
        )

    def _rebuild_recent(self) -> None:
        menu = self.recent_menu
        if menu is None:
            return
        menu.clear()
        existing = [p for p in self.config.recent if Path(p).is_file()]
        menu.setEnabled(bool(existing))
        for path in existing:
            action = QAction(Path(path).name, self)
            action.setToolTip(path)
            action.triggered.connect(lambda _=False, p=path: self.load_file(p))
            menu.addAction(action)

    def _build_player(self) -> None:
        self.player = QMediaPlayer(self)
        self.audio_out = QAudioOutput(self)
        self.audio_out.setVolume(self.volume.value() / 100.0)
        self.player.setAudioOutput(self.audio_out)
        self.player.setVideoOutput(self.video)
        self.player.positionChanged.connect(self._on_position_changed)
        self.player.playbackStateChanged.connect(self._on_playback_state)
        self.player.mediaStatusChanged.connect(self._on_media_status)
        self.player.errorOccurred.connect(self._on_player_error)
        self.volume.valueChanged.connect(lambda *_: self._apply_preview_volumes())

        # Seeks are coalesced so a fast scrub doesn't flood the demuxer.
        self._seek_timer = QTimer(self)
        self._seek_timer.setSingleShot(True)
        self._seek_timer.setInterval(40)
        self._seek_timer.timeout.connect(self._flush_seek)

        # The clock for the stretches no player can time (see _tick_gap).
        self._gap_timer = QTimer(self)
        self._gap_timer.setInterval(GAP_TICK_MS)
        self._gap_timer.timeout.connect(self._tick_gap)

    # ------------------------------------------------------------ file loading
    def open_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open video", self.config.last_open_dir, VIDEO_FILTER
        )
        if path:
            self.load_file(path)

    def import_dialog(self) -> None:
        """File ▸ Import: ADD a file to the edit, the way a drop on the window does."""
        media = sorted(VIDEO_SUFFIXES | AUDIO_SUFFIXES)
        path, _ = QFileDialog.getOpenFileName(
            self, "Import media", self.config.last_open_dir,
            f"Media files ({' '.join('*' + s for s in media)});;All files (*)",
        )
        if path:
            self.import_file(path)

    def import_file(self, path: str) -> None:
        kind = "audio" if Path(path).suffix.lower() in AUDIO_SUFFIXES else "video"
        if self.info is None:
            if kind == "video":
                self.load_file(path)  # nothing to add to yet: same as Open
            else:
                self._status("Open a video first, then import audio onto it.", 4000)
            return
        self.config.last_open_dir = str(Path(path).parent)
        if kind == "video":
            self.place_on_lane(path, kind, 0, "append")  # after what track 1 holds, Vegas-style
        else:
            self.add_audio_file(path)

    def load_file(self, path: str) -> None:
        # Absolute and symlink-free, so the recent list works from any directory.
        file_path = Path(path).expanduser().resolve()
        if not file_path.is_file():
            QMessageBox.warning(self, "ffcut", f"Not a file:\n{file_path}")
            return
        if not self._discard_ok():
            return
        self.config.last_open_dir = str(file_path.parent)
        self._status(f"Probing {file_path.name}…")
        for widget in (self.render_button, self.act_render_full, self.act_render_sel):
            widget.setEnabled(False)
        # A still-running probe of the previous file is simply outrun: its
        # result carries an older generation and _on_probed drops it.
        self._probe_gen += 1
        gen = self._probe_gen
        self._spawn(
            partial(probe, str(file_path)),
            lambda info: self._on_probed(gen, info),
            lambda message: self._on_probe_failed(gen, message),
        )

    def _discard_ok(self) -> bool:
        """Ask before an open or a close throws the edit away (it cannot be undone)."""
        if self.info is None or not self.timeline.can_undo():
            return True
        reply = QMessageBox.question(
            self, "ffcut", "Discard the current edit?",
            QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        return reply == QMessageBox.StandardButton.Discard

    def _spawn(
        self,
        fn: Callable[[], object],
        on_done: Callable[[object], None],
        on_failed: Callable[[str], None],
    ) -> _Worker:
        worker = _Worker(fn, self)
        worker.done.connect(on_done)
        worker.failed.connect(on_failed)
        worker.finished.connect(lambda: self._retire(worker))
        self._workers.add(worker)
        worker.start()
        return worker

    def _retire(self, worker: _Worker) -> None:
        """Drop our reference *before* the C++ object goes away."""
        self._workers.discard(worker)
        worker.deleteLater()

    def _on_probe_failed(self, gen: int, message: str) -> None:
        if gen != self._probe_gen or self._closing:
            return  # a probe cancelled by teardown is not an error to report
        self._status("Probe failed.")
        self._update_actions()  # the previously loaded file (if any) is still valid
        QMessageBox.critical(self, "ffcut — probe failed", message)

    def _on_probed(self, gen: int, info: MediaInfo) -> None:
        if gen != self._probe_gen:
            return  # a newer load_file() superseded this one
        self.info = info
        self.setWindowTitle(f"ffcut — {Path(info.path).name}")
        self.config.remember(info.path)
        self._rebuild_recent()

        # Asset jobs still running belong to the old file: their results are
        # dropped by generation, and any request queued behind them is stale.
        self._media_gen += 1
        for slot in (*self._thumbs.values(), *self._waves.values()):
            slot.pending = None
        # Whatever the previous file needed extra players for is gone with it.
        # Before set_media, not after: set_media re-plans the preview on its way
        # out, and sweeping afterwards would close players it had just built.
        self._drop_unused_aux(set())
        self._primary = None
        self._set_playing(False)
        self._runs, self._run, self._carry_audio = [], -1, True
        # Before set_media: it emits cutsChanged on its way out, which re-plans
        # the preview -- and a stale _video_path there reads as a source swap.
        self._video_path = self._play_path(info.path)
        self._in_swap = False
        self._load_seek_ms = None
        self.player.setSource(QUrl.fromLocalFile(self._video_path))
        self.timeline.set_media(info)  # builds the lanes and asks for assets
        self.player.pause()
        self._sync_preview()
        self._update_time_label(0.0)
        self._status(
            f"{Path(info.path).name} — {info.width}×{info.height} · "
            f"{info.fps:g} fps · {format_tc(info.duration, decimals=0)} · "
            f"{info.vcodec} · {len(info.audio)} audio track(s)"
        )
        self._update_actions()

    # -------------------------------------------------------- preview video
    def _sync_preview(self) -> None:
        """Re-read the timeline and make the whole preview agree with it.

        ONE place: the run list, whether the main player may carry audio, the
        audio players, and where the picture sits.  Everything that changes the
        edit ends up here — a cut, a lift, a drop, a lane, an undo.
        """
        if self._closing or self.info is None:
            return
        plan = self.timeline.preview_plan()
        self._runs = list(plan.video)
        self._carry_audio = self._one_file_no_gaps(plan.duration)
        self._sync_preview_audio(plan)
        # The run INDEX is meaningless across a re-plan, so re-seat from the
        # playhead, which the timeline has already clamped to the new edit.
        self._reseat(self.timeline.playhead())
        self._queue_proxies()
        self._sync_stage()

    # ---------------------------------------------------------------- text
    def _sync_stage(self) -> None:
        """Paused over a text block: show the still + titles; else the video."""
        if self._closing:
            return
        titles = (
            self.timeline.titles_at(self.timeline.playhead())
            if self.info is not None and not self._playing else []
        )
        if titles:
            self.stage.titles = titles
            self.stage.selected = self.timeline.clip_selection()
            if self.info is not None:
                self.stage.frame_size = QSize(self.info.width, self.info.height)
            if self.stage.frame is None:
                self._grab_frame()
            self.stage.setGeometry(self.video_frame.rect())
            self.video.setVisible(False)
            self.stage.show()
            self.stage.raise_()
            self.stage.update()
        elif not self.stage.isHidden():
            self.stage.hide()
            self.stage.frame = None
            self.video.setVisible(self._run >= 0)

    def _grab_frame(self) -> None:
        frame = self.video.videoSink().videoFrame()
        image = frame.toImage() if frame.isValid() and self._run >= 0 else QImage()
        self.stage.frame = None if image.isNull() else image

    def _on_video_frame(self, _frame: object) -> None:
        if not self.stage.isHidden():  # a seek while paused: follow it on the still
            self._grab_frame()
            self.stage.update()

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:  # noqa: N802
        if obj is self.video_frame and event.type() == QEvent.Type.Resize:
            self.stage.setGeometry(self.video_frame.rect())
        return super().eventFilter(obj, event)

    def _selected_title(self) -> object | None:
        for cid in self.timeline.clip_selection():
            clip = self.timeline.find_clip(cid)
            if clip is not None and clip.title is not None:
                return clip
        return None

    def add_text(self) -> None:
        if self.info is None:
            return
        text, ok = QInputDialog.getMultiLineText(self, "Add text", "Text:", "Text")
        if not ok or not text.strip():
            return
        at = self.timeline.edited_to_source(self.timeline.playhead())
        clip = self.timeline.add_title(at, text)
        self.timeline.select_block(clip)
        self._set_playing(False)
        self._status("Text added. Pause on it to drag it on the preview; "
                     "drag its corner to resize, Alt+drag its block edges to stretch it.", 8000)
        self._update_actions()

    def _edit_selected_text(self) -> None:
        clip = self._selected_title()
        if clip is not None:
            self._edit_text(clip.cid)  # type: ignore[attr-defined]

    def _edit_text(self, cid: int) -> None:
        clip = self.timeline.find_clip(cid)
        if clip is None or clip.title is None:
            return
        text, ok = QInputDialog.getMultiLineText(self, "Edit text", "Text:", clip.title.text)
        if ok and text.strip():
            self.timeline.set_title(clip, text=text)

    def _on_stage_picked(self, cid: int) -> None:
        clip = self.timeline.find_clip(cid)
        if clip is not None:
            self.timeline.select_block(clip)
            self.timeline.push_undo()  # the whole drag is one undo step

    def _on_stage_moved(self, cid: int, x: float, y: float, size: float) -> None:
        clip = self.timeline.find_clip(cid)
        if clip is not None:
            self.timeline.set_title(clip, undo=False, x=x, y=y, size=size)

    # ------------------------------------------------------- preview copies
    def _proxy_file(self, path: str) -> Path | None:
        """Where ``path``'s preview copy lives at the chosen quality; None = Full."""
        height = self.config.prefs.preview
        if height == "full":
            return None
        try:
            st = os.stat(path)
        except OSError:
            return None
        # Size + mtime in the key: a re-recorded file never reuses a stale copy.
        key = hashlib.sha1(f"{path}|{st.st_size}|{st.st_mtime_ns}".encode()).hexdigest()[:16]
        return PROXY_DIR / f"{key}-{height}p.mkv"

    def _play_path(self, path: str) -> str:
        """What the preview player opens for ``path``: its copy once built."""
        proxy = self._proxy_file(path)
        return str(proxy) if proxy is not None and proxy.is_file() else path

    def _queue_proxies(self) -> None:
        """Build missing copies for every file the picture uses, one at a time."""
        for run in self._runs:
            proxy = self._proxy_file(run.path)
            if (
                proxy is not None and not proxy.is_file()
                and run.path not in self._proxy_queue
                and run.path != self._proxy_busy
                and run.path not in self._proxy_failed
            ):
                self._proxy_queue.append(run.path)
        self._next_proxy()

    def _next_proxy(self) -> None:
        if self._proxy_busy is not None or self._closing:
            return
        while self._proxy_queue:
            path = self._proxy_queue.pop(0)
            proxy = self._proxy_file(path)
            if proxy is None or proxy.is_file():
                continue
            self._proxy_busy = path
            self._status(
                f"Building a {self.config.prefs.preview}p preview copy of {Path(path).name}"
                " in the background — the preview switches to it when it is done."
            )
            duration = self.info.duration if self.info is not None else 3600.0
            self._spawn(
                partial(
                    make_proxy, path, str(proxy), int(self.config.prefs.preview),
                    default_vaapi_device(), max(600.0, duration * 4),
                ),
                lambda _out, p=path: self._on_proxy_done(p, None),
                lambda message, p=path: self._on_proxy_done(p, message),
            )
            return

    def _on_proxy_done(self, path: str, error: str | None) -> None:
        self._proxy_busy = None
        if self._closing:
            return
        if error is not None:
            self._proxy_failed.add(path)  # no retry loop; the original keeps playing
            self._status(f"Preview copy of {Path(path).name} failed: {error}", 8000)
        else:
            self._status(f"Preview now plays a light copy of {Path(path).name}.", 6000)
            if self.info is not None:
                self._reseat(self.timeline.playhead())  # swaps the player onto it
        self._next_proxy()

    def _one_file_no_gaps(self, duration: float) -> bool:
        """True when the main player alone is the whole picture.

        That means every run comes from the loaded file AND the runs tile the
        edit end to end.  Only then are the player's clock and its audio the
        edit's: one gap and it has to park (taking its audio with it), one
        second file and it has to swap (taking its audio with it).  So this is
        exactly the condition for letting it carry a lane — see
        ``_sync_preview_audio``.
        """
        if self.info is None or not self._runs:
            return False
        cursor = 0.0
        for run in self._runs:
            if run.path != self.info.path or run.edited_start > cursor + RUN_EPS:
                return False
            cursor = run.edited_start + max(0.0, run.src_out - run.src_in)
        return cursor >= duration - RUN_EPS

    def _run_at(self, t_edited: float) -> int:
        """Index of the run covering edited time ``t``, or -1 for a black gap."""
        for index, run in enumerate(self._runs):
            length = max(0.0, run.src_out - run.src_in)
            if run.edited_start - RUN_EPS <= t_edited < run.edited_start + length:
                return index
        return -1

    def _run_end(self, run: PreviewVideo) -> float:
        return run.edited_start + max(0.0, run.src_out - run.src_in)

    def _reseat(self, t_edited: float, *, coalesce: bool = False) -> None:
        """Put the whole preview at edited time ``t`` — the only entry point.

        Only the picture: the caller says how the extra audio players follow,
        because a scrub, a re-plan and a cut the playback just crossed each
        want a different answer.
        """
        self._enter_run(self._run_at(t_edited), t_edited, coalesce=coalesce)

    def _enter_run(
        self, index: int, t_edited: float, *, coalesce: bool = False
    ) -> None:
        """Show run ``index`` at edited time ``t``; -1 means black.

        The one place the main player's source changes.  A swap cannot seek
        immediately — the backend drops a ``setPosition`` on media it has not
        loaded — so the target is parked in ``_load_seek_ms`` and applied from
        ``_on_media_status``.
        """
        self._run = index
        if index < 0:
            self._gap_edited = t_edited
            self._gap_mark = time.monotonic()
            self.video.setVisible(False)  # the frame behind it is black
            if self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
                self.player.pause()
            self._sync_gap_timer()
            return
        run = self._runs[index]
        self.video.setVisible(self.stage.isHidden())
        target_ms = max(0, int((run.src_in + (t_edited - run.edited_start)) * 1000))
        self._run_entered_ms = target_ms
        self._run_from_eof = self._entering_from_eof
        play = self._play_path(run.path)
        if play != self._video_path:
            self._in_swap = True
            self._video_path = play
            self._pending_seek_ms = None
            self._load_seek_ms = target_ms
            self.player.setSource(QUrl.fromLocalFile(play))
        elif self._load_seek_ms is not None:
            # A swap is still loading.  ``setPosition`` is dropped on media that
            # has not loaded, so the parked target is the only one that will be
            # applied — re-park it, or the FIRST seek wins and the picture lands
            # somewhere the playhead has already left.
            self._load_seek_ms = target_ms
        elif abs(self.player.position() - target_ms) > SEEK_SLOP_MS:
            # Only when it would actually move: arming a no-op seek would park
            # _pending_seek_ms, and that is what silences position reports.
            if coalesce:
                self._pending_seek_ms = target_ms
                if not self._seek_timer.isActive():
                    self._seek_timer.start()
            else:
                # Playback crossing a cut must not wait on the scrub timer --
                # 40 ms of the material the edit removed would play first.
                self._pending_seek_ms = None
                self.player.setPosition(target_ms)
        self._sync_gap_timer()
        if self._playing:
            self._resume()

    def _end_of_file(self) -> None:
        """The main player's FILE is over.  Advance the run exactly once.

        Both ``StoppedState`` and ``EndOfMedia`` report it, in the same turn.
        Either may arrive first and either may be the only one, so both call
        here — and the second is recognised by the player not having moved off
        the position the first one's advance put it at.  See
        :data:`EOF_PROGRESS_MS` for why this is not a time window.
        """
        if self._run < 0 or self._in_swap or self._load_seek_ms is not None:
            return
        if (
            self._run_from_eof
            and self.player.position() <= self._run_entered_ms + EOF_PROGRESS_MS
        ):
            return  # the pair's second signal: THIS entry was the first one's
        self._entering_from_eof = True
        try:
            self._advance_run()
        finally:
            self._entering_from_eof = False

    def _resume(self) -> None:
        """Start the main player — or step over a run whose file will not decode.

        ``play()`` on a player already in an error state emits NO new error, so
        ``_on_player_error`` never fires; and entering another run of the SAME
        file issues no ``setSource``, so there is no fresh error either.  Either
        way nothing else would ever move and the transport sits on "▶ Playing"
        with a frozen playhead — measured on two blocks of one undecodable file,
        from a paused seek AND from ordinary playback walking into them.

        Deferred so the stack unwinds before the next advance (this can be
        reached from inside ``_enter_run``).  ``_advance_run`` only moves
        forward, so a whole timeline of unplayable files walks to the end and
        stops rather than recursing.
        """
        if self.player.error() != QMediaPlayer.Error.NoError:
            QTimer.singleShot(0, self._step_over_dead_run)
            return
        self.player.play()

    def _step_over_dead_run(self) -> None:
        if (
            self._closing
            or not self._playing
            or self._run < 0
            or self.player.error() == QMediaPlayer.Error.NoError
        ):
            return
        self._advance_run()

    def _advance_run(self) -> None:
        """The current run is finished: move to whatever the edit has next."""
        if not 0 <= self._run < len(self._runs):
            return  # in a gap: the gap clock owns the time, not the player
        # An advance is a deliberate move, not a stale report from the demuxer,
        # so it is not what the post-seek guard is there to suppress: leaving it
        # armed stranded the playhead wherever the seek had put it.
        self._seek_guard_until = 0.0
        total = self.timeline.edited_duration()
        here = self._run_end(self._runs[self._run])
        if here >= total - RUN_EPS:
            self._publish_edited(total)
            self._set_playing(False)
            return
        here = min(here, total)
        self._reseat(here)
        # The extra tracks jump the cut with the picture, or they would be
        # playing the material that was just skipped.
        self._publish_edited(here, force=True)

    # -- the gap clock -------------------------------------------------------
    def _sync_gap_timer(self) -> None:
        """Run the clock exactly while the transport is playing over black."""
        want = self._playing and self._run < 0 and self.info is not None
        if want and not self._gap_timer.isActive():
            self._gap_mark = time.monotonic()
            self._gap_timer.start()
        elif not want and self._gap_timer.isActive():
            self._gap_timer.stop()

    def _tick_gap(self) -> None:
        """Advance edited time across a stretch no player can time for us."""
        if self._closing or self.info is None or self._run >= 0:
            self._sync_gap_timer()
            return
        now = time.monotonic()
        t = self._gap_edited + (now - self._gap_mark)
        self._gap_mark = now
        total = self.timeline.edited_duration()
        if t >= total - 1e-6:
            self._gap_edited = total
            self._publish_edited(total)
            self._set_playing(False)
            return
        index = self._run_at(t)
        back = index >= 0
        if back:
            self._enter_run(index, t)  # the picture comes back
        else:
            self._gap_edited = t
        self._publish_edited(t, force=back)

    # -- transport -----------------------------------------------------------
    def _set_playing(self, playing: bool) -> None:
        """The transport's state is ours, not the player's (see _tick_gap)."""
        self._playing = bool(playing) and self.info is not None
        if self._playing and self._run >= 0:
            self._resume()
        elif self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.player.pause()
        self._sync_gap_timer()
        self._refresh_transport()
        self._sync_stage()

    def _refresh_transport(self) -> None:
        self.transport.set_playing(self._playing)
        for aux in self._aux.values():
            aux.set_playing(self._playing)
        if self._playing:
            self.state_label.setText("▶  Playing")
        elif self.info is None:
            self.state_label.setText("■  Stopped")
        else:
            self.state_label.setText("❚❚  Paused")

    # -------------------------------------------------------- preview audio
    def _preview_sources(self, plan: PreviewPlan | None = None) -> list[_PreviewSource]:
        """Every source the preview should play, one player's worth per entry.

        ``TimelineWidget.preview_plan()`` hands back one entry per *contiguous
        run* — a cut, a lift or a lane above splits a clip into several — which
        is the right shape for a filter graph but not for a player.  Runs that
        share a lane, a file and a stream are collected into one
        :class:`_PreviewSource` with several spans, so one file on one lane is
        one player however many pieces the edit is in.  Gain and mute are the
        lane's, so the first run carries the pair for all of them.
        """
        rows_by_source: dict[tuple[str, int, int], list] = {}
        if plan is None:
            plan = self.timeline.preview_plan()
        for row in plan.audio:
            rows_by_source.setdefault((row.path, row.stream, row.lane), []).append(row)
        return [
            _PreviewSource(
                path, stream, lane, rows[0].gain_db, rows[0].muted,
                tuple((r.edited_start, r.src_in, r.src_out) for r in rows),
            )
            for (path, stream, lane), rows in rows_by_source.items()
        ]

    def _sync_preview_audio(
        self, plan: PreviewPlan | None = None, *, force: bool = True
    ) -> None:
        """Make the preview mix equal the render mix.

        The main player keeps the video and the FIRST embedded track (muted by
        its output when its lane is, so M never moves a track between players);
        every other unmuted source gets an :class:`_AuxPlayer`.  Those are built
        lazily and only for sources that are actually audible, so the ordinary
        one-track file still runs on exactly one QMediaPlayer.

        It keeps that track only while ``_carry_audio`` holds — one file, no
        gaps.  Otherwise the player has to swap sources and park over black,
        which would swap and park a lane's audio with it, so every lane goes to
        an aux player and the main output is muted.
        """
        if self._closing:
            return
        sources = self._preview_sources(plan)
        main_path = self.info.path if self.info is not None and self._carry_audio else ""
        self._primary = next((s for s in sources if s.path == main_path), None)
        if self._primary is not None:
            tracks = self.player.audioTracks()
            if tracks:
                want = max(0, min(len(tracks) - 1, int(self._primary.stream)))
                if self.player.activeAudioTrack() != want:
                    self.player.setActiveAudioTrack(want)

        primary_key = self._primary.key if self._primary is not None else None
        live = {s.key: s for s in sources if s.key != primary_key}
        for key, source in live.items():
            if key in self._aux:
                self._aux[key].source = source
            elif not source.muted:  # lazily: a silent lane does not need a player
                self._aux[key] = _AuxPlayer(source, self)
        self._drop_unused_aux(set(live))
        self._apply_preview_volumes()
        self._sync_aux_positions(force=force)

    def _drop_unused_aux(self, keep: set[tuple[str, int, int]]) -> None:
        """Close players for sources that are gone (new media, lane removed).

        A *muted* source is not gone — its player is parked, not destroyed, so
        toggling M does not cycle a QMediaPlayer through stop() every time.
        """
        for key in [k for k in self._aux if k not in keep]:
            self._aux.pop(key).close()

    def _apply_preview_volumes(self) -> None:
        """Master slider × each lane's gain, muted lanes silent.

        ``QAudioOutput`` tops out at 1.0, so a lane boosted above 0 dB cannot
        simply be played louder.  Clamping it would make the preview disagree
        with the render on the one thing a gain slider is for — the balance
        between lanes — so the whole mix is scaled instead: the loudest
        unmuted lane sits at unity and every other lane keeps its distance
        below it.  What you hear is the render's balance at a lower level.
        """
        primary = self._primary
        live = [s for s in (primary, *(a.source for a in self._aux.values()))
                if s is not None and not s.muted]
        peak = max((_gain_linear(s.gain_db) for s in live), default=1.0)
        scale = self.volume.value() / 100.0 / max(1.0, peak)
        silent = primary is None or primary.muted
        self.audio_out.setMuted(silent)
        self.audio_out.setVolume(0.0 if silent else scale * _gain_linear(primary.gain_db))
        for aux in self._aux.values():
            aux.apply_volume(scale)

    def _current_edited(self) -> float | None:
        """Where the preview is, in EDITED seconds.

        Read out of the run the player is in, never out of a global
        source→edited mapping: with two files on the timeline the player's
        position only means something against the clip it is decoding.  Over a
        black gap there is no player to ask, so the gap clock's own position is
        the answer.
        """
        if self._run < 0 or self._run >= len(self._runs):
            return self._gap_edited if self._runs or self.info is not None else None
        run = self._runs[self._run]
        position = (
            self._pending_seek_ms
            if self._pending_seek_ms is not None
            else self.player.position()
        )
        length = max(0.0, run.src_out - run.src_in)
        offset = min(length, max(0.0, position / 1000.0 - run.src_in))
        return run.edited_start + offset

    def _publish_edited(self, t_edited: float, *, force: bool = False) -> None:
        """Hand an edited position to everything that displays or follows it.

        ``force`` re-seeks the extra audio players instead of letting them
        drift: what playback crossing a cut needs, and what a plain tick must
        NOT do — a seek per position report stutters them.
        """
        if (
            self._pending_seek_ms is None
            and not self.timeline.is_scrubbing
            and time.monotonic() >= self._seek_guard_until
        ):
            self.timeline.set_playhead(t_edited)
        self._update_time_label(t_edited)
        self._sync_aux_positions(force=force, t_edited=t_edited)
        if not self._playing:
            self._sync_stage()

    def _sync_aux_positions(
        self, *, force: bool = False, t_edited: float | None = None
    ) -> None:
        """Put every extra player where edited time says, and silence the main
        one over any range its own lane has no audio for.

        That last part is the lift: the render writes ``volume=0`` over a lifted
        range, and the main player has no per-range control, so the output is
        muted for as long as the primary source has no span there.
        """
        if self._closing:
            return
        if t_edited is None:
            t_edited = self._current_edited()
        if t_edited is None:
            return
        for aux in self._aux.values():
            aux.follow(float(t_edited), self._playing, force=force)
        if self._primary is not None:
            silent = self._primary.muted or self._primary.source_at(float(t_edited)) is None
            if self.audio_out.isMuted() != silent:
                self.audio_out.setMuted(silent)

    def _on_track_mix_changed(self, *_args: object) -> None:
        """A lane's M or gain moved: it is the render mix *and* the preview."""
        self._sync_preview_audio(force=False)  # a re-seek per slider step stutters
        self._update_actions()

    # -------------------------------------------------------------- assets
    def _asset_root(self) -> Path:
        if self._asset_dir is None:
            self._asset_dir = Path(tempfile.mkdtemp(prefix="ffcut-assets-"))
            # closeEvent removes it on a clean exit; this catches everything
            # else (uncaught exception, SIGTERM from the session, tests).
            atexit.register(shutil.rmtree, str(self._asset_dir), ignore_errors=True)
        return self._asset_dir

    @staticmethod
    def _release(slot: _Slot, dispatch: Callable[[], None]) -> None:
        slot.worker = None
        dispatch()

    def _on_thumbs_needed(self, path: str, times: object) -> None:
        """The timeline wants frames of ``path`` at those SOURCE timestamps.

        It asks in source time, per file, because it is the only one that knows
        where each clip sits and which file it came from — a second video lane's
        clip has its own time base, and mapping again at delivery would land
        every frame on the wrong one.
        """
        slot = self._thumbs.setdefault(str(path), _Slot())
        slot.pending = [float(t) for t in times]  # type: ignore[union-attr]
        self._dispatch_thumbs(str(path))

    def _dispatch_thumbs(self, path: str) -> None:
        slot = self._thumbs.get(path)
        if self.info is None or slot is None or slot.worker is not None:
            return
        times = slot.pending
        slot.pending = None
        if not times:
            return
        gen = self._media_gen
        # One job at a time PER FILE, and each result is read (QPixmap) before
        # that file's next job starts — so one directory per file is safe.
        target = str(self._asset_root() / "thumbs" / _dir_key(path))
        slot.worker = self._spawn(
            partial(
                thumbnails_at, path, target, list(times), THUMB_W,
                fill_gaps=True, workers=self.config.prefs.jobs,
            ),
            lambda paths: self._on_thumbs_ready(gen, path, list(times), paths),
            lambda message: self._status(f"Thumbnails unavailable: {message}"),
        )
        slot.worker.finished.connect(
            lambda: self._release(slot, partial(self._dispatch_thumbs, path))
        )

    def _on_thumbs_ready(
        self, gen: int, path: str, times: list[float], paths: list[str]
    ) -> None:
        # Stale if the file changed, or a newer request is already queued.
        slot = self._thumbs.get(path)
        if gen == self._media_gen and slot is not None and slot.pending is None:
            self.timeline.set_thumbnails(paths, times, path)

    def _on_waveform_needed(self, track: int, start: float, end: float) -> None:
        slot = self._waves.setdefault(int(track), _Slot())
        slot.pending = (float(start), float(end))
        self._dispatch_wave(int(track))

    def _dispatch_wave(self, track: int) -> None:
        slot = self._waves[track]
        if self.info is None or slot.worker is not None or slot.pending is None:
            return
        start, end = slot.pending
        slot.pending = None
        source = self.timeline.wave_source(track)
        wanted = self.timeline.wave_request(track, start, end)
        if source is None or wanted is None:
            return
        # The timeline works the range out itself: once a block can be split
        # and slid, where the lane is on screen and where that is inside the
        # file are two different questions, and only it can answer the second.
        # A dropped file shorter than the video has no waveform past its end,
        # so the range it hands back is already clamped to what exists.
        path, stream, _duration = source
        src0, src1 = wanted
        gen = self._media_gen
        out = str(self._asset_root() / f"wave{track}.png")
        width = max(256, self.timeline.width())
        # WAVE_COLOR, not the active theme's trace colour: the lane re-inks the
        # PNG through its alpha at paint time, so one fetch serves both themes
        # and a theme switch never has to re-run ffmpeg.
        slot.worker = self._spawn(
            partial(
                waveform_png, path, stream, out, width, WAVE_H, src0, src1, WAVE_COLOR
            ),
            lambda png: self._on_wave_ready(gen, track, png, src0, src1, source),
            lambda message: self._status(
                f"Waveform for track {track + 1} unavailable: {message}"
            ),
        )
        slot.worker.finished.connect(
            lambda: self._release(slot, partial(self._dispatch_wave, track))
        )

    def _on_wave_ready(
        self, gen: int, track: int, png: str, src0: float, src1: float, source: object
    ) -> None:
        # The lane must still be the one fetched for: a lane removed mid-fetch
        # shifts the numbering, and a wrong trace is worse than none.
        if (
            gen == self._media_gen
            and self._waves[track].pending is None
            and self.timeline.wave_source(track) == source
        ):
            self.timeline.set_waveform(track, png, src0, src1)

    # ------------------------------------------------------------------ player
    def toggle_play(self) -> None:
        if self.info is None:
            return
        self._set_playing(not self._playing)

    def _on_media_status(self, status: QMediaPlayer.MediaStatus) -> None:
        # Teardown must not re-prime the player.  ``closeEvent``'s ``stop()``
        # delivers this signal *synchronously*, so priming here would call
        # pause() re-entrantly inside stop() -- which deadlocks against the
        # QFFmpeg audio renderer thread.  That is the original teardown hang.
        if self._closing:
            return
        if self._in_swap and self.player.source().toLocalFile() != self._video_path:
            # ``setSource`` walks the OUTGOING media back down first and
            # delivers ITS status changes synchronously, while ``source()`` is
            # still the old file and before the new one has begun loading.
            # Acting on those spends the parked seek on the file we are
            # leaving: measured, a swap then started the new clip at 0:00 (or
            # at the old file's duration) with no frame on screen, because the
            # priming pause had been used up too.  Wait for the status that
            # belongs to the source we actually asked for.
            return
        if status == QMediaPlayer.MediaStatus.LoadedMedia:
            self._in_swap = False
            # A source swap could not seek until now: the backend drops a
            # setPosition on media it has not loaded yet.
            if self._load_seek_ms is not None:
                self.player.setPosition(self._load_seek_ms)
                self._load_seek_ms = None
            # audioTracks() is only populated now, so this is where the main
            # player learns which embedded track it is carrying.
            self._sync_preview_audio()
            if self._playing:
                self.player.play()
            # Pausing a freshly loaded source primes the pipeline so the first
            # frame is on screen instead of a black box.
            elif self.player.playbackState() == QMediaPlayer.PlaybackState.StoppedState:
                self.player.pause()
        elif status == QMediaPlayer.MediaStatus.EndOfMedia and not self._in_swap:
            # This FILE ran out; the edit may not have — a second clip, or an
            # audio bed running on past the picture.
            self._end_of_file()
        elif status in (
            QMediaPlayer.MediaStatus.InvalidMedia,
            QMediaPlayer.MediaStatus.NoMedia,
        ):
            # Nothing will ever load, so the swap latches must not stay set:
            # they gate every position report the preview acts on.
            self._in_swap = False
            self._load_seek_ms = None

    def _on_player_error(self, error: QMediaPlayer.Error, message: str) -> None:
        if error == QMediaPlayer.Error.NoError or self._closing:
            return
        self._in_swap = False  # the seek it was waiting for will never land
        self._load_seek_ms = None
        self._status(f"Preview error: {message}")
        if self._playing and self._run >= 0:
            # Nothing will ever come off this run, so nothing will ever advance
            # past it: no position reports, and the gap clock does not run while
            # a run is current.  The transport sat on "▶ Playing" with a frozen
            # playhead.  Step over it — _advance_run only moves forward, so a
            # timeline of unplayable files walks to the end and stops.
            self._advance_run()

    def _on_playhead_moved(self, t_edited: float) -> None:
        """The user moved the playhead: put the preview where it now points."""
        # Update the readout now rather than waiting for the player to report
        # back — during a scrub that round trip is visibly late.
        self._update_time_label(float(t_edited))
        self._seek_edited(float(t_edited))

    def _seek_edited(self, t_edited: float) -> None:
        """Take the preview to an edited position: the right run, then a seek."""
        t = max(0.0, min(self.timeline.edited_duration(), float(t_edited)))
        self._seek_guard_until = time.monotonic() + SEEK_GUARD_S
        self._reseat(t, coalesce=True)
        # From the PENDING position, not the flushed one: a scrub coalesces the
        # main seeks and the extra tracks have to follow the pointer, not the
        # demuxer catching up behind it.
        self._sync_aux_positions(force=True, t_edited=t)

    def _goto_edited(self, t_edited: float) -> None:
        """Move playhead *and* preview to an edited-time position."""
        t = max(0.0, min(self.timeline.edited_duration(), float(t_edited)))
        self.timeline.set_playhead(t)
        self._seek_edited(t)
        self._update_time_label(t)

    def _flush_seek(self) -> None:
        if self._pending_seek_ms is not None:
            self.player.setPosition(self._pending_seek_ms)
            self._pending_seek_ms = None
            self._sync_aux_positions(force=True)

    def _on_position_changed(self, position_ms: int) -> None:
        """The main player reported where it is, inside the run it is playing.

        A position past the end of that run means the run is finished — which
        is what a cut, a lift, a clip boundary and the end of a file all look
        like from here, so there is one answer: move on to the next run.
        """
        if self.info is None or self._closing or not 0 <= self._run < len(self._runs):
            return
        if self._pending_seek_ms is not None or self._load_seek_ms is not None:
            return  # a seek we asked for has not landed; its target is the truth
        run = self._runs[self._run]
        position = position_ms / 1000.0
        if position >= run.src_out - RUN_EPS:
            self._advance_run()
            return
        self._publish_edited(run.edited_start + max(0.0, position - run.src_in))

    def _update_time_label(self, t_edited: float) -> None:
        total = self.timeline.edited_duration() if self.info else 0.0
        self.transport.time_label.setText(f"{format_tc(t_edited)} / {format_tc(total)}")
        # The seek line rides the same edited-time readout as the timecode, so
        # it follows playback, scrubs, skips and cuts without its own wiring.
        self.transport.seek.set_position(t_edited, total)

    def _skip(self, direction: int) -> None:
        if self.info is None:
            return
        step = self.config.prefs.skip_seconds * (1 if direction >= 0 else -1)
        self._goto_edited(self.timeline.playhead() + step)

    def _jump_cut(self, direction: int) -> None:
        if self.info is None:
            return
        here = self.timeline.playhead()
        points = self.timeline.model.splice_points_edited()
        if direction >= 0:
            target = next((p for p in points if p > here + 1e-3), None)
            self._goto_edited(
                target if target is not None else self.timeline.edited_duration()
            )
        else:
            target = next((p for p in reversed(points) if p < here - 1e-3), None)
            self._goto_edited(target if target is not None else 0.0)

    # -------------------------------------------------------------------- cuts
    def cut_selection(self) -> None:
        """Menu / Ctrl+S path: apply the current selection as a cut."""
        if self.info is None:
            return
        if self.timeline.selection() is None:
            self._status("Select a range on the timeline first.", 4000)
            return
        self.timeline.cut_selection()

    def undo(self) -> None:
        self.timeline.undo()

    def redo(self) -> None:
        self.timeline.redo()

    def _refresh_selection(self) -> None:
        """Status readout: the marked range, and which lanes an edit will hit
        whenever that is not simply all of them."""
        selection = self.timeline.selection()
        if selection is None:
            text = "No selection"
        else:
            start, end = selection
            text = (
                f"Selection  {format_tc(start)} → {format_tc(end)}"
                f"   ({format_tc(end - start)})"
            )
        if not self.timeline.all_armed():
            text += "  ·  lift: " + ", ".join(self.timeline.armed_names())
        blocks = len(self.timeline.clip_selection())
        if blocks:
            text += f"  ·  {blocks} block{'' if blocks == 1 else 's'} selected"
        self.selection_label.setText(text)
        self._update_actions()

    def _on_cuts_changed(self) -> None:
        self._update_time_label(self.timeline.playhead())
        # A cut re-maps edited time onto the file, so every run and every extra
        # track's spans move with it.
        self._sync_preview()
        self._update_actions()

    # ------------------------------------------------------------- preferences
    def open_preferences(self) -> None:
        dialog = PreferencesDialog(self.config.prefs, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        dialog.deleteLater()
        self.config.prefs = dialog.prefs()
        theme.apply(QApplication.instance(), self.config.prefs.theme)
        self.transport.set_skip_seconds(self.config.prefs.skip_seconds)
        self.config.save()
        self._sync_preview()  # swaps to (or back from) a preview copy

    # ------------------------------------------------------------------ render
    def _update_actions(self) -> None:
        has_media = self.info is not None
        segments = self.timeline.keep_segments_source() if has_media else []
        has_selection = has_media and self.timeline.selection() is not None

        self.transport.set_enabled_all(has_media)
        for action in (self.act_select_all, self.act_mark_in, self.act_mark_out):
            action.setEnabled(has_media)
        self.act_clear_sel.setEnabled(has_selection)
        self.act_cut.setEnabled(has_selection)
        self.act_undo.setEnabled(has_media and self.timeline.can_undo())
        self.act_redo.setEnabled(has_media and self.timeline.can_redo())

        blocks = len(self.timeline.clip_selection()) if has_media else 0
        self.act_split.setEnabled(has_media)
        self.act_add_text.setEnabled(has_media)
        self.act_group.setEnabled(blocks > 1)
        self.act_ungroup.setEnabled(has_media and self.timeline.has_grouped_selection())
        self.act_copy_blocks.setEnabled(blocks > 0)
        self.act_delete_blocks.setEnabled(blocks > 0)
        # NOT gated on has_media: with every block deleted the duration is 0,
        # and pasting one back is the way out (so is Ctrl+Z).  It used to be
        # enabled there and then answer "Nothing copied yet".
        self.act_paste_blocks.setEnabled(
            self.info is not None and self.timeline.can_paste()
        )

        empty = has_media and not segments
        self.warning_label.setText(
            "Everything is cut — there is nothing left to render. Press Ctrl+Z to undo."
            if empty
            else ""
        )
        self.warning_label.setVisible(empty)

        can_render = has_media and not empty
        self.render_button.setEnabled(can_render)
        self.act_render_full.setEnabled(can_render)
        self.act_render_sel.setEnabled(can_render and has_selection)
        self.act_render_sel.setToolTip(
            "Render only the marked range (cuts inside it are still honoured)."
            if has_selection
            else "Mark a selection on the timeline to enable this."
        )

        summary = self.options.summary()
        if has_media:
            kept = format_tc(total_output_seconds(segments), decimals=1)
            summary = (
                f"Keeping {kept} in {len(segments)} segment"
                f"{'' if len(segments) == 1 else 's'}  ·  {summary}"
            )
        self.summary_label.setText(summary)

    def render_full(self) -> None:
        self._render(self.timeline.keep_segments_source(), "whole edit")

    def render_selection(self) -> None:
        segments = self.timeline.selection_segments_source()
        if not segments:
            self._status("Mark a selection first (I / O, or drag).", 4000)
            return
        self._render(segments, "selection")

    def _render(self, segments: list[Segment], what: str) -> None:
        # One snapshot: a file loading behind the dialogs below must not swap
        # the media out from under these segments.
        info = self.info
        if info is None:
            return
        plan = self.timeline.render_plan()
        self._update_actions()
        if not segments:
            QMessageBox.warning(
                self, "ffcut", self.warning_label.text() or "Nothing to render."
            )
            return

        source = Path(info.path)
        suggested = str(
            Path(self.config.last_save_dir)
            / f"{source.stem}-{'sel' if what == 'selection' else 'cut'}{self.options.suffix}"
        )
        chosen, _ = QFileDialog.getSaveFileName(
            self,
            f"Render {what} to…",
            suggested,
            "MP4 video (*.mp4);;Matroska video (*.mkv);;All files (*)",
        )
        if not chosen:
            return
        chosen_path = Path(chosen)
        out_dir = chosen_path.parent
        self.config.last_save_dir = str(out_dir)

        names = [
            f"0:a:{lane.clips[0].stream}" + (f" {lane.gain_db:+.1f} dB" if lane.gain_db else "")
            if lane.clips and lane.clips[0].path == info.path
            else Path(lane.clips[0].path).name
            + (f" {lane.gain_db:+.1f} dB" if lane.gain_db else "")
            for lane in plan.audio
            if lane.clips and not lane.muted
        ]
        audio_text = (
            "video only (everything muted)" if not names else "audio " + ", ".join(names)
        )
        lifted = sum(len(lane.holes) for lane in (*plan.video, *plan.audio))
        extra_video = max(0, len(plan.video) - 1)
        total_out = total_output_seconds(segments)
        summary = (
            f"{what.capitalize()}: {len(segments)} segment"
            f"{'' if len(segments) == 1 else 's'} · "
            f"{format_tc(total_out, decimals=1)} of output · {audio_text}"
            + (f" · {extra_video} extra video track"
               f"{'' if extra_video == 1 else 's'}" if extra_video else "")
            + (f" · {lifted} lifted range{'' if lifted == 1 else 's'}" if lifted else "")
        )
        dialog = RenderOptionsDialog(
            out_dir, chosen_path.name, self.options, summary, self
        )
        accepted = dialog.exec() == QDialog.DialogCode.Accepted
        dialog.deleteLater()
        if not accepted:
            return

        self.options = dialog.options()
        self.config.render = self.options
        self.config.save()
        self._update_actions()

        out_path = dialog.out_path()
        # The save dialog already confirmed overwriting *chosen_path*; only ask
        # again if the options dialog changed the name or the container.
        if out_path.exists() and out_path != chosen_path:
            reply = QMessageBox.question(
                self,
                "ffcut",
                f"{out_path.name} already exists. Overwrite?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return
        inputs = {info.path, *(c.path for lane in (*plan.video, *plan.audio) for c in lane.clips)}
        if any(_same_file(out_path, p) for p in inputs):
            QMessageBox.critical(
                self, "ffcut", f"{out_path.name} is one of the files being edited.\n"
                "Pick another name — rendering over it would destroy the source."
            )
            return

        width, height = self.options.dimensions()
        settings = RenderSettings(
            segments=segments,
            plan=plan,
            fps=self.options.fps,
            tmix_frames=self.options.tmix_frames,
            width=width,
            height=height,
            out_path=str(part_path(out_path)),
            encoder=self.options.encoder,
            container=self.options.container,
            mbps=self.options.mbps,
            crf=self.options.crf,
            av1_preset=self.options.av1_preset,
        )
        try:
            argv = build_command(info, settings)
        except Exception as exc:
            QMessageBox.critical(
                self, "ffcut — cannot build command", f"{type(exc).__name__}: {exc}"
            )
            return

        self._set_playing(False)  # also stops the gap clock and the aux players

        job = RenderJob(argv, total_out)
        progress = RenderProgressDialog(job, out_path, self)
        result = progress.exec()
        progress.deleteLater()
        if result == QDialog.DialogCode.Accepted:
            self._render_succeeded(out_path)
        elif progress.was_cancelled:
            self._status("Render cancelled — partial output removed.", 5000)
        else:
            self._render_failed(progress.exit_code, progress.message)

    def _render_succeeded(self, out_path: Path) -> None:
        self._status(f"Rendered {out_path}", 8000)
        box = QMessageBox(self)
        box.setWindowTitle("ffcut")
        box.setIcon(QMessageBox.Icon.Information)
        box.setText("Render finished.")
        box.setInformativeText(str(out_path))
        open_button = box.addButton("Open folder", QMessageBox.ButtonRole.ActionRole)
        box.addButton(QMessageBox.StandardButton.Close)
        box.exec()
        box.deleteLater()
        if box.clickedButton() is open_button:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(out_path.parent)))

    def _render_failed(self, code: int, message: str) -> None:
        """Show the engine's own error text — VA-API failures are diagnosable."""
        self._status(f"Render failed (exit {code}).", 8000)
        box = QMessageBox(self)
        box.setWindowTitle("ffcut — render failed")
        box.setIcon(QMessageBox.Icon.Critical)
        box.setText(f"The render failed (exit code {code}).")
        if is_vaapi_encoder(self.options.encoder):
            box.setInformativeText(
                "This render used VA-API (GPU). VA-API can fail because of the driver "
                "or because the GPU's render node could not be opened — if that is "
                "what the message below says, re-run with the x264 (CPU) encoder, or "
                "switch to SVT-AV1 (also CPU)."
            )
        box.setDetailedText(message.strip() or "ffmpeg produced no error output.")
        box.exec()
        box.deleteLater()

    # ---------------------------------------------------------- drag and drop
    @staticmethod
    def _dropped_path(mime: QMimeData | None) -> tuple[str, str] | None:
        """First usable local file in a drop, as ``(path, "video" | "audio")``.

        A video REPLACES the loaded media, as it always has; an audio file ADDS
        a lane and keeps the video.  Anything else is rejected, so the drop
        cursor says no before the user lets go.
        """
        if mime is None or not mime.hasUrls():
            return None
        for url in mime.urls():
            path = Path(url.toLocalFile())
            if not (url.isLocalFile() and path.is_file()):
                continue
            suffix = path.suffix.lower()
            if suffix in VIDEO_SUFFIXES:
                return (str(path), "video")
            if suffix in AUDIO_SUFFIXES:
                return (str(path), "audio")
        return None

    def handle_drag(self, event: QDragEnterEvent) -> None:
        """Accept or refuse a drag, wherever in the window it is hovering."""
        dropped = self._dropped_path(event.mimeData())
        # An audio file needs somewhere to hang its lane; with nothing loaded
        # there is no timeline to add it to.
        if dropped and (dropped[1] == "video" or self.info is not None):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:  # noqa: N802
        self.handle_drag(event)

    def dragMoveEvent(self, event: QDragEnterEvent) -> None:  # noqa: N802
        self.handle_drag(event)

    def dropEvent(self, event: QDropEvent) -> None:  # noqa: N802
        self.handle_drop(event, self)

    def handle_drop(self, event: QDropEvent, source: QWidget) -> None:
        """Take a drop from anywhere in the window — the preview included.

        ``source`` is the widget ``event.position()`` is relative to, so a drop
        forwarded off the video stage can still be mapped onto the timeline and
        land on the lane the pointer was over.
        """
        dropped = self._dropped_path(event.mimeData())
        if dropped is None or (dropped[1] == "audio" and self.info is None):
            event.ignore()
            if dropped:
                self._status("Load a video first, then drop audio onto it.", 4000)
            return
        event.acceptProposedAction()
        path, kind = dropped
        target = self._drop_target(event, source)
        if target is not None:
            self.place_on_lane(path, kind, target[1], target[0])
        elif kind == "video":
            self.load_file(path)
        else:
            self.add_audio_file(path)

    def _drop_target(
        self, event: QDropEvent, source: QWidget
    ) -> tuple[str, int] | None:
        """``("place" | "append", lane)`` for a drop, or ``None``.

        ``None`` means "not aimed at a lane" — a drop on the preview, or on a
        window with nothing loaded — which keeps the old meaning: a video
        REPLACES the media and an audio file adds a lane of its own.  The
        pointer is routed through global coordinates because the event may have
        been forwarded from the video stage, whose origin is not the window's.
        """
        local = self.timeline.mapFromGlobal(source.mapToGlobal(event.position().toPoint()))
        if not self.timeline.rect().contains(local):
            return None
        return self.timeline.drop_target(local)

    def add_audio_file(self, path: str) -> None:
        """A drop that was not aimed at a lane: the file gets a lane of its own."""
        self.place_on_lane(path, "audio", None, "place")

    def place_on_lane(self, path: str, kind: str, lane: int | None, mode: str) -> None:
        """Put the dropped file on ``lane`` (``None`` = a new lane), probing it
        off the GUI thread through ``_spawn`` so teardown can kill the probe."""
        gen = self._media_gen
        self._status(f"Probing {Path(path).name}…")
        self._spawn(
            partial(probe if kind == "video" else probe_audio, str(path)),
            lambda info: self._on_clip_probed(gen, info, lane, mode),
            lambda message: self._on_clip_probe_failed(gen, message),
        )

    def _on_clip_probed(
        self, gen: int, info: MediaInfo | AudioFileInfo, lane: int | None, mode: str
    ) -> None:
        # A file loaded while the probe ran replaced the lanes this belonged to.
        if gen != self._media_gen or self._closing or self.info is None:
            return
        kind = "video" if isinstance(info, MediaInfo) else "audio"
        placed = self.timeline.add_clip(lane, info, mode)
        if not placed and lane is not None:
            # The wrong kind of lane — an mp3 let go over the video track, say.
            # A lane of its own beats a refusal the user has to read.
            lane, mode = None, "place"
            placed = self.timeline.add_clip(None, info, "place")
        if not placed:
            self._status(f"{Path(info.path).name} has no length to place.", 6000)
            return
        if lane is None:  # the lane add_clip just grew: last of its kind
            lane = (
                self.timeline.video_lane_count() - 1 if kind == "video"
                else self.timeline.lane_count() - 1
            )
        name = getattr(info, "label", None) or Path(info.path).name
        # Where it LANDED, not where it was aimed: a "place" drop onto a lane
        # something else reached first is pushed past it.
        where = f"at {format_tc(placed.start, decimals=0)}"
        tracks = len(getattr(info, "audio", ()) or ())
        note = ""
        if kind == "video" and tracks:
            note = (
                f"  Its {tracks} audio track{'' if tracks == 1 else 's'} came with it, "
                "grouped to the picture."
            )
        self._status(f"Added {name} to track {lane + 1} {where}.{note}", 8000)
        # Playable immediately: this is where a dropped mp3 gets its own player
        # and a second file's footage gets its runs.
        self._sync_preview()
        self._update_actions()

    def _on_clip_probe_failed(self, gen: int, message: str) -> None:
        if gen != self._media_gen or self._closing:
            return
        self._status("Probe failed.")
        QMessageBox.critical(self, "ffcut — probe failed", message)

    # ---------------------------------------------------------------- blocks
    # Thin wrappers: the timeline owns the edit, this owns the words for it.
    # Every one of them lands in _on_cuts_changed through _after_lane_change,
    # which re-plans the preview -- so a moved block is audible and visible
    # where it now is without any of these having to say so.
    def split_blocks(self) -> None:
        if not self.timeline.split_blocks():
            self._status(
                "Park the playhead over a block first — S splits what it crosses.",
                5000,
            )

    def delete_blocks(self) -> None:
        if not self.timeline.delete_blocks():
            self._status("Alt+click a block to select it, then Delete.", 5000)

    def group_blocks(self) -> None:
        if not self.timeline.group_blocks():
            self._status("Select two or more blocks to group them.", 5000)
        else:
            self._status("Grouped — they now move and delete together.", 5000)

    def ungroup_blocks(self) -> None:
        if self.timeline.ungroup_blocks():
            self._status("Ungrouped.", 4000)

    def copy_blocks(self) -> None:
        count = self.timeline.copy_blocks()
        self._status(
            f"Copied {count} block(s) — Ctrl+V pastes at the playhead." if count
            else "Alt+click a block to select it, then Copy.",
            5000,
        )
        self._update_actions()

    def paste_blocks(self) -> None:
        if not self.timeline.paste_blocks():
            self._status("Nothing copied yet.", 4000)

    def _on_blocks_changed(self, _selected: object = None) -> None:
        self._refresh_selection()

    def build_block_menu(self, lane: int | None) -> QMenu:
        """Right-click menu over a block: the selection's own verbs first.

        The track verbs stay on the bottom half, so one right-click still
        reaches both — the menu is about what is under the pointer, and a block
        sits on a track.
        """
        menu = QMenu(self)
        count = len(self.timeline.clip_selection())
        head = menu.addAction(f"{count} block(s) selected")
        head.setEnabled(False)
        menu.addSeparator()
        for text, slot, enabled in (
            ("Deselect", self.timeline.clear_block_selection, count > 0),
            ("Group", self.group_blocks, count > 1),
            ("Ungroup", self.ungroup_blocks, self.timeline.has_grouped_selection()),
            ("Split at playhead", self.split_blocks, True),
            ("Copy", self.copy_blocks, count > 0),
            ("Paste at playhead", self.paste_blocks, self.timeline.can_paste()),
            ("Delete", self.delete_blocks, count > 0),
            ("Edit text…", self._edit_selected_text, self._selected_title() is not None),
        ):
            action = menu.addAction(text)
            action.setEnabled(bool(enabled))
            action.triggered.connect(lambda _=False, fn=slot: fn())
        menu.addSeparator()
        # Parented to this menu, so the submenu and its actions live exactly as
        # long as it does -- addMenu does not take ownership.
        track = self.build_lane_menu(lane, parent=menu)
        track.setTitle("Track")
        menu.addMenu(track)
        return menu

    # ------------------------------------------------------------- lane menu
    def build_lane_menu(self, lane: int | None, parent: QWidget | None = None) -> QMenu:
        """The right-click menu for ``lane`` (``None`` = no lane under it)."""
        menu = QMenu(parent or self)
        for text, add in (
            ("Add video track", self.timeline.add_video_lane),
            ("Add audio track", self.timeline.add_audio_lane),
        ):
            action = menu.addAction(text)
            action.setToolTip("The new track is empty — drop a file on it.")
            # triggered carries a checked flag; the default swallows it.
            action.triggered.connect(
                lambda _=False, fn=add, what=text: self._add_lane(fn, what)
            )

        menu.addSeparator()
        action = menu.addAction("Add file to this track…")
        action.setEnabled(lane is not None)
        if lane is not None:
            action.triggered.connect(
                lambda _=False, n=int(lane): self._pick_clip_for(n)
            )

        menu.addSeparator()
        action = menu.addAction("Remove this track")
        # Lane 0 holds the loaded capture and the output geometry; File ▸ Open
        # is how that gets changed, so the timeline refuses to remove it.
        action.setEnabled(lane is not None and lane > 0)
        if action.isEnabled():
            action.triggered.connect(
                lambda _=False, n=int(lane or 0): self._remove_lane(n)
            )
        return menu

    def show_lane_menu(self, global_pos: object) -> None:
        """Right-click on a lane or its header: add and remove tracks.

        ``drop_target`` clamps a point just off the stack to the nearest lane,
        which is what a right-click two pixels into the ruler means as well.
        """
        if self.info is None:
            return
        local = self.timeline.mapFromGlobal(global_pos)  # type: ignore[arg-type]
        target = self.timeline.drop_target(local)
        lane = None if target is None else target[1]
        # A right-click ON a block is about that block.  Selecting it first is
        # what every editor does -- acting on something you cannot see selected
        # is the surprising half.
        hit = self.timeline.block_at(local)
        if hit is not None:
            if hit[1].cid not in self.timeline.clip_selection():
                self.timeline.select_block(hit[1])
            self.build_block_menu(lane).exec(global_pos)  # type: ignore[arg-type]
            return
        self.build_lane_menu(lane).exec(global_pos)  # type: ignore[arg-type]

    def _add_lane(self, add: Callable, what: str) -> None:
        """A new empty lane is a drop target — that is what makes it useful."""
        lane = int(add())
        self._status(
            f"{what.replace('Add', 'Added')} {lane + 1}. Drop a file on it.", 6000
        )
        self._update_actions()

    def _remove_lane(self, lane: int) -> None:
        if not self.timeline.remove_lane(lane):
            self._status("That track cannot be removed.", 4000)
            return
        self._drop_unused_aux(set())  # whatever is still wanted is re-made below
        self._sync_preview()
        self._update_actions()

    def _pick_clip_for(self, lane: int) -> None:
        """File picker for a lane, filtered by what that lane can hold."""
        kind = self.timeline.lane_kind(lane) or "video"
        if kind == "audio":
            suffixes, title = AUDIO_SUFFIXES, "Add audio to this track"
        else:
            suffixes, title = VIDEO_SUFFIXES, "Add video to this track"
        pattern = " ".join("*" + s for s in sorted(suffixes))
        path, _ = QFileDialog.getOpenFileName(
            self,
            title,
            self.config.last_open_dir,
            f"{kind.capitalize()} files ({pattern});;All files (*)",
        )
        if path:
            self.config.last_open_dir = str(Path(path).parent)
            self.place_on_lane(path, kind, lane, "append")

    # ------------------------------------------------------------------ close
    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
        if not self._closing and not self._discard_ok():
            event.ignore()
            return
        self._closing = True
        self.config.render = self.options
        self.config.save()
        # No more asset jobs: a queued worker-finished signal would otherwise
        # start the next pending one on a window that is going away, and a
        # QThread destroyed while still running aborts the process.
        self.info = None
        self._gap_timer.stop()
        self._playing = False
        for slot in (*self._thumbs.values(), *self._waves.values()):
            slot.pending = None
        # Kill the ffmpeg children *before* joining.  A worker sitting in
        # subprocess.communicate() pins the GUI thread here for as long as the
        # probe timeouts allow (WAVEFORM_TIMEOUT/THUMB_TIMEOUT, 20 s each) --
        # that join is what made closing feel frozen.  Killed, it unwinds at once.
        cancel_probes()
        deadline = time.monotonic() + WORKER_JOIN_S
        for worker in list(self._workers):
            left = int(max(0.0, deadline - time.monotonic()) * 1000)
            if worker.wait(left):
                continue
            # Over budget regardless: detach it, so destroying the window does
            # not destroy a running QThread -- which aborts the process.
            worker.setParent(None)
            _ORPHANS.append(worker)
        # Unlatch: the flag only ever needed to cover the join above, and this
        # module outlives the window (tests, and a second window per process).
        resume_probes()
        if self._asset_dir is not None:
            shutil.rmtree(self._asset_dir, ignore_errors=True)
            self._asset_dir = None

        # Everything durable is flushed by now, which is deliberate: the player
        # teardown below is the one step left that can wedge.  stop() delivers
        # mediaStatusChanged synchronously (see _on_media_status, guarded), and
        # independently of that it sometimes deadlocks against the QFFmpeg audio
        # renderer thread with no Python frame to blame.  Bound it rather than
        # hand the user a window that will not close.
        watchdog = threading.Timer(PLAYER_TEARDOWN_S, _force_exit)
        watchdog.daemon = True
        watchdog.start()
        try:
            # The auxiliary audio players belong to this step, not an earlier
            # one: each is a QMediaPlayer with the same stop() risk as the main
            # one, so they get the same watchdog.  Each latches its own
            # _closing first (see _AuxPlayer.close) — the re-entrancy guard
            # that _on_media_status has, one per player.
            while self._aux:
                self._aux.popitem()[1].close()
            self.player.stop()
            # Drop the source while Python is still alive: it releases the
            # ffmpeg decoder and the file handle deterministically, instead of
            # leaving a loaded QMediaPlayer to be torn down at interpreter exit.
            self.player.setSource(QUrl())
        finally:
            watchdog.cancel()
        super().closeEvent(event)
