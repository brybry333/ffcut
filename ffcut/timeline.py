"""ffcut timeline — edited-time cut model + multi-lane zoomable widget.

``CutList`` is the Qt-free model: SOURCE-time cuts plus the source<->edited
mapping the whole v2 UI rests on.  Cuts COLLAPSE, so nothing above it ever draws
or skips a gap.  Its undo stack carries an opaque UI snapshot, which is how the
view and playhead come back with the edit they belonged to.

``TimelineWidget`` stacks lanes right of a fixed gutter of lane headers: video
lanes first, then audio lanes.  Each lane holds an ORDERED LIST OF CLIPS, not
one file — dropping onto an empty lane places a clip at 0:00, dropping onto a
lane that already has content appends after it.  The gutter never scrolls or
zooms; everything right of it draws, hit-tests, scrolls and zooms entirely in
EDITED time.  Assets are requested by signal and cached in SOURCE coordinates,
so a later cut re-collapses pixels it already has instead of waiting on ffmpeg.

Video lanes are a PRIORITY STACK.  "Add video track" puts the new lane
*underneath* the main one, and that is the point: lane 0 paints over lane 1, so
lane 1 is what shows through wherever lane 0 has a hole (a lift) or has run out
of footage.  ``render.visible_spans`` walks exactly that stack, and the render
and the preview both come out of it.  Audio lanes have no priority; they mix.

The timeline runs to the END OF THE LONGEST LANE, not to the video's duration.
A music bed longer than the footage extends the edit and the render pads the
video with black (``tpad``) and every short audio source with silence
(``apad``) to match.

Two kinds of edit share one history:

* a **ripple cut** removes a range from every lane and the timeline collapses —
  that is ``CutList``'s cut list, and it alone owns the source<->edited mapping;
* a **lift** removes a range from the *armed* lanes only, leaving a hole.  Holes
  are stored per lane in SOURCE time, so they survive a later ripple cut and map
  through ``source_to_edited`` like everything else.  The timeline length does
  not change, so the lanes stay in sync.

Three conventions to know before touching this file:

* Every public time in/out of ``TimelineWidget`` is EDITED time unless the name
  says ``_source``.
* ``set_playhead()`` is programmatic and does NOT emit ``playheadMoved``; only a
  user gesture does.  That is what keeps player->timeline->player from looping.
* Two index spaces meet here.  **Lane ids** are positions in the lane stack and
  are what ``lane_kind``/``add_clip``/``remove_lane``/``drop_target``, the ARMED
  set and ``armedChanged`` speak in.  **Audio-track ordinals** count audio lanes
  only and are what the ``track*`` signals, ``waveformNeeded``, ``set_waveform``
  and ``wave_source`` speak in — a video lane inserted above them must not
  renumber the app's waveform router.  ``_audio_ordinal`` and ``wave_source``
  are the only crossings.  Holes are keyed by a third thing again — the lane's
  stable KEY — so inserting a lane cannot move them.
* Every lane armed is the normal state and wears no badge; only a subset is
  flagged in the gutter.
"""

from __future__ import annotations

import bisect
import math
import os
from dataclasses import dataclass, replace
from collections.abc import Iterable
from typing import NamedTuple

from PyQt6.QtCore import QEvent, QPoint, QPointF, QRectF, QSize, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import (
    QColor,
    QFont,
    QFontMetrics,
    QKeyEvent,
    QLinearGradient,
    QMouseEvent,
    QPainter,
    QPaintEvent,
    QPen,
    QPixmap,
    QPolygonF,
    QWheelEvent,
)
from PyQt6.QtWidgets import (
    QApplication,
    QGridLayout,
    QLabel,
    QPushButton,
    QScrollBar,
    QSizePolicy,
    QSlider,
    QWidget,
)

from .probe import AudioFileInfo, MediaInfo
from .render import (
    AudioClip,
    AudioLane,
    Segment,
    TextClip,
    TimelinePlan,
    VideoClip,
    VideoLane,
    visible_spans,
)
from .theme import TIMELINE_MODES, TOKENS, variant as theme_variant

__all__ = [
    "Clip",
    "CutList",
    "Lane",
    "PreviewAudio",
    "PreviewPlan",
    "PreviewVideo",
    "TimelineWidget",
    "format_tc",
]

Range = tuple[float, float]

# --- model tuning -----------------------------------------------------------
MIN_KEEP: float = 0.02
"""Keep-segments shorter than this are degenerate and get dropped."""

EDGE_SNAP: float = 0.05
"""A cut edge this close to 0 / duration snaps flush to it."""

# --- view tuning ------------------------------------------------------------
MIN_SPAN: float = 1.0
"""Zoom-in limit: one second across the full track width."""

ZOOM_STEP: float = 1.25
PAN_FRACTION: float = 0.08
RIGHT_ROOM_FRAC: float = 0.2
RIGHT_ROOM_MIN: float = 10.0
"""Empty room the view keeps past the edit's end: a fifth of it, never under 10 s."""
REQ_DEBOUNCE_MS: int = 150

# --- painting metrics -------------------------------------------------------
GUTTER_W: int = 170
PAD: int = 6
RULER_H: int = 18
VIDEO_LANE_H: int = 70
#: Audio lanes are deliberately taller than the header strictly needs.  At 40 px
#: the waveform was a hairline you could not read at full-file zoom, which was
#: the whole point of having one lane per track.
AUDIO_LANE_H: int = 62
LANE_GAP: int = 2
#: Slim horizontal scrollbar under the lanes, spanning the track area only.
SCROLL_H: int = 11
SCROLL_GAP: int = 3
#: Scrollbar ticks per second of edited time.  Integer ticks are what makes the
#: view<->bar round trip exact (see ``_sync_scrollbar``).
SCROLL_SCALE: int = 1000
HANDLE_PX: int = 7
DRAG_THRESHOLD_PX: int = 3
#: How close, in pixels, a dragged block snaps to a neighbour's edge, 0:00 or
#: the playhead.  A pixel tolerance rather than a time one, so zoomed in a
#: frame-accurate drag is still possible -- see ``TimelineWidget._snap``.
SNAP_PX: float = 7.0
#: How close to a block's left/right edge an Alt+hover grabs the TRIM handle
#: instead of the block itself.  Deliberately smaller than ``HANDLE_PX``: the
#: middle of a block is the common gesture (slide it) and the edge is the rare
#: one, so the edge zone must not eat the middle of a short block.
EDGE_GRAB_PX: float = 6.0
#: The fade handle's corner box, in pixels: this wide from the block's edge and
#: this far down from the top of the lane.  It sits INSIDE the trim zone and
#: wins there — the top corners are the fade, the rest of the edge is the trim,
#: which is the Vegas layout.
FADE_GRAB_PX: float = 11.0
FADE_GRAB_H: float = 13.0
#: A block narrower than this is all handle and nothing else: at that width the
#: trim and fade zones would leave no middle to grab, so a block this small has
#: no handles at all and can only be moved.  Zoom in to trim it.
MIN_BLOCK_PX: float = 30.0
#: Shortest a block may be trimmed to.  Under a frame or two there is nothing
#: left to grab at any zoom, and ``render._clean_plan`` drops a clip shorter
#: than ``EPS`` from the graph entirely — silently, which is the failure mode
#: this project keeps hitting.
MIN_BLOCK_LEN: float = 0.05
#: The cursor each block handle wears under Alt.  The two fade cursors match
#: the DIAGONAL the fade is painted as — "/" at the top-left corner, "\" at the
#: top-right — rather than the corner-resize convention, because the ramp on
#: screen is the thing the pointer is about to drag.
BLOCK_CURSORS: dict[str, Qt.CursorShape] = {
    "move": Qt.CursorShape.OpenHandCursor,
    "left": Qt.CursorShape.SizeHorCursor,
    "right": Qt.CursorShape.SizeHorCursor,
    "fade-in": Qt.CursorShape.SizeBDiagCursor,
    "fade-out": Qt.CursorShape.SizeFDiagCursor,
}
MIN_THUMBS: int = 8
MAX_THUMBS: int = 200
#: Ceiling on the per-lane waveform peak normalisation, so a near-silent lane is
#: lifted into visibility without being blown up into a solid wall.
WAVE_NORM_MAX: float = 6.0
#: Vertical headroom left around a normalised waveform, in pixels.
WAVE_INSET: float = 3.0

# --- palette ----------------------------------------------------------------
# The lane colours live in theme.py (``theme.TIMELINE_MODES``) so the strip and
# the app chrome cannot drift apart.  They are looked up at PAINT time, never at
# import time: a module-level ``C_BG = QColor(TIMELINE["lane_bg"])`` freezes the
# dark set into the process, which is exactly why the strip used to stay dark in
# light mode however the theme was switched.  ``_palette()`` resolves the active
# variant on every repaint and caches one immutable set per mode, so a live
# theme switch costs nothing and lands immediately.


class _Lanes:
    """Every colour the lane strip paints with, for ONE theme variant."""

    __slots__ = (
        "mode", "bg", "gutter", "border", "ruler_bg", "tick", "tick_text",
        "accent", "sel_fill", "splice", "playhead", "hover", "chip_bg",
        "chip_text", "label", "empty_text", "muted", "wave_bg", "wave",
        "armed", "armed_tint", "hole", "video_top", "video_bottom",
        "block_sel", "block_edge", "group",
    )

    def __init__(self, mode: str) -> None:
        t = TIMELINE_MODES.get(mode) or TIMELINE_MODES["dark"]

        def c(key: str, alpha: int | None = None) -> QColor:
            colour = QColor(t[key])
            if alpha is not None:
                colour.setAlpha(alpha)
            return colour

        dark = QColor(t["lane_bg"]).lightness() < 128
        self.mode = mode
        self.bg = c("lane_bg")
        self.gutter = c("lane_bg").lighter(135) if dark else c("lane_bg").darker(105)
        self.border = c("lane_edge")
        self.ruler_bg = c("lane_bg").lighter(125) if dark else c("lane_bg").darker(103)
        self.tick = c("ruler_text").darker(135) if dark else c("ruler_text").lighter(125)
        self.tick_text = c("ruler_text")
        self.accent = c("sel")
        self.sel_fill = c("sel", 64)
        self.splice = c("splice", 230)
        self.playhead = c("playhead")
        self.hover = QColor(255, 255, 255, 55) if dark else QColor(0, 0, 0, 40)
        chip = c("lane_bg").darker(150) if dark else QColor(0x2B, 0x38, 0x45)
        chip.setAlpha(228)
        self.chip_bg = chip
        self.chip_text = QColor(0xEF, 0xF0, 0xF1)
        self.label = c("ruler_text").lighter(125) if dark else c("ruler_text").darker(125)
        self.empty_text = c("ruler_text").darker(125) if dark else c("ruler_text").lighter(115)
        self.muted = c("lane_bg", 165)
        self.wave_bg = c("wave_bg")
        self.wave = c("wave")
        self.armed = c("armed")
        self.armed_tint = c("armed", 30)
        self.hole = c("hole")
        base = QColor(t["lane_bg"])
        self.video_top = base.lighter(150) if dark else base.darker(106)
        self.video_bottom = base.lighter(118) if dark else base.darker(118)
        # A selected block is outlined twice: a bright inner line that survives
        # a dark thumbnail, and a dark outer one that survives a bright
        # waveform.  The accent on its own does neither.
        self.block_sel = c("block_sel")
        self.block_edge = QColor(0, 0, 0, 200) if dark else QColor(255, 255, 255, 225)
        self.group = c("group")


_PALETTES: dict[str, _Lanes] = {}


def _palette(mode: str | None = None) -> _Lanes:
    """Lane colours for ``mode`` (default: whatever the app palette reads as).

    Call this from ``paintEvent``/``_content``, not at import: that is the whole
    point — see the comment above.
    """
    if mode is None:
        mode = theme_variant(QApplication.instance())
    if mode not in _PALETTES:
        _PALETTES[mode] = _Lanes(mode)
    return _PALETTES[mode]


#: Handed to ``probe.waveform_png`` so the PNG has a known ink.  The lane
#: re-tints the trace to the ACTIVE theme at paint time (``_tinted``), so a
#: waveform fetched in one theme still reads correctly in the other.
WAVE_COLOR: str = TIMELINE_MODES["dark"]["wave"]

_TICK_STEPS: tuple[float, ...] = (
    0.1, 0.2, 0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600, 7200,
)

GAIN_MIN_DB: int = -30
GAIN_MAX_DB: int = 12


def format_tc(seconds: float, *, decimals: int = 2) -> str:
    """``h:mm:ss.cc`` (hours only when needed).  Negatives clamp to zero."""
    value = max(0.0, round(float(seconds), decimals))
    whole = int(value)
    frac = value - whole
    hours, rem = divmod(whole, 3600)
    minutes, secs = divmod(rem, 60)
    tail = f"{frac:.{decimals}f}"[1:] if decimals > 0 else ""
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}{tail}"
    return f"{minutes:02d}:{secs:02d}{tail}"


def _normalize(ranges: list[Range], duration: float) -> list[Range]:
    """Clamp to ``[0, duration]``, sort, and merge overlapping/touching ranges."""
    cleaned: list[Range] = []
    for start, end in ranges:
        lo, hi = (start, end) if start <= end else (end, start)
        lo = max(0.0, min(duration, lo))
        hi = max(0.0, min(duration, hi))
        if lo <= EDGE_SNAP:
            lo = 0.0
        if hi >= duration - EDGE_SNAP:
            hi = duration
        if hi - lo > 1e-6:
            cleaned.append((lo, hi))
    if not cleaned:
        return []
    cleaned.sort()
    merged: list[Range] = [cleaned[0]]
    for lo, hi in cleaned[1:]:
        plo, phi = merged[-1]
        # Merge when they overlap, or when the sliver between them would be too
        # short to survive as a keep-segment anyway.
        if lo <= phi + MIN_KEEP:
            merged[-1] = (plo, max(phi, hi))
        else:
            merged.append((lo, hi))
    return merged


def _overlaps(span: Range, ranges: Iterable[Range]) -> list[Range]:
    """The parts of ``span`` that fall inside any of ``ranges``, in order."""
    lo, hi = span
    out: list[Range] = []
    for a, b in ranges:
        x0, x1 = max(lo, a), min(hi, b)
        if x1 - x0 > 1e-9:
            out.append((x0, x1))
    return out


def _sample(spans: list[Range], count: int) -> list[float]:
    """``count`` evenly spaced points across ``spans`` treated as one run.

    Cell midpoints, like the thumbnail strip wants: the i-th of n sits at
    ``(i + 0.5)/n`` of the total width, so the outer two are half a cell in
    from the ends instead of sitting on them.
    """
    total = sum(hi - lo for lo, hi in spans)
    if total <= 1e-9 or count <= 0:
        return []
    out: list[float] = []
    for i in range(count):
        want = total * (i + 0.5) / count
        for lo, hi in spans:
            if want <= hi - lo:
                out.append(lo + want)
                break
            want -= hi - lo
        else:
            out.append(spans[-1][1])
    return out


def _subtract(span: Range, ranges: Iterable[Range]) -> list[Range]:
    """``span`` with every range in ``ranges`` cut out of it, in order."""
    pieces = [span]
    for a, b in sorted(ranges):
        nxt: list[Range] = []
        for lo, hi in pieces:
            if b <= lo or a >= hi:
                nxt.append((lo, hi))
                continue
            if lo < a:
                nxt.append((lo, a))
            if b < hi:
                nxt.append((b, hi))
        pieces = nxt
    return [(lo, hi) for lo, hi in pieces if hi - lo > 1e-9]


class _Snapshot(NamedTuple):
    """One history entry: the whole edit plus the UI state it belonged to."""

    cuts: list[Range]
    holes: dict[int, list[Range]]
    ui: dict | None


class CutList:
    """Deleted source ranges, per-lane holes, the edited mapping, and undo/redo.

    Pure model: no Qt import, no widget knowledge.  ``ui`` arguments are opaque
    dicts the widget uses to park its view/playhead alongside each history
    entry; the model only stores and hands them back.

    Two edit kinds, one history:

    * **cuts** are global and ripple — they shorten the edit and define
      ``source_to_edited``.  This is the only owner of that mapping; giving each
      lane its own cut list would desync the lanes, which is exactly what the
      lift is there to avoid.
    * **holes** belong to one lane each and do not.  They are kept in SOURCE
      time so a later ripple cut re-maps them for free.
    """

    MAX_UNDO = 256

    def __init__(self, duration: float = 0.0) -> None:
        self._duration: float = max(0.0, float(duration))
        self._cuts: list[Range] = []
        self._holes: dict[int, list[Range]] = {}
        self._undo: list[_Snapshot] = []
        self._redo: list[_Snapshot] = []
        self.restored_ui: dict | None = None
        """UI snapshot handed back by the most recent ``undo``/``redo``."""
        self._keeps: list[Range] | None = None
        self._starts: list[float] = []
        self._cum: list[float] = []
        self._edited: float = 0.0

    # -- state ---------------------------------------------------------------
    @property
    def duration(self) -> float:
        """Source duration in seconds."""
        return self._duration

    @property
    def cuts(self) -> list[Range]:
        return list(self._cuts)

    @property
    def can_undo(self) -> bool:
        return bool(self._undo)

    @property
    def can_redo(self) -> bool:
        return bool(self._redo)

    def reset(self, duration: float) -> None:
        """New media loaded: fresh duration, no cuts or holes, empty history."""
        self._duration = max(0.0, float(duration))
        self._cuts.clear()
        self._holes.clear()
        self._undo.clear()
        self._redo.clear()
        self.restored_ui = None
        self._invalidate()

    def _invalidate(self) -> None:
        self._keeps = None

    def _copy_holes(self) -> dict[int, list[Range]]:
        return {lane: list(ranges) for lane, ranges in self._holes.items() if ranges}

    def _push(self, ui: dict | None) -> None:
        self._undo.append(_Snapshot(list(self._cuts), self._copy_holes(), ui))
        if len(self._undo) > self.MAX_UNDO:
            del self._undo[0]
        self._redo.clear()

    def push_state(self, ui: dict | None) -> None:
        """Record an undo step for an edit this model does not itself own.

        Adding or removing a CLIP changes neither the cut list nor the holes,
        but it must still be undoable — and it must share one history with the
        cuts, or Ctrl+Z would walk two stacks in an order nobody can predict.
        The caller parks the pre-edit lane layout in ``ui``; ``undo`` hands it
        straight back, exactly as it does for the playhead and the view.
        """
        self._push(ui)

    def set_duration(self, duration: float) -> bool:
        """Re-length the timeline (a clip was added, moved or removed).

        NOT an undo step and NOT a reset: cuts and holes are kept, only clamped
        into the new length.  The undo step belongs to whatever moved the clip,
        which pushes it with :meth:`push_state` before calling this.
        """
        value = max(0.0, float(duration))
        if abs(value - self._duration) < 1e-9:
            return False
        self._duration = value
        # Clamp rather than re-normalise: re-running _normalize would snap a cut
        # that ended flush with the OLD duration out to the new one, silently
        # eating the material a newly appended clip just added.
        self._cuts = [(lo, min(hi, value)) for lo, hi in self._cuts if lo < value - 1e-9]
        self._cuts = [(lo, hi) for lo, hi in self._cuts if hi - lo > 1e-6]
        self._holes = {
            lane: [(lo, min(hi, value)) for lo, hi in ranges if lo < value - 1e-9]
            for lane, ranges in self._holes.items()
        }
        self._holes = {k: v for k, v in self._holes.items() if v}
        self._invalidate()
        return True

    # -- mutation ------------------------------------------------------------
    def add_cuts(self, ranges: list[Range], ui: dict | None = None) -> bool:
        """Delete every range in ``ranges`` as ONE undoable step.

        ``ui`` is the pre-edit UI snapshot; ``undo`` hands it back.  Holes are
        left exactly as they are: they are source-time, so the ripple re-maps
        them without touching the stored ranges.
        """
        if self._duration <= 0.0:
            return False
        candidate = _normalize(self._cuts + list(ranges), self._duration)
        if candidate == self._cuts:
            return False
        self._push(ui)
        self._cuts = candidate
        self._invalidate()
        return True

    def add_cut(self, start: float, end: float, ui: dict | None = None) -> bool:
        """Delete ``[start, end]`` (source time).  True when the model changed."""
        return self.add_cuts([(start, end)], ui)

    def add_holes(
        self, per_lane: dict[int, list[Range]], ui: dict | None = None
    ) -> bool:
        """LIFT ``per_lane`` source ranges as ONE undoable step.

        Nothing is removed from the timeline: the edit keeps its length and the
        lanes stay in sync, which is the whole difference from a ripple cut.
        """
        if self._duration <= 0.0:
            return False
        candidate = self._copy_holes()
        for lane, ranges in per_lane.items():
            merged = _normalize(candidate.get(lane, []) + list(ranges), self._duration)
            if merged:
                candidate[int(lane)] = merged
            else:
                candidate.pop(int(lane), None)
        if candidate == self._copy_holes():
            return False
        self._push(ui)
        self._holes = candidate
        return True

    def holes(self, lane: int) -> list[Range]:
        """SOURCE ranges lifted out of ``lane``, in time order."""
        return list(self._holes.get(int(lane), ()))

    def keep_holes(self, lanes: Iterable[int]) -> None:
        """Forget holes for lanes that no longer exist (a track-list rebuild).

        Lanes are addressed by their STABLE key, not their position, so
        inserting "Video track 2" between the video and the audio cannot slide
        a hole onto a lane it was never cut from.  Not undoable: the lanes
        those holes described are gone.
        """
        keep = {int(lane) for lane in lanes}
        self._holes = {
            lane: ranges for lane, ranges in self._holes.items() if lane in keep
        }

    def all_holes(self) -> dict[int, list[Range]]:
        """Every lane's holes, ``{lane: [(start, end), ...]}`` in SOURCE time."""
        return self._copy_holes()

    def hole_spans_edited(self, lane: int) -> list[Range]:
        """This lane's holes as EDITED spans, split by any cut inside them."""
        spans: list[Range] = []
        for start, end in self._holes.get(int(lane), ()):
            spans.extend(self.edited_spans_for_source(start, end))
        return spans

    def undo(self, ui: dict | None = None) -> bool:
        """Step back one edit.  ``ui`` is parked for the matching ``redo``."""
        if not self._undo:
            return False
        self._redo.append(_Snapshot(list(self._cuts), self._copy_holes(), ui))
        self._cuts, self._holes, self.restored_ui = self._undo.pop()
        self._invalidate()
        return True

    def redo(self, ui: dict | None = None) -> bool:
        if not self._redo:
            return False
        self._undo.append(_Snapshot(list(self._cuts), self._copy_holes(), ui))
        self._cuts, self._holes, self.restored_ui = self._redo.pop()
        self._invalidate()
        return True

    # -- keeps ---------------------------------------------------------------
    def _build(self) -> None:
        """Recompute keeps + edited-time offsets (cached until a mutation)."""
        if self._keeps is not None:
            return
        keeps: list[Range] = []
        if self._duration > 0.0:
            cursor = 0.0
            for lo, hi in self._cuts:
                if lo - cursor >= MIN_KEEP:
                    keeps.append((cursor, lo))
                cursor = max(cursor, hi)
            if self._duration - cursor >= MIN_KEEP:
                keeps.append((cursor, self._duration))
        cum: list[float] = []
        total = 0.0
        for lo, hi in keeps:
            cum.append(total)
            total += hi - lo
        self._keeps = keeps
        self._starts = [lo for lo, _ in keeps]
        self._cum = cum
        self._edited = total

    def keep_segments(self) -> list[Range]:
        """Surviving SOURCE ranges, in time order.  Empty means nothing left."""
        self._build()
        return list(self._keeps or ())

    # -- edited-time mapping -------------------------------------------------
    def edited_duration(self) -> float:
        """Length of the edit: the sum of the keep-segment durations."""
        self._build()
        return self._edited

    def source_to_edited(self, t_source: float) -> float | None:
        """Edited position of a source timestamp, or None if it was cut away.

        Cut boundaries are shared: the end of one keep and the start of the next
        both map to the same edited instant (the splice point).
        """
        self._build()
        if not self._keeps:
            return None
        t = max(0.0, min(self._duration, float(t_source)))
        i = bisect.bisect_right(self._starts, t) - 1
        if i < 0:
            return None  # inside a cut that starts at 0
        lo, hi = self._keeps[i]
        if t > hi:
            return None  # inside the cut that follows keep i
        return self._cum[i] + (t - lo)

    def edited_to_source(self, t_edited: float) -> float:
        """Source timestamp of an edited position.  Always defined.

        At a splice point this returns the start of the FOLLOWING keep, which is
        what a seek wants: land after the removed material, never inside it.
        """
        self._build()
        if not self._keeps:
            return 0.0
        t = max(0.0, min(self._edited, float(t_edited)))
        i = max(0, bisect.bisect_right(self._cum, t) - 1)
        lo, hi = self._keeps[i]
        return min(hi, lo + (t - self._cum[i]))

    def keep_index_at_edited(self, t_edited: float) -> int:
        """Index of the keep-segment an edited position lands in (-1 if none).

        A splice point belongs to the segment that STARTS there.
        """
        self._build()
        if not self._keeps:
            return -1
        t = max(0.0, min(self._edited, float(t_edited)))
        return max(0, bisect.bisect_right(self._cum, t) - 1)

    def keep_bounds_at_edited(self, t_edited: float) -> Range | None:
        """Edited ``(start, end)`` of the keep-segment under ``t_edited``."""
        i = self.keep_index_at_edited(t_edited)
        if i < 0:
            return None
        lo, hi = self._keeps[i]  # type: ignore[index]
        start = self._cum[i]
        return (start, start + (hi - lo))

    def splice_points_edited(self) -> list[float]:
        """Edited positions where material was removed, in order.

        Only interior joins: a cut at the very start or the very end shortens
        the edit rather than splicing two survivors together, so it has no
        marker.
        """
        self._build()
        return list(self._cum[1:])

    def source_ranges_for_edited(self, start: float, end: float) -> list[Range]:
        """SOURCE ranges covered by an edited span — cuts inside it honoured."""
        self._build()
        if not self._keeps:
            return []
        lo, hi = (start, end) if start <= end else (end, start)
        lo = max(0.0, min(self._edited, float(lo)))
        hi = max(0.0, min(self._edited, float(hi)))
        out: list[Range] = []
        for i, (s, e) in enumerate(self._keeps):
            c0 = self._cum[i]
            c1 = c0 + (e - s)
            x0 = max(lo, c0)
            x1 = min(hi, c1)
            if x1 - x0 > 1e-9:
                out.append((s + (x0 - c0), s + (x1 - c0)))
        return out

    def spans_for_source(
        self, start: float, end: float
    ) -> list[tuple[float, float, float, float]]:
        """A SOURCE range as ``(edited_start, edited_end, src_start, src_end)``
        pieces, one per keep-segment it crosses.

        Both halves together, because the two are NOT recoverable from each
        other at a splice: ``edited_to_source`` deliberately answers a splice
        instant with the start of the FOLLOWING keep (that is what a seek
        wants), so asking it for the END of a piece lands past the cut.  Anyone
        who needs the pair — the lane painter mapping pixels back into a file —
        has to get it from the one walk that knows both.
        """
        self._build()
        lo, hi = (start, end) if start <= end else (end, start)
        out: list[tuple[float, float, float, float]] = []
        for i, (s, e) in enumerate(self._keeps or ()):
            x0 = max(lo, s)
            x1 = min(hi, e)
            if x1 - x0 > 1e-9:
                c0 = self._cum[i]
                out.append((c0 + (x0 - s), c0 + (x1 - s), x0, x1))
        return out

    def edited_spans_for_source(self, start: float, end: float) -> list[Range]:
        """EDITED spans covered by a SOURCE range — the inverse of
        :meth:`source_ranges_for_edited`.  A cut inside the range splits it.
        """
        return [(a, b) for a, b, _s0, _s1 in self.spans_for_source(start, end)]

    def next_keep_time(self, t_source: float) -> float | None:
        """First kept source timestamp at or after ``t``; None if nothing is.

        The preview plays the untouched source file, so playback uses this to
        jump the material the edit removed.
        """
        self._build()
        for start, end in self._keeps or ():
            if t_source < start:
                return start
            if start <= t_source < end:
                return t_source
        return None


@dataclass(frozen=True)
class Title:
    """What a text block draws: centre ``x``/``y`` and font height ``size`` as
    fractions of the frame (see :class:`render.TextClip`)."""

    text: str
    x: float = 0.5
    y: float = 0.85
    size: float = 0.08
    color: str = "#ffffff"


#: A text block has no file, so its "file" is a long virtual one with the block
#: in the middle: both edges can then be dragged out as far as anyone wants.
TEXT_SOURCE: float = 2_000_000.0
TEXT_DEFAULT_LEN: float = 5.0


class Clip:
    """One piece of media placed on a lane, in TIMELINE seconds — a BLOCK.

    ``start`` is where it begins on the uncut timeline, ``src_in`` where it
    begins inside its own file, ``length`` how much of it plays and ``source``
    the full duration of the file.  A lane holds an ordered, non-overlapping
    list of these — dropping a file on an empty lane places one at 0:00,
    dropping onto a lane that already has content appends one after the last —
    and Alt+drag slides one along its lane, splitting, deleting and grouping
    like an event in an NLE.

    ``cid`` is a stable per-widget id: the selection, the clipboard and the
    undo snapshots all address blocks by it, because a block's INDEX on its
    lane moves whenever a neighbour is dragged past it.  ``group`` is 0 for a
    loose block, or an id shared by blocks that select, move and delete as one
    — which is how a dropped video's picture stays welded to its sound.

    ``source`` is what makes edge-trimming possible at all: it is the file's
    WHOLE duration, so ``src_in`` and ``src_in + length`` can be walked back
    out towards 0 and ``source`` to restore material a split or a trim cut off.
    A clip whose file duration was never probed carries ``source == length``
    and so has nothing to give back — honest, rather than pretending to
    material that may not be there.

    ``fade_in``/``fade_out`` are seconds of ramp at the block's own two ends,
    0.0 for none.  They are a property of the BLOCK, not of the lane or the
    timeline: trimming an edge keeps them (clamped to what still fits) and a
    split gives each half the fade at its OUTER end, because the inner ends are
    a join that was never there before.  The render applies them in the clip's
    own SOURCE time (``render._fade_filters``/``_audio_heads``), so a cut
    inside a fade keeps the brightness the surviving frames had.
    """

    __slots__ = (
        "path", "kind", "start", "src_in", "length", "stream", "label", "source",
        "cid", "group", "fade_in", "fade_out", "title",
    )

    def __init__(
        self,
        path: str,
        kind: str,
        start: float = 0.0,
        src_in: float = 0.0,
        length: float = 0.0,
        stream: int = 0,
        label: str = "",
        source: float = 0.0,
        cid: int = 0,
        group: int = 0,
        fade_in: float = 0.0,
        fade_out: float = 0.0,
    ) -> None:
        self.path = path
        self.kind = kind
        self.start = float(start)
        self.src_in = float(src_in)
        self.length = float(length)
        self.stream = int(stream)
        self.label = label or os.path.basename(path)
        self.source = float(source or length)
        self.cid = int(cid)
        self.group = int(group)
        self.fade_in = max(0.0, float(fade_in))
        self.fade_out = max(0.0, float(fade_out))
        self.title: Title | None = None  # set = a text block (no file behind it)

    @property
    def src_out(self) -> float:
        return self.src_in + self.length

    @property
    def end(self) -> float:
        return self.start + self.length

    def source_at(self, t: float) -> float:
        return self.src_in + (float(t) - self.start)

    def copy(self) -> Clip:
        """An exact duplicate, ``cid``, group and fades included.

        Used for the undo snapshots, so the id has to come along: restoring an
        edit that renamed every block would drop the selection on the floor.
        The fades ride along for the same reason — an undo that restored a
        block without its fade would silently change the render.
        """
        clip = Clip(
            self.path, self.kind, self.start, self.src_in, self.length,
            self.stream, self.label, self.source, self.cid, self.group,
            self.fade_in, self.fade_out,
        )
        clip.title = self.title
        return clip

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"Clip({self.label!r}, {self.kind}, start={self.start:g}, "
            f"in={self.src_in:g}, len={self.length:g})"
        )


class Lane:
    """One timeline lane: a kind, a stable key and an ordered list of clips.

    ``key`` never changes and never gets reused, so holes (which are stored
    against it) survive lanes being inserted above them.  The lane's POSITION
    in :attr:`TimelineWidget._lanes` is its priority and its id in the public
    API; the key is internal bookkeeping.

    Video lanes are a priority stack — lane 0 paints over lane 1 — which is why
    "Add video track" puts the new lane *underneath* the main one: it is the
    footage that shows through wherever the main lane has a hole or has run
    out.  Audio lanes have no priority; they all mix.
    """

    __slots__ = ("key", "kind", "clips", "muted", "gain_db", "header")

    def __init__(self, key: int, kind: str) -> None:
        self.key = int(key)
        self.kind = kind
        self.clips: list[Clip] = []
        self.muted = False
        self.gain_db = 0.0
        self.header: _AudioLaneHeader | None = None

    def end(self) -> float:
        """Timeline time the lane's last clip stops at (0.0 when empty)."""
        return max((c.end for c in self.clips), default=0.0)

    def snapshot(self) -> dict:
        return {
            "key": self.key,
            "kind": self.kind,
            "muted": self.muted,
            "gain_db": self.gain_db,
            "clips": [c.copy() for c in self.clips],
        }


class PreviewVideo(NamedTuple):
    """One contiguous stretch of footage the preview must show."""

    path: str
    src_in: float
    src_out: float
    edited_start: float


class PreviewAudio(NamedTuple):
    """One contiguous stretch of audio the preview must play.

    ``lane`` is the lane's stable key: two lanes holding the same file and
    stream are two things to play, each at its own gain and mute.
    """

    path: str
    stream: int
    gain_db: float
    muted: bool
    edited_start: float
    src_in: float
    src_out: float
    lane: int


class PreviewPlan(NamedTuple):
    """What the preview has to play, in EDITED time — cuts already collapsed.

    Every entry is one contiguous run: a clip that a cut (or a lift, or a lane
    above it) interrupts arrives as several entries, because that is the only
    shape a player can actually be driven from.  ``duration`` is the length of
    the edit, so a progress bar needs nothing else.
    """

    duration: float
    video: list[PreviewVideo]
    audio: list[PreviewAudio]


def _scrollbar_qss(mode: str) -> str:
    """QSS for the slim timeline scrollbar: a silver-blue glass pill on a
    recessed groove, no arrow buttons.  The pill wears the play dome's own
    gradient (:data:`theme.TOKENS`), so it belongs to the chrome; the groove is
    taken from the ACTIVE lane palette so it belongs to the strip."""
    lanes = _palette(mode)
    t = TOKENS["light"]
    face = QColor(t["play_face2"])
    groove = lanes.bg.darker(115) if mode == "dark" else lanes.bg.darker(108)
    return f"""
QScrollBar#timelineScroll:horizontal {{
    background: {groove.name()};
    border: 1px solid {lanes.border.name()};
    border-radius: {SCROLL_H // 2}px;
    height: {SCROLL_H}px;
    margin: 0;
}}
QScrollBar#timelineScroll::handle:horizontal {{
    min-width: 26px;
    border: 1px solid {t['play_rim']};
    border-top-color: {t['gloss_top']};
    border-radius: {SCROLL_H // 2 - 1}px;
    margin: 1px;
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 {t['play_face1']}, stop:0.48 {face.lighter(112).name()},
        stop:0.52 {face.name()}, stop:1 {face.darker(112).name()});
}}
QScrollBar#timelineScroll::handle:horizontal:hover {{
    border-color: {t['accent']};
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 {t['play_face1']}, stop:0.48 {face.lighter(125).name()},
        stop:0.52 {face.lighter(108).name()}, stop:1 {face.name()});
}}
/* Zoomed all the way out there is nothing to scroll.  Qt still draws a handle
   filling the whole groove; sink it into the groove instead. */
QScrollBar#timelineScroll::handle:horizontal:disabled {{
    background: {lanes.bg.name()};
    border-color: {lanes.border.darker(115).name()};
}}
QScrollBar#timelineScroll::add-line:horizontal,
QScrollBar#timelineScroll::sub-line:horizontal {{ width: 0; height: 0; }}
QScrollBar#timelineScroll::add-page:horizontal,
QScrollBar#timelineScroll::sub-page:horizontal {{ background: transparent; }}
"""


class _NoWheelSlider(QSlider):
    """Slider that lets the wheel through, so scrolling never nudges a gain."""

    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802
        event.ignore()


class _ElidedLabel(QLabel):
    """A label that elides to whatever width the layout gives it.

    The full text goes in the tooltip.  ``Ignored`` horizontally, so a long
    track title can never push its header wider than the gutter — which is
    what used to clip labels mid-word.
    """

    def __init__(self, text: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._full = text
        self.setToolTip(text)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self._elide()

    def set_full(self, text: str, tooltip: str | None = None) -> None:
        self._full = text
        self.setToolTip(tooltip or text)
        self._elide()

    def _elide(self) -> None:
        self.setText(
            self.fontMetrics().elidedText(
                self._full, Qt.TextElideMode.ElideRight, max(1, self.width())
            )
        )

    def resizeEvent(self, event: object) -> None:  # noqa: N802
        super().resizeEvent(event)  # type: ignore[arg-type]
        self._elide()


class _AudioLaneHeader(QWidget):
    """Gutter header for one audio lane: label, mute toggle, gain slider.

    Serves an embedded track, a dropped audio file and a lane holding several
    of either — mute and gain are lane-wide, so they behave identically for
    all of them.

    ``index`` is the LANE index, the same id ``lane_kind``/``add_clip``/
    ``remove_lane`` and the armed set use.  Colours come from the active lane
    palette via :meth:`apply_palette` rather than the inherited widget palette,
    because the gutter is painted by the timeline, not by the app stylesheet —
    a press anywhere that is not the M button or the slider arms the lane.
    """

    muteChanged = pyqtSignal(int, bool)
    gainChanged = pyqtSignal(int, float)
    headerClicked = pyqtSignal(int, bool)  # lane index, ctrl held

    def __init__(self, index: int, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.index = int(index)
        font = QFont(self.font())
        font.setPointSizeF(max(7.0, font.pointSizeF() - 1.5))
        self.setFont(font)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip(
            "Click to edit this lane only; Ctrl+click to add or remove it. "
            "Every lane armed ripples the whole timeline; a subset lifts, "
            "leaving the rest in sync."
        )

        self.label = _ElidedLabel("", self)  # text arrives via _sync_headers
        self.mute = QPushButton("M")
        self.mute.setCheckable(True)
        self.mute.setFixedSize(19, 16)
        self.mute.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.mute.setToolTip("Mute: leave this track out of the render mix")
        self.slider = _NoWheelSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(GAIN_MIN_DB, GAIN_MAX_DB)
        self.slider.setValue(0)
        self.slider.setPageStep(3)
        self.slider.setFixedHeight(14)
        self.slider.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.slider.setToolTip(
            "Gain applied to this track in the render and the preview.\n"
            "Above 0 dB the preview keeps the balance by turning the other "
            "lanes down — it cannot play louder than unity."
        )
        self.gain_label = QLabel("0 dB")
        self.gain_label.setFixedWidth(40)
        self.gain_label.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )

        grid = QGridLayout(self)
        grid.setContentsMargins(6, 3, 6, 3)
        grid.setHorizontalSpacing(4)
        grid.setVerticalSpacing(1)
        grid.setAlignment(Qt.AlignmentFlag.AlignTop)  # a tall lane keeps its controls up top
        grid.addWidget(self.label, 0, 0)
        grid.addWidget(self.mute, 0, 1)
        grid.addWidget(self.slider, 1, 0)
        grid.addWidget(self.gain_label, 1, 1)
        grid.setColumnStretch(0, 1)

        self.mute.toggled.connect(self._on_mute)
        self.slider.valueChanged.connect(self._on_gain)
        self.apply_palette(_palette())

    # -- look ----------------------------------------------------------------
    def apply_palette(self, pal: _Lanes) -> None:
        """Re-ink for the active theme.  Called on every live theme switch."""
        face = pal.bg.lighter(160) if pal.mode == "dark" else pal.bg.darker(108)
        edge = pal.border
        self.setStyleSheet(
            f"QLabel{{color:{pal.label.name()};background:transparent;}}"
            f"QPushButton{{color:{pal.tick_text.name()};background:{face.name()};"
            f"border:1px solid {edge.name()};border-radius:3px;padding:0;"
            "font-weight:bold;}"
            "QPushButton:checked{background:#c0453f;border-color:#e06a63;color:#fff;}"
        )

    # -- state ---------------------------------------------------------------
    def _on_mute(self, muted: bool) -> None:
        self.muteChanged.emit(self.index, muted)

    def _on_gain(self, value: int) -> None:
        self.gain_label.setText(f"{value:+d} dB" if value else "0 dB")
        self.gainChanged.emit(self.index, float(value))

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self.headerClicked.emit(
                self.index,
                bool(event.modifiers() & Qt.KeyboardModifier.ControlModifier),
            )
            event.accept()
            return
        super().mousePressEvent(event)


class _Wave(NamedTuple):
    """A waveform PNG plus the SOURCE span it covers and its normalised crop."""

    pix: QPixmap
    src0: float
    src1: float
    top: float  # source-rect top inside the PNG, in pixels
    height: float  # source-rect height, in pixels
    #: ``wave_source`` of the lane when the PNG arrived.  The cache is keyed by
    #: audio-lane ORDINAL, and removing a lane hands that ordinal to the next
    #: one — which must not wear the removed lane's trace in the meantime.
    source: tuple[str, int, float] | None = None


def _wave_crop(pix: QPixmap) -> tuple[float, float]:
    """Source-rect ``(top, height)`` that peak-normalises a waveform PNG.

    ``showwavespic``'s own density scaling squashes a long file hard: measured
    on ffmpeg 9.0.1, a FULL-SCALE sine spread over a strip lights only ~8% of
    its height, and a -20 dBFS track drawn linearly lights nothing at all.  The
    lane therefore crops to the trace's own bounding box and stretches that to
    the lane height — per-lane peak normalisation, so what you compare between
    lanes is SHAPE, which is what tells you where the speech is.

    The bounds are taken from a single smooth-scaled column: Qt area-averages
    the rows in C++ (far cheaper than walking pixels from Python) and averaging
    also stops one isolated click from setting the peak.  Capped at
    :data:`WAVE_NORM_MAX`, so a near-silent lane is not blown up into a wall.
    """
    height = pix.height()
    if height <= 1:
        return (0.0, float(max(1, height)))
    column = pix.toImage().scaled(
        1,
        height,
        Qt.AspectRatioMode.IgnoreAspectRatio,
        Qt.TransformationMode.SmoothTransformation,
    )
    alphas = [column.pixelColor(0, y).alpha() for y in range(height)]
    peak = max(alphas, default=0)
    if peak <= 0:
        return (0.0, float(height))
    floor = max(1, peak // 12)
    lit = [y for y, alpha in enumerate(alphas) if alpha >= floor]
    if not lit:
        return (0.0, float(height))
    centre = height / 2.0
    # showwavespic draws symmetrically about the centre, so keep the crop
    # symmetric too: an asymmetric one would slide the trace off-centre.
    half = max(centre - lit[0], lit[-1] + 1 - centre, centre / WAVE_NORM_MAX)
    half = min(half, centre)
    return (centre - half, 2.0 * half)


class TimelineWidget(QWidget):
    """Multi-lane, edited-time timeline.  See module docstring for the rules."""

    playheadMoved = pyqtSignal(float)  # the USER moved it; the app should seek
    selectionChanged = pyqtSignal(object)  # (start, end) edited, or None
    cutsChanged = pyqtSignal()  # keep-list changed (cut, lift, undo, redo, media)
    playPauseRequested = pyqtSignal()  # Space, while the timeline had focus
    trackMuteChanged = pyqtSignal(int, bool)  # track index, muted
    trackGainChanged = pyqtSignal(int, float)  # track index, gain dB
    thumbsNeeded = pyqtSignal(str, object)  # file, SOURCE timestamps inside it
    waveformNeeded = pyqtSignal(int, float, float)  # track, start, end (edited)
    armedChanged = pyqtSignal(object)  # frozenset of lane ids (0 = video)
    blockSelectionChanged = pyqtSignal(object)  # frozenset of selected clip ids
    #: A block gesture could not do what was asked.  The widget has no status
    #: bar of its own and a silently refused drag is the worst of both worlds,
    #: so the window shows the reason.
    blockHint = pyqtSignal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.model = CutList()
        self._info: MediaInfo | None = None

        self._playhead: float = 0.0
        self._mark_in: float | None = None
        self._mark_out: float | None = None
        self._emitted_sel: Range | None = None

        self._view: Range = (0.0, 1.0)
        #: Lanes in PRIORITY/display order: video lanes first (0 = the main
        #: one, painting over the rest), then audio lanes.  The index in this
        #: list is the lane id the whole public API speaks in.
        self._lanes: list[Lane] = []
        self._lane_seq: int = 0
        self._mode: str = _palette().mode
        self._pal: _Lanes = _palette(self._mode)
        #: Lane ids an edit applies to.  Every lane armed => ripple; a subset
        #: => lift.  Loads arm everything, so nothing changes out of the box.
        self._armed: set[int] = {0}
        #: The same set by lane KEY, which is what survives a lane being
        #: inserted or removed above an armed one (see ``_lanes_changed``).
        self._armed_keys: set[int] = set()

        # Assets are cached in SOURCE coordinates so new cuts re-collapse them.
        self._thumbs: dict[str, list[tuple[float, QPixmap]]] = {}
        self._cells: dict[tuple[str, float], list[tuple[float, float, QPixmap]]] | None = None
        self._waves: dict[int, _Wave] = {}
        self._asked: tuple[float, float, int] | None = None
        # Composed lane content, rebuilt only when its inputs change.
        self._content_pm: QPixmap | None = None
        self._content_key: tuple | None = None
        #: Waveform PNGs re-inked for the active theme: track -> (ink, the
        #: source PNG's cacheKey, the tinted copy).  The PNG's identity is part
        #: of the key because a lane is handed a NEW one on every cut and zoom.
        self._tinted: dict[int, tuple[str, int, QPixmap]] = {}
        self._gen_cuts: int = 0
        self._gen_assets: int = 0

        # -- blocks ----------------------------------------------------------
        #: Hands out ``Clip.cid``.  Never reused, so a stale selection or
        #: clipboard entry can only fail to match, never match the wrong block.
        self._clip_seq: int = 0
        self._group_seq: int = 0
        #: Selected block ids.  Blocks, not lanes: the armed set is a different
        #: thing entirely (which lanes an edit applies to).
        self._selected: set[int] = set()
        #: ``(lane KEY, clip)`` pairs from the last Copy, starts relative to the
        #: earliest of them.  The KEY, not the index: "Add video track" inserts
        #: above the audio lanes and renumbers every one of them, so an index
        #: copied before that lands on somebody else's lane afterwards.
        self._clipboard: list[tuple[int, Clip]] = []

        self._drag: str | None = None
        self._press_x: float = 0.0
        self._press_view: Range = (0.0, 1.0)
        self._press_lane: int | None = None
        self._moved: bool = False
        self._anchor: float = 0.0
        self._hover_x: float | None = None
        #: Alt+drag state: where the grab started in TIMELINE time, what each
        #: dragged block started at, how far the lane lets it go, and whether
        #: the undo step has been pushed yet (a click must not push one).
        self._block_anchor: float = 0.0
        self._block_from: dict[int, float] = {}
        self._block_limits: Range = (0.0, 0.0)
        #: The pointer passed the drag threshold.  NOT the same as
        #: ``_block_pushed``: a block wedged between two neighbours cannot move
        #: along its lane and so pushes no undo step, but it can still be
        #: dragged onto another one.
        self._block_moved: bool = False
        self._block_pushed: bool = False
        #: Alt+drag on an EDGE (``"left"``/``"right"``) or on a fade corner
        #: (``"in"``/``"out"``).  ``_edge_from``/``_fade_from`` hold what every
        #: affected block looked like when the drag STARTED, keyed by cid: a
        #: trim recomputes from those origins on every mouse move rather than
        #: accumulating, so dragging out and back lands exactly where it began
        #: and a fade clamped away by a shrink comes back when it grows again.
        self._edge_kind: str = ""
        self._edge_cid: int = 0
        self._edge_anchor: float = 0.0
        self._edge_from: dict[int, tuple[float, float, float, float, float]] = {}
        self._edge_limits: Range = (0.0, 0.0)
        self._edge_refit: bool = False
        self._fade_kind: str = ""
        self._fade_anchor: float = 0.0
        self._fade_from: dict[int, float] = {}

        self._req = QTimer(self)
        self._req.setSingleShot(True)
        self._req.setInterval(REQ_DEBOUNCE_MS)
        self._req.timeout.connect(self._emit_asset_requests)

        self._hbar = QScrollBar(Qt.Orientation.Horizontal, self)
        self._hbar.setObjectName("timelineScroll")
        self._hbar.setFixedHeight(SCROLL_H)
        self._hbar.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._hbar.setToolTip("Scroll the timeline; Ctrl+wheel over a lane zooms")
        self._hbar.setStyleSheet(_scrollbar_qss(self._mode))
        self._hbar.valueChanged.connect(self._on_scroll)

        self.setToolTip(
            "Drag to mark a range, click to seek.\n"
            "Double-click starts over: it selects just the block under the "
            "pointer (or the section, over empty lane).\n"
            "Alt+click a block to select it, Alt+drag to move it.  Ctrl+click "
            "adds another block to the selection.\n"
            "Alt+drag a block's LEFT or RIGHT edge to trim it — out restores "
            "material a cut took off, in gives it back.\n"
            "Alt+drag a block's TOP-LEFT or TOP-RIGHT corner for a fade in / "
            "fade out.\n"
            "S splits at the playhead; right-click a block to group, copy or delete."
        )
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)
        self.setAutoFillBackground(False)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._relayout_lanes()

    # ================================================================ media
    def set_media(self, info: MediaInfo | None) -> None:
        """Load (or clear, with ``None``) the media the timeline represents.

        Builds the default lane stack: one video lane holding the whole file,
        then one audio lane per embedded track.  Every previously added lane
        and clip is discarded — this is "open a different capture", not a drop.
        """
        self._info = info
        self._playhead = 0.0
        self._mark_in = self._mark_out = None
        self._emitted_sel = None
        self._thumbs = {}
        self._waves.clear()
        self._clear_lanes()
        if info is not None:
            video = self._new_lane("video")
            picture = self._new_clip(
                info.path, "video", 0.0, 0.0, info.duration, source=info.duration
            )
            video.clips.append(picture)
            self._lanes.append(video)
            # The capture's picture and its sound arrive welded together, so
            # Alt+dragging one drags the other and they cannot drift apart.
            # Right-click ▸ Ungroup when you want to move one on its own.
            self.weld_group([picture, *self._embedded_audio(info, info.audio, 0.0)])
        self.model.reset(info.duration if info else 0.0)
        self._lanes_changed(keep_all=True)
        self._set_view(0.0, self._view_total())
        self._asked = None
        self._req.start()  # always ask, even if the view happened not to change
        self.selectionChanged.emit(None)
        self.cutsChanged.emit()
        self.update()

    def _clear_lanes(self) -> None:
        for lane in self._lanes:
            self._drop_header(lane)
        self._lanes = []

    @staticmethod
    def _drop_header(lane: Lane) -> None:
        if lane.header is not None:
            lane.header.setParent(None)  # gone from the gutter now, not next tick
            lane.header.deleteLater()
            lane.header = None

    def _new_lane(self, kind: str) -> Lane:
        self._lane_seq += 1
        return Lane(self._lane_seq, kind)

    def _total_duration(self) -> float:
        """Length of the longest lane — the timeline runs to the last frame.

        Deliberately NOT the video's duration.  A music bed longer than the
        footage extends the edit and the render pads the video with black to
        match (``tpad``); trimming the audio to the video, which is what v3
        did, threw away material the user had just dropped in.
        """
        return max((lane.end() for lane in self._lanes), default=0.0)

    def has_media(self) -> bool:
        return self.model.duration > 0.0

    def set_thumbnails(
        self, paths: list[str], times_source: list[float], path: str | None = None
    ) -> None:
        """Accept thumbnails, one per SOURCE timestamp in ``times_source``.

        Delivered in source time on purpose: the app converted the request at
        dispatch, and a cut made while ffmpeg was running would make converting
        again at delivery land every thumb on the wrong frame.  Stored by
        source timestamp, a later cut just drops the thumbs that no longer
        exist; a pixmap that fails to load is skipped WITH its timestamp, so it
        never shifts the rest of the strip.

        ``path`` says which FILE they came from and defaults to the loaded
        media, which is what the current app delivers.  Clips from another file
        draw a labelled placeholder until someone delivers thumbs for it.
        """
        key = path or (self._info.path if self._info else "")
        thumbs = [
            (float(t), pix)
            for t, pix in zip(times_source, (QPixmap(p) for p in paths))
            if not pix.isNull()
        ]
        if not thumbs:
            return
        self._thumbs[key] = thumbs
        self._cells = None
        self._gen_assets += 1
        self.update()

    def set_waveform(self, track: int, png_path: str, src_start: float, src_end: float) -> None:
        """Accept a waveform image for audio track ``track`` over a SOURCE span.

        ``track`` is the AUDIO-LANE ordinal (0-based over audio lanes only),
        the same number ``waveformNeeded`` asks with — video lanes do not
        shift it.

        It is drawn per keep-segment, so any cut inside the span collapses
        instead of smearing the audio out of sync.  The vertical crop that
        peak-normalises the trace is computed once, here, not per repaint.
        """
        pix = QPixmap(png_path)
        if pix.isNull() or src_end - src_start <= 1e-6:
            return
        top, height = _wave_crop(pix)
        self._waves[int(track)] = _Wave(
            pix, float(src_start), float(src_end), top, height,
            self.wave_source(int(track)),
        )
        self._gen_assets += 1
        self.update()

    # ================================================================ theme
    def _refresh_theme(self) -> None:
        """Re-resolve the lane palette after a theme switch and repaint.

        Lane colours are looked up per paint, so the only state to rebuild here
        is the stuff Qt caches for us: the scrollbar's QSS, the header
        stylesheets, and the composed lane pixmap.
        """
        mode = theme_variant(QApplication.instance())
        if mode == self._mode:
            return
        self._mode = mode
        self._pal = _palette(mode)
        self._hbar.setStyleSheet(_scrollbar_qss(mode))
        for header in self._headers:
            header.apply_palette(self._pal)
        self._content_key = None
        self._tinted.clear()
        self.update()

    def changeEvent(self, event: QEvent) -> None:  # noqa: N802
        super().changeEvent(event)
        if event.type() in (QEvent.Type.PaletteChange, QEvent.Type.ApplicationPaletteChange):
            self._refresh_theme()

    # ================================================================ lanes
    @property
    def _headers(self) -> list[_AudioLaneHeader]:
        """Every audio lane's gutter header, in lane order.

        Video lanes have no child widget — their gutter row is painted — so
        this is shorter than ``self._lanes`` whenever a video lane exists.
        """
        return [lane.header for lane in self._lanes if lane.header is not None]

    def lanes(self) -> list[Lane]:
        """The lane stack, in priority/display order.  Live objects, not copies."""
        return list(self._lanes)

    def lane_kind(self, lane: int) -> str:
        """``"video"`` or ``"audio"`` for a lane id; ``""`` when there is none."""
        if 0 <= int(lane) < len(self._lanes):
            return self._lanes[int(lane)].kind
        return ""

    def lane_clips(self, lane: int) -> list[Clip]:
        """The clips on ``lane``, in time order (empty for an unknown lane)."""
        if 0 <= int(lane) < len(self._lanes):
            return list(self._lanes[int(lane)].clips)
        return []

    def video_lane_count(self) -> int:
        return sum(1 for lane in self._lanes if lane.kind == "video")

    def add_video_lane(self) -> int:
        """Append an empty video lane UNDERNEATH the existing video lanes.

        "Underneath" is priority, not just pixels: lane 0 keeps painting over
        it, and the new lane is what shows through wherever lane 0 has a hole
        (the lift) or has run out of footage.  That is the only reason a second
        video lane is worth having — see :func:`render.visible_spans`.
        """
        return self._insert_lane("video", self.video_lane_count())

    def add_title(self, at: float, text: str) -> Clip:
        """A text block at TIMELINE time ``at`` on a lane of its own.  Undoable.

        It goes on the first video lane that holds only titles (or is empty
        and is not the main lane), else on a new one -- "its own video lane".
        Titles are drawn over the finished picture, so that lane's position in
        the priority stack does not matter.
        """
        keep_all = self.all_armed()
        self.model.push_state(self._ui_state())
        lane = next(
            (
                ln for i, ln in enumerate(self._lanes)
                if ln.kind == "video" and i > 0
                and all(c.title is not None for c in ln.clips)
            ),
            None,
        )
        if lane is None:
            lane = self._new_lane("video")
            self._lanes.insert(self.video_lane_count(), lane)
        start = self._free_start(lane, TEXT_DEFAULT_LEN, max(0.0, float(at)), set())
        clip = self._new_clip(
            "", "video", start, TEXT_SOURCE / 2, TEXT_DEFAULT_LEN,
            label="Text", source=TEXT_SOURCE,
        )
        clip.title = Title(text)
        lane.clips.append(clip)
        lane.clips.sort(key=lambda c: c.start)
        self._after_lane_change(keep_all)
        return clip

    def set_title(self, clip: Clip, *, undo: bool = True, **changes: object) -> None:
        """Change a text block's text/position/size.  ``undo=False`` while a
        drag that already pushed its one undo step is still moving it."""
        if clip.title is None:
            return
        if undo:
            self.model.push_state(self._ui_state())
        clip.title = replace(clip.title, **changes)  # type: ignore[arg-type]
        self.update()
        self.cutsChanged.emit()

    def push_undo(self) -> None:
        """Open one undo step for a gesture that then calls ``set_title(undo=False)``."""
        self.model.push_state(self._ui_state())

    def titles_at(self, t_edited: float) -> list[Clip]:
        """Text blocks showing at EDITED time ``t``, in lane order."""
        if self.model.edited_duration() <= 0.0:
            return []
        t = self.model.edited_to_source(t_edited)
        return [
            c for lane in self._lanes if lane.kind == "video"
            for c in lane.clips if c.title is not None and c.start <= t < c.end
        ]

    def find_clip(self, cid: int) -> Clip | None:
        return next((c for lane in self._lanes for c in lane.clips if c.cid == cid), None)

    def add_audio_lane(self) -> int:
        """Append an empty audio lane at the bottom.  Audio lanes all mix."""
        return self._insert_lane("audio", len(self._lanes))

    def _insert_lane(self, kind: str, at: int) -> int:
        keep_all = self.all_armed()
        self.model.push_state(self._ui_state())
        self._lanes.insert(at, self._new_lane(kind))
        self._after_lane_change(keep_all)
        return at

    def remove_lane(self, lane: int) -> bool:
        """Drop a lane and everything on it.  Undoable.  False if refused.

        Lane 0 is the main video lane and cannot be removed: it is where
        ``set_media`` puts the loaded capture, and the render takes the output
        geometry from it.  Use ``File ▸ Open`` to change the media instead.
        """
        index = int(lane)
        if not 0 < index < len(self._lanes):
            return False
        keep_all = self.all_armed()
        self.model.push_state(self._ui_state())
        self._drop_header(self._lanes.pop(index))
        self._after_lane_change(keep_all)
        return True

    def add_clip(
        self,
        lane: int | None,
        info: MediaInfo | AudioFileInfo,
        mode: str,
        *,
        with_audio: bool = True,
    ) -> Clip | None:
        """Put a probed file on ``lane``.  Undoable.  ``None`` when it will not fit.

        Returns the block it placed, so a caller that has more to do with it
        can.

        ``mode`` is ``"place"`` (start at 0:00, for an empty lane) or
        ``"append"`` (start where the lane's existing content ends).  The kind
        has to match the lane: a video file needs a video lane, audio needs an
        audio lane — :meth:`drop_target` already resolves which lane that is.
        ``lane=None`` grows a new lane of the right kind for the clip, as ONE
        undo step, which is what "just drop it anywhere" means.

        A VIDEO brings its sound with it (``with_audio``): one audio lane per
        embedded track, each clip starting where the picture does, and the
        whole set GROUPED — so sliding the picture slides the sound, which is
        the only way a second capture stays in sync once blocks can move.
        That is the same thing :meth:`set_media` does for the loaded file; a
        drop used to be the one place it did not happen.

        Adding a clip can make the timeline LONGER; everything else follows
        from :meth:`_total_duration`.
        """
        kind = "video" if isinstance(info, MediaInfo) else "audio"
        duration = float(info.duration)
        if duration <= 0.0:
            return None
        if lane is None:
            target = self._new_lane(kind)
            at = self.video_lane_count() if kind == "video" else len(self._lanes)
        elif 0 <= int(lane) < len(self._lanes) and self._lanes[int(lane)].kind == kind:
            target, at = self._lanes[int(lane)], None
        else:
            return None
        keep_all = self.all_armed()
        self.model.push_state(self._ui_state())
        if at is not None:
            self._lanes.insert(at, target)
        # ``mode`` was decided at DROP time by ``drop_target``, but the clip is
        # only placed once an async probe comes back — so "place on an empty
        # lane" can be stale by the time it lands.  Two files dropped on one
        # empty lane both resolved to "place" and both went to 0.0, which is
        # the no-overlap rule broken by a plain gesture: ``visible_spans``
        # takes the first clip and the second file silently never renders.
        # Ask for 0.0, take the first spot it actually fits.
        wanted = 0.0 if mode == "place" else target.end()
        clip = self._new_clip(
            info.path, kind, self._free_start(target, duration, wanted, set()), 0.0,
            duration, label=getattr(info, "label", ""), source=duration,
        )
        target.clips.append(clip)
        target.clips.sort(key=lambda c: c.start)
        tracks = getattr(info, "audio", ()) if with_audio and kind == "video" else ()
        self.weld_group([clip, *self._embedded_audio(info, tracks, clip.start)])
        self._after_lane_change(keep_all)
        return clip

    def _embedded_audio(
        self, info: MediaInfo, tracks: Iterable[object], start: float
    ) -> list[Clip]:
        """One new audio lane per embedded track, each aligned to ``start``.

        Caller owns the undo step — this is half of one edit, never a whole one.
        """
        made: list[Clip] = []
        for track in tracks:
            bed = self._new_lane("audio")
            bed.clips.append(
                self._new_clip(
                    info.path, "audio", start, 0.0, float(info.duration),
                    stream=getattr(track, "index", 0),
                    label=getattr(track, "label", ""),
                    source=float(info.duration),
                )
            )
            self._lanes.append(bed)
            made.append(bed.clips[0])
        return made

    def drop_target(self, pos: QPointF | QPoint) -> tuple[str, int] | None:
        """Resolve a widget-local drop position to ``(mode, lane)``.

        ``("place", lane)`` on an empty lane, ``("append", lane)`` on one that
        already has content — which is the user's rule verbatim: drop on the
        new track and it lands there, drop on the main track and it goes on the
        end of what is already there.  ``None`` when there are no lanes at all.

        A position above or below the stack clamps to the nearest lane rather
        than failing, because a drop that lands two pixels into the ruler is
        still obviously aimed at the top lane.
        """
        if not self._lanes:
            return None
        y = float(pos.y())
        lane = self._lane_at(y)
        if lane is None:
            lane = 0 if y < self._lane_rect(0).top() else len(self._lanes) - 1
        return ("append" if self._lanes[lane].clips else "place", lane)

    # =============================================================== blocks
    # A clip is a BLOCK.  Alt+click selects it (Ctrl+Alt adds or removes),
    # Alt+drag slides it along its lane, and the right-click menu splits,
    # groups, copies and deletes the selection.  PLAIN click and drag are
    # untouched: they still seek and mark a range, which is the cut workflow
    # this tool exists for and the one gesture that must not be retrained.
    #
    # Blocks live in TIMELINE time like everything else on a lane, so a drag is
    # just a new ``Clip.start`` and the render picks it up for free (the graph
    # already places a clip with ``adelay``/``trim`` — see render._audio_heads
    # and render.visible_spans).  Two rules the model depends on:
    #
    # * blocks on one lane NEVER overlap — ``_audio_heads`` lays them end to end
    #   through ``amix`` and ``visible_spans`` takes the first clip covering an
    #   instant, so an overlap would silently drop one of them.  Every mutation
    #   here goes through ``_move_limits`` or ``_free_start``;
    # * a GAP is legal and means black (video) or silence (audio).  That is what
    #   Delete leaves behind, and the render already emits a ``color`` source
    #   for it.
    def _new_clip(self, *args: object, **kwargs: object) -> Clip:
        """A clip carrying the next id.  The ONLY way to make one here."""
        self._clip_seq += 1
        kwargs["cid"] = self._clip_seq
        return Clip(*args, **kwargs)  # type: ignore[arg-type]

    def clip_selection(self) -> frozenset[int]:
        """The selected block ids."""
        return frozenset(self._selected)

    def selected_blocks(self) -> list[tuple[int, Clip]]:
        """``(lane index, clip)`` for every selected block, in lane/time order."""
        return [
            (index, clip)
            for index, lane in enumerate(self._lanes)
            for clip in lane.clips
            if clip.cid in self._selected
        ]

    def block_at(self, pos: QPointF | QPoint) -> tuple[int, Clip] | None:
        """The block under a widget-local point, or ``None``.

        The inverse of what ``_clip_boxes`` draws: the x maps back through the
        cut list to timeline time, so a click lands on the block the pixel came
        from even when a cut has collapsed the material between them.
        """
        x, y = float(pos.x()), float(pos.y())
        if not self._in_track(x) or not self.has_media():
            return None
        lane = self._lane_at(y)
        if lane is None:
            return None
        t = self.model.edited_to_source(self._t_for_clamped(x))
        for clip in self._lanes[lane].clips:
            if clip.start - 1e-9 <= t <= clip.end + 1e-9:
                return (lane, clip)
        return None

    def block_edges(self, clip: Clip) -> Range | None:
        """The block's outermost LEFT and RIGHT x in widget pixels.

        Through the cut mapping, not from ``_x_for(clip.start)``: a cut inside
        the block splits it into several boxes and a cut at its very start
        collapses the pixel its start maps to.  The outermost edges of those
        boxes are where the handles live, which is exactly what the user sees.
        ``None`` when every frame of the block is cut away — there is no edge
        on screen to grab.
        """
        spans = self.model.edited_spans_for_source(clip.start, clip.end)
        if not spans:
            return None
        return (self._x_for(spans[0][0]), self._x_for(spans[-1][1]))

    def block_handle(self, pos: QPointF | QPoint) -> tuple[int, Clip, str] | None:
        """What an Alt+press at ``pos`` grabs: ``(lane, clip, handle)``.

        ``handle`` is ``"move"`` (the body — slide it along the lane),
        ``"left"``/``"right"`` (an edge — trim it, restoring or giving back
        material inside the clip's own file) or ``"fade-in"``/``"fade-out"``
        (a top corner — set the fade length).  This is Sony Vegas's event
        layout: the corners are the fades, the rest of the edge is the trim,
        the middle is the move.

        The corner box wins inside the edge zone because it is the smaller
        target of the two, and both are refused on a block narrower than
        ``MIN_BLOCK_PX`` — at that width the handles would cover the whole
        block and it could never be moved again without zooming in.
        """
        hit = self.block_at(pos)
        if hit is None:
            return None
        index, clip = hit
        edges = self.block_edges(clip)
        if edges is None:
            return (index, clip, "move")
        x0, x1 = edges
        if x1 - x0 < MIN_BLOCK_PX:
            return (index, clip, "move")
        x, y = float(pos.x()), float(pos.y())
        if y - self._lane_rect(index).top() <= FADE_GRAB_H:
            if x - x0 <= FADE_GRAB_PX:
                return (index, clip, "fade-in")
            if x1 - x <= FADE_GRAB_PX:
                return (index, clip, "fade-out")
        if x - x0 <= EDGE_GRAB_PX:
            return (index, clip, "left")
        if x1 - x <= EDGE_GRAB_PX:
            return (index, clip, "right")
        return (index, clip, "move")

    def select_block(self, clip: Clip | None, *, toggle: bool = False) -> None:
        """Select ``clip`` and everything grouped with it (``None`` clears).

        ``toggle`` is Ctrl+Alt+click: it adds the group, or removes it when the
        whole group is already in the selection.
        """
        if clip is None:
            self.clear_block_selection()
            return
        cids = self._group_cids(clip)
        if not toggle:
            self._selected = set(cids)
        elif cids <= self._selected:
            self._selected -= cids
        else:
            self._selected |= cids
        self._announce_blocks()

    def clear_block_selection(self) -> None:
        if self._selected:
            self._selected.clear()
            self._announce_blocks()

    def _group_cids(self, clip: Clip) -> set[int]:
        """Every block that moves with ``clip`` — itself, or its whole group."""
        if not clip.group:
            return {clip.cid}
        return {
            other.cid
            for lane in self._lanes
            for other in lane.clips
            if other.group == clip.group
        }

    def _announce_blocks(self) -> None:
        self._content_key = None  # the selection is painted into the composite
        self.blockSelectionChanged.emit(frozenset(self._selected))
        self.update()

    def _prune_selection(self) -> None:
        """Forget ids that no longer exist (an undo, a delete, a new file)."""
        live = {clip.cid for lane in self._lanes for clip in lane.clips}
        if not self._selected <= live:
            self._selected &= live
            self._announce_blocks()

    # -- placement arithmetic ------------------------------------------------
    def _move_limits(self) -> Range:
        """How far the selection may slide, as ``(most negative, most positive)``.

        Computed ONCE at the start of a drag, from where the blocks then were:
        recomputing per mouse-move would let a block creep past a neighbour one
        pixel at a time.  Blocks stop against their neighbours and against 0:00
        rather than pushing them, because a push would move material the user
        did not grab.
        """
        lo, hi = -1e18, 1e18
        for index, clip in self.selected_blocks():
            lo = max(lo, -clip.start)
            for other in self._lanes[index].clips:
                if other.cid in self._selected:
                    continue
                if other.end <= clip.start + 1e-9:
                    lo = max(lo, other.end - clip.start)
                elif other.start >= clip.end - 1e-9:
                    hi = min(hi, other.start - clip.end)
        return (min(0.0, lo), max(0.0, hi))

    @staticmethod
    def _free_start(lane: Lane, length: float, wanted: float, ignore: set[int]) -> float:
        """First start at or after ``wanted`` where ``length`` fits on ``lane``.

        Used where a block ARRIVES somewhere rather than sliding there (paste,
        a drag onto another lane): pushing it right to the first free spot is
        predictable, and it keeps the no-overlap rule without silently eating
        whatever was already there.
        """
        others = sorted(
            (c for c in lane.clips if c.cid not in ignore), key=lambda c: c.start
        )
        start = max(0.0, wanted)
        for other in others:
            if start < other.end - 1e-9 and other.start < start + length - 1e-9:
                start = other.end
        return start

    def _snap_tolerance(self) -> float:
        """Snap radius in SECONDS for the current zoom.

        The radius is fixed in PIXELS, so zoomed out it forgives a lot and
        zoomed in it forgives almost nothing: a frame-accurate drag is still
        possible without turning snapping off.
        """
        span = max(1e-9, self._view[1] - self._view[0])
        return SNAP_PX * span / max(1.0, self._track_rect().width())

    def _snap_candidates(self, ignore: set[int]) -> list[float]:
        """Timeline times a moving edge is worth landing flush on.

        0:00, the playhead and every edge of every block that is NOT itself
        moving.  Shared by the move drag and the edge trim so they snap to the
        same things — a trimmed edge that would not sit flush against the
        neighbour a move snaps to would be its own kind of surprise.
        """
        edges = [0.0, self.model.edited_to_source(self._playhead)]
        for lane in self._lanes:
            for clip in lane.clips:
                if clip.cid not in ignore:
                    edges.extend((clip.start, clip.end))
        return edges

    def _snap_time(self, t: float, ignore: set[int]) -> float:
        """Snap ONE moving edge (a trim) to the nearest candidate, or leave it."""
        best, error = t, self._snap_tolerance()
        for edge in self._snap_candidates(ignore):
            if abs(t - edge) < error:
                best, error = edge, abs(t - edge)
        return best

    def _snap_delta(self, delta: float) -> float:
        """Nudge a drag so a moving EDGE lands flush on something meaningful.

        Candidates are 0:00, the playhead and every stationary block's edges —
        which is what stops a drag leaving a 3 ms sliver of black between two
        blocks that were meant to butt together.  The tolerance is in PIXELS,
        not seconds, so zoomed out it forgives a lot and zoomed in it forgives
        almost nothing: a frame-accurate drag is still possible.
        """
        tol = self._snap_tolerance()
        edges = self._snap_candidates(self._selected)
        best, error = delta, tol
        for _index, clip in self.selected_blocks():
            origin = self._block_from.get(clip.cid, clip.start)
            for moving in (origin + delta, origin + clip.length + delta):
                for edge in edges:
                    if abs(moving - edge) < error:
                        error = abs(moving - edge)
                        best = delta + (edge - moving)
        return best

    # -- edits ---------------------------------------------------------------
    def _push_block_edit(self) -> None:
        """Make the drag undoable, once, at the moment it first changes something.

        Pushing on PRESS would leave an undo step behind every plain Alt+click;
        pushing when the pointer merely moves would leave one behind a block
        that had nowhere to go.
        """
        if not self._block_pushed:
            self.model.push_state(self._ui_state())
            self._block_pushed = True

    def move_blocks(self, delta: float) -> None:
        """Slide the selection by ``delta`` timeline seconds, clamped.

        Live, during a drag: the undo step was pushed when the drag started, so
        this may be called as often as the pointer moves.
        """
        lo, hi = self._block_limits
        delta = max(lo, min(hi, delta))
        for index, clip in self.selected_blocks():
            clip.start = self._block_from.get(clip.cid, clip.start) + delta
        for lane in self._lanes:
            lane.clips.sort(key=lambda c: c.start)
        # GROW only, mid-drag.  Shrinking clamps the cut list to the new length
        # (``CutList.set_duration``), and a drag that goes left and comes back
        # would drop cuts on the way out and not restore them on the way in.
        # The release settles the real length once, through _after_lane_change.
        self.model.set_duration(max(self.model.duration, self._total_duration()))
        self._cells = None
        self._gen_cuts += 1
        self._content_key = None
        self.update()

    # -- edge trimming -------------------------------------------------------
    # Vegas event trimming: grab a block's EDGE instead of its middle and the
    # block's own in/out points move instead of its position.  Dragging the
    # right edge out restores material a split or an earlier trim cut off, up
    # to the end of the file itself; dragging the left edge out moves ``start``
    # and ``src_in`` by the SAME amount, so the restored seconds appear before
    # the block and the picture already in it does not shift on screen.
    #
    # Two rules it must not break.  A trim can never overlap a neighbour (the
    # no-overlap rule: ``visible_spans`` silently drops one of an overlapping
    # pair), and it can never ask for material the file does not have —
    # ``src_in >= 0`` and ``src_out <= source``, or ffmpeg trims to nothing and
    # exits 0 with a short file.  ``_trim_limits`` is the one place both live.
    def trim_members(self, clip: Clip) -> list[tuple[int, Clip]]:
        """``(lane, clip)`` for everything a trim of ``clip`` moves.

        A GROUP trims together — a dropped video's picture and its sound share
        an in-point, and pulling one of them out while the other stayed put is
        exactly the desync the group exists to prevent.  Loose blocks trim
        alone.  The limits are intersected across the members (see
        :meth:`_trim_limits`), so a picture whose file has 3 more seconds and a
        sound whose file has 1 both stop after 1.
        """
        cids = self._group_cids(clip)
        return [
            (index, other)
            for index, lane in enumerate(self._lanes)
            for other in lane.clips
            if other.cid in cids
        ]

    def _trim_limits(self, members: list[tuple[int, Clip]], edge: str) -> Range:
        """How far this EDGE may travel, as ``(most negative, most positive)``.

        Computed once at the start of the drag, like :meth:`_move_limits`, and
        intersected over every member so a group keeps one length.  Positive is
        always rightwards on the timeline: on the right edge that GROWS the
        block, on the left edge it SHRINKS it.
        """
        lo, hi = -1e18, 1e18
        for index, clip in members:
            # Everything except THIS clip is a neighbour, group members
            # included: two grouped blocks sitting back to back on one lane
            # both move the same edge by the same delta, so the one behind does
            # not get out of the way — excluding it would walk them straight
            # into an overlap, which ``visible_spans`` resolves by silently
            # dropping one of them.
            others = [c for c in self._lanes[index].clips if c.cid != clip.cid]
            if edge == "right":
                # Shrink to the minimum, grow to whichever comes first: the end
                # of the file's own material or the next block on the lane.
                lo = max(lo, MIN_BLOCK_LEN - clip.length)
                room = clip.source - clip.src_out
                after = [c.start for c in others if c.start >= clip.end - 1e-9]
                if after:
                    room = min(room, min(after) - clip.end)
                hi = min(hi, room)
            else:
                # Out (left) is bounded by the file's head, by 0:00 and by the
                # block before it; in (right) by the minimum length.
                room = min(clip.src_in, clip.start)
                before = [c.end for c in others if c.end <= clip.start + 1e-9]
                if before:
                    room = min(room, clip.start - max(before))
                lo = max(lo, -room)
                hi = min(hi, clip.length - MIN_BLOCK_LEN)
        return (min(0.0, lo), max(0.0, hi))

    def trim_blocks(self, delta: float) -> None:
        """Move the grabbed edge by ``delta`` timeline seconds, clamped.

        Live, during a drag, so it is recomputed from the origins recorded at
        the press rather than accumulated — the undo step was pushed when the
        drag first changed something.  Fades are re-clamped from their ORIGINAL
        lengths every time, so a drag that shrinks a block past its fade and
        then grows it back leaves the fade exactly as it was.
        """
        lo, hi = self._edge_limits
        delta = max(lo, min(hi, delta))
        for _index, clip in self._blocks_by_cid(self._edge_from):
            start, src_in, length, fade_in, fade_out = self._edge_from[clip.cid]
            if self._edge_kind == "right":
                clip.length = length + delta
            else:
                clip.start = start + delta
                clip.src_in = src_in + delta
                clip.length = length - delta
            clip.fade_in = min(fade_in, clip.length)
            clip.fade_out = min(fade_out, clip.length)
        for lane in self._lanes:
            lane.clips.sort(key=lambda c: c.start)
        # GROW only, mid-drag: ``CutList.set_duration`` CLAMPS the cut list, so
        # a trim that shortens the timeline and comes back would drop cuts on
        # the way out and not restore them.  The release settles it once.
        self.model.set_duration(max(self.model.duration, self._total_duration()))
        self._cells = None
        self._gen_cuts += 1
        self._content_key = None
        self.update()

    def trim_block(self, clip: Clip, edge: str, delta: float) -> float:
        """Trim one block's ``"left"``/``"right"`` edge as ONE undoable edit.

        The whole gesture in a single call — limits, mutation and the tail —
        for callers that are not a drag: a test, a keyboard nudge, a menu item.
        Returns the delta actually applied, which is 0.0 when the edge had
        nowhere to go (and then nothing is pushed, so Ctrl+Z is not wasted on
        an edit that did not happen).
        """
        members = self.trim_members(clip)
        lo, hi = self._trim_limits(members, edge)
        applied = max(lo, min(hi, float(delta)))
        if abs(applied) <= 1e-9:
            return 0.0
        keep_all = self.all_armed()
        self.model.push_state(self._ui_state())
        self._edge_kind = edge
        self._edge_cid = clip.cid
        self._edge_from = {
            c.cid: (c.start, c.src_in, c.length, c.fade_in, c.fade_out)
            for _, c in members
        }
        self._edge_limits = (lo, hi)
        self.trim_blocks(applied)
        self._edge_from.clear()
        self._edge_kind = ""
        self._after_lane_change(keep_all)
        return applied

    def set_block_fade(self, clip: Clip, which: str, seconds: float) -> bool:
        """Set a fade on a block and its group as ONE undoable edit.

        The non-drag counterpart of the corner handle.  Returns False (and
        pushes nothing) when every affected block already has that fade.
        """
        members = self.trim_members(clip)
        wanted = [
            (c, max(0.0, min(float(seconds), c.length))) for _, c in members
        ]
        if all(
            abs(value - (c.fade_in if which == "in" else c.fade_out)) <= 1e-9
            for c, value in wanted
        ):
            return False
        keep_all = self.all_armed()
        self.model.push_state(self._ui_state())
        for c, value in wanted:
            self.set_fade(c, which, value)
        self._after_lane_change(keep_all)
        return True

    def _blocks_by_cid(self, cids: Iterable[int]) -> list[tuple[int, Clip]]:
        """``(lane, clip)`` for a set of ids, skipping ones that are gone."""
        wanted = set(cids)
        return [
            (index, clip)
            for index, lane in enumerate(self._lanes)
            for clip in lane.clips
            if clip.cid in wanted
        ]

    # -- fades ---------------------------------------------------------------
    def set_fade(self, clip: Clip, which: str, seconds: float) -> None:
        """Set one fade on one block, clamped to what the block can hold.

        The cap is the block's WHOLE length, not half of it: fading a short
        block all the way up from black is a real look, and two fades that
        overlap render as the product of two ramps rather than as a failure —
        ``fade``/``afade`` are independent filters.  Capping at half would
        forbid the honest case to prevent a harmless one.
        """
        value = max(0.0, min(float(seconds), clip.length))
        if clip.title is not None:
            value = 0.0  # shortcut: titles do not fade, add drawtext alpha if wanted
        if which == "in":
            clip.fade_in = value
        else:
            clip.fade_out = value

    def _fade_target(self, clip: Clip, delta: float) -> float:
        """What ``delta`` seconds of fade drag would make this block's fade.

        The handles pull INWARD: the fade-in handle grows as it travels right,
        the fade-out handle grows as it travels left, which is the direction
        the painted ramp runs.  No snapping — a fade has nothing meaningful to
        land flush against, and snapping one to a neighbour's edge would just
        make short fades impossible to set.
        """
        was = self._fade_from.get(clip.cid, 0.0)
        want = was + (delta if self._fade_kind == "in" else -delta)
        return max(0.0, min(want, clip.length))

    def fade_changes(self, delta: float) -> bool:
        """Would this fade drag actually change anything?

        Asked BEFORE the first mutation, because that is when the undo step has
        to be pushed: a handle already at 0 being dragged further out is a
        no-op, and a no-op must not leave a Ctrl+Z behind (the same rule an
        Alt+click on a wedged block follows).
        """
        return any(
            abs(self._fade_target(clip, delta) - self._fade_from[clip.cid]) > 1e-9
            for _index, clip in self._blocks_by_cid(self._fade_from)
        )

    def fade_blocks(self, delta: float) -> None:
        """Drag the grabbed fade handle by ``delta`` seconds, live.

        Recomputed from the lengths recorded at the press rather than
        accumulated, so a drag out and back lands exactly where it started.
        """
        for _index, clip in self._blocks_by_cid(self._fade_from):
            self.set_fade(clip, self._fade_kind, self._fade_target(clip, delta))
        self._content_key = None
        self.update()

    def move_blocks_to_lane(self, target: int) -> bool:
        """Move the selection onto ``target``, appended after its content.

        Dropping a block on another track and having it land where the pointer
        happened to be would silently overlap whatever is there, so it goes on
        the END of that track instead — the same rule a file dropped on a lane
        with content already follows.  An EMPTY target keeps the block where it
        was, because there is nothing to append to.  Refused across kinds: a
        video block has no meaning on an audio lane.
        """
        if not self.can_move_to_lane(target):
            return False
        blocks = self.selected_blocks()
        destination = self._lanes[target]
        source = self._lanes[{index for index, _ in blocks}.pop()]
        base = min(clip.start for _, clip in blocks)
        at = destination.end() if destination.clips else base
        for _, clip in blocks:
            source.clips.remove(clip)
            clip.start = at + (clip.start - base)
            destination.clips.append(clip)
        destination.clips.sort(key=lambda c: c.start)
        return True

    def can_move_to_lane(self, target: int) -> bool:
        """Would :meth:`move_blocks_to_lane` do anything?

        Asked separately so a drag can decide whether to push an undo step
        BEFORE it mutates anything — pushing first and then finding the move
        refused would leave an undo step that undoes nothing.
        """
        blocks = self.selected_blocks()
        if not blocks or not 0 <= target < len(self._lanes):
            return False
        lanes = {index for index, _ in blocks}
        if len(lanes) != 1 or target in lanes:
            return False
        # Any group, even one whose members all sit on this lane: the README
        # promises "Ungroup first", and a group is what keeps blocks in step.
        if any(clip.group for _, clip in blocks):
            return False
        return all(clip.kind == self._lanes[target].kind for _, clip in blocks)

    def _group_shift(self, members: list[tuple[Lane, Clip]], at: float) -> float:
        """How far one pasted GROUP has to move right to fit, as one offset.

        Every member moves by the same amount or the group arrives welded but
        out of sync — a picture seconds from its own sound, which no drag can
        fix without ungrouping first.  Taking the largest single push is not
        enough: shifting the group can walk one member into something ELSE, so
        the offset is iterated until nobody needs to move again.

        Asked when the group is PLACED, against the lanes as they are then,
        earlier pastes included — and the members then go exactly there, with
        no second ``_free_start``.  Worked out up front for every group, a
        block pasted before it pushed one member and not the other.

        Loose blocks (``group == 0``) never come here.  They are unrelated to
        each other and each finds its own spot — sharing one offset dragged a
        block on an empty lane hundreds of seconds away because some other
        block in the same paste had nowhere to go.
        """
        offset = 0.0
        # Each pass steps the group past at least one blocking clip, so the
        # count of clips on the lanes involved bounds it; the cap is only a
        # guard against a float that will not settle.
        cap = sum(len(lane.clips) for lane, _ in members) + 4
        for _ in range(cap):
            need = max(
                self._free_start(lane, c.length, at + c.start + offset, set())
                - (at + c.start + offset)
                for lane, c in members
            )
            if need <= 1e-9:
                break
            offset += need
        return offset

    def split_blocks(self) -> bool:
        """Razor every block the playhead crosses in two.  Undoable.

        With blocks selected it splits those; with none it splits every block
        on an ARMED lane, which is the razor-across-the-tracks gesture.  It
        changes not one frame of the edit — it makes two blocks that can be
        moved apart, which is the whole point.

        A GROUP is split too, into two groups: the heads keep the old id and
        the tails share a fresh one.  Leaving both halves in one group would
        weld them back together and make the split useless — while dropping the
        group entirely would let a picture's tail slide away from its own
        sound's tail, which is the thing groups exist to prevent.
        """
        if not self.has_media():
            return False
        at = self.model.edited_to_source(self._playhead)
        targets: list[tuple[Lane, Clip]] = []
        for index, lane in enumerate(self._lanes):
            for clip in lane.clips:
                # The trim's floor, not the cut list's: a split must not make a
                # block shorter than a trim is allowed to leave one.
                if not clip.start + MIN_BLOCK_LEN <= at <= clip.end - MIN_BLOCK_LEN:
                    continue
                if clip.cid in self._selected or (
                    not self._selected and self.is_armed(index)
                ):
                    targets.append((lane, clip))
        if not targets:
            return False
        keep_all = self.all_armed()
        self.model.push_state(self._ui_state())
        tails: dict[int, list[Clip]] = {}
        for lane, clip in targets:
            tail = self._new_clip(
                clip.path, clip.kind, at, clip.source_at(at), clip.end - at,
                clip.stream, clip.label, clip.source,
            )
            tail.title = clip.title
            clip.length = at - clip.start
            # Each half keeps the fade at its OUTER end and neither gets one at
            # the razor: the new inner edges are a join that did not exist a
            # moment ago, and a fade there would dip the picture in the middle
            # of a cut nobody asked to see.  Both are clamped to their new
            # length, so splitting inside a long fade shortens it rather than
            # leaving a fade longer than the block carrying it.
            self.set_fade(tail, "out", clip.fade_out)
            self.set_fade(tail, "in", 0.0)
            self.set_fade(clip, "in", clip.fade_in)
            self.set_fade(clip, "out", 0.0)
            lane.clips.append(tail)
            lane.clips.sort(key=lambda c: c.start)
            if clip.group:
                tails.setdefault(clip.group, []).append(tail)
            if clip.cid in self._selected:
                self._selected.add(tail.cid)
        for members in tails.values():
            # A group of one is not a group: the other members of the original
            # were not split, so this tail has nothing left to stay in step with.
            self.weld_group(members)
        self._after_lane_change(keep_all)
        self._announce_blocks()
        return True

    def delete_blocks(self) -> bool:
        """Remove the selected blocks, leaving a GAP where they were.

        Not a ripple: everything else stays exactly where it is, which is what
        makes the survivors worth dragging.  A video gap is black (or whatever
        video lane is underneath, showing through), an audio gap is silence.
        """
        if not self._selected:
            return False
        keep_all = self.all_armed()
        self.model.push_state(self._ui_state())
        for lane in self._lanes:
            lane.clips = [c for c in lane.clips if c.cid not in self._selected]
        self._selected.clear()
        self._after_lane_change(keep_all)
        self._announce_blocks()
        return True

    def group_blocks(self) -> bool:
        """Weld the selection together: from now on they select and move as one."""
        blocks = self.selected_blocks()
        if len(blocks) < 2:
            return False
        first = blocks[0][1]
        if first.group and self._group_cids(first) == {c.cid for _, c in blocks}:
            return False  # already exactly this group: not an undoable edit
        keep_all = self.all_armed()
        self.model.push_state(self._ui_state())
        self._group_seq += 1
        for _, clip in blocks:
            clip.group = self._group_seq
        self._after_lane_change(keep_all)
        self._announce_blocks()
        return True

    def ungroup_blocks(self) -> bool:
        grouped = [clip for _, clip in self.selected_blocks() if clip.group]
        if not grouped:
            return False
        keep_all = self.all_armed()
        self.model.push_state(self._ui_state())
        for clip in grouped:
            clip.group = 0
        self._after_lane_change(keep_all)
        self._announce_blocks()
        return True

    def has_grouped_selection(self) -> bool:
        return any(clip.group for _, clip in self.selected_blocks())

    def weld_group(self, clips: Iterable[Clip]) -> int:
        """Put ``clips`` in a fresh group WITHOUT an undo step of its own.

        For a caller already inside one: dropping a video welds its picture to
        its sound, and that has to undo as the single drop it was.  Fewer than
        two members is not a group and is left alone — ``split_blocks`` leans on
        that for the tail of a block whose group-mates were not split.
        """
        members = [c for c in clips]
        if len(members) < 2:
            return 0
        self._group_seq += 1
        for clip in members:
            clip.group = self._group_seq
        return self._group_seq

    def copy_blocks(self) -> int:
        """Park the selection on the clipboard.  Returns how many."""
        blocks = self.selected_blocks()
        if not blocks:
            return 0
        base = min(clip.start for _, clip in blocks)
        self._clipboard = []
        for index, clip in blocks:
            copy = clip.copy()
            copy.start = clip.start - base  # relative: paste lands at the playhead
            self._clipboard.append((self._lanes[index].key, copy))
        return len(self._clipboard)

    def can_paste(self) -> bool:
        """Something copied AND somewhere to put it.

        Deliberately not ``has_media()``: deleting every block takes the
        duration to 0, and a paste is exactly how you put something back.  It
        does check that a clipboard entry still resolves to a live lane of the
        right kind — otherwise the menu offered a paste that then answered
        "nothing copied", which is not what happened at all.
        """
        by_key = {lane.key: lane for lane in self._lanes}
        return any(
            (lane := by_key.get(key)) is not None and lane.kind == clip.kind
            for key, clip in self._clipboard
        )

    def paste_blocks(self) -> bool:
        """Put the clipboard back on its lanes, starting at the playhead.

        A pasted block keeps its file, its in-point and its length; what it
        does not keep is its identity — fresh ids, and a fresh group id per
        group, or pasting a grouped pair would weld the copies to the
        originals.  Where the playhead would overlap what is already on the
        lane, the paste is pushed right to the first spot it fits.
        """
        # Not ``has_media()``: deleting every block takes the duration to 0 and
        # that must not be a one-way door — a paste is exactly how you put
        # something back.
        if not self._clipboard or not self._lanes:
            return False
        by_key = {lane.key: lane for lane in self._lanes}
        at = self.model.edited_to_source(self._playhead)
        placed: list[tuple[Lane, Clip]] = []
        for key, clip in self._clipboard:
            lane = by_key.get(key)
            if lane is not None and lane.kind == clip.kind:
                placed.append((lane, clip))
        if not placed:
            return False
        keep_all = self.all_armed()
        self.model.push_state(self._ui_state())
        # Groups first, each placed whole at ONE offset (see _group_shift);
        # loose blocks after, each finding its own spot around them.
        placed.sort(key=lambda pair: not pair[1].group)
        regroup: dict[int, int] = {}
        shift: dict[int, float] = {}
        fresh_ids: set[int] = set()
        for lane, clip in placed:
            if clip.group and clip.group not in regroup:
                self._group_seq += 1
                regroup[clip.group] = self._group_seq
                shift[clip.group] = self._group_shift(
                    [(other_lane, c) for other_lane, c in placed if c.group == clip.group],
                    at,
                )
            start = (
                at + clip.start + shift[clip.group]
                if clip.group
                else self._free_start(lane, clip.length, at + clip.start, set())
            )
            fresh = self._new_clip(
                clip.path, clip.kind, start,
                clip.src_in, clip.length, clip.stream, clip.label, clip.source,
                group=regroup.get(clip.group, 0),
                # Its fades are part of what was copied: a pasted block that
                # came out full-volume where the original ramped would be a
                # difference nobody could see on the strip.
                fade_in=clip.fade_in, fade_out=clip.fade_out,
            )
            fresh.title = clip.title
            lane.clips.append(fresh)
            lane.clips.sort(key=lambda c: c.start)
            fresh_ids.add(fresh.cid)
        self._selected = fresh_ids
        self._after_lane_change(keep_all)
        self._announce_blocks()
        return True

    def _after_lane_change(self, keep_all: bool) -> None:
        """Tail of an undoable structural change: re-length, re-ask, announce."""
        # A view that was showing the whole edit should still be showing the
        # whole edit after a clip made it longer — otherwise dropping a music
        # bed past the end of the video hides the very thing just added.
        was_fit = (
            self._view[0] <= 1e-6
            and self._view[1] >= self._view_total() - 1e-6
        )
        self._lanes_changed(keep_all)
        if was_fit:
            self._view = (0.0, self._view_total())
        self._after_model_change()
        self.cutsChanged.emit()

    def _lanes_changed(self, keep_all: bool) -> None:
        """The lane stack is different: re-key holes, re-length, re-arm, re-header."""
        self.model.keep_holes(lane.key for lane in self._lanes)
        self._prune_selection()
        self.model.set_duration(self._total_duration())
        # Re-armed by KEY: the ids are positions, and "Add video track" or a
        # removed lane above an armed one renumbers everything under it —
        # re-arming the old ids lifted the neighbouring lane instead.
        if keep_all:
            self._arm_all()
        else:
            self.set_armed_lanes(
                index
                for index, lane in enumerate(self._lanes)
                if lane.key in self._armed_keys
            )
        self._sync_headers()
        # Keyed by AUDIO-LANE ordinal, so bound it by the audio lane count --
        # len(self._lanes) counts the video lanes too and leaves a removed
        # lane's trace behind to be handed to whatever lands on that ordinal.
        audio_lanes = sum(1 for lane in self._lanes if lane.kind == "audio")
        self._waves = {k: v for k, v in self._waves.items() if k < audio_lanes}
        self._cells = None
        self._gen_assets += 1
        self._gen_cuts += 1
        self._relayout_lanes()

    def _sync_headers(self) -> None:
        """Give every audio lane a header, every video lane none, ids in order."""
        for index, lane in enumerate(self._lanes):
            if lane.kind != "audio":
                self._drop_header(lane)
                continue
            if lane.header is None:
                lane.header = self._make_header(index, lane)
            lane.header.index = index
            lane.header.label.set_full(*self._lane_label(index, lane))
            lane.header.show()

    def _lane_label(self, index: int, lane: Lane) -> tuple[str, str]:
        """``(text, tooltip)`` for an audio lane header."""
        if not lane.clips:
            text = f"audio {self._audio_ordinal(index) + 1} · empty"
            return text, "Drop an audio file here"
        first = lane.clips[0]
        embedded = self._info is not None and first.path == self._info.path
        text = first.label if embedded else "♪ " + first.label
        if len(lane.clips) > 1:
            text = f"{text}  +{len(lane.clips) - 1}"
        tip = "\n".join(
            f"{format_tc(c.start, decimals=1)}  {c.label}" for c in lane.clips
        )
        return text, tip

    def _make_header(self, index: int, lane: Lane) -> _AudioLaneHeader:
        header = _AudioLaneHeader(index, self)
        header.apply_palette(self._pal)
        header.mute.setChecked(lane.muted)
        header.slider.setValue(int(lane.gain_db))
        header.muteChanged.connect(self._on_mute)
        header.gainChanged.connect(self._on_gain)
        header.headerClicked.connect(self._lane_clicked)
        header.show()
        return header

    def add_audio_file(self, info: AudioFileInfo) -> int:
        """A dropped file on a NEW audio lane at 0:00.  Returns the lane id (-1 if refused)."""
        return len(self._lanes) - 1 if self.add_clip(None, info, "place") else -1


    def wave_source(self, track: int) -> tuple[str, int, float] | None:
        """``(file, stream, file duration)`` the waveform for audio-lane ordinal
        ``track`` comes from — its FIRST block — or ``None`` for an empty lane.

        One asset per lane, and this is the block that defines it.  A block on
        the same lane from a different file or stream draws its outline and its
        name instead of somebody else's trace: a waveform under the wrong audio
        is worse than no waveform at all.
        """
        audio = [lane for lane in self._lanes if lane.kind == "audio"]
        if not 0 <= int(track) < len(audio) or not audio[int(track)].clips:
            return None
        first = audio[int(track)].clips[0]
        return (first.path, first.stream, first.source)

    def wave_request(self, track: int, vs: float, ve: float) -> tuple[float, float] | None:
        """SOURCE range the lane's waveform has to cover for the view ``[vs, ve]``.

        The union over every visible block on the lane that shares the lane's
        file and stream, mapped through each block's OWN placement — which is
        the whole point: once a block can be split and slid, ``edited_to_source``
        against the lane as a whole is no longer the same thing as the position
        inside the file.
        """
        source = self.wave_source(track)
        if source is None:
            return None
        path, stream, duration = source
        audio = [lane for lane in self._lanes if lane.kind == "audio"]
        lo, hi = float("inf"), float("-inf")
        for clip in audio[int(track)].clips:
            if clip.path != path or clip.stream != stream:
                continue
            # Both halves from the one walk: ``edited_to_source`` answers a
            # span END at a splice with the next keep's start (see
            # ``spans_for_source``), which asked for the whole cut as well.
            for e0, e1, t0, _t1 in self.model.spans_for_source(clip.start, clip.end):
                x0, x1 = max(vs, e0), min(ve, e1)
                if x1 - x0 > 1e-9:
                    lo = min(lo, clip.source_at(t0 + (x0 - e0)))
                    hi = max(hi, clip.source_at(t0 + (x1 - e0)))
        lo, hi = max(0.0, lo), min(hi, duration if duration > 0 else hi)
        return (lo, hi) if hi - lo > 1e-6 else None

    # ----------------------------------------------------------- render plan
    def render_plan(self) -> TimelinePlan:
        """The whole timeline as :mod:`ffcut.render` wants it — every lane,
        every clip, every hole, and the timeline's own duration."""
        video: list[VideoLane] = []
        audio: list[AudioLane] = []
        texts: list[TextClip] = []
        for lane in self._lanes:
            holes = self.model.holes(lane.key)
            if lane.kind == "video":
                titles = [c for c in lane.clips if c.title is not None]
                texts += [
                    TextClip(c.start, c.end, t.text, t.x, t.y, t.size, t.color)
                    for c in titles if (t := c.title) is not None
                ]
                footage = [c for c in lane.clips if c.title is None]
                if titles and not footage:
                    continue  # a text-only lane is an overlay, not a picture layer
                video.append(
                    VideoLane(
                        [VideoClip(c.path, c.start, c.src_in, c.src_out,
                                   c.fade_in, c.fade_out) for c in footage],
                        holes,
                    )
                )
            else:
                audio.append(
                    AudioLane(
                        [AudioClip(c.path, c.stream, c.start, c.src_in, c.src_out,
                                   c.fade_in, c.fade_out)
                         for c in lane.clips],
                        lane.gain_db,
                        lane.muted,
                        holes,
                    )
                )
        return TimelinePlan(max(self.model.duration, 0.0), video, audio, texts)

    def preview_plan(self) -> PreviewPlan:
        """What the preview must play, in EDITED time.

        Video entries follow the lane priority stack (a hole in lane 0 shows
        lane 1 through it); audio entries are every unmuted clip minus its
        lane's holes.  Each entry is one contiguous run, so a cut, a lift or a
        lane above simply yields more entries rather than a gap to guess at.
        """
        plan = self.render_plan()
        keeps = self.model.keep_segments()
        video: list[PreviewVideo] = []
        for lo, hi, clip in visible_spans(plan):
            if clip is None:
                continue
            for a, b in _overlaps((lo, hi), keeps):
                edited = self.model.source_to_edited(a)
                if edited is None:
                    continue
                video.append(
                    PreviewVideo(
                        clip.path,
                        clip.source_at(a),
                        clip.source_at(b),
                        edited,
                    )
                )
        audio: list[PreviewAudio] = []
        for lane in self._lanes:
            if lane.kind != "audio":
                continue
            holes = self.model.holes(lane.key)
            for clip in lane.clips:
                for lo, hi in _subtract((clip.start, clip.end), holes):
                    for a, b in _overlaps((lo, hi), keeps):
                        edited = self.model.source_to_edited(a)
                        if edited is None:
                            continue
                        audio.append(
                            PreviewAudio(
                                clip.path,
                                clip.stream,
                                lane.gain_db,
                                lane.muted,
                                edited,
                                clip.source_at(a),
                                clip.source_at(b),
                                lane.key,
                            )
                        )
        video.sort(key=lambda e: e.edited_start)
        audio.sort(key=lambda e: e.edited_start)
        return PreviewPlan(self.model.edited_duration(), video, audio)

    # -- lane id <-> audio-track ordinal ------------------------------------
    # Two index spaces meet here, deliberately:
    #   * LANE ids (video lanes then audio lanes) are what the lane API, the
    #     armed set and ``armedChanged`` speak in;
    #   * AUDIO-TRACK ordinals (audio lanes only, 0-based) are what the
    #     ``track*`` signals, ``waveformNeeded``/``set_waveform`` and
    #     ``wave_source`` speak in — the app's waveform router is built on them.
    # Everything crossing between them goes through these two helpers.
    def _audio_ordinal(self, lane: int) -> int:
        return sum(1 for other in self._lanes[: int(lane)] if other.kind == "audio")

    def _on_mute(self, lane: int, muted: bool) -> None:
        if 0 <= lane < len(self._lanes):
            self._lanes[lane].muted = bool(muted)
            self.trackMuteChanged.emit(self._audio_ordinal(lane), muted)
        self.update()

    def _on_gain(self, lane: int, gain_db: float) -> None:
        if 0 <= lane < len(self._lanes):
            self._lanes[lane].gain_db = float(gain_db)
            self.trackGainChanged.emit(self._audio_ordinal(lane), gain_db)
        self.update()

    # ================================================================ armed
    def lane_count(self) -> int:
        """Number of lanes, video included."""
        return len(self._lanes)

    def armed_lanes(self) -> frozenset[int]:
        """Lane ids an edit currently applies to (0 = video, 1 + track)."""
        return frozenset(self._armed)

    def is_armed(self, lane: int) -> bool:
        return int(lane) in self._armed

    def all_armed(self) -> bool:
        """True when an edit would ripple the whole timeline."""
        return len(self._armed) >= self.lane_count()

    def set_armed_lanes(self, lanes: Iterable[int]) -> None:
        """Arm exactly ``lanes``.  An empty/invalid set falls back to all lanes.

        There is no useful "nothing armed" state — every edit would be a no-op
        — so it is not reachable, by this path or by Ctrl+click.
        """
        valid = {int(lane) for lane in lanes if 0 <= int(lane) < self.lane_count()}
        self._apply_armed(valid or set(range(self.lane_count())))

    def _arm_all(self) -> None:
        self._apply_armed(set(range(self.lane_count())))

    def _apply_armed(self, lanes: set[int]) -> None:
        self._armed_keys = {
            self._lanes[lane].key for lane in lanes if 0 <= lane < len(self._lanes)
        }
        if lanes == self._armed:
            return
        self._armed = lanes
        self.armedChanged.emit(frozenset(lanes))
        self.update()

    def _flagged(self, lane: int) -> bool:
        """Whether ``lane`` wears the armed badge: only when a SUBSET is armed."""
        return int(lane) in self._armed and not self.all_armed()

    def armed_names(self) -> list[str]:
        """Short names of the armed lanes, in lane order, for a status readout."""
        primary = self._info.path if self._info else None
        names: list[str] = []
        videos = 0
        audios = 0
        for lane in self._lanes:
            if lane.kind == "video":
                videos += 1
                names.append("video" if videos == 1 else f"video {videos}")
            else:
                audios += 1
                first = lane.clips[0] if lane.clips else None
                names.append(
                    f"audio {audios}"
                    if first is None or first.path == primary
                    else os.path.basename(first.path)
                )
        return [names[lane] for lane in sorted(self._armed) if lane < len(names)]

    def _lane_clicked(self, lane: int, ctrl: bool) -> None:
        """Header press: exclusive select, or Ctrl+click to toggle."""
        if not 0 <= lane < self.lane_count():
            return
        if ctrl:
            lanes = set(self._armed)
            if lane in lanes:
                lanes.discard(lane)
                if not lanes:  # never leave the user with a dead edit
                    return
            else:
                lanes.add(lane)
            self._apply_armed(lanes)
        else:
            self._apply_armed({lane})
        self.setFocus(Qt.FocusReason.MouseFocusReason)

    # ============================================================= playhead
    def playhead(self) -> float:
        return self._playhead

    def set_playhead(self, t_edited: float) -> None:
        """Move the playhead WITHOUT emitting (programmatic / player-driven)."""
        self._apply_playhead(t_edited, emit=False)

    def _apply_playhead(self, t_edited: float, *, emit: bool) -> None:
        value = max(0.0, min(self.model.edited_duration(), float(t_edited)))
        if abs(value - self._playhead) < 1e-9:
            return
        self._playhead = value
        if self._drag is None:
            self._follow(value)
        if emit:
            self.playheadMoved.emit(value)
        self.update()

    @property
    def is_scrubbing(self) -> bool:
        """True while a gesture owns the playhead/selection (app: skip updates)."""
        return self._drag in ("scrub", "select", "mark-in", "mark-out")

    # ============================================================ selection
    def selection(self) -> Range | None:
        """Current selection in edited time, or None.

        A lone in-point runs to the end of the edit, a lone out-point from the
        start — so Delete always has something coherent to remove.
        """
        total = self.model.edited_duration()
        if total <= 0.0 or (self._mark_in is None and self._mark_out is None):
            return None
        lo = 0.0 if self._mark_in is None else self._mark_in
        hi = total if self._mark_out is None else self._mark_out
        lo = max(0.0, min(total, lo))
        hi = max(0.0, min(total, hi))
        if hi - lo <= 1e-6:
            return None
        return (lo, hi)

    def set_mark_in(self) -> None:
        """In-point at the playhead.  Preserves the out-point (swaps if later)."""
        if self.model.edited_duration() <= 0.0:
            return
        t = self._playhead
        if self._mark_out is not None and self._mark_out < t:
            self._mark_in, self._mark_out = self._mark_out, t
        else:
            self._mark_in = t
        self._sel_changed()

    def set_mark_out(self) -> None:
        """Out-point at the playhead.  Preserves the in-point (swaps if earlier)."""
        if self.model.edited_duration() <= 0.0:
            return
        t = self._playhead
        if self._mark_in is not None and self._mark_in > t:
            self._mark_in, self._mark_out = t, self._mark_in
        else:
            self._mark_out = t
        self._sel_changed()

    def clear_selection(self) -> None:
        self._mark_in = self._mark_out = None
        self._sel_changed()

    def select_all(self) -> None:
        if self.model.edited_duration() > 0.0:
            self._set_marks(0.0, self.model.edited_duration())

    def _set_marks(self, a: float, b: float) -> None:
        total = self.model.edited_duration()
        lo, hi = (a, b) if a <= b else (b, a)
        self._mark_in = max(0.0, min(total, lo))
        self._mark_out = max(0.0, min(total, hi))
        self._sel_changed()

    def _sel_changed(self) -> None:
        current = self.selection()
        if current != self._emitted_sel:
            self._emitted_sel = current
            self.selectionChanged.emit(current)
        self.update()

    # ================================================================= cuts
    def cut_selection(self) -> bool:
        """Remove the selected span.  True when the model actually changed.

        Which kind of removal depends on what is armed:

        * **every lane armed -> ripple**: the range is deleted and the timeline
          collapses, exactly as it always has;
        * **a subset armed -> lift**: the range is blanked on those lanes only
          and the timeline keeps its length, so the other lanes stay in sync.

        The selection is edited-time and therefore gapless; it can still span
        earlier splices, so it maps to one or more SOURCE ranges that are
        removed as a single undo step either way.  A ripple clears the
        selection (the material under it is gone); a lift keeps it, so you can
        arm another lane and lift again.
        """
        sel = self.selection()
        if sel is None:
            return False
        ranges = self.model.source_ranges_for_edited(*sel)
        if not ranges:
            return False
        if self.all_armed():
            if not self.model.add_cuts(ranges, self._ui_state()):
                return False
            self._mark_in = self._mark_out = None
            self._playhead = min(sel[0], self.model.edited_duration())
        else:
            # Holes are stored against the lane's STABLE key, not its position,
            # so inserting a lane above one later cannot move its holes.
            per_lane = {
                self._lanes[lane].key: list(ranges)
                for lane in self._armed
                if 0 <= lane < len(self._lanes)
            }
            if not self.model.add_holes(per_lane, self._ui_state()):
                return False
        self._after_model_change()
        self.cutsChanged.emit()
        return True

    def undo(self) -> bool:
        if not self.model.can_undo:
            return False
        self.model.undo(self._ui_state())
        self._restore_ui(self.model.restored_ui)
        self._after_model_change()
        self.cutsChanged.emit()
        return True

    def redo(self) -> bool:
        if not self.model.can_redo:
            return False
        self.model.redo(self._ui_state())
        self._restore_ui(self.model.restored_ui)
        self._after_model_change()
        self.cutsChanged.emit()
        return True

    def can_undo(self) -> bool:
        return self.model.can_undo

    def can_redo(self) -> bool:
        return self.model.can_redo

    def edited_duration(self) -> float:
        return self.model.edited_duration()

    def keep_segments_source(self) -> list[Segment]:
        """SOURCE keep-ranges for a full render."""
        return [Segment(start, end) for start, end in self.model.keep_segments()]

    def selection_segments_source(self) -> list[Segment] | None:
        """SOURCE ranges under the selection only, cuts honoured.  None if none."""
        sel = self.selection()
        if sel is None:
            return None
        ranges = self.model.source_ranges_for_edited(*sel)
        return [Segment(start, end) for start, end in ranges] or None

    def edited_to_source(self, t_edited: float) -> float:
        return self.model.edited_to_source(t_edited)

    def source_to_edited(self, t_source: float) -> float | None:
        return self.model.source_to_edited(t_source)

    def _ui_state(self) -> dict:
        """The UI half of a history entry: playhead, view AND the lane stack.

        The lanes ride along because adding or removing a clip has to be
        undoable too, and one history is the only way Ctrl+Z can stay
        predictable — see :meth:`CutList.push_state`.
        """
        return {
            "playhead": self._playhead,
            "view": self._view,
            "lanes": [lane.snapshot() for lane in self._lanes],
        }

    def _restore_ui(self, state: dict | None) -> None:
        """Put the playhead/view/lanes back where the restored edit left them."""
        if not state:
            return
        lanes = state.get("lanes")
        if lanes is not None:
            self._restore_lanes(lanes)
        self._playhead = float(state.get("playhead", self._playhead))
        view = state.get("view")
        if view:
            self._view = (float(view[0]), float(view[1]))

    def _restore_lanes(self, snapshot: list[dict]) -> None:
        """Rebuild the lane stack from a history entry, keys and all."""
        keep_all = self.all_armed()
        # Mute and gain are not undo steps, so an undo must not walk them back
        # either: a lane that is still here keeps the mix it has NOW.  Only a
        # lane the undo brings back takes the mix it had when it went.
        mix = {lane.key: (lane.muted, lane.gain_db) for lane in self._lanes}
        if [lane.key for lane in self._lanes] != [s["key"] for s in snapshot]:
            self._clear_lanes()
            self._lanes = [Lane(s["key"], s["kind"]) for s in snapshot]
            self._lane_seq = max([self._lane_seq] + [lane.key for lane in self._lanes])
        for lane, s in zip(self._lanes, snapshot):
            lane.clips = [c.copy() for c in s["clips"]]
            lane.muted, lane.gain_db = mix.get(lane.key, (s["muted"], s["gain_db"]))
            # Ids never go backwards: a redo must not hand a fresh block the id
            # of one an undo has just brought back.
            for clip in lane.clips:
                self._clip_seq = max(self._clip_seq, clip.cid)
                self._group_seq = max(self._group_seq, clip.group)
            if lane.header is not None:  # the widget must show the restored mix
                lane.header.mute.setChecked(lane.muted)
                lane.header.slider.setValue(int(lane.gain_db))
        self._lanes_changed(keep_all)

    def _after_model_change(self) -> None:
        """Clamp everything to the new edit length and re-ask for assets."""
        total = self.model.edited_duration()
        self._cells = None
        self._gen_cuts += 1
        self._playhead = max(0.0, min(total, self._playhead))
        if self._mark_in is not None:
            self._mark_in = max(0.0, min(total, self._mark_in))
        if self._mark_out is not None:
            self._mark_out = max(0.0, min(total, self._mark_out))
        self._set_view(*self._view)
        self._asked = None
        self._req.start()
        self._sel_changed()

    # ================================================================= view
    def view_range(self) -> Range:
        return self._view

    def set_view_range(self, start: float, end: float) -> None:
        self._set_view(start, end)

    def zoom_to_fit(self) -> None:
        self._set_view(0.0, self._view_total())

    def _view_total(self) -> float:
        """The edit plus empty room past its end, as in Vegas/Premiere, so a block
        can be dragged (or a file dropped) beyond the last one."""
        total = self.model.edited_duration()
        return total + max(RIGHT_ROOM_MIN, total * RIGHT_ROOM_FRAC) if total > 0.0 else 0.0

    def _max_span(self) -> float:
        return max(MIN_SPAN, self._view_total())

    def _set_view(self, start: float, end: float) -> None:
        """Clamp a candidate range into something drawable, then adopt it."""
        total = self._max_span()
        span = max(min(MIN_SPAN, total), min(total, float(end) - float(start)))
        start = max(0.0, min(total - span, float(start)))
        view = (start, start + span)
        if view != self._view:
            self._view = view
            self._req.start()
            self.update()
        # Always re-sync, even when the candidate clamped to the view we already
        # had: the scrollbar may be the thing that proposed it.
        self._sync_scrollbar()

    def _sync_scrollbar(self) -> None:
        """Push the view onto the scrollbar, with its signals muted.

        Value is ``round(view_start * SCROLL_SCALE)``, so a user gesture that
        set the bar to ``v`` comes straight back as ``v``: the round trip is
        exact and there is no feedback loop and no drift.
        """
        bar = self._hbar
        total = self._view_total()
        start, end = self._view
        span = end - start
        blocked = bar.blockSignals(True)
        try:
            if total <= 0.0 or span >= total - 1e-9:
                bar.setRange(0, 0)
                bar.setPageStep(1)
                bar.setValue(0)
                bar.setEnabled(False)
                return
            bar.setEnabled(True)
            bar.setRange(0, max(1, round((total - span) * SCROLL_SCALE)))
            bar.setPageStep(max(1, round(span * SCROLL_SCALE)))
            bar.setSingleStep(max(1, round(span * PAN_FRACTION * SCROLL_SCALE)))
            bar.setValue(max(0, min(bar.maximum(), round(start * SCROLL_SCALE))))
        finally:
            bar.blockSignals(blocked)

    def _on_scroll(self, value: int) -> None:
        """The user dragged the bar: pan without changing the span."""
        start, end = self._view
        span = end - start
        target = value / SCROLL_SCALE
        self._set_view(target, target + span)

    def _zoom_at(self, x: float, factor: float) -> None:
        """Zoom so the frame under ``x`` stays under ``x`` (request 3)."""
        track = self._track_rect()
        if track.width() <= 0:
            return
        x = max(track.left(), min(track.right(), x))
        anchor = self._t_for(x)
        frac = (x - track.left()) / track.width()
        vs, ve = self._view
        total = self._max_span()
        span = max(min(MIN_SPAN, total), min(total, (ve - vs) / factor))
        self._set_view(anchor - frac * span, anchor - frac * span + span)

    def _follow(self, t: float) -> None:
        """Keep the playhead on screen while zoomed in."""
        vs, ve = self._view
        if vs - 1e-9 <= t <= ve + 1e-9:
            return
        span = ve - vs
        self._set_view(t - span * 0.1, t + span * 0.9)

    def _emit_asset_requests(self) -> None:
        """Ask the app for assets for the visible range (debounced)."""
        if not self.has_media() or self.model.edited_duration() <= 0.0:
            return
        vs, ve = self._view
        count = self._thumb_count()
        want = (round(vs, 3), round(ve, 3), count)
        if want == self._asked:
            return
        self._asked = want
        for path, times in self._thumb_requests(vs, ve, count).items():
            self.thumbsNeeded.emit(path, times)
        for track in range(sum(1 for lane in self._lanes if lane.kind == "audio")):
            self.waveformNeeded.emit(track, vs, ve)

    def _thumb_requests(
        self, vs: float, ve: float, count: int
    ) -> dict[str, list[float]]:
        """SOURCE timestamps to fetch, per FILE, for the view ``[vs, ve]``.

        Sampled per CLIP, not once across the view, for two reasons: a clip on
        a lower video lane draws its whole length in its own row whether or not
        the lane above lets it through, and a clip from another file needs
        timestamps in *that* file — which only this widget can work out, since
        it owns both the clip's placement and the cut mapping.

        Several clips of one file share a request; ``_thumb_cells`` keeps each
        clip to the frames that fall inside it.
        """
        out: dict[str, list[float]] = {}
        span = max(1e-9, ve - vs)
        for lane in self._lanes:
            if lane.kind != "video":
                continue
            for clip in lane.clips:
                visible = _overlaps(
                    (vs, ve), self.model.edited_spans_for_source(clip.start, clip.end)
                )
                width = sum(hi - lo for lo, hi in visible)
                if width <= 1e-9 or clip.title is not None:
                    continue
                wanted = max(2, round(count * width / span))
                times = out.setdefault(clip.path, [])
                for edited in _sample(visible, wanted):
                    source = clip.source_at(self.model.edited_to_source(edited))
                    times.append(round(max(0.0, source), 3))
        return {path: sorted(set(times)) for path, times in out.items() if times}

    def _thumb_count(self) -> int:
        cell = max(80.0, self._lane_rect(0).height() * 16.0 / 9.0 * 0.8)
        width = max(1.0, self._track_rect().width())
        return max(MIN_THUMBS, min(MAX_THUMBS, round(width / cell) or MIN_THUMBS))

    # ============================================================= geometry
    def _base_heights(self) -> list[int]:
        base = [
            VIDEO_LANE_H if lane.kind == "video" else AUDIO_LANE_H
            for lane in self._lanes
        ]
        return base or [VIDEO_LANE_H]

    def _chrome_height(self) -> int:
        """Minimum height: ruler, base-size lanes, scrollbar, padding."""
        base = self._base_heights()
        lanes = sum(base) + (len(base) - 1) * LANE_GAP
        return PAD + RULER_H + 2 + lanes + SCROLL_GAP + SCROLL_H + PAD

    def _lane_heights(self) -> list[int]:
        """Base lane heights plus an even share of whatever the splitter gave
        us beyond the minimum: a taller pane means taller lanes, not dead
        space under them."""
        base = self._base_heights()
        extra = max(0, self.height() - self._chrome_height()) // len(base)
        return [h + extra for h in base]

    def _relayout_lanes(self) -> None:
        """Lane count changed: re-floor the widget, then place the headers."""
        self.setMinimumHeight(self._chrome_height())
        self._place_headers()
        self.updateGeometry()
        self.update()

    def _place_headers(self) -> None:
        for index, lane in enumerate(self._lanes):
            if lane.header is None:
                continue
            rect = self._lane_rect(index)
            lane.header.setGeometry(
                PAD, int(rect.top()), GUTTER_W - 2 * PAD, int(rect.height())
            )
        # The bar spans the TRACK area only: the gutter is fixed furniture and
        # must never slide with it.
        bar = self._scroll_rect()
        self._hbar.setGeometry(
            int(bar.left()), int(bar.top()), int(bar.width()), int(bar.height())
        )

    def _scroll_rect(self) -> QRectF:
        lanes = self._lanes_rect()
        return QRectF(
            lanes.left(),
            lanes.bottom() + SCROLL_GAP,
            max(1.0, lanes.width()),
            SCROLL_H,
        )

    def _lane_rect(self, lane: int) -> QRectF:
        """Content rect of a lane (lane 0 = video).  Excludes the gutter."""
        heights = self._lane_heights()
        top = PAD + RULER_H + 2 + sum(heights[:lane]) + lane * LANE_GAP
        track = self._track_rect()
        return QRectF(track.left(), top, track.width(), heights[lane])

    def _lanes_rect(self) -> QRectF:
        first = self._lane_rect(0)
        last = self._lane_rect(max(0, len(self._lanes) - 1))
        return QRectF(
            first.left(), first.top(), first.width(), last.bottom() - first.top()
        )

    def _lane_at(self, y: float) -> int | None:
        """Which lane a vertical position lands in; None above or below them.

        Lane gaps round to the lane above, so a press never falls in a crack.
        """
        for lane in range(max(1, self.lane_count())):
            rect = self._lane_rect(lane)
            if rect.top() - LANE_GAP <= y <= rect.bottom() + LANE_GAP:
                return lane
        return None

    def _track_rect(self) -> QRectF:
        return QRectF(
            GUTTER_W, PAD, max(1.0, self.width() - GUTTER_W - PAD), self.height() - 2 * PAD
        )

    def _ruler_rect(self) -> QRectF:
        track = self._track_rect()
        return QRectF(track.left(), PAD, track.width(), RULER_H)

    def _x_for(self, t: float) -> float:
        track = self._track_rect()
        vs, ve = self._view
        span = max(1e-9, ve - vs)
        return track.left() + (float(t) - vs) / span * track.width()

    def _t_for(self, x: float) -> float:
        track = self._track_rect()
        vs, ve = self._view
        if track.width() <= 0:
            return vs
        return vs + (float(x) - track.left()) / track.width() * (ve - vs)

    def _t_for_clamped(self, x: float) -> float:
        return max(0.0, min(self.model.edited_duration(), self._t_for(x)))

    def _source_for_x(self, x: float) -> float:
        """TIMELINE time under ``x``, extended PAST the end of the edit.

        ``_t_for_clamped`` stops at the edit's end and ``edited_to_source``
        stops at the last keep, which is right for a seek and wrong for a trim:
        the whole point of dragging the last block's right edge out is to go
        somewhere the timeline does not reach yet.  Past the end there are no
        more cuts, so edited time and timeline time run 1:1 and the extension
        is exact rather than a guess.
        """
        total = self.model.edited_duration()
        t = self._t_for(x)
        if t <= total:
            return self.model.edited_to_source(max(0.0, t))
        return self.model.edited_to_source(total) + (t - total)

    def resizeEvent(self, event: object) -> None:  # noqa: N802
        super().resizeEvent(event)  # type: ignore[arg-type]
        self._place_headers()
        self._asked = None  # the thumb count scales with width
        self._req.start()

    def sizeHint(self) -> QSize:  # noqa: N802
        return QSize(900, self._chrome_height())

    # ================================================================ paint
    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802
        self._pal = _palette(theme_variant(QApplication.instance()))
        painter = QPainter(self)
        painter.setRenderHints(
            QPainter.RenderHint.Antialiasing
            | QPainter.RenderHint.TextAntialiasing
            | QPainter.RenderHint.SmoothPixmapTransform
        )
        self._paint_gutter(painter)
        if not self.has_media() or self.model.edited_duration() <= 0.0:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(self._pal.ruler_bg)
            painter.drawRoundedRect(self._ruler_rect(), 3, 3)
            self._paint_empty(painter)
            painter.end()
            return

        # Ruler + lanes + splices depend only on the view and the edit, so they
        # are composed once and blitted; only the overlays follow the pointer.
        # Both go under one clip: the gutter has already been painted and it is
        # fixed furniture, so nothing that scrolls may land on it.  The overlays
        # each clamp themselves, but "clamped" has been wrong before — measured,
        # a playhead just off the left edge of the view still draws its line and
        # marker 5 px into the gutter (``_paint_playhead`` deliberately paints
        # up to 8 px outside, so a playhead about to scroll in is visible).
        painter.save()
        self._clip_to(painter, self._track_rect())
        painter.drawPixmap(0, 0, self._content())
        self._paint_selection(painter)
        self._paint_hover(painter)
        self._paint_playhead(painter)
        painter.restore()
        painter.end()

    def _content(self) -> QPixmap:
        """Cached composite of everything that does not move with the playhead."""
        ratio = self.devicePixelRatioF()
        key = (
            self.width(), self.height(), self._view, ratio,
            self._gen_cuts, self._gen_assets, self._pal.mode,
            tuple(lane.muted for lane in self._lanes),
            frozenset(self._armed),  # armed lanes are framed differently
            frozenset(self._selected),  # selected blocks are painted in here
            # Fades are painted in here too, and ``set_fade`` is a plain model
            # setter that bumps no generation counter — without this a fade set
            # through the API would render and not draw, which is this
            # project's whole failure family pointed at its own strip.
            tuple(
                (c.cid, c.fade_in, c.fade_out)
                for lane in self._lanes
                for c in lane.clips
            ),
        )
        if self._content_key == key and self._content_pm is not None:
            return self._content_pm
        pixmap = QPixmap(round(self.width() * ratio), round(self.height() * ratio))
        pixmap.setDevicePixelRatio(ratio)
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHints(
            QPainter.RenderHint.Antialiasing
            | QPainter.RenderHint.TextAntialiasing
            | QPainter.RenderHint.SmoothPixmapTransform
        )
        # Belt and braces for the gutter.  ``paintEvent`` blits this pixmap OVER
        # a gutter it has already painted, so ANYTHING a lane painter lets
        # escape lands on top of the track labels — which is exactly how the
        # thumbnail flood showed up.  Nothing composed here belongs left of the
        # track area, so the whole composite is clipped to it at the source and
        # a future escape is a missing pixel instead of a ruined gutter.
        self._clip_to(painter, self._track_rect())
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(self._pal.ruler_bg)
        painter.drawRoundedRect(self._ruler_rect(), 3, 3)
        for index, lane in enumerate(self._lanes):
            if lane.kind == "video":
                self._paint_video_lane(painter, index)
            else:
                self._paint_audio_lane(painter, index)
        self._paint_splices(painter)
        self._paint_ruler(painter)
        painter.end()
        self._content_pm, self._content_key = pixmap, key
        return pixmap

    def _clip_to(self, painter: QPainter, rect: QRectF) -> None:
        """Narrow ``painter``'s clip to ``rect``: INTERSECT, never replace.

        ``QPainter.setClipRect`` defaults to ``ReplaceClip``, and that default
        is what flooded the gutter with thumbnails.  ``_paint_video_lane``
        clipped to the lane (which excludes the gutter) and then clipped again
        to each thumbnail cell — replacing the lane clip with a box that, zoomed
        in, starts thousands of pixels to the LEFT of the track area.  The cell
        straddling the gutter edge then painted across the whole gutter; worst
        in the moment after a zoom, when the stale wide cells are still on
        screen and one of them spans the entire view.

        Every clip in this widget is a NARROWING of the one outside it, so they
        all go through here and none of them can widen what it was given.
        """
        painter.setClipRect(rect, Qt.ClipOperation.IntersectClip)

    def _paint_gutter(self, painter: QPainter) -> None:
        pal = self._pal
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(pal.gutter)
        painter.drawRect(QRectF(0, 0, GUTTER_W, self.height()))
        # Everything armed is the normal state and wears no badge.  A SUBSET is
        # what needs flagging: those lanes are tinted and lit down the left
        # edge, so which tracks the next edit will hit is readable at a glance.
        for lane in range(self.lane_count()):
            if self._flagged(lane):
                rect = self._lane_rect(lane)
                painter.setBrush(pal.armed_tint)
                painter.drawRect(QRectF(0, rect.top(), GUTTER_W - 1, rect.height()))
                painter.setBrush(pal.armed)
                painter.drawRect(QRectF(0, rect.top(), 3, rect.height()))
        painter.setPen(QPen(pal.border, 1.0))
        painter.drawLine(QPointF(GUTTER_W - 0.5, 0), QPointF(GUTTER_W - 0.5, self.height()))
        font = QFont(painter.font())
        font.setPointSizeF(7.6)
        painter.setFont(font)
        painter.setPen(pal.tick_text)
        ruler = QRectF(PAD, PAD, GUTTER_W - 2 * PAD, RULER_H)
        painter.drawText(
            ruler,
            int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
            f"EDIT {format_tc(self.model.edited_duration(), decimals=1)}",
        )
        painter.drawText(
            ruler,
            int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter),
            "RIPPLE" if self.all_armed() else "LIFT",
        )
        # Video lanes have no child header widget — their gutter row is drawn
        # here, so a second video lane needs no new widget class.
        font.setBold(True)
        painter.setFont(font)
        seen = 0
        for index, lane in enumerate(self._lanes):
            if lane.kind != "video":
                continue
            seen += 1
            rect = self._lane_rect(index)
            flag = pal.armed.lighter(135) if pal.mode == "dark" else pal.armed.darker(125)
            painter.setPen(flag if self._flagged(index) else pal.label)
            if seen == 1:
                label = "VIDEO"
                if self._info is not None:
                    label = f"VIDEO  {self._info.width}×{self._info.height}"
            else:
                label = f"VIDEO {seen}"
            painter.drawText(
                QRectF(PAD + 3, rect.top(), GUTTER_W - 2 * PAD - 3, 16),
                int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
                label,
            )
            if not lane.clips:
                painter.setPen(pal.empty_text)
                painter.drawText(
                    QRectF(PAD + 3, rect.top() + 15, GUTTER_W - 2 * PAD - 3, 14),
                    int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
                    "drop a clip here",
                )

    def _paint_empty(self, painter: QPainter) -> None:
        pal = self._pal
        rect = self._lanes_rect()
        painter.setPen(QPen(pal.border, 1.4, Qt.PenStyle.DashLine))
        painter.setBrush(pal.bg)
        painter.drawRoundedRect(rect.adjusted(0.5, 0.5, -0.5, -0.5), 5, 5)
        painter.setPen(pal.empty_text)
        if self.has_media():
            message = "every frame is cut   ·   Ctrl+Z to undo"
        elif self._info is not None:
            # A file IS loaded; the timeline just has nothing left on it — the
            # last block was deleted.  Pointing at File ▸ Open here would send
            # the user to reload the capture they have not actually lost.
            message = "nothing left on the timeline   ·   Ctrl+Z to undo"
        else:
            message = "Drop a video here   ·   File ▸ Open…"
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, message)

    def _paint_video_lane(self, painter: QPainter, index: int) -> None:
        """Paint one video lane: its clips, then its lifted holes.

        Clips are drawn where they sit on the timeline, not stretched across
        it, so an appended clip and a gap are both visible.  Every clip gets
        real frames — ``_thumb_requests`` asks per file — and a clip whose
        thumbnails have not arrived yet gets a labelled block, which is honest
        about what is there without pretending to a frame it has not decoded.
        """
        pal = self._pal
        lane_model = self._lanes[index]
        lane = self._lane_rect(index)
        painter.save()
        self._clip_to(painter, lane)
        painter.setPen(Qt.PenStyle.NoPen)
        grad = QLinearGradient(lane.topLeft(), lane.bottomLeft())
        grad.setColorAt(0.0, pal.video_top)
        grad.setColorAt(1.0, pal.video_bottom)
        painter.setBrush(grad)
        painter.drawRect(lane)
        if not lane_model.clips:
            painter.setPen(pal.empty_text)
            painter.drawText(
                lane, Qt.AlignmentFlag.AlignCenter, "empty — drop a video here"
            )
        for clip, box, _t0, _t1 in self._clip_boxes(lane_model, lane):
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(pal.bg)
            painter.drawRect(box)
            cells = self._thumb_cells(clip) if clip.title is None else None
            if clip.title is not None:
                self._paint_clip_block(painter, box, "T  " + clip.title.text.replace("\n", " "))
            elif cells:
                painter.save()
                self._clip_to(painter, box)
                for left, right, pix in cells:
                    cx0, cx1 = self._x_for(left), self._x_for(right)
                    if cx1 < box.left() - 1:
                        continue
                    if cx0 > box.right() + 1:
                        break
                    target = QRectF(cx0, box.top(), max(1.0, cx1 - cx0), box.height())
                    painter.drawPixmap(target, pix, self._cover_source(pix, target))
                painter.restore()
            elif clip.path not in self._thumbs:
                painter.setPen(pal.empty_text)
                painter.drawText(box, Qt.AlignmentFlag.AlignCenter, "generating thumbnails…")
            else:
                self._paint_clip_block(painter, box, clip.label)
            if len(lane_model.clips) > 1 or index > 0:
                self._outline(painter, box)
            self._paint_block_state(painter, clip, box)
        self._paint_holes(painter, lane_model.key, lane)
        painter.restore()
        self._frame(painter, lane, armed=self._flagged(index))

    def _clip_boxes(
        self, lane_model: Lane, lane: QRectF
    ) -> Iterable[tuple[Clip, QRectF, float, float]]:
        """Each visible ``(clip, box, t0, t1)`` on ``lane``, one box per edited
        span — a cut inside a clip splits it, exactly as it splits the render.

        ``t0``/``t1`` are the box's TIMELINE edges, carried along rather than
        re-derived from its pixels: a box can start off-screen to the left, and
        ``edited_to_source`` answers a splice with the next keep's start, so
        both routes back are wrong exactly where it matters.
        """
        for clip in lane_model.clips:
            for ed0, ed1, t0, t1 in self.model.spans_for_source(clip.start, clip.end):
                x0, x1 = self._x_for(ed0), self._x_for(ed1)
                if x1 >= lane.left() - 1 and x0 <= lane.right() + 1:
                    box = QRectF(x0, lane.top(), max(1.0, x1 - x0), lane.height())
                    yield clip, box, t0, t1

    def _outline(self, painter: QPainter, box: QRectF) -> None:
        """Hairline around one clip, so several clips on a lane read as several."""
        painter.setPen(QPen(self._pal.border.lighter(130), 1.0))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(box.adjusted(0.5, 0.5, -0.5, -0.5))

    def _paint_block_state(
        self, painter: QPainter, clip: Clip, box: QRectF
    ) -> None:
        """Selection and group marks on one block.

        Selected: a heavy accent border and a wash, so a multi-block selection
        reads at a glance.  Grouped: a solid accent bar along the bottom edge,
        which is what says "this and the other one move together" without
        needing a legend.
        """
        pal = self._pal
        if clip.cid in self._selected:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(pal.sel_fill)
            painter.drawRect(box)
            # Two borders, not one: a bright line alone disappears against a
            # blue waveform, a dark line alone disappears against a dark
            # thumbnail.  One inside the other reads on anything.
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(pal.block_edge, 1.0))
            painter.drawRect(box.adjusted(0.5, 0.5, -0.5, -0.5))
            painter.setPen(QPen(pal.block_sel, 2.0))
            painter.drawRect(box.adjusted(2.0, 2.0, -2.0, -2.0))
        if clip.group and box.width() > 8.0:
            # Last, and inset: the group bar has to survive the selection
            # border rather than be painted under it.
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(pal.group)
            painter.drawRect(
                QRectF(box.left() + 4.0, box.bottom() - 6.0, box.width() - 8.0, 3.0)
            )
        self._paint_block_fades(painter, clip, box)

    def _fade_span_x(self, clip: Clip, which: str) -> Range | None:
        """``(x0, x1)`` the fade ramp covers on screen, or ``None``.

        Through the cut mapping like :meth:`block_edges`, so a fade with a cut
        inside it is drawn over the pixels that survived the cut — which is
        also exactly what the render does with it, since both read the clip's
        own source time.
        """
        length = clip.fade_in if which == "in" else clip.fade_out
        if length <= 1e-6:
            return None
        a, b = (
            (clip.start, clip.start + length)
            if which == "in"
            else (clip.end - length, clip.end)
        )
        spans = self.model.edited_spans_for_source(a, b)
        if not spans:
            return None
        return (self._x_for(spans[0][0]), self._x_for(spans[-1][1]))

    def _paint_block_fades(self, painter: QPainter, clip: Clip, box: QRectF) -> None:
        """The block's fade ramps, drawn the way Vegas draws them.

        A wedge of shade whose height IS the attenuation — full height where
        the material is silent/black, nothing where it is at full level — with
        the ramp itself picked out along the hypotenuse.  Drawn per BOX and
        clipped to it: a clip that a cut has split into several boxes gets one
        wedge spanning them all, and because the boxes are disjoint the
        translucent wash is never laid down twice over the same pixel.

        A SELECTED block with no fade shows a small grip in each empty corner.
        That is the only discoverability the gesture has — the cursor only
        changes once the pointer is already on the handle — and it costs
        nothing, because the selection is already part of the cached composite.
        """
        pal = self._pal
        shade = QColor(0, 0, 0, 120)
        for which in ("in", "out"):
            span = self._fade_span_x(clip, which)
            if span is None:
                continue
            x0, x1 = span
            if x1 - x0 < 1.0 or x1 < box.left() - 1 or x0 > box.right() + 1:
                continue
            top, bottom = box.top(), box.bottom()
            wedge = (
                QPolygonF([QPointF(x0, bottom), QPointF(x0, top), QPointF(x1, top)])
                if which == "in"
                else QPolygonF([QPointF(x1, bottom), QPointF(x1, top), QPointF(x0, top)])
            )
            painter.save()
            painter.setClipRect(box)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(shade)
            painter.drawPolygon(wedge)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(pal.block_sel, 1.6))
            painter.drawLine(
                QPointF(x0, bottom if which == "in" else top),
                QPointF(x1, top if which == "in" else bottom),
            )
            painter.restore()
        if clip.cid not in self._selected:
            return
        edges = self.block_edges(clip)
        if edges is None or edges[1] - edges[0] < MIN_BLOCK_PX:
            return
        painter.save()
        painter.setClipRect(box)
        # Two-tone like the selection outline, and INSET past it: a grip in the
        # accent alone disappears into the selected block's own border, which is
        # the same colour and sits on the same pixels.
        painter.setBrush(pal.block_sel)
        painter.setPen(QPen(pal.block_edge, 1.0))
        grip = 6.0
        top = box.top() + 3.0
        if clip.fade_in <= 1e-6:
            painter.drawRect(QRectF(edges[0] + 3.0, top, grip, grip))
        if clip.fade_out <= 1e-6:
            painter.drawRect(QRectF(edges[1] - 3.0 - grip, top, grip, grip))
        painter.restore()

    def _paint_clip_block(self, painter: QPainter, box: QRectF, label: str) -> None:
        """A clip with no thumbnails: a soft block carrying its filename."""
        pal = self._pal
        painter.setPen(Qt.PenStyle.NoPen)
        grad = QLinearGradient(box.topLeft(), box.bottomLeft())
        grad.setColorAt(0.0, pal.wave_bg.lighter(112))
        grad.setColorAt(1.0, pal.wave_bg)
        painter.setBrush(grad)
        painter.drawRect(box)
        if box.width() < 28:
            return
        font = QFont(painter.font())
        font.setPointSizeF(8.0)
        font.setBold(False)
        painter.setFont(font)
        painter.setPen(pal.label)
        painter.drawText(
            box.adjusted(6, 0, -6, 0),
            int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
            QFontMetrics(font).elidedText(
                label, Qt.TextElideMode.ElideMiddle, int(box.width()) - 12
            ),
        )

    def _paint_holes(self, painter: QPainter, lane_key: int, rect: QRectF) -> None:
        """Black out this lane's lifted ranges.  Caller owns the clip."""
        spans = self.model.hole_spans_edited(lane_key)
        if not spans:
            return
        pal = self._pal
        vs, ve = self._view
        for start, end in spans:
            if end < vs or start > ve:
                continue
            x0 = self._x_for(start)
            x1 = self._x_for(end)
            hole = QRectF(x0, rect.top(), max(1.0, x1 - x0), rect.height())
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(pal.hole)
            painter.drawRect(hole)
            painter.setPen(QPen(pal.border, 1.0, Qt.PenStyle.DashLine))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(hole.adjusted(0.5, 0.5, -0.5, -0.5))

    def _thumb_cells(self, clip: Clip) -> list[tuple[float, float, QPixmap]]:
        """Thumbnails as edited-time cells; cached until cuts or assets change.

        Cell edges sit halfway between neighbouring thumbs, so the strip is
        gapless whatever the spacing; the two outer edges mirror the inner gap.
        Timestamps are shifted from the clip's own source time onto the
        timeline, so an appended clip's frames land under the appended clip —
        and frames that fall outside THIS clip are dropped, because two clips
        of one file share one cache of that file's thumbnails.
        """
        thumbs = self._thumbs.get(clip.path)
        if not thumbs:
            return []
        shift = clip.start - clip.src_in
        if self._cells is None:
            self._cells = {}
        key = (clip.path, round(shift, 6), round(clip.src_in, 6), round(clip.length, 6))
        if key in self._cells:
            return self._cells[key]
        points = [
            (edited, pix)
            for src_t, pix in thumbs
            if clip.src_in - 1e-6 <= src_t <= clip.src_out + 1e-6
            and (edited := self.model.source_to_edited(src_t + shift)) is not None
        ]
        cells: list[tuple[float, float, QPixmap]] = []
        n = len(points)
        for i, (edited, pix) in enumerate(points):
            prev_t = points[i - 1][0] if i else None
            next_t = points[i + 1][0] if i + 1 < n else None
            gap = (next_t - edited) if next_t is not None else (
                (edited - prev_t) if prev_t is not None else 1.0
            )
            left = (prev_t + edited) / 2.0 if prev_t is not None else edited - gap / 2.0
            right = (next_t + edited) / 2.0 if next_t is not None else edited + gap / 2.0
            cells.append((left, right, pix))
        self._cells[key] = cells
        return cells

    @staticmethod
    def _cover_source(pix: QPixmap, target: QRectF) -> QRectF:
        """Centre-crop source rect so the pixmap fills ``target`` (cover fit)."""
        pw, ph = float(pix.width()), float(pix.height())
        if pw <= 0 or ph <= 0 or target.height() <= 0:
            return QRectF(0, 0, pw, ph)
        target_ar = target.width() / target.height()
        if pw / ph > target_ar:
            w = ph * target_ar
            return QRectF((pw - w) / 2.0, 0.0, w, ph)
        h = pw / target_ar
        return QRectF(0.0, (ph - h) / 2.0, pw, h)

    def _tinted_wave(self, track: int, asset: _Wave) -> QPixmap:
        """The waveform PNG re-inked in the ACTIVE theme's trace colour.

        ``probe.waveform_png`` bakes a colour into the PNG, and the app may
        have fetched it under the other theme.  Replacing the RGB while keeping
        the alpha (``CompositionMode_SourceIn``) is cheap and cached, so a theme
        switch never has to re-run ffmpeg.  (Do NOT move the tint into the
        request instead: one fetch has to serve both themes.)

        The cache key is the ink AND the PNG the ink went on — ``cacheKey()``
        is Qt's own per-pixmap serial, so a new asset can never match an old
        entry.  Keyed by track alone (which is what it used to be), the FIRST
        waveform a lane ever received was returned for the rest of the session:
        ``set_waveform`` replaces ``_waves[track]``, and ``_paint_wave`` then
        stretched that stale PNG across whatever SOURCE span the new asset
        claims.  That is the "the waveform never updates after a cut" bug, and
        it is also the combed trace after a hard zoom — the request path was
        working the whole time, the answer was being thrown away at paint time.
        """
        ink = self._pal.wave.name()
        cached = self._tinted.get(track)
        if cached is not None and cached[0] == ink and cached[1] == asset.pix.cacheKey():
            return cached[2]
        pixmap = QPixmap(asset.pix.size())
        pixmap.setDevicePixelRatio(asset.pix.devicePixelRatio())
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        painter.drawPixmap(0, 0, asset.pix)
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceIn)
        painter.fillRect(pixmap.rect(), self._pal.wave)
        painter.end()
        self._tinted[track] = (ink, asset.pix.cacheKey(), pixmap)
        return pixmap

    def _paint_wave(
        self, painter: QPainter, index: int, track: int, asset: _Wave, lane: QRectF
    ) -> None:
        """Draw the lane's trace UNDER EACH BLOCK, one box at a time.

        Per block, not once across the lane: a block can be split and slid now,
        so the map from screen x to a position inside the file is the block's
        own placement and nothing else.  Drawing the trace across the lane
        instead smears it over gaps and shows the wrong audio under a moved
        block — silently, which is the failure mode this project keeps hitting.

        A block from another file or stream is skipped; its outline and name
        already say what it is (see :meth:`wave_source`).
        """
        lane_model = self._lanes[index]
        source = self.wave_source(track)
        if source is None or source != asset.source:
            return  # somebody else's trace: the next fetch brings this lane's
        path, stream, _duration = source
        span = asset.src1 - asset.src0
        if span <= 1e-9:
            return
        pix = self._tinted_wave(track, asset)
        pw = float(pix.width())
        # A little headroom: a peak-normalised trace that touches both lane
        # edges reads as clipped, and the frame stops being visible.
        top = lane.top() + WAVE_INSET
        height = lane.height() - 2 * WAVE_INSET
        for clip, box, t0, t1 in self._clip_boxes(lane_model, lane):
            if clip.path != path or clip.stream != stream:
                continue
            # The box is ONE edited span of ONE block, so source time runs
            # linearly across it: its two TIMELINE edges are the whole mapping.
            a, b = clip.source_at(t0), clip.source_at(t1)
            a, b = max(a, asset.src0), min(b, asset.src1)
            if b - a <= 1e-9:
                continue
            u0, u1 = (a - asset.src0) / span, (b - asset.src0) / span
            painter.drawPixmap(
                QRectF(box.left(), top, max(1.0, box.width()), height),
                pix,
                # The vertical crop is the per-lane peak normalisation; see
                # _wave_crop for why a raw PNG is a hairline on a long file.
                QRectF(u0 * pw, asset.top, max(1.0, (u1 - u0) * pw), asset.height),
            )

    def _paint_audio_lane(self, painter: QPainter, index: int) -> None:
        pal = self._pal
        lane_model = self._lanes[index]
        track = self._audio_ordinal(index)
        lane = self._lane_rect(index)
        muted = lane_model.muted
        painter.save()
        self._clip_to(painter, lane)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(pal.bg if muted else pal.wave_bg)
        painter.drawRect(lane)

        painter.setPen(QPen(pal.border.darker(120), 1.0))
        mid = lane.center().y()
        painter.drawLine(QPointF(lane.left(), mid), QPointF(lane.right(), mid))

        if not lane_model.clips:
            painter.setPen(pal.empty_text)
            painter.drawText(
                lane, Qt.AlignmentFlag.AlignCenter, "empty — drop an audio file here"
            )
        elif len(lane_model.clips) > 1:
            # Clip outlines first, so a lane with several clips reads as several
            # clips even before any waveform has arrived for them.
            for _clip, box, _t0, _t1 in self._clip_boxes(lane_model, lane):
                self._outline(painter, box)

        asset = self._waves.get(track)
        if asset is None:
            if lane_model.clips:
                painter.setPen(pal.empty_text)
                painter.drawText(lane, Qt.AlignmentFlag.AlignCenter, "…")
        else:
            self._paint_wave(painter, index, track, asset, lane)
        self._paint_holes(painter, lane_model.key, lane)
        # After the waveform, so a selected block still reads as selected under
        # its own trace.
        for clip, box, _t0, _t1 in self._clip_boxes(lane_model, lane):
            self._paint_block_state(painter, clip, box)
        if muted:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(pal.muted)
            painter.drawRect(lane)
        painter.restore()
        self._frame(painter, lane, armed=self._flagged(index))

    def _frame(self, painter: QPainter, rect: QRectF, *, armed: bool = False) -> None:
        pal = self._pal
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(pal.armed if armed else pal.border, 1.6 if armed else 1.0))
        painter.drawRect(rect.adjusted(0.5, 0.5, -0.5, -0.5))

    def _paint_splices(self, painter: QPainter) -> None:
        """Thin marker at each former cut — an affordance, never a gap."""
        points = self.model.splice_points_edited()
        if not points:
            return
        vs, ve = self._view
        lo = bisect.bisect_left(points, vs)
        hi = bisect.bisect_right(points, ve)
        rect = self._lanes_rect()
        painter.save()
        # -6 at the top is headroom for the marker triangles, which sit above
        # the lanes; the ±1 sideways is antialiasing slack for a splice landing
        # exactly on an edge.  Intersected, so the slack cannot reach past the
        # composite's own clip and into the gutter.
        self._clip_to(painter, rect.adjusted(-1, -6, 1, 1))
        for t in points[lo:hi]:
            x = self._x_for(t)
            painter.setPen(QPen(self._pal.splice, 1.0))
            painter.drawLine(QPointF(x, rect.top()), QPointF(x, rect.bottom()))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(self._pal.splice)
            painter.drawPolygon(
                QPolygonF(
                    [
                        QPointF(x - 3, rect.top() - 4),
                        QPointF(x + 3, rect.top() - 4),
                        QPointF(x, rect.top() + 1),
                    ]
                )
            )
        painter.restore()

    def _paint_selection(self, painter: QPainter) -> None:
        """Paint the selection across the ARMED lanes only.

        That is the affordance the lift rests on: the overlay shows exactly
        which tracks Delete is about to touch, so ripple and lift are never
        ambiguous before the fact.
        """
        sel = self.selection()
        if sel is None:
            return
        lanes = self._lanes_rect()
        x0 = max(lanes.left(), self._x_for(sel[0]))
        x1 = min(lanes.right(), self._x_for(sel[1]))
        if x1 < lanes.left() or x0 > lanes.right():
            return
        width = max(1.0, x1 - x0)
        bands = [
            self._lane_rect(lane) for lane in range(self.lane_count()) if self.is_armed(lane)
        ]
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(self._pal.sel_fill)
        for band in bands:
            painter.drawRect(QRectF(x0, band.top(), width, band.height()))
        rect = QRectF(x0, bands[0].top(), width, bands[-1].bottom() - bands[0].top())
        painter.setPen(QPen(self._pal.accent, 1.6))
        for band in bands:
            painter.drawLine(
                QPointF(x0, band.top()), QPointF(x0 + width, band.top())
            )
            painter.drawLine(
                QPointF(x0, band.bottom()), QPointF(x0 + width, band.bottom())
            )

        handle_h = max(16.0, rect.height() * 0.3)
        y = rect.center().y() - handle_h / 2
        painter.setPen(QPen(QColor(0x10, 0x14, 0x17, 200), 1.0))
        painter.setBrush(self._pal.accent)
        for mark in sel:
            x = self._x_for(mark)
            if lanes.left() - 3 <= x <= lanes.right() + 3:
                painter.drawRoundedRect(QRectF(x - 2.5, y, 5.0, handle_h), 2.5, 2.5)

        left_chip = self._chip(painter, rect.left() + 3, rect.top() + 3, format_tc(sel[0]))
        # The out chip only when it fits beside the in chip: on a narrow
        # selection the two would otherwise print over each other.
        if rect.right() - 3 - left_chip.right() > left_chip.width() + 6:
            self._chip(
                painter, rect.right() - 3, rect.top() + 3, format_tc(sel[1]), align="right"
            )
        if rect.width() > 140:
            self._chip(
                painter,
                rect.center().x(),
                rect.center().y() - 8,
                format_tc(sel[1] - sel[0]),
                align="center",
                bg=QColor(0x1E, 0x63, 0x8C, 235),
            )

    def _paint_ruler(self, painter: QPainter) -> None:
        ruler = self._ruler_rect()
        vs, ve = self._view
        span = max(1e-9, ve - vs)
        font = QFont(painter.font())
        font.setPointSizeF(7.8)
        font.setBold(False)
        painter.setFont(font)
        metrics = QFontMetrics(font)
        px_per_sec = ruler.width() / span
        step = _TICK_STEPS[-1]
        for candidate in _TICK_STEPS:
            if candidate * px_per_sec >= 76:
                step = candidate
                break
        decimals = 1 if step < 1 else 0
        t = math.ceil(vs / step) * step
        while t <= ve + 1e-9:
            x = self._x_for(t)
            painter.setPen(QPen(self._pal.tick, 1.0))
            painter.drawLine(QPointF(x, ruler.bottom() - 4), QPointF(x, ruler.bottom()))
            label = format_tc(t, decimals=decimals)
            width = metrics.horizontalAdvance(label)
            tx = x + 3
            if tx + width > ruler.right():
                tx = x - 3 - width
            painter.setPen(self._pal.tick_text)
            painter.drawText(
                QRectF(tx, ruler.top(), width + 2, ruler.height()),
                int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
                label,
            )
            t += step

    def _paint_hover(self, painter: QPainter) -> None:
        if self._hover_x is None or self._drag is not None:
            return
        lanes = self._lanes_rect()
        x = float(self._hover_x)
        if not (lanes.left() <= x <= lanes.right()):
            return
        painter.setPen(QPen(self._pal.hover, 1.0))
        painter.drawLine(QPointF(x, lanes.top()), QPointF(x, lanes.bottom()))

    def _paint_playhead(self, painter: QPainter) -> None:
        lanes = self._lanes_rect()
        x = self._x_for(self._playhead)
        if not (lanes.left() - 8 <= x <= lanes.right() + 8):
            return
        top, bottom = lanes.top(), lanes.bottom()
        painter.setPen(QPen(QColor(0, 0, 0, 160), 3.0))
        painter.drawLine(QPointF(x, top), QPointF(x, bottom))
        painter.setPen(QPen(self._pal.playhead, 1.3))
        painter.drawLine(QPointF(x, top), QPointF(x, bottom))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(self._pal.playhead)
        painter.drawPolygon(
            QPolygonF(
                [QPointF(x - 5, top - 5), QPointF(x + 5, top - 5), QPointF(x, top + 3)]
            )
        )
        self._chip(painter, x, bottom - 16, format_tc(self._playhead), align="center")

    def _chip(
        self,
        painter: QPainter,
        x: float,
        y: float,
        text: str,
        *,
        align: str = "left",
        bg: QColor | None = None,
    ) -> QRectF:
        font = QFont(painter.font())
        font.setPointSizeF(8.0)
        font.setBold(False)
        painter.setFont(font)
        metrics = QFontMetrics(font)
        w = metrics.horizontalAdvance(text) + 9
        h = metrics.height() + 2
        left = x - w / 2 if align == "center" else (x - w if align == "right" else x)
        track = self._track_rect()
        left = max(track.left() + 1, min(track.right() - w - 1, left))
        rect = QRectF(left, y, w, h)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(bg if bg is not None else self._pal.chip_bg)
        painter.drawRoundedRect(rect, 3, 3)
        painter.setPen(self._pal.chip_text)
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, text)
        return rect

    # ================================================================ mouse
    def _hit(self, x: float) -> str:
        """What a press at ``x`` grabs.

        Mark handles win over the playhead where they coincide — which they
        always do right after I/O, since both marks are set AT the playhead.
        The playhead can be put anywhere with a plain click; a mark can only be
        adjusted by its handle, so the handle is the one worth protecting.
        """
        sel = self.selection()
        if sel is not None:
            if abs(x - self._x_for(sel[0])) <= HANDLE_PX:
                return "mark-in"
            if abs(x - self._x_for(sel[1])) <= HANDLE_PX:
                return "mark-out"
        if abs(x - self._x_for(self._playhead)) <= HANDLE_PX:
            return "scrub"
        return "strip"

    def _in_track(self, x: float) -> bool:
        return x >= GUTTER_W

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        x, y = event.position().x(), event.position().y()
        if not self.has_media():
            super().mousePressEvent(event)
            return
        if not self._in_track(x):
            # The gutter: the video lane has no child header widget, so its
            # arm/ctrl-arm click is handled here — and so are the few pixels of
            # margin either side of an audio header, which is where the armed
            # strip is painted.  An audio header itself is a real widget and
            # handles its own press.
            lane = self._lane_at(y)
            if event.button() == Qt.MouseButton.LeftButton and lane is not None:
                self._lane_clicked(
                    lane, bool(event.modifiers() & Qt.KeyboardModifier.ControlModifier)
                )
                return
            super().mousePressEvent(event)
            return
        if event.button() == Qt.MouseButton.MiddleButton:
            self._drag = "pan"
            self._press_x = x
            self._press_view = self._view
            return
        if event.button() != Qt.MouseButton.LeftButton:
            super().mousePressEvent(event)
            return
        self.setFocus(Qt.FocusReason.MouseFocusReason)
        self._press_x = x
        self._press_lane = self._lane_at(y)
        self._moved = False
        mods = event.modifiers()
        ctrl = bool(mods & Qt.KeyboardModifier.ControlModifier)
        if mods & Qt.KeyboardModifier.AltModifier:
            # Alt is the BLOCK gesture: select what is under the pointer and be
            # ready to slide it.  It is a modifier rather than the plain drag
            # because the plain drag marks a range, and that is the cut
            # workflow — retraining it to move blocks would cost more than the
            # modifier does.
            self._press_block(event.position(), mods)
            return
        if ctrl and self._selected and self.block_at(event.position()) is not None:
            # Once a block IS selected, plain Ctrl+click adds the next one —
            # no Alt needed, because that is the gesture everyone reaches for.
            # With nothing selected it stays a seek, so the cut workflow never
            # sees it.
            self._press_block(event.position(), mods)
            return
        hit = self._hit(x)
        if hit == "scrub":
            self._drag = "scrub"
        elif hit in ("mark-in", "mark-out"):
            self._drag = hit
        else:
            self._drag = "pending"
            self._anchor = self._t_for_clamped(x)
        self.update()

    def _press_block(self, pos: QPointF, mods: Qt.KeyboardModifier) -> None:
        """Alt+press: select the block under the pointer and arm the gesture.

        WHICH gesture depends on where in the block the press landed
        (:meth:`block_handle`): the middle slides it, an edge trims it, a top
        corner sets a fade.  Ctrl+Alt is always the plain add/remove click —
        picking blocks for a multi-selection must not turn into a trim because
        the pointer happened to be near an edge.
        """
        hit = self.block_at(pos)
        if hit is None:
            self.clear_block_selection()
            self._drag = None
            return
        _lane, clip = hit
        toggle = bool(mods & Qt.KeyboardModifier.ControlModifier)
        if toggle or clip.cid not in self._selected:
            self.select_block(clip, toggle=toggle)
        if clip.cid not in self._selected:
            self._drag = None  # Ctrl+Alt just removed it; nothing to slide
            return
        handle = "move"
        if not toggle:
            grabbed = self.block_handle(pos)
            handle = grabbed[2] if grabbed is not None else "move"
        self._block_moved = False
        self._block_pushed = False
        if handle in ("left", "right"):
            self._press_edge(pos, clip, handle)
            return
        if handle in ("fade-in", "fade-out"):
            self._press_fade(pos, clip, handle.split("-", 1)[1])
            return
        self._drag = "block"
        self._block_anchor = self.model.edited_to_source(self._t_for_clamped(pos.x()))
        self._block_from = {c.cid: c.start for _, c in self.selected_blocks()}
        self._block_limits = self._move_limits()

    def _press_edge(self, pos: QPointF, clip: Clip, edge: str) -> None:
        """Alt+press on a block's left/right edge: arm a TRIM of its group."""
        members = self.trim_members(clip)
        self._drag = "trim"
        self._edge_kind = edge
        self._edge_cid = clip.cid  # the grabbed block is what the snap follows
        # Trimming the LAST block out makes the timeline longer, and the view is
        # hard-clamped to the edit's length -- so a fit view stops being fit the
        # moment the drag grows anything, and ``_after_lane_change`` (which
        # tests fitness AFTER the fact) would leave the restored seconds off the
        # right of the strip.  Remember the answer from BEFORE the drag instead.
        self._edge_refit = (
            self._view[0] <= 1e-6
            and self._view[1] >= self._view_total() - 1e-6
        )
        self._edge_anchor = self._source_for_x(pos.x())
        self._edge_from = {
            c.cid: (c.start, c.src_in, c.length, c.fade_in, c.fade_out)
            for _, c in members
        }
        self._edge_limits = self._trim_limits(members, edge)

    def _press_fade(self, pos: QPointF, clip: Clip, which: str) -> None:
        """Alt+press on a block's top corner: arm a FADE drag on its group."""
        members = self.trim_members(clip)
        self._drag = "fade"
        self._fade_kind = which
        self._fade_anchor = self._source_for_x(pos.x())
        self._fade_from = {
            c.cid: (c.fade_in if which == "in" else c.fade_out) for _, c in members
        }

    def _trim_edge_target(self, here: float) -> float:
        """Pointer time -> the trim delta, after snapping the MOVING edge.

        The grab offset is kept (the edge travels with the pointer rather than
        jumping to it, exactly like a move), and the snap is applied to where
        the edge would land — the block's other edge and its neighbours are
        what a trim wants to sit flush against.
        """
        start, _src_in, length, _fade_in, _fade_out = self._edge_from[self._edge_cid]
        origin = start + length if self._edge_kind == "right" else start
        wanted = self._snap_time(
            origin + (here - self._edge_anchor), set(self._edge_from)
        )
        return wanted - origin

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        x = event.position().x()
        self._hover_x = x if self._in_track(x) else None
        if self._drag is None:
            if self._in_track(x) and self.has_media():
                alt = bool(event.modifiers() & Qt.KeyboardModifier.AltModifier)
                grabbed = self.block_handle(event.position()) if alt else None
                self.setCursor(
                    BLOCK_CURSORS[grabbed[2]]
                    if grabbed is not None
                    else Qt.CursorShape.SizeHorCursor
                    if self._hit(x) != "strip"
                    else Qt.CursorShape.PointingHandCursor
                )
            else:
                self.setCursor(Qt.CursorShape.ArrowCursor)
            self.update()
            return

        if self._drag in ("trim", "fade"):
            # Same threshold as a move: a couple of pixels of jitter under an
            # Alt+click must not leave an undo step (or a 3 ms trim) behind.
            if not self._block_moved:
                if abs(x - self._press_x) < DRAG_THRESHOLD_PX:
                    return
                self._block_moved = True
            self._moved = True
            here = self._source_for_x(x)
            if self._drag == "fade":
                delta = here - self._fade_anchor
                if not self._block_pushed and not self.fade_changes(delta):
                    return  # a handle already at 0 pulled further out
                self._push_block_edit()
                self.fade_blocks(delta)
                return
            # Snap the MOVING EDGE, not the block: what a trim wants to land on
            # is a neighbour's edge or the playhead, and the edge's own target
            # position is the only thing that can be tested against those.
            moving = self._trim_edge_target(here)
            lo, hi = self._edge_limits
            delta = max(lo, min(hi, moving))
            if delta == 0.0 and not self._block_pushed:
                return  # nowhere to go: no undo step for an edit that is not one
            self._push_block_edit()
            self.trim_blocks(delta)
            return

        if self._drag == "block":
            if not self._block_moved:
                if abs(x - self._press_x) < DRAG_THRESHOLD_PX:
                    return
                self._block_moved = True
                self.setCursor(Qt.CursorShape.ClosedHandCursor)
            self._moved = True
            here = self.model.edited_to_source(self._t_for_clamped(x))
            delta = self._snap_delta(here - self._block_anchor)
            lo, hi = self._block_limits
            if max(lo, min(hi, delta)) == 0.0 and not self._block_pushed:
                # Wedged between its neighbours, or back where it started: no
                # undo step for an edit that has not happened.  The drag stays
                # armed, because letting go over ANOTHER lane is still a move.
                return
            self._push_block_edit()
            self.move_blocks(delta)
            return

        if self._drag == "pan":
            vs, ve = self._press_view
            span = ve - vs
            track = self._track_rect()
            shift = (self._press_x - x) / max(1.0, track.width()) * span
            self._set_view(vs + shift, ve + shift)
            return
        if self._drag == "pending":
            if abs(x - self._press_x) < DRAG_THRESHOLD_PX:
                return
            self._drag = "select"
            # Dragging inside an armed lane keeps the armed set (that is how you
            # edit several lanes at once: Ctrl+click the headers, then drag).
            # Dragging inside an UNarmed lane re-arms to that lane alone, which
            # is the "just drag over the video track" half of the same gesture.
            if self._press_lane is not None and not self.is_armed(self._press_lane):
                self._apply_armed({self._press_lane})

        self._moved = True
        t = self._t_for_clamped(x)
        if self._drag == "scrub":
            self._apply_playhead(t, emit=True)
        elif self._drag == "select":
            self._set_marks(self._anchor, t)
        elif self._drag == "mark-in":
            other = self._mark_out if self._mark_out is not None else self.edited_duration()
            self._set_marks(t, other)
        elif self._drag == "mark-out":
            other = self._mark_in if self._mark_in is not None else 0.0
            self._set_marks(other, t)
        self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        drag = self._drag
        self._drag = None
        if drag == "block":
            self._release_block(event)
            return
        if drag in ("trim", "fade"):
            self._release_edge()
            return
        if drag is None:
            super().mouseReleaseEvent(event)
            return
        if drag in ("pending", "select") and not self._moved:
            # Plain click: seek ONLY.  It must not disturb marks or selection —
            # clicking to park the playhead before pressing O is the whole point.
            self._apply_playhead(self._t_for_clamped(event.position().x()), emit=True)
        self.update()

    def _release_block(self, event: QMouseEvent) -> None:
        """End an Alt+drag: land on another lane if that is where it was let go.

        A block let go over a DIFFERENT track of the same kind is appended to
        the end of that track's content, which is the user's own rule and the
        only landing that cannot silently overlap what is already there.
        """
        self.setCursor(Qt.CursorShape.OpenHandCursor)
        if not self._block_moved:
            return  # a click, not a drag: the selection it made is the result
        target = self._lane_at(event.position().y())
        blocks = self.selected_blocks()
        lanes = {index for index, _ in blocks}
        if blocks and target is not None and self.can_move_to_lane(target):
            self._push_block_edit()  # before it mutates, and only if it will
        # ``blocks`` can be EMPTY here: Escape or Delete during the drag clears
        # the selection while the button is still down.  Reading a "first
        # block" then is an unhandled StopIteration inside a reimplemented
        # virtual, which PyQt turns into qFatal — it aborted the process.
        if blocks and target is not None and target not in lanes:
            if self.move_blocks_to_lane(target):
                self._announce_blocks()
            elif any(clip.group for _, clip in blocks):
                self.blockHint.emit(
                    "Grouped blocks stay on their own tracks — right-click ▸ "
                    "Ungroup to move one of them across."
                )
            elif len(lanes) == 1:
                self.blockHint.emit(
                    f"A {blocks[0][1].kind} block cannot "
                    f"go on a {self.lane_kind(target)} track."
                )
        pushed = self._block_pushed
        # A drag is in TIMELINE time and jumps a whole cut at a splice, so the
        # neighbour clamp can park a block entirely inside a cut: gone from
        # the strip, the preview and the render, and no pixel left to click it
        # by.  Take the move back rather than let a block vanish silently.
        vanished = pushed and any(
            not self.model.spans_for_source(clip.start, clip.end)
            and self.model.spans_for_source(
                self._block_from.get(clip.cid, clip.start),
                self._block_from.get(clip.cid, clip.start) + clip.length,
            )
            for _, clip in self.selected_blocks()
        )
        self._block_moved = self._block_pushed = False
        self._block_from.clear()
        if vanished:
            self.undo()
            self.blockHint.emit(
                "That would hide the block inside a cut — move it somewhere "
                "the edit still shows."
            )
        elif pushed:
            self._after_lane_change(self.all_armed())
        self.update()

    def _release_edge(self) -> None:
        """End a trim or a fade drag: settle the timeline length, once.

        Nothing here depends on WHERE it was let go — an edge trim cannot
        change lanes, which is the whole reason it is a separate gesture from
        the slide.  The undo step was pushed the moment the drag first changed
        something, so an Alt+click on an edge that went nowhere leaves no step
        behind and ``_after_lane_change`` is not called at all.
        """
        pushed = self._block_pushed
        self._block_moved = self._block_pushed = False
        self._edge_from.clear()
        self._fade_from.clear()
        self._edge_kind = self._fade_kind = ""
        if pushed:
            # A trim can shorten the LAST block, and only this settles the
            # timeline back down (``trim_blocks`` may only grow it mid-drag).
            self._after_lane_change(self.all_armed())
            if self._edge_refit:
                self.zoom_to_fit()
        self._edge_refit = False
        self.update()

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        """Double-click: forget what was selected, select what is under here.

        The user's words: "if you double clicked, it kindof behaved as if you
        deselected and selected where you double clicked".  So it never adds to
        a selection and never leaves a stale one behind — whatever was selected
        before, what is selected after is exactly what the pointer is on:

        * over a BLOCK: that block and its group (the same unit Alt+click
          selects, so a picture and its sound still come together), with the
          time marks set to the SECTION of it under the cursor — the block's
          visible extent narrowed to the keep-segment the pointer is in.  On an
          uncut block those are the same thing; on a cut one, marking the whole
          block would have thrown away the older gesture, because the main
          capture is one block covering the entire timeline and a double-click
          anywhere on it would always mark the entire edit;
        * over bare lane (a gap, an empty lane): the keep-segment under the
          cursor, which is what a double-click has always done (request 8).

        Single click is untouched: a plain click still ONLY seeks.
        """
        x = event.position().x()
        if (
            event.button() != Qt.MouseButton.LeftButton
            or not self._in_track(x)
            or self.model.edited_duration() <= 0.0
        ):
            return
        hit = self.block_at(event.position())
        if hit is not None:
            # ``select_block`` without ``toggle`` REPLACES the selection, which
            # is the "deselect, then select this" in one step and one
            # blockSelectionChanged — clearing first would announce twice.
            self.select_block(hit[1])
            extent = self._block_extent(hit[1])
            if extent is not None:
                section = self.model.keep_bounds_at_edited(self._t_for_clamped(x))
                if section is not None:
                    extent = (
                        max(extent[0], section[0]),
                        min(extent[1], section[1]),
                    )
                self._set_marks(*extent)
            return
        self.clear_block_selection()
        bounds = self.model.keep_bounds_at_edited(self._t_for_clamped(x))
        if bounds is not None:
            self._set_marks(*bounds)

    def _block_extent(self, clip: Clip) -> Range | None:
        """A block's EDITED extent — the range a double-click on it marks.

        Read off the same spans ``_clip_boxes`` paints, so the marks land on
        the block's visible edges: a cut INSIDE the block has collapsed part of
        it, and the block's own ``start``/``end`` are source time, which stopped
        meaning screen position the moment the first cut was made.  ``None``
        when no part of the block survives the cut list.
        """
        spans = self.model.spans_for_source(clip.start, clip.end)
        if not spans:
            return None
        return (spans[0][0], spans[-1][1])

    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802
        if not self.has_media() or self.model.edited_duration() <= 0.0:
            event.ignore()
            return
        delta = event.angleDelta().y() or event.angleDelta().x()
        if delta == 0:
            event.ignore()
            return
        notches = delta / 120.0
        mods = event.modifiers()
        if mods & Qt.KeyboardModifier.ControlModifier:
            self._zoom_at(event.position().x(), ZOOM_STEP**notches)
        else:
            vs, ve = self._view
            fast = 3.0 if mods & Qt.KeyboardModifier.ShiftModifier else 1.0
            shift = -notches * (ve - vs) * PAN_FRACTION * fast
            self._set_view(vs + shift, ve + shift)
        event.accept()

    def leaveEvent(self, event: object) -> None:  # noqa: N802
        self._hover_x = None
        self.update()

    # ============================================================= keyboard
    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802
        key = event.key()
        mods = event.modifiers()
        ctrl = bool(mods & Qt.KeyboardModifier.ControlModifier)
        shift = bool(mods & Qt.KeyboardModifier.ShiftModifier)

        if key in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            # Blocks selected: Delete means those blocks, leaving a gap.  That
            # is what was just clicked on, so it wins over the range.
            self.delete_blocks() or self.cut_selection()
        elif key == Qt.Key.Key_Escape:
            self.clear_block_selection()
            self.clear_selection()
        # S, Ctrl+C/V/G/Shift+G and Ctrl+Z/Shift+Z are the window's QAction
        # shortcuts (app.py), and a shortcut wins over keyPressEvent — branches
        # for them here never ran.  Ctrl+Y has no action, so it stays.
        elif key == Qt.Key.Key_Space:
            self.playPauseRequested.emit()
        elif key == Qt.Key.Key_I:
            self.set_mark_in()
        elif key == Qt.Key.Key_O:
            self.set_mark_out()
        elif key == Qt.Key.Key_A and not ctrl:
            self._arm_all()  # back to ripple, from any subset
        elif ctrl and key == Qt.Key.Key_Y:
            self.redo()
        elif key in (Qt.Key.Key_Plus, Qt.Key.Key_Equal):
            self._zoom_at(self._x_for(self._playhead), ZOOM_STEP)
        elif key == Qt.Key.Key_Minus:
            self._zoom_at(self._x_for(self._playhead), 1.0 / ZOOM_STEP)
        elif key == Qt.Key.Key_0:
            self.zoom_to_fit()
        elif key in (Qt.Key.Key_Left, Qt.Key.Key_Right):
            step = 10.0 if ctrl else (0.1 if shift else 1.0)
            delta = step if key == Qt.Key.Key_Right else -step
            self._apply_playhead(self._playhead + delta, emit=True)
        elif key == Qt.Key.Key_Home:
            self._apply_playhead(0.0, emit=True)
        elif key == Qt.Key.Key_End:
            self._apply_playhead(self.model.edited_duration(), emit=True)
        else:
            super().keyPressEvent(event)
            return
        event.accept()
