"""Tests for the edited-time cut model and the multi-lane timeline widget.

The :class:`CutList` half needs no QApplication.  The widget half runs with
QT_QPA_PLATFORM=offscreen and never shows a window: events are built by hand and
handed to the handlers, which keeps the assertions deterministic.
"""

from __future__ import annotations

import os
import random
import sys
import tempfile
import time
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt6.QtCore import QEvent, QPoint, QPointF, Qt  # noqa: E402
from PyQt6.QtGui import (  # noqa: E402
    QColor,
    QImage,
    QKeyEvent,
    QMouseEvent,
    QPainter,
    QPixmap,
    QWheelEvent,
)
from PyQt6.QtWidgets import QApplication  # noqa: E402

_app = QApplication.instance() or QApplication(sys.argv)

from ffcut.probe import AudioTrack, MediaInfo  # noqa: E402
from ffcut.timeline import (  # noqa: E402
    GUTTER_W,
    MIN_SPAN,
    CutList,
    TimelineWidget,
    format_tc,
)

WIDTH = 1000
_TMP = tempfile.TemporaryDirectory(prefix="ffcut-timeline-tests-")


def info_for(duration: float = 2780.0, n_audio: int = 3, title: str = "T{}") -> MediaInfo:
    return MediaInfo(
        "/tmp/x.mp4", duration, 1664, 936, 120.0, "h264",
        [AudioTrack(i, "aac", 2, 48000, "eng", title.format(i + 1)) for i in range(n_audio)],
    )


def pump(seconds: float) -> None:
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        _app.processEvents()
        time.sleep(0.005)


def times(n: int, lo: float = 0.0, hi: float = 2780.0) -> list[float]:
    """``n`` evenly spaced SOURCE timestamps — what the app hands set_thumbnails."""
    return [lo + (hi - lo) * (i + 0.5) / n for i in range(n)]


def rgb_distance(a: QColor, b: QColor) -> int:
    """Manhattan RGB distance — enough to say which of two palette colours a
    grabbed pixel is, without pinning exact values through antialiasing."""
    return (
        abs(a.red() - b.red()) + abs(a.green() - b.green()) + abs(a.blue() - b.blue())
    )


class TestFormatTc(unittest.TestCase):
    def test_shapes(self) -> None:
        self.assertEqual(format_tc(2780.0), "46:20.00")
        self.assertEqual(format_tc(3671.5, decimals=1), "1:01:11.5")
        self.assertEqual(format_tc(1200.0, decimals=0), "20:00")
        self.assertEqual(format_tc(-3), "00:00.00")


class TestCutList(unittest.TestCase):
    """Keep-list behaviour: what the renderer is handed."""

    def test_no_cuts_keeps_everything(self) -> None:
        self.assertEqual(CutList(100.0).keep_segments(), [(0.0, 100.0)])

    def test_mid_cut_splits_keeps(self) -> None:
        m = CutList(100.0)
        self.assertTrue(m.add_cut(20.0, 30.0))
        self.assertEqual(m.keep_segments(), [(0.0, 20.0), (30.0, 100.0)])
        self.assertAlmostEqual(m.edited_duration(), 90.0)

    def test_edges_snap_and_overlaps_merge(self) -> None:
        m = CutList(100.0)
        m.add_cut(0.01, 5.0)
        m.add_cut(95.0, 99.99)
        m.add_cut(20.0, 30.0)
        m.add_cut(15.0, 40.0)
        self.assertEqual(m.cuts, [(0.0, 5.0), (15.0, 40.0), (95.0, 100.0)])
        self.assertEqual(m.keep_segments(), [(5.0, 15.0), (40.0, 95.0)])

    def test_keep_segments_are_what_the_renderer_gets(self) -> None:
        """Three late keeps on a 46-minute file — the shape used in the e2e run."""
        m = CutList(2780.02)
        for a, b in [(0, 1200), (1203, 2000), (2003, 2500), (2503, 2780.02)]:
            m.add_cut(a, b)
        self.assertEqual(
            m.keep_segments(), [(1200.0, 1203.0), (2000.0, 2003.0), (2500.0, 2503.0)]
        )
        self.assertAlmostEqual(m.edited_duration(), 9.0)

    def test_noop_cut_does_not_touch_history(self) -> None:
        m = CutList(100.0)
        m.add_cut(10.0, 20.0)
        self.assertFalse(m.add_cut(12.0, 18.0))  # already inside a cut
        self.assertFalse(m.add_cut(5.0, 5.0000001))  # zero width
        self.assertTrue(m.can_undo)
        m.undo()
        self.assertFalse(m.can_undo)

    def test_undo_redo(self) -> None:
        m = CutList(100.0)
        m.add_cut(10.0, 20.0)
        m.add_cut(50.0, 60.0)
        self.assertTrue(m.undo())
        self.assertEqual(m.cuts, [(10.0, 20.0)])
        self.assertTrue(m.redo())
        self.assertEqual(m.cuts, [(10.0, 20.0), (50.0, 60.0)])
        self.assertTrue(m.undo() and m.undo())
        self.assertFalse(m.undo())
        self.assertEqual(m.cuts, [])
        m.add_cut(1.0, 2.0)  # new edit clears redo
        self.assertFalse(m.can_redo)

    def test_undo_carries_a_ui_snapshot(self) -> None:
        m = CutList(100.0)
        m.add_cut(10.0, 20.0, {"playhead": 7.0})
        self.assertTrue(m.undo({"playhead": 10.0}))
        self.assertEqual(m.restored_ui, {"playhead": 7.0})
        self.assertTrue(m.redo())
        self.assertEqual(m.restored_ui, {"playhead": 10.0})

    def test_add_cuts_is_one_history_step(self) -> None:
        m = CutList(100.0)
        m.add_cuts([(10.0, 20.0), (50.0, 60.0)])
        self.assertEqual(m.cuts, [(10.0, 20.0), (50.0, 60.0)])
        m.undo()
        self.assertEqual(m.cuts, [])

    def test_next_keep_time_for_playback_skipping(self) -> None:
        m = CutList(100.0)
        m.add_cut(15.0, 40.0)
        self.assertEqual(m.next_keep_time(20.0), 40.0)
        self.assertEqual(m.next_keep_time(50.0), 50.0)
        m.add_cut(40.0, 100.0)
        self.assertIsNone(m.next_keep_time(20.0))

    def test_reset_clears_everything(self) -> None:
        m = CutList(100.0)
        m.add_cut(1.0, 2.0)
        m.reset(50.0)
        self.assertEqual(m.duration, 50.0)
        self.assertEqual(m.cuts, [])
        self.assertFalse(m.can_undo)
        self.assertEqual(m.edited_duration(), 50.0)


class TestEditedTimeMapping(unittest.TestCase):
    """Request 6: cuts collapse.  Any error here corrupts a render."""

    def test_nothing_cut_is_the_identity(self) -> None:
        m = CutList(100.0)
        self.assertEqual(m.edited_duration(), 100.0)
        self.assertEqual(m.splice_points_edited(), [])
        for t in (0.0, 0.001, 33.3, 99.999, 100.0):
            self.assertAlmostEqual(m.source_to_edited(t), t)
            self.assertAlmostEqual(m.edited_to_source(t), t)
            self.assertEqual(m.keep_index_at_edited(t), 0)

    def test_mid_cut_collapses(self) -> None:
        m = CutList(100.0)
        m.add_cut(20.0, 30.0)
        self.assertAlmostEqual(m.edited_duration(), 90.0)
        self.assertEqual(m.splice_points_edited(), [20.0])
        # source -> edited
        self.assertAlmostEqual(m.source_to_edited(0.0), 0.0)
        self.assertAlmostEqual(m.source_to_edited(19.999), 19.999)
        self.assertAlmostEqual(m.source_to_edited(20.0), 20.0)  # cut's left edge
        self.assertIsNone(m.source_to_edited(20.001))
        self.assertIsNone(m.source_to_edited(25.0))
        self.assertIsNone(m.source_to_edited(29.999))
        self.assertAlmostEqual(m.source_to_edited(30.0), 20.0)  # cut's right edge
        self.assertAlmostEqual(m.source_to_edited(30.5), 20.5)
        self.assertAlmostEqual(m.source_to_edited(100.0), 90.0)
        # edited -> source
        self.assertAlmostEqual(m.edited_to_source(0.0), 0.0)
        self.assertAlmostEqual(m.edited_to_source(19.999), 19.999)
        self.assertAlmostEqual(m.edited_to_source(20.0), 30.0)  # splice: post-cut side
        self.assertAlmostEqual(m.edited_to_source(20.5), 30.5)
        self.assertAlmostEqual(m.edited_to_source(90.0), 100.0)
        self.assertAlmostEqual(m.edited_to_source(1e6), 100.0)  # clamped
        self.assertAlmostEqual(m.edited_to_source(-5.0), 0.0)
        # which segment
        self.assertEqual(m.keep_index_at_edited(19.999), 0)
        self.assertEqual(m.keep_index_at_edited(20.0), 1)  # splice joins the later
        self.assertEqual(m.keep_index_at_edited(90.0), 1)
        self.assertEqual(m.keep_bounds_at_edited(5.0), (0.0, 20.0))
        self.assertEqual(m.keep_bounds_at_edited(50.0), (20.0, 90.0))

    def test_cut_at_the_very_start(self) -> None:
        m = CutList(100.0)
        m.add_cut(0.0, 10.0)
        self.assertAlmostEqual(m.edited_duration(), 90.0)
        self.assertEqual(m.splice_points_edited(), [])  # nothing was spliced together
        self.assertIsNone(m.source_to_edited(0.0))
        self.assertIsNone(m.source_to_edited(9.999))
        self.assertAlmostEqual(m.source_to_edited(10.0), 0.0)
        self.assertAlmostEqual(m.source_to_edited(55.0), 45.0)
        self.assertAlmostEqual(m.edited_to_source(0.0), 10.0)
        self.assertAlmostEqual(m.edited_to_source(90.0), 100.0)
        self.assertEqual(m.keep_index_at_edited(0.0), 0)

    def test_cut_at_the_very_end(self) -> None:
        m = CutList(100.0)
        m.add_cut(90.0, 100.0)
        self.assertAlmostEqual(m.edited_duration(), 90.0)
        self.assertEqual(m.splice_points_edited(), [])
        self.assertAlmostEqual(m.source_to_edited(90.0), 90.0)
        self.assertIsNone(m.source_to_edited(90.001))
        self.assertIsNone(m.source_to_edited(100.0))
        self.assertAlmostEqual(m.edited_to_source(90.0), 90.0)

    def test_cuts_at_both_ends(self) -> None:
        m = CutList(100.0)
        m.add_cut(0.0, 10.0)
        m.add_cut(90.0, 100.0)
        self.assertAlmostEqual(m.edited_duration(), 80.0)
        self.assertEqual(m.splice_points_edited(), [])
        self.assertAlmostEqual(m.edited_to_source(0.0), 10.0)
        self.assertAlmostEqual(m.edited_to_source(80.0), 90.0)

    def test_near_touching_cuts_merge_too(self) -> None:
        """A sliver shorter than MIN_KEEP cannot survive, so the cuts merge."""
        m = CutList(100.0)
        m.add_cut(20.0, 30.0)
        m.add_cut(30.01, 40.0)
        self.assertEqual(m.cuts, [(20.0, 40.0)])
        self.assertAlmostEqual(m.edited_duration(), 80.0)
        self.assertEqual(m.splice_points_edited(), [20.0])
        self.assertIsNone(m.source_to_edited(35.0))

    def test_separate_cuts_give_two_splices(self) -> None:
        m = CutList(100.0)
        m.add_cut(20.0, 30.0)
        m.add_cut(50.0, 60.0)
        self.assertAlmostEqual(m.edited_duration(), 80.0)
        self.assertEqual(m.splice_points_edited(), [20.0, 40.0])
        self.assertAlmostEqual(m.source_to_edited(65.0), 45.0)
        self.assertAlmostEqual(m.edited_to_source(45.0), 65.0)
        self.assertEqual(m.keep_index_at_edited(45.0), 2)

    def test_cut_spanning_several_existing_cuts(self) -> None:
        m = CutList(100.0)
        m.add_cut(10.0, 20.0)
        m.add_cut(30.0, 40.0)
        m.add_cut(60.0, 70.0)
        m.add_cut(15.0, 65.0)
        self.assertEqual(m.cuts, [(10.0, 70.0)])
        self.assertAlmostEqual(m.edited_duration(), 40.0)
        self.assertEqual(m.splice_points_edited(), [10.0])
        self.assertAlmostEqual(m.source_to_edited(70.0), 10.0)
        self.assertIsNone(m.source_to_edited(25.0))

    def test_everything_cut(self) -> None:
        m = CutList(100.0)
        m.add_cut(0.0, 100.0)
        self.assertEqual(m.keep_segments(), [])
        self.assertEqual(m.edited_duration(), 0.0)
        self.assertEqual(m.splice_points_edited(), [])
        self.assertIsNone(m.source_to_edited(0.0))
        self.assertIsNone(m.source_to_edited(50.0))
        self.assertEqual(m.edited_to_source(0.0), 0.0)
        self.assertEqual(m.edited_to_source(50.0), 0.0)
        self.assertEqual(m.keep_index_at_edited(0.0), -1)
        self.assertIsNone(m.keep_bounds_at_edited(0.0))
        self.assertEqual(m.source_ranges_for_edited(0.0, 100.0), [])

    def test_no_media_is_inert(self) -> None:
        m = CutList(0.0)
        self.assertEqual(m.edited_duration(), 0.0)
        self.assertIsNone(m.source_to_edited(0.0))
        self.assertEqual(m.edited_to_source(3.0), 0.0)
        self.assertFalse(m.add_cut(1.0, 2.0))

    def test_edited_round_trip_is_exact_everywhere(self) -> None:
        m = CutList(500.0)
        for a, b in [(0, 17), (100, 140), (300, 301), (499, 500)]:
            m.add_cut(a, b)
        total = m.edited_duration()
        steps = 400
        previous = -1.0
        for i in range(steps + 1):
            t_ed = total * i / steps
            t_src = m.edited_to_source(t_ed)
            back = m.source_to_edited(t_src)
            self.assertIsNotNone(back, f"{t_ed} -> {t_src} fell inside a cut")
            self.assertAlmostEqual(back, t_ed, places=9)
            self.assertGreaterEqual(t_src, previous)  # monotonic
            previous = t_src

    def test_source_to_edited_is_monotonic_and_total_adds_up(self) -> None:
        rng = random.Random(20260912)
        for _ in range(60):
            duration = rng.uniform(5.0, 3000.0)
            m = CutList(duration)
            for _ in range(rng.randint(0, 8)):
                a = rng.uniform(0, duration)
                m.add_cut(a, a + rng.uniform(0.0, duration / 4))
            self.assertAlmostEqual(
                m.edited_duration(), sum(b - a for a, b in m.keep_segments()), places=9
            )
            last = -1.0
            for i in range(200):
                t = duration * i / 199
                edited = m.source_to_edited(t)
                if edited is None:
                    continue
                self.assertGreaterEqual(edited + 1e-9, last)
                last = edited
                self.assertLessEqual(edited, m.edited_duration() + 1e-9)

    def test_source_ranges_for_edited_honours_cuts_inside(self) -> None:
        m = CutList(100.0)
        m.add_cut(20.0, 30.0)
        # A selection that straddles the splice comes back as two source ranges.
        self.assertEqual(m.source_ranges_for_edited(10.0, 30.0), [(10.0, 20.0), (30.0, 40.0)])
        # Wholly inside one keep-segment.
        self.assertEqual(m.source_ranges_for_edited(0.0, 5.0), [(0.0, 5.0)])
        self.assertEqual(m.source_ranges_for_edited(25.0, 30.0), [(35.0, 40.0)])
        # Whole edit == every keep-segment.
        self.assertEqual(m.source_ranges_for_edited(0.0, 90.0), m.keep_segments())
        # Reversed input is normalised; zero width yields nothing.
        self.assertEqual(m.source_ranges_for_edited(30.0, 10.0), [(10.0, 20.0), (30.0, 40.0)])
        self.assertEqual(m.source_ranges_for_edited(40.0, 40.0), [])

    def test_cutting_an_edited_span_twice_is_stable(self) -> None:
        """Cut, then cut again in edited space: source ranges stay consistent."""
        m = CutList(100.0)
        m.add_cuts(m.source_ranges_for_edited(20.0, 30.0))
        self.assertEqual(m.keep_segments(), [(0.0, 20.0), (30.0, 100.0)])
        m.add_cuts(m.source_ranges_for_edited(10.0, 30.0))
        self.assertEqual(m.keep_segments(), [(0.0, 10.0), (40.0, 100.0)])
        self.assertAlmostEqual(m.edited_duration(), 70.0)
        self.assertEqual(m.splice_points_edited(), [10.0])


class TestHoles(unittest.TestCase):
    """Lifted ranges: per-lane, source-time, undoable, cut-proof.

    Holes are the whole reason a per-lane edit does not desync the lanes, so
    everything here is about them staying put in SOURCE time while the ripple
    cuts move edited time around underneath.
    """

    def test_holes_start_empty_and_are_per_lane(self) -> None:
        m = CutList(100.0)
        self.assertEqual(m.holes(0), [])
        self.assertEqual(m.all_holes(), {})
        self.assertTrue(m.add_holes({0: [(10.0, 20.0)]}))
        self.assertEqual(m.holes(0), [(10.0, 20.0)])
        self.assertEqual(m.holes(1), [])
        self.assertEqual(m.all_holes(), {0: [(10.0, 20.0)]})

    def test_holes_do_not_shorten_the_edit(self) -> None:
        m = CutList(100.0)
        m.add_holes({0: [(10.0, 40.0)]})
        self.assertAlmostEqual(m.edited_duration(), 100.0)
        self.assertEqual(m.keep_segments(), [(0.0, 100.0)])
        self.assertEqual(m.splice_points_edited(), [])

    def test_overlapping_holes_merge_and_a_repeat_is_a_no_op(self) -> None:
        m = CutList(100.0)
        m.add_holes({2: [(10.0, 20.0)]})
        m.add_holes({2: [(18.0, 30.0)]})
        self.assertEqual(m.holes(2), [(10.0, 30.0)])
        self.assertFalse(m.add_holes({2: [(12.0, 15.0)]}), "already covered")
        self.assertFalse(m.add_holes({2: []}))
        self.assertFalse(m.add_holes({}))

    def test_one_lift_across_several_lanes_is_one_undo_step(self) -> None:
        m = CutList(100.0)
        m.add_holes({0: [(10.0, 20.0)], 3: [(10.0, 20.0)]})
        self.assertEqual(m.all_holes(), {0: [(10.0, 20.0)], 3: [(10.0, 20.0)]})
        self.assertTrue(m.undo())
        self.assertEqual(m.all_holes(), {})
        self.assertTrue(m.redo())
        self.assertEqual(m.all_holes(), {0: [(10.0, 20.0)], 3: [(10.0, 20.0)]})

    def test_undo_unwinds_cuts_and_holes_in_one_history(self) -> None:
        m = CutList(100.0)
        m.add_holes({1: [(10.0, 20.0)]})
        m.add_cut(50.0, 60.0)
        self.assertAlmostEqual(m.edited_duration(), 90.0)
        m.undo()  # the cut
        self.assertAlmostEqual(m.edited_duration(), 100.0)
        self.assertEqual(m.holes(1), [(10.0, 20.0)])
        m.undo()  # the lift
        self.assertEqual(m.holes(1), [])

    def test_keep_holes_forgets_vanished_lanes(self) -> None:
        m = CutList(100.0)
        m.add_holes({7: [(1.0, 2.0)], 9: [(5.0, 6.0)]})
        m.keep_holes([7])  # lane 9 was removed; its holes go with it
        self.assertEqual(m.all_holes(), {7: [(1.0, 2.0)]})

    def test_reset_clears_holes(self) -> None:
        m = CutList(100.0)
        m.add_holes({0: [(10.0, 20.0)]})
        m.reset(50.0)
        self.assertEqual(m.all_holes(), {})
        self.assertFalse(m.can_undo)

    # -- the important one --------------------------------------------------
    def test_hole_ranges_survive_a_later_ripple_cut(self) -> None:
        """A hole is SOURCE-time, so a cut before it must not move it.

        If holes were stored in edited time, this cut would slide every one of
        them 10 s earlier and the render would silence the wrong footage.
        """
        m = CutList(100.0)
        m.add_holes({1: [(60.0, 70.0)]})
        self.assertEqual(m.hole_spans_edited(1), [(60.0, 70.0)])
        m.add_cut(10.0, 20.0)  # 10 s removed BEFORE the hole
        self.assertEqual(m.holes(1), [(60.0, 70.0)], "stored range moved")
        self.assertEqual(m.hole_spans_edited(1), [(50.0, 60.0)], "did not re-map")
        m.undo()
        self.assertEqual(m.hole_spans_edited(1), [(60.0, 70.0)])

    def test_a_cut_inside_a_hole_splits_its_edited_span(self) -> None:
        m = CutList(100.0)
        m.add_holes({0: [(20.0, 60.0)]})
        m.add_cut(30.0, 40.0)
        # 20..30 and 40..60 survive; edited time has them adjacent at 20 and 30.
        self.assertEqual(m.hole_spans_edited(0), [(20.0, 30.0), (30.0, 50.0)])

    def test_a_hole_entirely_inside_a_cut_has_no_edited_span(self) -> None:
        m = CutList(100.0)
        m.add_holes({0: [(30.0, 40.0)]})
        m.add_cut(20.0, 50.0)
        self.assertEqual(m.hole_spans_edited(0), [])
        self.assertEqual(m.holes(0), [(30.0, 40.0)], "kept, so undo restores it")
        m.undo()
        self.assertEqual(m.hole_spans_edited(0), [(30.0, 40.0)])

    def test_edited_spans_for_source_inverts_source_ranges_for_edited(self) -> None:
        rng = random.Random(20260913)
        for _ in range(40):
            duration = rng.uniform(20.0, 1000.0)
            m = CutList(duration)
            for _ in range(rng.randint(0, 5)):
                a = rng.uniform(0, duration)
                m.add_cut(a, a + rng.uniform(0.0, duration / 5))
            total = m.edited_duration()
            if total <= 1.0:
                continue
            lo = rng.uniform(0.0, total)
            hi = min(total, lo + rng.uniform(0.0, total / 3))
            source = m.source_ranges_for_edited(lo, hi)
            back: list[tuple[float, float]] = []
            for a, b in source:
                back.extend(m.edited_spans_for_source(a, b))
            covered = sum(b - a for a, b in back)
            self.assertAlmostEqual(covered, sum(b - a for a, b in source), places=6)
            for a, b in back:
                self.assertGreaterEqual(a, lo - 1e-6)
                self.assertLessEqual(b, hi + 1e-6)


class _WidgetCase(unittest.TestCase):
    """Shared offscreen widget fixture; no window is ever shown."""

    def setUp(self) -> None:
        self.w = TimelineWidget()
        self.w.resize(WIDTH, self.w.sizeHint().height())
        self.w.set_media(info_for())
        self.seeks: list[float] = []
        self.selections: list[object] = []
        self.w.playheadMoved.connect(self.seeks.append)
        self.w.selectionChanged.connect(self.selections.append)

    def tearDown(self) -> None:
        # deleteLater() is only honoured when DeferredDelete is flushed; with no
        # event loop above us that must be explicit, or every widget survives
        # to interpreter teardown.
        self.w.deleteLater()
        QApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)

    def assertSel(self, expect: tuple[float, float] | None) -> None:  # noqa: N802
        """Selection check that tolerates time<->pixel round-off."""
        sel = self.w.selection()
        if expect is None:
            self.assertIsNone(sel)
            return
        self.assertIsNotNone(sel)
        assert sel is not None
        self.assertAlmostEqual(sel[0], expect[0], places=3)
        self.assertAlmostEqual(sel[1], expect[1], places=3)

    # -- synthetic input ----------------------------------------------------
    def x_for(self, t: float) -> float:
        return self.w._x_for(t)

    def press(self, x: float, button: Qt.MouseButton = Qt.MouseButton.LeftButton) -> None:
        self.w.mousePressEvent(
            QMouseEvent(
                QEvent.Type.MouseButtonPress, QPointF(x, 60.0), QPointF(x, 60.0),
                button, button, Qt.KeyboardModifier.NoModifier,
            )
        )

    def move(self, x: float, button: Qt.MouseButton = Qt.MouseButton.LeftButton) -> None:
        self.w.mouseMoveEvent(
            QMouseEvent(
                QEvent.Type.MouseMove, QPointF(x, 60.0), QPointF(x, 60.0),
                Qt.MouseButton.NoButton, button, Qt.KeyboardModifier.NoModifier,
            )
        )

    def release(self, x: float, button: Qt.MouseButton = Qt.MouseButton.LeftButton) -> None:
        self.w.mouseReleaseEvent(
            QMouseEvent(
                QEvent.Type.MouseButtonRelease, QPointF(x, 60.0), QPointF(x, 60.0),
                button, Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier,
            )
        )

    def click(self, x: float) -> None:
        self.press(x)
        self.release(x)

    def double_click(self, x: float, y: float = 60.0) -> None:
        """y matters now: a double-click over a BLOCK selects the block, so the
        keep-segment cases have to be aimed at a lane with nothing on it."""
        self.w.mouseDoubleClickEvent(
            QMouseEvent(
                QEvent.Type.MouseButtonDblClick, QPointF(x, y), QPointF(x, y),
                Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton,
                Qt.KeyboardModifier.NoModifier,
            )
        )

    def drag(self, x0: float, x1: float) -> None:
        self.press(x0)
        self.move((x0 + x1) / 2)
        self.move(x1)
        self.release(x1)

    def wheel(self, x: float, notches: float, mods=Qt.KeyboardModifier.NoModifier) -> None:
        self.w.wheelEvent(
            QWheelEvent(
                QPointF(x, 60.0), QPointF(x, 60.0), QPoint(0, 0),
                QPoint(0, int(120 * notches)), Qt.MouseButton.NoButton, mods,
                Qt.ScrollPhase.NoScrollPhase, False,
            )
        )

    def key(self, key: Qt.Key, mods=Qt.KeyboardModifier.NoModifier) -> None:
        self.w.keyPressEvent(QKeyEvent(QEvent.Type.KeyPress, key, mods))


class TestMarksAndSelection(_WidgetCase):
    """Request 5 — the reported bug: marking out reset the selection."""

    def test_reported_sequence_mark_in_click_elsewhere_mark_out(self) -> None:
        self.w.set_playhead(100.0)
        self.w.set_mark_in()
        self.assertEqual(self.w.selection(), (100.0, 2780.0))
        # The user now clicks where they want the out-point.  v1 cleared the
        # selection here, which is what "it resets it" meant.
        self.click(self.x_for(400.0))
        self.assertAlmostEqual(self.w.playhead(), 400.0, places=3)
        self.assertEqual(self.w.selection(), (100.0, 2780.0))
        self.w.set_mark_out()
        self.assertSel((100.0, 400.0))

    def test_click_never_clears_a_finished_selection(self) -> None:
        self.w.set_playhead(100.0)
        self.w.set_mark_in()
        self.w.set_playhead(400.0)
        self.w.set_mark_out()
        self.click(self.x_for(1500.0))
        self.assertEqual(self.w.selection(), (100.0, 400.0))
        self.assertAlmostEqual(self.w.playhead(), 1500.0, places=3)

    def test_mark_out_preserves_in_and_mark_in_preserves_out(self) -> None:
        self.w.set_playhead(300.0)
        self.w.set_mark_in()
        self.w.set_playhead(900.0)
        self.w.set_mark_out()
        self.assertEqual(self.w.selection(), (300.0, 900.0))
        self.w.set_playhead(500.0)
        self.w.set_mark_in()  # move in, out must stay
        self.assertEqual(self.w.selection(), (500.0, 900.0))
        self.w.set_playhead(1200.0)
        self.w.set_mark_out()  # move out, in must stay
        self.assertEqual(self.w.selection(), (500.0, 1200.0))

    def test_out_before_in_swaps_instead_of_discarding(self) -> None:
        self.w.set_playhead(900.0)
        self.w.set_mark_in()
        self.w.set_playhead(300.0)
        self.w.set_mark_out()
        self.assertEqual(self.w.selection(), (300.0, 900.0))

    def test_in_after_out_swaps_instead_of_discarding(self) -> None:
        self.w.set_playhead(300.0)
        self.w.set_mark_out()
        self.assertEqual(self.w.selection(), (0.0, 300.0))
        self.w.set_playhead(900.0)
        self.w.set_mark_in()
        self.assertEqual(self.w.selection(), (300.0, 900.0))

    def test_selection_survives_zoom_pan_seek_and_playback(self) -> None:
        self.w.set_playhead(100.0)
        self.w.set_mark_in()
        self.w.set_playhead(400.0)
        self.w.set_mark_out()
        expect = (100.0, 400.0)
        self.w.set_view_range(50.0, 500.0)
        self.assertEqual(self.w.selection(), expect)
        self.wheel(self.x_for(200.0), 3, Qt.KeyboardModifier.ControlModifier)  # zoom
        self.assertEqual(self.w.selection(), expect)
        self.wheel(self.x_for(200.0), -2)  # pan
        self.assertEqual(self.w.selection(), expect)
        for t in (0.0, 120.0, 260.0, 2779.0):  # playback ticks
            self.w.set_playhead(t)
        self.assertEqual(self.w.selection(), expect)
        self.w.zoom_to_fit()
        self.assertEqual(self.w.selection(), expect)

    def test_keyboard_i_and_o(self) -> None:
        self.w.set_playhead(600.0)
        self.key(Qt.Key.Key_I)
        self.w.set_playhead(650.0)
        self.key(Qt.Key.Key_O)
        self.assertEqual(self.w.selection(), (600.0, 650.0))

    def test_only_a_drag_replaces_the_selection(self) -> None:
        self.w.set_playhead(100.0)
        self.w.set_mark_in()
        self.w.set_playhead(400.0)
        self.w.set_mark_out()
        self.drag(self.x_for(1000.0), self.x_for(1400.0))
        self.assertSel((1000.0, 1400.0))

    def test_dragging_a_handle_moves_only_that_mark(self) -> None:
        self.w.set_playhead(1000.0)
        self.w.set_mark_in()
        self.w.set_playhead(1400.0)
        self.w.set_mark_out()
        # A mark handle wins over the playhead when they coincide, which they do
        # here: both marks were set AT the playhead.
        self.press(self.x_for(1400.0))
        self.move(self.x_for(1600.0))
        self.release(self.x_for(1600.0))
        self.assertSel((1000.0, 1600.0))
        self.assertAlmostEqual(self.w.playhead(), 1400.0, places=3)
        self.press(self.x_for(1000.0))  # and the in handle
        self.move(self.x_for(800.0))
        self.release(self.x_for(800.0))
        self.assertSel((800.0, 1600.0))

    def test_clear_and_select_all(self) -> None:
        self.w.select_all()
        self.assertEqual(self.w.selection(), (0.0, 2780.0))
        self.w.clear_selection()
        self.assertIsNone(self.w.selection())
        self.key(Qt.Key.Key_Escape)
        self.assertIsNone(self.w.selection())

    def test_selection_changed_is_emitted_once_per_change(self) -> None:
        self.selections.clear()
        self.w.set_playhead(100.0)
        self.w.set_mark_in()
        self.w.set_mark_in()  # same playhead, no change
        self.assertEqual(self.selections, [(100.0, 2780.0)])

    def test_set_playhead_does_not_emit_but_gestures_do(self) -> None:
        self.seeks.clear()
        self.w.set_playhead(123.0)
        self.assertEqual(self.seeks, [])
        self.click(self.x_for(500.0))
        self.assertEqual(len(self.seeks), 1)
        self.assertAlmostEqual(self.seeks[0], 500.0, places=3)


class TestDoubleClickSelect(_WidgetCase):
    """Request 8 — double-click selects the keep-segment under the cursor.

    Only where there is no BLOCK under the cursor, since the "deselect and
    select what I double-clicked" request: over a block the gesture selects the
    block instead (:class:`TestDoubleClickResetsTheSelection` in
    ``test_blocks.py`` owns that half).  Every lane ``set_media`` builds is
    covered end to end by the capture's own blocks, so these aim at a lane with
    nothing on it — which is also a gap, and an empty lane, in one.
    """

    def bare_y(self) -> float:
        """Centre of a lane with no block on it."""
        return self.w._lane_rect(self.w.add_audio_lane()).center().y()

    def test_without_cuts_selects_the_whole_edit(self) -> None:
        self.double_click(self.x_for(1234.0), self.bare_y())
        self.assertEqual(self.w.selection(), (0.0, 2780.0))

    def test_selects_the_segment_the_cursor_is_in(self) -> None:
        y = self.bare_y()
        self.w.set_view_range(0.0, 2780.0)
        self.w._set_marks(1000.0, 1100.0)
        self.w.cut_selection()  # splice at edited 1000, edit is now 2680 long
        self.assertEqual(self.w.model.splice_points_edited(), [1000.0])
        self.double_click(self.x_for(500.0), y)
        self.assertEqual(self.w.selection(), (0.0, 1000.0))
        self.double_click(self.x_for(2000.0), y)
        self.assertEqual(self.w.selection(), (1000.0, 2680.0))

    def test_boundary_exactly_on_the_splice_takes_the_later_segment(self) -> None:
        y = self.bare_y()
        self.w._set_marks(1000.0, 1100.0)
        self.w.cut_selection()
        self.w.set_view_range(0.0, 2680.0)  # whole seconds per pixel: x_for(splice) is exact
        splice = self.w.model.splice_points_edited()[0]
        self.double_click(self.x_for(splice), y)
        self.assertEqual(self.w.selection(), (1000.0, 2680.0))
        self.double_click(self.x_for(splice) - 1.0, y)  # one pixel earlier
        self.assertEqual(self.w.selection(), (0.0, 1000.0))

    def test_delete_after_double_click_removes_the_segment(self) -> None:
        y = self.bare_y()
        self.w._set_marks(1000.0, 1100.0)
        self.w.cut_selection()
        self.double_click(self.x_for(2000.0), y)  # the tail segment
        self.assertEqual(self.w.clip_selection(), frozenset(), "no block was under it")
        self.key(Qt.Key.Key_Delete)  # no blocks selected: Delete is the range
        self.assertAlmostEqual(self.w.edited_duration(), 1000.0)
        self.assertEqual(
            [(s.start, s.end) for s in self.w.keep_segments_source()], [(0.0, 1000.0)]
        )

    def test_a_double_click_on_bare_lane_forgets_a_block_selection(self) -> None:
        """The other half of "deselect, then select where I clicked": over bare
        lane there is nothing to select, so the blocks are simply dropped."""
        y = self.bare_y()
        self.w.select_block(self.w.lane_clips(0)[0])
        self.assertTrue(self.w.clip_selection())
        self.double_click(self.x_for(1234.0), y)
        self.assertEqual(self.w.clip_selection(), frozenset())
        self.assertEqual(self.w.selection(), (0.0, 2780.0))

    def test_gutter_clicks_are_ignored(self) -> None:
        self.double_click(GUTTER_W / 2)
        self.assertIsNone(self.w.selection())
        self.assertEqual(self.w.clip_selection(), frozenset())
        self.click(GUTTER_W / 2)
        self.assertEqual(self.seeks, [])


class TestCuttingInEditedTime(_WidgetCase):
    """Request 6 at the widget level."""

    def test_cut_collapses_and_parks_the_playhead_on_the_splice(self) -> None:
        self.w._set_marks(600.0, 700.0)
        self.w.cut_selection()
        self.assertAlmostEqual(self.w.edited_duration(), 2680.0)
        self.assertAlmostEqual(self.w.playhead(), 600.0)
        self.assertIsNone(self.w.selection())
        self.assertEqual(
            [(s.start, s.end) for s in self.w.keep_segments_source()],
            [(0.0, 600.0), (700.0, 2780.0)],
        )

    def test_a_selection_across_a_splice_cuts_both_source_ranges(self) -> None:
        self.w._set_marks(600.0, 700.0)
        self.w.cut_selection()  # splice at edited 600
        self.w._set_marks(500.0, 800.0)  # straddles it
        segments = self.w.selection_segments_source()
        assert segments is not None
        self.assertEqual(
            [(s.start, s.end) for s in segments], [(500.0, 600.0), (700.0, 900.0)]
        )
        self.w.cut_selection()
        self.assertAlmostEqual(self.w.edited_duration(), 2380.0)
        self.assertEqual(
            [(s.start, s.end) for s in self.w.keep_segments_source()],
            [(0.0, 500.0), (900.0, 2780.0)],
        )

    def test_selection_segments_source_is_none_without_a_selection(self) -> None:
        self.assertIsNone(self.w.selection_segments_source())

    def test_cuts_changed_fires_once_per_delete(self) -> None:
        changed: list[int] = []
        self.w.cutsChanged.connect(lambda: changed.append(1))
        self.w._set_marks(10.0, 20.0)
        self.key(Qt.Key.Key_Delete)
        self.assertEqual(len(changed), 1)
        self.key(Qt.Key.Key_Delete)  # nothing selected any more
        self.assertEqual(len(changed), 1)

    def test_undo_redo_restores_duration_playhead_and_view(self) -> None:
        self.w.set_view_range(500.0, 1500.0)
        self.w.set_playhead(700.0)
        self.w._set_marks(600.0, 700.0)
        self.assertFalse(self.w.can_undo())
        self.w.cut_selection()
        self.assertTrue(self.w.can_undo())
        self.assertTrue(self.w.undo())
        self.assertAlmostEqual(self.w.edited_duration(), 2780.0)
        self.assertAlmostEqual(self.w.playhead(), 700.0)
        self.assertEqual(self.w.view_range(), (500.0, 1500.0))
        self.assertTrue(self.w.can_redo())
        self.assertTrue(self.w.redo())
        self.assertAlmostEqual(self.w.edited_duration(), 2680.0)
        self.assertAlmostEqual(self.w.playhead(), 600.0)
        self.assertFalse(self.w.redo())

    def test_everything_cut_then_undo(self) -> None:
        self.w.select_all()
        self.w.cut_selection()
        self.assertEqual(self.w.edited_duration(), 0.0)
        self.assertEqual(self.w.keep_segments_source(), [])
        self.assertIsNone(self.w.selection())
        self.assertEqual(self.w.playhead(), 0.0)
        self.w.render(QPixmap(WIDTH, self.w.height()))  # must not crash
        self.w.undo()
        self.assertAlmostEqual(self.w.edited_duration(), 2780.0)

    def test_view_is_clamped_to_the_shorter_edit(self) -> None:
        self.w.select_all()
        self.w._set_marks(0.0, 2700.0)
        self.w.cut_selection()
        start, end = self.w.view_range()
        self.assertGreaterEqual(start, 0.0)
        self.assertLessEqual(end, max(MIN_SPAN, self.w._view_total()) + 1e-9)
        self.assertGreater(end, start)

    def test_seam_conversions_match_the_model(self) -> None:
        self.w._set_marks(600.0, 700.0)
        self.w.cut_selection()
        self.assertAlmostEqual(self.w.edited_to_source(600.0), 700.0)
        self.assertAlmostEqual(self.w.source_to_edited(800.0), 700.0)
        self.assertIsNone(self.w.source_to_edited(650.0))


class TestZoomAndPan(_WidgetCase):
    """Request 3 — anchored zoom, pan, limits, debounced asset requests."""

    def test_zoom_keeps_the_frame_under_the_cursor(self) -> None:
        for x in (GUTTER_W + 1.0, 300.0, 512.0, 800.0, float(WIDTH - 7)):
            self.w.zoom_to_fit()
            before = self.w._t_for(x)
            for _ in range(6):
                self.wheel(x, 1, Qt.KeyboardModifier.ControlModifier)
                self.assertAlmostEqual(self.w._t_for(x), before, places=6)
            for _ in range(6):
                self.wheel(x, -1, Qt.KeyboardModifier.ControlModifier)
                self.assertAlmostEqual(self.w._t_for(x), before, places=6)

    def test_zoom_anchor_holds_for_fractional_notches(self) -> None:
        self.w.set_view_range(300.0, 900.0)
        x = 640.0
        before = self.w._t_for(x)
        self.wheel(x, 0.5, Qt.KeyboardModifier.ControlModifier)
        self.assertAlmostEqual(self.w._t_for(x), before, places=6)
        self.wheel(x, -0.25, Qt.KeyboardModifier.ControlModifier)
        self.assertAlmostEqual(self.w._t_for(x), before, places=6)

    def test_zoom_clamped_at_the_left_does_not_invert(self) -> None:
        """Anchor at and near zero: the view clamps, the span stays sane."""
        self.w.zoom_to_fit()
        self.wheel(float(GUTTER_W), 4, Qt.KeyboardModifier.ControlModifier)
        start, end = self.w.view_range()
        self.assertAlmostEqual(start, 0.0)  # the anchored frame IS 0
        self.assertGreater(end, start)
        self.w.set_view_range(0.0, 2780.0)
        self.wheel(GUTTER_W + 2.0, 4, Qt.KeyboardModifier.ControlModifier)
        start, end = self.w.view_range()
        self.assertGreaterEqual(start, 0.0)
        self.assertGreater(end - start, MIN_SPAN - 1e-9)
        self.assertLess(start, 10.0)  # still pinned near the grabbed frame

    def test_zoom_limits(self) -> None:
        for _ in range(80):
            self.wheel(500.0, 1, Qt.KeyboardModifier.ControlModifier)
        start, end = self.w.view_range()
        self.assertAlmostEqual(end - start, MIN_SPAN, places=6)
        for _ in range(120):
            self.wheel(500.0, -1, Qt.KeyboardModifier.ControlModifier)
        self.assertEqual(self.w.view_range(), (0.0, self.w._view_total()))

    def test_plain_wheel_pans_without_changing_the_span(self) -> None:
        self.w.set_view_range(1000.0, 1100.0)
        span = 100.0
        self.wheel(500.0, -1)
        start, end = self.w.view_range()
        self.assertAlmostEqual(end - start, span, places=6)
        self.assertGreater(start, 1000.0)
        self.wheel(500.0, 1)
        self.assertAlmostEqual(self.w.view_range()[0], 1000.0, places=6)

    def test_shift_wheel_pans_further(self) -> None:
        self.w.set_view_range(1000.0, 1100.0)
        self.wheel(500.0, -1)
        plain = self.w.view_range()[0] - 1000.0
        self.w.set_view_range(1000.0, 1100.0)
        self.wheel(500.0, -1, Qt.KeyboardModifier.ShiftModifier)
        self.assertGreater(self.w.view_range()[0] - 1000.0, plain)

    def test_pan_clamps_to_the_edit(self) -> None:
        self.w.set_view_range(0.0, 100.0)
        for _ in range(200):
            self.wheel(500.0, 1)
        self.assertEqual(self.w.view_range(), (0.0, 100.0))
        for _ in range(1500):
            self.wheel(500.0, -1)
        # Past the edit's end by the room on the right, and no further.
        self.assertAlmostEqual(self.w.view_range()[1], self.w._view_total(), places=6)

    def test_middle_drag_pans(self) -> None:
        self.w.set_view_range(1000.0, 1200.0)
        self.press(600.0, Qt.MouseButton.MiddleButton)
        self.move(500.0, Qt.MouseButton.MiddleButton)
        self.release(500.0, Qt.MouseButton.MiddleButton)
        start, end = self.w.view_range()
        self.assertGreater(start, 1000.0)
        self.assertAlmostEqual(end - start, 200.0, places=6)

    def test_set_view_range_rejects_degenerate_ranges(self) -> None:
        self.w.set_view_range(500.0, 500.0)
        start, end = self.w.view_range()
        self.assertAlmostEqual(end - start, MIN_SPAN, places=6)
        self.w.set_view_range(900.0, 100.0)  # inverted
        start, end = self.w.view_range()
        self.assertGreater(end, start)
        self.w.set_view_range(-500.0, 9e9)
        self.assertEqual(self.w.view_range(), (0.0, self.w._view_total()))

    def test_asset_requests_are_debounced_and_cover_the_visible_range(self) -> None:
        thumbs: list[tuple[str, list[float]]] = []
        waves: list[tuple[int, float, float]] = []
        self.w.thumbsNeeded.connect(lambda p, t: thumbs.append((p, list(t))))
        self.w.waveformNeeded.connect(lambda t, a, b: waves.append((t, a, b)))
        for i in range(10):  # a zoom gesture is many events
            self.wheel(500.0, 1, Qt.KeyboardModifier.ControlModifier)
            self.assertEqual(thumbs, [], "emitted mid-gesture (not debounced)")
        pump(0.3)
        self.assertEqual(len(thumbs), 1, "one request per file on the timeline")
        path, times = thumbs[0]
        self.assertEqual(path, self.w._info.path)
        start, end = self.w.view_range()
        self.assertGreaterEqual(len(times), 8)
        # Asked in SOURCE time, inside the visible range, in order.
        self.assertEqual(times, sorted(times))
        self.assertGreaterEqual(times[0], start)
        self.assertLessEqual(times[-1], end)
        self.assertEqual([t for t, _, _ in waves], [0, 1, 2])  # one per audio lane
        self.assertEqual((waves[0][1], waves[0][2]), self.w.view_range())
        thumbs.clear()
        pump(0.3)
        self.assertEqual(thumbs, [], "re-asked for a range it already has")

    def test_playhead_follows_when_it_leaves_the_view(self) -> None:
        self.w.set_view_range(0.0, 100.0)
        self.w.set_playhead(500.0)
        start, end = self.w.view_range()
        self.assertLessEqual(start, 500.0)
        self.assertGreaterEqual(end, 500.0)
        self.assertAlmostEqual(end - start, 100.0, places=6)

    def test_zoom_keys(self) -> None:
        self.w.zoom_to_fit()
        self.key(Qt.Key.Key_Plus)
        self.assertLess(self.w.view_range()[1] - self.w.view_range()[0], 2780.0)
        self.key(Qt.Key.Key_0)
        self.assertEqual(self.w.view_range(), (0.0, self.w._fit_total()))


class TestLanesAndMix(_WidgetCase):
    """Request 1 (timeline half) — lanes own the mix, no second checkbox panel."""

    def test_one_lane_per_audio_track(self) -> None:
        self.assertEqual(len(self.w._headers), 3)
        lanes = [self.w._lane_rect(i) for i in range(4)]
        for above, below in zip(lanes, lanes[1:]):
            self.assertLessEqual(above.bottom(), below.top())
        for lane in lanes:
            self.assertGreaterEqual(lane.left(), GUTTER_W)

    def test_headers_sit_in_the_fixed_gutter(self) -> None:
        for header in self.w._headers:
            self.assertLess(header.geometry().right(), GUTTER_W)
        before = [h.geometry() for h in self.w._headers]
        self.w.set_view_range(10.0, 20.0)  # zooming must not move the gutter
        self.assertEqual([h.geometry() for h in self.w._headers], before)

    def mix(self) -> list:
        """The audio lanes as the render will see them."""
        return self.w.render_plan().audio

    def test_default_mix_is_every_track_at_unity(self) -> None:
        mix = self.mix()
        self.assertEqual([m.clips[0].stream for m in mix], [0, 1, 2])
        self.assertTrue(all(m.gain_db == 0.0 and not m.muted for m in mix))

    def test_mute_toggle_emits_and_feeds_the_mix(self) -> None:
        events: list[tuple[int, bool]] = []
        self.w.trackMuteChanged.connect(lambda i, m: events.append((i, m)))
        self.w._headers[1].mute.click()
        self.assertEqual(events, [(1, True)])
        self.assertEqual([m.muted for m in self.mix()], [False, True, False])
        self.w._headers[1].mute.click()
        self.assertEqual(events[-1], (1, False))
        self.assertEqual([m.muted for m in self.mix()], [False] * 3)

    def test_gain_slider_emits_db_and_feeds_the_mix(self) -> None:
        events: list[tuple[int, float]] = []
        self.w.trackGainChanged.connect(lambda i, g: events.append((i, g)))
        self.w._headers[2].slider.setValue(-6)
        self.assertEqual(events, [(2, -6.0)])
        self.assertEqual([m.gain_db for m in self.mix()], [0.0, 0.0, -6.0])
        self.assertEqual(self.w._headers[2].gain_label.text(), "-6 dB")
        self.w._headers[2].slider.setValue(0)
        self.assertEqual(self.w._headers[2].gain_label.text(), "0 dB")

    def test_gain_range_is_minus30_to_plus12(self) -> None:
        slider = self.w._headers[0].slider
        self.assertEqual((slider.minimum(), slider.maximum()), (-30, 12))
        slider.setValue(-99)
        self.assertEqual(self.mix()[0].gain_db, -30.0)
        slider.setValue(99)
        self.assertEqual(self.mix()[0].gain_db, 12.0)

    def test_wheel_over_a_gain_slider_does_not_change_it(self) -> None:
        slider = self.w._headers[0].slider
        slider.setValue(3)
        slider.wheelEvent(
            QWheelEvent(
                QPointF(5, 5), QPointF(5, 5), QPoint(0, 0), QPoint(0, 120),
                Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier,
                Qt.ScrollPhase.NoScrollPhase, False,
            )
        )
        self.assertEqual(slider.value(), 3)

    def test_lane_controls_do_not_steal_keyboard_focus(self) -> None:
        for header in self.w._headers:
            self.assertEqual(header.mute.focusPolicy(), Qt.FocusPolicy.NoFocus)
            self.assertEqual(header.slider.focusPolicy(), Qt.FocusPolicy.NoFocus)

    def test_fewer_tracks_refloor_the_height(self) -> None:
        tall = self.w.minimumHeight()
        self.w.set_media(info_for(n_audio=1))
        self.assertEqual(len(self.w._headers), 1)
        self.assertLess(self.w.minimumHeight(), tall)
        self.assertEqual(len(self.mix()), 1)

    def test_extra_height_goes_to_the_lanes(self) -> None:
        """The splitter can hand the timeline more than its minimum; the lanes
        grow into it instead of leaving a dead strip underneath."""
        floor = self.w.minimumHeight()
        self.assertEqual(self.w.height(), floor)
        base = [self.w._lane_rect(i).height() for i in range(4)]
        self.w.resize(WIDTH, floor + 100)
        grown = [self.w._lane_rect(i).height() for i in range(4)]
        self.assertEqual([g - b for g, b in zip(grown, base)], [25.0] * 4)
        self.assertGreater(self.w._scroll_rect().top(), self.w._lane_rect(3).bottom())
        self.assertLessEqual(self.w._scroll_rect().bottom(), self.w.height())

    def test_long_labels_elide_and_keep_the_full_text_in_the_tooltip(self) -> None:
        title = "a track title far too long for the gutter to hold"
        self.w.set_media(info_for(n_audio=1, title=title))
        self.w.show()  # a hidden widget gets no resize events, so no layout
        pump(0.05)
        label = self.w._headers[0].label
        self.assertTrue(label.text().endswith("…"), label.text())
        self.assertLess(label.width(), GUTTER_W)
        self.assertIn(title, label.toolTip())
        self.w.hide()

    def test_video_only_media_has_no_audio_lanes(self) -> None:
        self.w.set_media(info_for(n_audio=0))
        self.assertEqual(self.mix(), [])
        self.assertEqual(self.w._headers, [])
        self.w.render(QPixmap(WIDTH, self.w.height()))


class TestAssetsAndPainting(_WidgetCase):
    """Assets are cached in source time and never generated in paintEvent."""

    def _thumb_paths(self, count: int) -> list[str]:
        directory = _TMP.name
        paths = []
        for i in range(count):
            path = os.path.join(directory, f"t{i}.png")
            if not os.path.exists(path):
                image = QImage(32, 18, QImage.Format.Format_RGB32)
                image.fill(0x202020 + i)
                image.save(path)
            paths.append(path)
        return paths

    def test_thumbnails_map_to_gapless_cells_and_collapse_with_cuts(self) -> None:
        paths = self._thumb_paths(10)
        self.w.set_thumbnails(paths, times(10))
        cells = self.w._thumb_cells(self.w.lane_clips(0)[0])
        self.assertEqual(len(cells), 10)
        self.assertAlmostEqual(cells[0][0], 0.0, places=3)
        self.assertAlmostEqual(cells[-1][1], 2780.0, places=3)
        for (_, right, _), (left, _, _) in zip(cells, cells[1:]):
            self.assertAlmostEqual(right, left, places=9)  # no gap, no overlap
        # Cut the middle: thumbs inside the cut disappear, the rest stay in
        # order and still tile the (shorter) edit without a gap.
        self.w._set_marks(1000.0, 2000.0)
        self.w.cut_selection()
        cells = self.w._thumb_cells(self.w.lane_clips(0)[0])
        self.assertLess(len(cells), 10)
        for (_, right, _), (left, _, _) in zip(cells, cells[1:]):
            self.assertAlmostEqual(right, left, places=9)
        self.assertLessEqual(cells[-1][0], self.w.edited_duration() + 1e-6)

    def test_thumbnails_are_stored_in_source_time(self) -> None:
        """Delivery is by source timestamp; an unloadable path is dropped WITH
        its timestamp, so it cannot shift the frames after it."""
        paths = self._thumb_paths(3)
        self.w._set_marks(100.0, 200.0)
        self.w.cut_selection()
        self.w.set_thumbnails(
            [paths[0], "/nonexistent.jpg", paths[1], paths[2]], [50.0, 150.0, 250.0, 350.0]
        )
        stored = self.w._thumbs["/tmp/x.mp4"]
        self.assertEqual([t for t, _ in stored], [50.0, 250.0, 350.0])
        # 150 fell inside the cut anyway; 250/350 collapse to edited 150/250
        self.assertEqual(len(self.w._thumb_cells(self.w.lane_clips(0)[0])), 3)
        self.w.set_thumbnails(["/nonexistent.jpg"], [1.0])  # nothing usable: keep what we had
        self.assertEqual(len(self.w._thumbs["/tmp/x.mp4"]), 3)

    def test_waveform_delivery_is_stored_per_track(self) -> None:
        path = self._thumb_paths(1)[0]
        self.w.set_waveform(1, path, 0.0, 2780.0)
        self.assertIn(1, self.w._waves)
        src0, src1 = self.w._waves[1].src0, self.w._waves[1].src1
        self.assertEqual((src0, src1), (0.0, 2780.0))
        self.w.set_waveform(0, "/nonexistent.png", 0.0, 2780.0)
        self.assertNotIn(0, self.w._waves)

    def test_paint_every_state_without_crashing(self) -> None:
        target = QPixmap(WIDTH, self.w.height())
        fresh = TimelineWidget()
        fresh.resize(WIDTH, fresh.sizeHint().height())
        fresh.render(QPixmap(WIDTH, fresh.height()))  # no media
        fresh.deleteLater()
        self.w.render(target)  # media, no assets
        self.w.set_thumbnails(self._thumb_paths(12), times(12))
        self.w.set_waveform(0, self._thumb_paths(1)[0], 0.0, 2780.0)
        self.w.render(target)
        self.w._set_marks(100.0, 900.0)
        self.w.cut_selection()
        self.w.set_view_range(50.0, 120.0)  # zoomed, with a splice off-screen
        self.w.render(target)
        self.w.set_view_range(0.0, 2780.0)
        self.w._headers[0].mute.click()
        self.w.render(target)

    def test_lane_content_is_cached_across_playhead_moves(self) -> None:
        """Scrubbing must not recompose the lanes (46-min timeline stays smooth)."""
        self.w.set_thumbnails(self._thumb_paths(12), times(12))
        target = QPixmap(WIDTH, self.w.height())
        self.w.render(target)
        first = self.w._content()
        for t in (10.0, 20.0, 30.0):
            self.w.set_playhead(t)
            self.w.render(target)
            self.assertIs(self.w._content(), first, "playhead move rebuilt the lanes")
        self.move(400.0)  # hover
        self.assertIs(self.w._content(), first)
        self.w._set_marks(100.0, 200.0)  # selection is an overlay too
        self.assertIs(self.w._content(), first)

    def test_lane_content_cache_invalidates_on_real_changes(self) -> None:
        self.w.render(QPixmap(WIDTH, self.w.height()))
        first = self.w._content()
        self.w.set_view_range(0.0, 100.0)
        second = self.w._content()
        self.assertIsNot(second, first)
        self.w._set_marks(10.0, 20.0)
        self.w.cut_selection()
        third = self.w._content()
        self.assertIsNot(third, second)
        self.w.set_thumbnails(self._thumb_paths(6), times(6, 0.0, 100.0))
        fourth = self.w._content()
        self.assertIsNot(fourth, third)
        self.w._headers[0].mute.click()
        self.assertIsNot(self.w._content(), fourth)

    def test_hover_and_leave_do_not_crash(self) -> None:
        self.move(500.0)
        self.assertIsNotNone(self.w._hover_x)
        self.w.leaveEvent(QEvent(QEvent.Type.Leave))
        self.assertIsNone(self.w._hover_x)


class TestGutterSurvivesAZoom(_WidgetCase):
    """The reported glitch: "the video thumbnail seems to glitch out sometimes
    when zooming in and out, it extends to the left panel, looks bad".

    A thumbnail CELL is edited-time, so zoomed hard enough one cell is wider
    than the screen and starts thousands of pixels to the LEFT of the track
    area — which is exactly what the moment after a zoom looks like, while the
    stale wide cells are still on screen and the re-sized ones have not
    arrived.  ``_paint_video_lane`` clipped the lane (gutter excluded) and then
    clipped each cell with a bare ``setClipRect``, whose default operation is
    ReplaceClip: the cell straddling the gutter edge REPLACED the lane clip
    with its own off-screen box and painted across the whole gutter.
    """

    GARISH = 0xFFFF00FF  # magenta: nothing in either lane palette is near it

    def garish_thumbs(self, count: int) -> list[str]:
        paths = []
        for i in range(count):
            path = os.path.join(_TMP.name, f"garish{i}.png")
            if not os.path.exists(path):
                image = QImage(32, 18, QImage.Format.Format_RGB32)
                image.fill(self.GARISH)
                image.save(path)
            paths.append(path)
        return paths

    def zoom_onto_a_straddling_cell(self) -> tuple[float, float]:
        """Four thumbs over 46 minutes, then 1.2 s of view: one cell has to
        cross the gutter edge.  Returns its ``(left x, right x)``, asserted
        first so the test can never pass by having nothing to draw."""
        self.w.set_thumbnails(self.garish_thumbs(4), times(4))
        self.w.set_view_range(1390.5, 1391.7)  # inside the third cell, not on its edge
        crossing = [
            (self.w._x_for(left), self.w._x_for(right))
            for left, right, _pix in self.w._thumb_cells(self.w.lane_clips(0)[0])
            if self.w._x_for(left) < GUTTER_W < self.w._x_for(right)
        ]
        self.assertEqual(len(crossing), 1, "the zoom produced no straddling cell")
        self.assertLess(crossing[0][0], 0.0, "it has to start off-screen to the left")
        return crossing[0]

    def test_a_hard_zoom_does_not_flood_the_gutter_with_thumbnails(self) -> None:
        self.zoom_onto_a_straddling_cell()
        pixmap = self.w.grab()
        ratio = pixmap.devicePixelRatio()
        image = pixmap.toImage()
        band = self.w._lane_rect(0)  # the video lane's row of the gutter
        gutter, garish = self.w._pal.gutter, QColor.fromRgb(self.GARISH)
        for fraction in (0.35, 0.5, 0.75):  # below the "VIDEO 1664×936" label
            y = band.top() + band.height() * fraction
            for x in (4, 40, 90, 140, 160):
                colour = image.pixelColor(round(x * ratio), round(y * ratio))
                self.assertGreater(
                    rgb_distance(colour, garish), 60,
                    f"thumbnail painted over the gutter at ({x}, {round(y)})",
                )
                self.assertEqual(colour.name(), gutter.name())

    def test_the_composed_lanes_paint_nothing_over_the_gutter_at_all(self) -> None:
        """Second line of defence.  ``paintEvent`` blits the composite OVER the
        gutter it has just painted, so the gutter's safety cannot rest on every
        lane painter being well behaved: the composite itself is clipped to the
        track area, and its gutter strip must be entirely transparent."""
        self.zoom_onto_a_straddling_cell()
        content = self.w._content()
        image = content.toImage()
        ratio = content.devicePixelRatio()
        opaque = [
            (x, y)
            for y in range(0, image.height(), 4)
            for x in range(0, round(GUTTER_W * ratio), 4)
            if image.pixelColor(x, y).alpha() > 0
        ]
        self.assertEqual(opaque[:8], [], f"{len(opaque)} composed pixels in the gutter")

    def test_a_playhead_just_off_the_left_edge_stays_off_the_gutter(self) -> None:
        """Found while auditing the clips, not reported: ``_paint_playhead``
        paints up to 8 px OUTSIDE the track on purpose, so a playhead about to
        scroll into view is still visible — and 5 px of that landed on the
        gutter (measured: 16 gutter pixels wearing the playhead's colours).
        The overlays are drawn straight onto the widget, after the gutter, so
        they are clipped as a group in ``paintEvent``."""
        self.w.set_playhead(1000.0)
        span = 60.0
        start = 1000.0 + 5.0 * span / self.w._track_rect().width()
        self.w.set_view_range(start, start + span)  # playhead 5 px left of the track
        self.assertAlmostEqual(self.w._x_for(1000.0), GUTTER_W - 5.0, places=3)
        pixmap = self.w.grab()
        ratio = pixmap.devicePixelRatio()
        image = pixmap.toImage()
        band = self.w._lane_rect(0)
        gutter = self.w._pal.gutter.name()
        for fraction in (0.3, 0.5, 0.9):  # below the gutter's own lane label
            y = round((band.top() + band.height() * fraction) * ratio)
            for x in range(150, GUTTER_W - 1):
                self.assertEqual(
                    image.pixelColor(round(x * ratio), y).name(), gutter,
                    f"the playhead painted on the gutter at ({x}, {y})",
                )

    def test_a_lane_painter_that_escapes_still_cannot_reach_the_gutter(self) -> None:
        """The belt on its own.  A lane painter that paints the whole widget —
        a future clip bug, or none at all — must still be stopped by the
        composite's own clip, because the blit lands on top of the gutter."""
        width, height = self.w.width(), self.w.height()
        self.w._paint_video_lane = lambda painter, index: painter.fillRect(
            0, 0, width, height, QColor.fromRgb(self.GARISH)
        )
        self.w._content_key = None
        image = self.w._content().toImage()
        ratio = self.w._content().devicePixelRatio()
        for y in range(0, image.height(), 16):
            for x in range(0, round(GUTTER_W * ratio), 16):
                self.assertEqual(
                    image.pixelColor(x, y).alpha(), 0, f"escaped to the gutter at {x},{y}"
                )


class TestWaveformFollowsEveryNewAsset(_WidgetCase):
    """The reported bug: "when an audio is cut, the waveform doesnt seem to
    update.  this is NOT good.  it needs to update on every cut".

    ``_tinted_wave`` caches the re-inked PNG so a theme switch never re-runs
    ffmpeg, but it used to key that cache on the track and the ink ALONE.
    ``set_waveform`` puts a brand new ``_Wave`` in ``_waves[track]``; from the
    second one on, the cache handed back the FIRST PNG for the rest of the
    session and ``_paint_wave`` stretched it over whatever SOURCE span the new
    asset claimed.  The request path was working the whole time.
    """

    def wave_png(self, side: str) -> str:
        """A 200x240 waveform PNG with its trace on ONE side.

        Two traces that differ in SHAPE, not in height: ``_wave_crop``
        peak-normalises the vertical, so a tall PNG and a short one are drawn
        identically and would prove nothing.
        """
        path = os.path.join(_TMP.name, f"wave-{side}.png")
        if not os.path.exists(path):
            image = QImage(200, 240, QImage.Format.Format_ARGB32)
            image.fill(0)  # transparent: the trace IS the alpha
            painter = QPainter(image)
            painter.fillRect(
                0 if side == "left" else 100, 40, 100, 160, QColor(0x33, 0x99, 0xEE)
            )
            painter.end()
            image.save(path)
        return path

    @staticmethod
    def heavier_side(image: QImage, left: float, right: float) -> str:
        """Which half of ``[left, right]`` carries more of the opaque pixels."""
        middle = (left + right) / 2.0
        counts = [0, 0]
        for y in range(0, image.height(), 3):
            for x in range(round(left), round(right), 3):
                if image.pixelColor(x, y).alpha() > 128:
                    counts[0 if x < middle else 1] += 1
        return "left" if counts[0] > counts[1] else "right"

    def lane_trace_side(self, lane: int) -> str:
        """Which half of an audio lane the drawn trace is in, off a real grab."""
        pixmap = self.w.grab()
        ratio = pixmap.devicePixelRatio()
        image = pixmap.toImage()
        rect = self.w._lane_rect(lane)
        wave, background = self.w._pal.wave, self.w._pal.wave_bg
        counts = [0, 0]
        for y in range(round(rect.top()) + 6, round(rect.bottom()) - 6, 3):
            for x in range(round(rect.left()) + 3, round(rect.right()) - 3, 3):
                colour = image.pixelColor(round(x * ratio), round(y * ratio))
                if rgb_distance(colour, wave) < rgb_distance(colour, background):
                    counts[0 if x < rect.center().x() else 1] += 1
        self.assertGreater(sum(counts), 50, "no trace was drawn at all")
        return "left" if counts[0] > counts[1] else "right"

    def test_a_second_waveform_replaces_the_first_instead_of_being_dropped(self) -> None:
        self.w.set_waveform(0, self.wave_png("left"), 0.0, 2780.0)
        first = self.w._tinted_wave(0, self.w._waves[0])
        self.assertEqual(self.heavier_side(first.toImage(), 0, first.width()), "left")
        # A cut is exactly this: _after_model_change re-asks, the app renders a
        # fresh PNG over the surviving source range, set_waveform takes it.
        self.w._set_marks(100.0, 600.0)
        self.w.cut_selection()
        self.w.set_waveform(0, self.wave_png("right"), 500.0, 2780.0)
        second = self.w._tinted_wave(0, self.w._waves[0])
        self.assertEqual(
            self.heavier_side(second.toImage(), 0, second.width()), "right",
            "the cache handed back the waveform from before the cut",
        )

    def test_the_lane_really_redraws_when_a_new_waveform_arrives(self) -> None:
        """The same thing measured where the user sees it: on the lane's own
        pixels.  Same source span both times, so ONLY the new PNG can move the
        trace — a stale one leaves the lane pixel-identical."""
        lane = next(i for i, model in enumerate(self.w.lanes()) if model.kind == "audio")
        self.w.set_waveform(0, self.wave_png("left"), 0.0, 2780.0)
        self.assertEqual(self.lane_trace_side(lane), "left")
        self.w.set_waveform(0, self.wave_png("right"), 0.0, 2780.0)
        self.assertEqual(
            self.lane_trace_side(lane), "right", "the lane kept the first PNG"
        )

    def test_the_tint_is_still_cached_and_still_a_tint(self) -> None:
        """The fix is a better KEY, not a disabled cache: the PNG is fetched
        once in WAVE_COLOR and re-inked through its alpha at paint time, and a
        per-paint re-ink (or a per-request tint) would undo that."""
        self.w.set_waveform(0, self.wave_png("left"), 0.0, 2780.0)
        asset = self.w._waves[0]
        once = self.w._tinted_wave(0, asset)
        self.assertIs(self.w._tinted_wave(0, asset), once, "re-inked on every paint")
        image = once.toImage()
        inks = {
            image.pixelColor(x, y).name()
            for y in range(0, image.height(), 7)
            for x in range(0, image.width(), 7)
            if image.pixelColor(x, y).alpha() == 255
        }
        self.assertEqual(inks, {self.w._pal.wave.name()})


class TestMediaLifecycle(_WidgetCase):
    def test_loading_media_resets_everything(self) -> None:
        self.w._set_marks(10.0, 20.0)
        self.w.cut_selection()
        self.w.set_playhead(100.0)
        self.w.set_media(info_for(duration=600.0, n_audio=1))
        self.assertEqual(self.w.edited_duration(), 600.0)
        self.assertEqual(self.w.playhead(), 0.0)
        self.assertIsNone(self.w.selection())
        self.assertFalse(self.w.can_undo())
        self.assertEqual(self.w.view_range(), (0.0, 720.0))  # 600 s + a fifth of room
        self.assertEqual(len(self.w.render_plan().audio), 1)

    def test_clearing_media(self) -> None:
        self.w.set_media(None)
        self.assertFalse(self.w.has_media())
        self.assertEqual(self.w.edited_duration(), 0.0)
        self.assertEqual(self.w.keep_segments_source(), [])
        self.assertIsNone(self.w.selection())
        self.w.set_mark_in()  # must be inert, not crash
        self.assertIsNone(self.w.selection())
        self.wheel(500.0, 1, Qt.KeyboardModifier.ControlModifier)
        self.w.render(QPixmap(WIDTH, self.w.height()))

    def test_playhead_is_clamped_to_the_edit(self) -> None:
        self.w.set_playhead(9e9)
        self.assertAlmostEqual(self.w.playhead(), 2780.0)
        self.w.set_playhead(-5.0)
        self.assertEqual(self.w.playhead(), 0.0)


class TestWaveNormalisation(unittest.TestCase):
    """Per-lane peak normalisation: the fix for the flat full-file waveform."""

    def strip(self, half_height: int, total: int = 100) -> QPixmap:
        """A centred band ``2 * half_height`` px tall, like showwavespic draws."""
        image = QImage(40, total, QImage.Format.Format_ARGB32)
        image.fill(0)
        centre = total // 2
        for y in range(centre - half_height, centre + half_height):
            for x in range(40):
                image.setPixel(x, y, 0xFF5FC3F0)
        return QPixmap.fromImage(image)

    def test_a_thin_trace_is_cropped_to_itself(self) -> None:
        from ffcut.timeline import _wave_crop

        top, height = _wave_crop(self.strip(10))
        self.assertAlmostEqual(top, 40.0, delta=1.5)
        self.assertAlmostEqual(height, 20.0, delta=3.0)

    def test_the_crop_stays_centred(self) -> None:
        from ffcut.timeline import _wave_crop

        for half in (4, 10, 25, 49):
            top, height = _wave_crop(self.strip(half))
            self.assertAlmostEqual(top + height / 2.0, 50.0, delta=0.6, msg=str(half))

    def test_a_full_height_trace_is_not_cropped(self) -> None:
        from ffcut.timeline import _wave_crop

        top, height = _wave_crop(self.strip(50))
        self.assertAlmostEqual(top, 0.0, delta=0.5)
        self.assertAlmostEqual(height, 100.0, delta=0.5)

    def test_the_gain_is_capped(self) -> None:
        """A near-silent lane must be lifted, not blown up into a solid wall."""
        from ffcut.timeline import WAVE_NORM_MAX, _wave_crop

        _top, height = _wave_crop(self.strip(1))
        self.assertGreaterEqual(height, 100.0 / WAVE_NORM_MAX - 1e-6)

    def test_an_empty_png_is_left_alone(self) -> None:
        from ffcut.timeline import _wave_crop

        blank = QPixmap(40, 100)
        blank.fill(Qt.GlobalColor.transparent)
        self.assertEqual(_wave_crop(blank), (0.0, 100.0))
        self.assertEqual(_wave_crop(QPixmap()), (0.0, 1.0))

    def test_set_waveform_stores_the_crop(self) -> None:
        import os

        widget = TimelineWidget()
        widget.resize(WIDTH, widget.sizeHint().height())
        widget.set_media(info_for())
        path = os.path.join(_TMP.name, "wave-norm.png")
        self.strip(10).save(path)
        widget.set_waveform(0, path, 0.0, 2780.0)
        wave = widget._waves[0]
        self.assertLess(wave.height, wave.pix.height())
        self.assertGreater(wave.top, 0.0)
        widget.render(QPixmap(WIDTH, widget.height()))
        widget.deleteLater()
        QApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


class _LaneCase(_WidgetCase):
    """Helpers for gestures that care WHICH lane they land in."""

    def lane_y(self, lane: int) -> float:
        return self.w._lane_rect(lane).center().y()

    def press_in(self, x: float, lane: int) -> None:
        y = self.lane_y(lane)
        self.w.mousePressEvent(
            QMouseEvent(
                QEvent.Type.MouseButtonPress, QPointF(x, y), QPointF(x, y),
                Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton,
                Qt.KeyboardModifier.NoModifier,
            )
        )

    def move_in(self, x: float, lane: int) -> None:
        y = self.lane_y(lane)
        self.w.mouseMoveEvent(
            QMouseEvent(
                QEvent.Type.MouseMove, QPointF(x, y), QPointF(x, y),
                Qt.MouseButton.NoButton, Qt.MouseButton.LeftButton,
                Qt.KeyboardModifier.NoModifier,
            )
        )

    def drag_in(self, lane: int, x0: float, x1: float) -> None:
        self.press_in(x0, lane)
        self.move_in(x1, lane)
        self.w.mouseReleaseEvent(
            QMouseEvent(
                QEvent.Type.MouseButtonRelease,
                QPointF(x1, self.lane_y(lane)), QPointF(x1, self.lane_y(lane)),
                Qt.MouseButton.LeftButton, Qt.MouseButton.NoButton,
                Qt.KeyboardModifier.NoModifier,
            )
        )

    def click_header(self, lane: int, ctrl: bool = False) -> None:
        """Press the gutter header of ``lane`` the way a user would.

        The video lane has no child widget, so its press goes to the timeline;
        an audio lane's header IS a widget and handles its own press.
        """
        mods = (
            Qt.KeyboardModifier.ControlModifier if ctrl
            else Qt.KeyboardModifier.NoModifier
        )
        if lane == 0:
            y = self.lane_y(0)
            self.w.mousePressEvent(
                QMouseEvent(
                    QEvent.Type.MouseButtonPress, QPointF(20.0, y), QPointF(20.0, y),
                    Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton, mods,
                )
            )
            return
        header = self.w._headers[lane - 1]
        header.mousePressEvent(
            QMouseEvent(
                QEvent.Type.MouseButtonPress, QPointF(4.0, 4.0), QPointF(4.0, 4.0),
                Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton, mods,
            )
        )


class TestArming(_LaneCase):
    """Per-track selection: which lanes the next edit will touch."""

    def test_loading_arms_every_lane(self) -> None:
        self.assertEqual(self.w.lane_count(), 4)  # video + 3 audio
        self.assertEqual(self.w.armed_lanes(), frozenset({0, 1, 2, 3}))
        self.assertTrue(self.w.all_armed())
        self.assertFalse(any(self.w._flagged(lane) for lane in range(4)), "no badge by default")

    def test_header_click_selects_that_lane_exclusively(self) -> None:
        seen: list[object] = []
        self.w.armedChanged.connect(seen.append)
        self.click_header(2)
        self.assertEqual(self.w.armed_lanes(), frozenset({2}))
        self.assertFalse(self.w.all_armed())
        self.assertEqual(seen, [frozenset({2})])
        self.assertEqual([self.w._flagged(lane) for lane in range(4)], [False, False, True, False])

    def test_gutter_presses_arm_the_lane_under_them(self) -> None:
        """The video lane has no header widget, and the armed strip beside an
        audio header is bare gutter: both presses land on the timeline."""
        self.click_header(0)
        self.assertEqual(self.w.armed_lanes(), frozenset({0}))
        y = self.lane_y(2)
        self.w.mousePressEvent(
            QMouseEvent(
                QEvent.Type.MouseButtonPress, QPointF(1.0, y), QPointF(1.0, y),
                Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton,
                Qt.KeyboardModifier.NoModifier,
            )
        )
        self.assertEqual(self.w.armed_lanes(), frozenset({2}))

    def test_ctrl_click_toggles_a_lane_in_and_out(self) -> None:
        self.click_header(0)  # video alone
        self.click_header(2, ctrl=True)  # + audio track 1
        self.assertEqual(self.w.armed_lanes(), frozenset({0, 2}))
        self.click_header(2, ctrl=True)  # and back out
        self.assertEqual(self.w.armed_lanes(), frozenset({0}))

    def test_ctrl_click_never_leaves_nothing_armed(self) -> None:
        self.click_header(1)
        self.click_header(1, ctrl=True)  # would empty the set
        self.assertEqual(self.w.armed_lanes(), frozenset({1}))

    def test_set_armed_lanes_filters_and_falls_back_to_all(self) -> None:
        self.w.set_armed_lanes([1, 99, -4])
        self.assertEqual(self.w.armed_lanes(), frozenset({1}))
        self.w.set_armed_lanes([])
        self.assertTrue(self.w.all_armed())

    def test_a_key_re_arms_everything(self) -> None:
        self.click_header(1)
        self.key(Qt.Key.Key_A)
        self.assertTrue(self.w.all_armed())

    def test_dragging_in_an_unarmed_lane_re_arms_to_that_lane(self) -> None:
        """'Drag over the video track to select only the video track.'"""
        self.click_header(3)  # only the last audio lane is armed
        self.drag_in(0, self.x_for(100.0), self.x_for(400.0))
        self.assertEqual(self.w.armed_lanes(), frozenset({0}))
        self.assertSel((100.0, 400.0))

    def test_dragging_in_an_armed_lane_keeps_the_whole_armed_set(self) -> None:
        """'Ctrl+click both, then drag over it.'"""
        self.click_header(0)
        self.click_header(1, ctrl=True)
        self.drag_in(1, self.x_for(100.0), self.x_for(400.0))
        self.assertEqual(self.w.armed_lanes(), frozenset({0, 1}))

    def test_dragging_with_everything_armed_changes_nothing(self) -> None:
        self.drag_in(2, self.x_for(100.0), self.x_for(400.0))
        self.assertTrue(self.w.all_armed())

    def test_a_plain_click_does_not_re_arm(self) -> None:
        """Only a DRAG re-arms; a click is still just a seek."""
        self.click_header(1)
        self.press_in(self.x_for(500.0), 0)
        self.w.mouseReleaseEvent(
            QMouseEvent(
                QEvent.Type.MouseButtonRelease,
                QPointF(self.x_for(500.0), self.lane_y(0)),
                QPointF(self.x_for(500.0), self.lane_y(0)),
                Qt.MouseButton.LeftButton, Qt.MouseButton.NoButton,
                Qt.KeyboardModifier.NoModifier,
            )
        )
        self.assertEqual(self.w.armed_lanes(), frozenset({1}))

    def test_removing_a_lane_forgets_its_holes(self) -> None:
        self.click_header(3)
        self.w._set_marks(100.0, 200.0)
        self.w.cut_selection()
        self.assertEqual(self.w.render_plan().audio[2].holes, [(100.0, 200.0)])
        self.w.remove_lane(3)
        self.assertEqual(self.w.model.all_holes(), {})

    def test_new_media_re_arms_everything(self) -> None:
        self.click_header(1)
        self.w.set_media(info_for(duration=600.0, n_audio=2))
        self.assertEqual(self.w.armed_lanes(), frozenset({0, 1, 2}))


class TestLiftVsRipple(_LaneCase):
    """All lanes armed => ripple delete.  A subset => lift."""

    def video_holes(self) -> list:
        return self.w.render_plan().video[0].holes

    def audio_holes(self) -> list:
        return [lane.holes for lane in self.w.render_plan().audio]

    def test_everything_armed_still_ripples(self) -> None:
        self.w._set_marks(600.0, 700.0)
        self.assertTrue(self.w.cut_selection())
        self.assertAlmostEqual(self.w.edited_duration(), 2680.0)
        self.assertEqual(self.w.model.all_holes(), {})
        self.assertIsNone(self.w.selection())

    def test_a_subset_lifts_without_changing_the_length(self) -> None:
        self.click_header(0)  # video only
        self.w._set_marks(600.0, 700.0)
        self.assertTrue(self.w.cut_selection())
        self.assertAlmostEqual(self.w.edited_duration(), 2780.0, places=6)
        self.assertEqual(
            [(s.start, s.end) for s in self.w.keep_segments_source()],
            [(0.0, 2780.0)],
            "a lift must not touch the keep-list",
        )
        self.assertEqual(self.video_holes(), [(600.0, 700.0)])
        self.assertEqual(self.audio_holes(), [[], [], []])

    def test_a_lift_keeps_the_selection_so_you_can_lift_another_lane(self) -> None:
        self.click_header(0)
        self.w._set_marks(600.0, 700.0)
        self.w.cut_selection()
        self.assertIsNotNone(self.w.selection())
        self.click_header(2)
        self.w.cut_selection()
        self.assertEqual(self.video_holes(), [(600.0, 700.0)])
        self.assertEqual(self.audio_holes(), [[], [(600.0, 700.0)], []])

    def test_a_lift_on_several_armed_lanes_is_one_undo_step(self) -> None:
        self.click_header(1)
        self.click_header(3, ctrl=True)
        self.w._set_marks(100.0, 200.0)
        self.w.cut_selection()
        self.assertEqual(self.audio_holes(), [[(100.0, 200.0)], [], [(100.0, 200.0)]])
        self.assertTrue(self.w.undo())
        self.assertEqual(self.audio_holes(), [[], [], []])
        self.assertTrue(self.w.redo())
        self.assertEqual(len(self.audio_holes()[0]), 1)

    def test_a_lift_then_a_ripple_cut_keeps_the_hole_on_the_right_footage(self) -> None:
        self.click_header(1)
        self.w._set_marks(600.0, 700.0)
        self.w.cut_selection()  # lift on audio track 0
        self.w.clear_selection()
        self.w.set_armed_lanes(range(self.w.lane_count()))
        self.w._set_marks(100.0, 200.0)
        self.w.cut_selection()  # ripple, 100 s BEFORE the hole
        self.assertAlmostEqual(self.w.edited_duration(), 2680.0)
        self.assertEqual(self.audio_holes()[0], [(600.0, 700.0)])
        self.assertEqual(
            self.w.model.hole_spans_edited(self.w.lanes()[1].key),
            [(500.0, 600.0)],
            "did not re-map",
        )

    def test_holes_paint_without_crashing(self) -> None:
        self.click_header(0)
        self.w._set_marks(600.0, 700.0)
        self.w.cut_selection()
        self.click_header(2)
        self.w._set_marks(800.0, 900.0)
        self.w.cut_selection()
        self.w.render(QPixmap(WIDTH, self.w.height()))


class TestExternalAudioLane(_WidgetCase):
    """A dropped audio file becomes its own lane, aligned to 0:00."""

    def audio_info(self, path: str = "/tmp/bgm.mp3", duration: float = 900.0):
        from ffcut.probe import AudioFileInfo

        return AudioFileInfo(path, duration, "mp3", 2, 44100)

    def test_add_audio_file_appends_a_lane_in_one_undo_step(self) -> None:
        self.assertEqual(len(self.w._headers), 3)
        lane = self.w.add_audio_file(self.audio_info())
        self.assertEqual(lane, 4)  # lane id, video lane included
        self.assertEqual(self.w.lane_kind(lane), "audio")
        self.assertEqual(len(self.w._headers), 4)
        self.assertEqual(self.w.lane_count(), 5)
        self.assertEqual([c.path for c in self.w.lane_clips(lane)], ["/tmp/bgm.mp3"])
        self.w.render(QPixmap(WIDTH, self.w.height()))
        self.w.undo()  # the lane AND its clip: a drop is one edit
        self.assertEqual(self.w.lane_count(), 4)
        self.assertFalse(self.w.can_undo())

    def test_the_lane_is_taller_than_before_and_headers_stay_in_the_gutter(self) -> None:
        before = self.w.height()
        self.w.add_audio_file(self.audio_info())
        self.assertGreater(self.w.height(), before)
        for header in self.w._headers:
            self.assertLess(header.geometry().right(), GUTTER_W)

    def test_an_external_lane_is_its_own_plan_entry_with_its_own_mix(self) -> None:
        self.w.add_audio_file(self.audio_info())
        header = self.w._headers[3]
        header.slider.setValue(-6)
        header.mute.click()
        lanes = self.w.render_plan().audio
        self.assertEqual([lane.clips[0].path for lane in lanes[:3]], ["/tmp/x.mp4"] * 3)
        self.assertEqual(lanes[3].clips[0].path, "/tmp/bgm.mp3")
        self.assertEqual((lanes[3].gain_db, lanes[3].muted), (-6.0, True))
        self.assertTrue(all(m.gain_db == 0.0 and not m.muted for m in lanes[:3]))

    def test_an_external_lane_can_be_armed_and_lifted(self) -> None:
        self.w.add_audio_file(self.audio_info())
        self.w.set_armed_lanes([4])
        self.w._set_marks(100.0, 200.0)
        self.w.cut_selection()
        self.assertEqual(self.w.render_plan().audio[3].holes, [(100.0, 200.0)])
        self.assertAlmostEqual(self.w.edited_duration(), 2780.0)

    def test_wave_source_follows_the_lane_not_the_drop_order(self) -> None:
        """An empty lane inserted before a dropped file must not steal its
        waveform: the router asks per lane, not 'embedded first, then drops'."""
        empty = self.w.add_audio_lane()
        self.w.add_audio_file(self.audio_info())
        self.assertIsNone(self.w.wave_source(empty - 1))
        self.assertEqual(self.w.wave_source(4), ("/tmp/bgm.mp3", 0, 900.0))
        self.assertEqual(self.w.wave_source(1), ("/tmp/x.mp4", 1, 2780.0))
        self.assertIsNone(self.w.wave_source(9))

    def test_adding_a_lane_keeps_a_subset_armed(self) -> None:
        self.w.set_armed_lanes([0])
        self.w.add_audio_file(self.audio_info())
        self.assertEqual(self.w.armed_lanes(), frozenset({0}))
        self.assertFalse(self.w.is_armed(4))

    def test_adding_a_lane_with_everything_armed_arms_it_too(self) -> None:
        self.w.add_audio_file(self.audio_info())
        self.assertEqual(self.w.armed_lanes(), frozenset({0, 1, 2, 3, 4}))
        self.assertTrue(self.w.is_armed(4))

    def test_the_new_lane_asks_for_a_waveform(self) -> None:
        waves: list[tuple[int, float, float]] = []
        self.w.waveformNeeded.connect(lambda t, a, b: waves.append((t, a, b)))
        self.w.add_audio_file(self.audio_info())
        pump(0.3)
        self.assertIn(3, [t for t, _, _ in waves])

    def test_new_media_drops_external_lanes(self) -> None:
        self.w.add_audio_file(self.audio_info())
        self.w.set_media(info_for(duration=600.0, n_audio=1))
        self.assertEqual(len(self.w.render_plan().audio), 1)
        self.assertEqual(len(self.w._headers), 1)

class TestScrollbar(_WidgetCase):
    """The slim horizontal bar under the lanes."""

    def bar(self):
        return self.w._hbar

    def test_it_spans_the_track_area_only(self) -> None:
        geometry = self.bar().geometry()
        self.assertEqual(geometry.left(), GUTTER_W)
        self.assertLessEqual(geometry.right(), WIDTH)
        self.assertLessEqual(geometry.height(), 14)
        self.assertGreater(geometry.top(), self.w._lane_rect(3).bottom())

    def test_the_gutter_never_moves_with_it(self) -> None:
        before = [h.geometry() for h in self.w._headers]
        self.w.set_view_range(0.0, 100.0)
        self.bar().setValue(500_000)
        self.assertEqual([h.geometry() for h in self.w._headers], before)
        self.assertEqual(self.bar().geometry().left(), GUTTER_W)

    def test_scrollable_right_after_opening(self) -> None:
        """'I want to scroll as far as it says I can when I first open it':
        the fitted view leaves the bar a real handle to drag past the end."""
        self.w.zoom_to_fit()
        bar = self.bar()
        self.assertTrue(bar.isEnabled())
        self.assertGreater(bar.maximum(), 0)
        bar.setValue(bar.maximum())
        self.assertAlmostEqual(self.w.view_range()[1], self.w._view_total(), places=3)
        self.assertGreater(self.w.view_range()[1], self.w.edited_duration() * 1.5)
        self.w.set_view_range(0.0, 1e9)  # zoomed all the way out: nothing left to scroll
        self.assertFalse(bar.isEnabled())

    def test_zooming_rescales_it(self) -> None:
        self.w.set_view_range(0.0, 100.0)
        bar = self.bar()
        total = self.w._view_total()  # the edit plus the room past its end
        self.assertEqual(bar.maximum(), round((total - 100.0) * 1000))
        self.assertEqual(bar.pageStep(), 100_000)
        self.w.set_view_range(0.0, 500.0)
        self.assertEqual(bar.maximum(), round((total - 500.0) * 1000))
        self.assertEqual(bar.pageStep(), 500_000)

    def test_panning_moves_it(self) -> None:
        self.w.set_view_range(0.0, 100.0)
        self.assertEqual(self.bar().value(), 0)
        self.wheel(500.0, -1)  # plain wheel = pan right
        self.assertGreater(self.bar().value(), 0)
        self.assertAlmostEqual(
            self.bar().value() / 1000.0, self.w.view_range()[0], places=3
        )

    def test_round_trip_has_no_drift(self) -> None:
        """bar -> view -> bar must be the identity, or scrolling creeps."""
        self.w.set_view_range(0.0, 60.0)
        bar = self.bar()
        for value in (0, 1, 999, 1000, 123_456, 1_000_000, bar.maximum()):
            bar.setValue(value)
            self.assertEqual(bar.value(), value, "the bar corrected itself")
            start, end = self.w.view_range()
            self.assertAlmostEqual(start, value / 1000.0, places=9)
            self.assertAlmostEqual(end - start, 60.0, places=9, msg="a pan changed the span")

    def test_repeated_scrolling_returns_to_the_same_place(self) -> None:
        self.w.set_view_range(300.0, 360.0)
        bar = self.bar()
        origin = bar.value()
        view = self.w.view_range()
        for _ in range(200):
            bar.setValue(bar.value() + bar.singleStep())
            bar.setValue(bar.value() - bar.singleStep())
        self.assertEqual(bar.value(), origin)
        self.assertEqual(self.w.view_range(), view)

    def test_a_shorter_edit_rescales_it(self) -> None:
        self.w.set_view_range(0.0, 100.0)
        self.w.select_all()
        self.w._set_marks(0.0, 2000.0)
        self.w.cut_selection()
        bar = self.bar()
        self.assertLessEqual(bar.maximum(), round(self.w._view_total() * 1000))
        self.assertLessEqual(bar.value(), bar.maximum())


if __name__ == "__main__":
    unittest.main(verbosity=2)
