"""Clips as BLOCKS: select, slide, split, group, copy, delete.

Widget-level and pure-model, like ``test_tracks.py`` — nothing here imports
``ffcut.app``.  The block API is what the GUI is written against, so it is
tested on its own side of the seam.

Two invariants are worth more than any of the individual verbs, and most of
this file exists to pin them:

* blocks on one lane NEVER overlap.  ``render._audio_heads`` lays a lane's
  clips end to end through ``amix`` and ``render.visible_spans`` takes the
  FIRST clip covering an instant, so an overlap does not fail — it silently
  drops one of them, with exit code 0, which is this project's whole bug
  family (see HANDOFF).
* a moved block is a rendered block.  The graph places a clip with ``adelay``
  (audio) and a per-piece ``trim`` (video), so the render already follows
  ``Clip.start``; ``TestMovedBlocksReachTheGraph`` proves the wiring rather
  than trusting it.
"""

from __future__ import annotations

import json
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt6.QtCore import QPointF, Qt  # noqa: E402
from PyQt6.QtGui import QMouseEvent  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

_app = QApplication.instance() or QApplication(sys.argv)

from ffcut.probe import AudioFileInfo, AudioTrack, MediaInfo  # noqa: E402
from ffcut.render import build_filter_graph, visible_spans  # noqa: E402
from ffcut.timeline import GUTTER_W, MIN_BLOCK_LEN, TimelineWidget  # noqa: E402

WIDTH = 1000
HEIGHT = 420
FFMPEG = shutil.which("ffmpeg")


def video(path: str = "/tmp/main.mp4", duration: float = 120.0, n_audio: int = 1) -> MediaInfo:
    return MediaInfo(
        path, duration, 1664, 936, 60.0, "h264",
        [AudioTrack(i, "aac", 2, 48000, "eng", f"T{i + 1}") for i in range(n_audio)],
    )


def audio(path: str = "/tmp/bgm.mp3", duration: float = 200.0) -> AudioFileInfo:
    return AudioFileInfo(path, duration, "mp3", 2, 44100)


class _Case(unittest.TestCase):
    def setUp(self) -> None:
        self.w = TimelineWidget()
        self.w.resize(WIDTH, HEIGHT)
        self.w.set_media(video())
        self.addCleanup(self.w.deleteLater)

    # -- helpers -------------------------------------------------------------
    def clips(self, lane: int) -> list:
        return self.w.lane_clips(lane)

    def starts(self, lane: int) -> list[float]:
        return [round(c.start, 3) for c in self.clips(lane)]

    def point(self, t_edited: float, lane: int) -> QPointF:
        """A widget point over ``lane`` at edited time ``t``."""
        rect = self.w._lane_rect(lane)
        return QPointF(self.w._x_for(t_edited), rect.center().y())

    def press(self, t_edited: float, lane: int, mods: Qt.KeyboardModifier) -> None:
        pos = self.point(t_edited, lane)
        self.w.mousePressEvent(
            QMouseEvent(
                QMouseEvent.Type.MouseButtonPress, pos, QPointF(self.w.mapToGlobal(pos.toPoint())),
                Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton, mods,
            )
        )

    def move_to(self, t_edited: float, lane: int, mods: Qt.KeyboardModifier) -> None:
        pos = self.point(t_edited, lane)
        self.w.mouseMoveEvent(
            QMouseEvent(
                QMouseEvent.Type.MouseMove, pos, QPointF(self.w.mapToGlobal(pos.toPoint())),
                Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton, mods,
            )
        )

    def release(self, t_edited: float, lane: int) -> None:
        pos = self.point(t_edited, lane)
        self.w.mouseReleaseEvent(
            QMouseEvent(
                QMouseEvent.Type.MouseButtonRelease, pos,
                QPointF(self.w.mapToGlobal(pos.toPoint())), Qt.MouseButton.LeftButton,
                Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier,
            )
        )

    def dbl(self, t_edited: float, lane: int) -> None:
        pos = self.point(t_edited, lane)
        self.w.mouseDoubleClickEvent(
            QMouseEvent(
                QMouseEvent.Type.MouseButtonDblClick, pos,
                QPointF(self.w.mapToGlobal(pos.toPoint())), Qt.MouseButton.LeftButton,
                Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
            )
        )

    def alt_drag(self, lane: int, grab: float, drop: float, to_lane: int | None = None) -> None:
        alt = Qt.KeyboardModifier.AltModifier
        self.press(grab, lane, alt)
        self.move_to(drop, lane, alt)
        self.release(drop, to_lane if to_lane is not None else lane)

    def split_at(self, t_edited: float) -> bool:
        self.w.set_playhead(t_edited)
        return self.w.split_blocks()


# --------------------------------------------------------------------------- #
# identity
# --------------------------------------------------------------------------- #
class TestBlockIdentity(_Case):
    def test_every_block_gets_its_own_id(self) -> None:
        self.w.set_media(video(n_audio=3))
        ids = [c.cid for i in range(self.w.lane_count()) for c in self.clips(i)]
        self.assertEqual(len(ids), len(set(ids)), f"duplicate ids: {ids}")
        self.assertTrue(all(ids))

    def test_ids_are_never_reused_after_an_undo(self) -> None:
        """A redo must not hand a fresh block the id an undo just brought back —
        the selection and the clipboard both address blocks by id."""
        first = self.clips(0)[0].cid
        lane = self.w.add_video_lane()
        self.w.add_clip(lane, video("/tmp/b.mp4", 30.0), "place")
        self.w.undo()
        self.w.add_clip(lane, video("/tmp/c.mp4", 30.0), "place")
        fresh = {c.cid for i in range(self.w.lane_count()) for c in self.clips(i)}
        self.assertIn(first, fresh)
        self.assertEqual(len(fresh), len({*fresh}))

    def test_a_copy_carries_the_id_and_the_group(self) -> None:
        clip = self.clips(0)[0]
        clip.group = 7
        twin = clip.copy()
        self.assertEqual((twin.cid, twin.group), (clip.cid, 7))


# --------------------------------------------------------------------------- #
# selection
# --------------------------------------------------------------------------- #
class TestSelection(_Case):
    def test_alt_click_selects_the_block_under_the_pointer(self) -> None:
        self.press(60.0, 0, Qt.KeyboardModifier.AltModifier)
        self.assertEqual(len(self.w.clip_selection()), 2, "picture + its sound")
        self.assertIn(self.clips(0)[0].cid, self.w.clip_selection())

    def test_a_plain_click_still_only_seeks(self) -> None:
        """The cut workflow is the thing that must not be retrained."""
        moved: list[float] = []
        self.w.playheadMoved.connect(moved.append)
        self.press(60.0, 0, Qt.KeyboardModifier.NoModifier)
        self.release(60.0, 0)
        self.assertEqual(self.w.clip_selection(), frozenset())
        self.assertTrue(moved, "a plain click must still seek")

    def test_ctrl_alt_click_adds_and_removes(self) -> None:
        self.w.add_video_lane()
        self.w.add_clip(1, video("/tmp/b.mp4", 30.0), "place", with_audio=False)
        both = Qt.KeyboardModifier.AltModifier | Qt.KeyboardModifier.ControlModifier
        self.press(10.0, 0, Qt.KeyboardModifier.AltModifier)
        before = set(self.w.clip_selection())
        self.press(10.0, 1, both)
        self.assertEqual(len(self.w.clip_selection()), len(before) + 1)
        self.press(10.0, 1, both)  # again: removes it
        self.assertEqual(set(self.w.clip_selection()), before)

    def test_ctrl_click_adds_once_a_block_is_already_selected(self) -> None:
        """The user's own words: "if i ctrl click them they are both selected".
        With nothing selected Ctrl+click is still a seek, so the cut workflow
        never sees this."""
        self.w.add_video_lane()
        self.w.add_clip(1, video("/tmp/b.mp4", 30.0), "place", with_audio=False)
        ctrl = Qt.KeyboardModifier.ControlModifier
        moved: list[float] = []
        self.w.playheadMoved.connect(moved.append)

        self.press(10.0, 1, ctrl)  # nothing selected yet
        self.release(10.0, 1)
        self.assertEqual(self.w.clip_selection(), frozenset(), "Ctrl+click selected early")
        self.assertTrue(moved, "Ctrl+click stopped seeking")

        self.press(10.0, 0, Qt.KeyboardModifier.AltModifier)
        before = set(self.w.clip_selection())
        self.press(10.0, 1, ctrl)  # now it adds
        self.assertEqual(len(self.w.clip_selection()), len(before) + 1)
        self.press(10.0, 1, ctrl)  # and removes
        self.assertEqual(set(self.w.clip_selection()), before)

    def test_alt_click_on_empty_lane_space_clears(self) -> None:
        self.press(60.0, 0, Qt.KeyboardModifier.AltModifier)
        self.assertTrue(self.w.clip_selection())
        lane = self.w.add_video_lane()  # empty
        self.press(60.0, lane, Qt.KeyboardModifier.AltModifier)
        self.assertEqual(self.w.clip_selection(), frozenset())

    def test_a_deleted_block_leaves_the_selection(self) -> None:
        self.w.select_block(self.clips(0)[0])
        self.w.delete_blocks()
        self.assertEqual(self.w.clip_selection(), frozenset())

    def test_block_at_reads_through_a_cut(self) -> None:
        """x -> edited -> timeline: a click lands on the block the pixel came
        from even when a cut has collapsed material in front of it."""
        self.w._set_marks(10.0, 30.0)
        self.w.cut_selection()
        hit = self.w.block_at(self.point(50.0, 0))
        self.assertIsNotNone(hit)
        self.assertEqual(hit[0], 0)


# --------------------------------------------------------------------------- #
# double-click
# --------------------------------------------------------------------------- #
class TestDoubleClickResetsTheSelection(_Case):
    """The user's request: "if you double clicked, it kindof behaved as if you
    deselected and selected where you double clicked".

    So a double-click starts from nothing every time: whatever was selected
    before, what is selected after is exactly what the pointer is on — the
    block under it (with its group, and its extent marked), or, over bare lane,
    the keep-segment under it and NO blocks (``test_timeline`` owns that half).
    """

    def setUp(self) -> None:
        super().setUp()
        # 120 s video + one embedded audio track, welded into one group by
        # set_media; a second video lane carrying a 30 s clip of another file
        # at 0:00 gives a block that is NOT the whole timeline to aim at.
        self.other = self.w.add_video_lane()
        self.w.add_clip(self.other, video("/tmp/b.mp4", 30.0), "place", with_audio=False)

    def test_double_click_selects_the_block_under_the_cursor(self) -> None:
        self.dbl(10.0, self.other)
        clip = self.clips(self.other)[0]
        self.assertEqual(self.w.clip_selection(), frozenset({clip.cid}))

    def test_double_click_marks_the_blocks_extent(self) -> None:
        self.dbl(10.0, self.other)
        sel = self.w.selection()
        self.assertIsNotNone(sel)
        self.assertAlmostEqual(sel[0], 0.0, places=3)
        self.assertAlmostEqual(sel[1], 30.0, places=3)  # the 30 s clip, not the edit

    def test_double_click_takes_the_whole_group(self) -> None:
        """Picture and its sound come together, exactly as Alt+click does —
        double-clicking the picture and pressing Delete must not orphan audio."""
        self.dbl(60.0, 0)
        selected = {c.cid for _i, c in self.w.selected_blocks()}
        sound = next(
            i for i in range(self.w.lane_count()) if self.w.lane_kind(i) == "audio"
        )
        self.assertEqual(len(selected), 2, "picture + its embedded sound")
        self.assertIn(self.clips(0)[0].cid, selected)
        self.assertIn(self.clips(sound)[0].cid, selected)

    def test_double_click_replaces_whatever_was_selected(self) -> None:
        """The "deselect" half.  A previous selection on another lane must be
        gone, not added to — that is what Ctrl+click is for."""
        self.press(60.0, 0, Qt.KeyboardModifier.AltModifier)
        before = set(self.w.clip_selection())
        self.assertTrue(before)
        self.dbl(10.0, self.other)
        after = set(self.w.clip_selection())
        self.assertEqual(after, {self.clips(self.other)[0].cid})
        self.assertFalse(after & before, "the old selection survived the double-click")

    def test_double_click_over_a_gap_clears_instead_of_selecting(self) -> None:
        """The 30 s clip sits at 0:00 on that lane; past its end is bare lane."""
        self.press(60.0, 0, Qt.KeyboardModifier.AltModifier)
        self.assertTrue(self.w.clip_selection())
        self.dbl(90.0, self.other)
        self.assertEqual(self.w.clip_selection(), frozenset())
        self.assertEqual(self.w.selection(), (0.0, 120.0))  # the keep-segment

    def test_a_cut_block_marks_only_the_SECTION_under_the_cursor(self) -> None:
        """The block is selected whole, but the MARKS are the piece you clicked.

        Marking the whole block instead would have thrown the older gesture
        away: the main capture is one block covering the entire timeline, so a
        double-click anywhere on it would always mark the entire edit and
        "select the section between two splices" would be unreachable.
        """
        self.w._set_marks(5.0, 15.0)  # inside the 30 s clip on both lanes
        self.w.cut_selection()
        self.assertAlmostEqual(self.w.edited_duration(), 110.0)

        self.dbl(2.0, self.other)  # in the first section, 0..5
        self.assertTrue(self.w.clip_selection(), "the block itself is still selected")
        first = self.w.selection()
        self.assertIsNotNone(first)
        self.assertAlmostEqual(first[0], 0.0, places=3)
        self.assertAlmostEqual(first[1], 5.0, places=3)

        self.dbl(12.0, self.other)  # in the second section, 5..20
        second = self.w.selection()
        self.assertIsNotNone(second)
        self.assertAlmostEqual(second[0], 5.0, places=3)
        self.assertAlmostEqual(second[1], 20.0, places=3)

    def test_an_uncut_block_marks_its_whole_extent(self) -> None:
        """With no cut inside it the section IS the block, so nothing is lost."""
        self.dbl(2.0, self.other)
        sel = self.w.selection()
        self.assertIsNotNone(sel)
        self.assertAlmostEqual(sel[0], 0.0, places=3)
        self.assertAlmostEqual(sel[1], 30.0, places=3)

    def test_delete_after_a_double_click_removes_that_block(self) -> None:
        """Delete means the blocks when blocks are selected, so double-click
        then Delete is "get rid of this clip" — a GAP, not a ripple."""
        self.dbl(10.0, self.other)
        self.w.delete_blocks()
        self.assertEqual(self.clips(self.other), [])
        self.assertAlmostEqual(self.w.edited_duration(), 120.0, places=3)
        self.assertEqual(self.starts(0), [0.0], "the main video was not touched")

    def test_a_single_click_is_still_only_a_seek(self) -> None:
        """The load-bearing one: a double-click in Qt arrives AFTER a press and
        release pair, so the block behaviour must not leak into the single
        click that the cut workflow is built on."""
        moved: list[float] = []
        self.w.playheadMoved.connect(moved.append)
        self.press(10.0, self.other, Qt.KeyboardModifier.NoModifier)
        self.release(10.0, self.other)
        self.assertEqual(self.w.clip_selection(), frozenset())
        self.assertIsNone(self.w.selection())
        self.assertTrue(moved)


# --------------------------------------------------------------------------- #
# moving
# --------------------------------------------------------------------------- #
class TestMoving(_Case):
    def setUp(self) -> None:
        super().setUp()
        # Two blocks on the video lane: 0-60 and 60-120.
        self.assertTrue(self.split_at(60.0))
        self.w.clear_block_selection()

    def test_split_makes_two_blocks_and_changes_nothing_else(self) -> None:
        self.assertEqual(self.starts(0), [0.0, 60.0])
        self.assertEqual([round(c.length, 3) for c in self.clips(0)], [60.0, 60.0])
        self.assertAlmostEqual(self.w.edited_duration(), 120.0, places=3)

    def test_alt_drag_slides_a_block_along_its_lane(self) -> None:
        self.w.select_block(self.clips(0)[1])
        self.w.set_playhead(0.0)  # keep the playhead out of the snap candidates
        self.alt_drag(0, 90.0, 110.0)
        self.assertAlmostEqual(self.clips(0)[1].start, 80.0, delta=0.6)
        self.assertAlmostEqual(self.clips(0)[0].start, 0.0, places=6)

    def test_a_block_stops_against_its_neighbour_instead_of_overlapping(self) -> None:
        """The no-overlap rule: ``visible_spans`` takes the first clip covering
        an instant, so an overlap silently drops one of them."""
        self.w.select_block(self.clips(0)[1])
        self.alt_drag(0, 90.0, 10.0)  # dragged hard left, into block 0
        first, second = self.clips(0)
        self.assertAlmostEqual(second.start, first.end, places=6)
        self.assertGreaterEqual(second.start + 1e-9, first.end)

    def test_a_block_cannot_be_dragged_before_zero(self) -> None:
        self.w.select_block(self.clips(0)[0])
        self.alt_drag(0, 30.0, 0.0)
        self.assertAlmostEqual(self.clips(0)[0].start, 0.0, places=6)

    def test_a_drag_is_one_undo_step_and_a_click_is_none(self) -> None:
        self.w.select_block(self.clips(0)[1])
        before = self.w.can_undo()
        self.press(90.0, 0, Qt.KeyboardModifier.AltModifier)
        self.release(90.0, 0)  # a click: no movement
        self.assertEqual(self.w.can_undo(), before, "a click pushed an undo step")

        self.alt_drag(0, 90.0, 110.0)
        moved = self.clips(0)[1].start
        self.assertGreater(moved, 60.0)
        self.assertTrue(self.w.undo())
        self.assertEqual(self.starts(0), [0.0, 60.0], "one Ctrl+Z did not undo the drag")

    def test_dragging_onto_another_lane_appends_to_its_end(self) -> None:
        """The user's own rule: a block let go on another track goes on the END
        of what is already there, which is the only landing that cannot
        silently overlap it."""
        lane = self.w.add_video_lane()
        self.w.add_clip(lane, video("/tmp/b.mp4", 30.0), "place", with_audio=False)
        self.assertEqual(self.starts(lane), [0.0])
        self.w.select_block(self.clips(0)[1])
        self.w.ungroup_blocks()  # its sound cannot follow it onto a video lane
        self.w.select_block(self.clips(0)[1])
        self.alt_drag(0, 90.0, 100.0, to_lane=lane)
        self.assertEqual(self.starts(0), [0.0], "it did not leave its old lane")
        self.assertEqual(self.starts(lane), [0.0, 30.0], "it did not land on the end")

    def test_a_grouped_block_is_not_moved_across_lanes_alone(self) -> None:
        """Its sound lives on an audio lane and cannot follow it onto a video
        one, so moving the picture alone would break the sync the group is
        there to keep.  The horizontal move still stands."""
        lane = self.w.add_video_lane()
        self.w.select_block(self.clips(0)[1])  # picture + sound
        self.alt_drag(0, 90.0, 100.0, to_lane=lane)
        self.assertEqual(len(self.clips(0)), 2, "the picture left its sound behind")
        self.assertEqual(self.starts(lane), [])

    def test_a_block_is_not_dropped_onto_a_lane_of_the_other_kind(self) -> None:
        self.w.select_block(self.clips(0)[1])
        self.alt_drag(0, 90.0, 100.0, to_lane=1)  # lane 1 is audio
        self.assertEqual(len(self.clips(0)), 2, "a video block went to an audio lane")

    def test_a_moved_block_can_make_the_timeline_longer(self) -> None:
        self.w.select_block(self.clips(0)[1])
        self.w.set_playhead(0.0)
        self.alt_drag(0, 90.0, 150.0)
        self.assertGreater(self.w.edited_duration(), 120.0)

    def test_a_gap_is_left_behind_not_closed(self) -> None:
        self.w.select_block(self.clips(0)[1])
        self.w.set_playhead(0.0)
        self.alt_drag(0, 90.0, 110.0)
        self.assertGreater(self.clips(0)[1].start, self.clips(0)[0].end + 1.0)


# --------------------------------------------------------------------------- #
# split
# --------------------------------------------------------------------------- #
class TestSplit(_Case):
    def test_split_needs_the_playhead_inside_a_block(self) -> None:
        self.assertFalse(self.split_at(0.0), "at the very start there is nothing to cut")
        self.assertFalse(self.split_at(120.0))

    def test_split_with_nothing_selected_razors_every_armed_lane(self) -> None:
        self.assertTrue(self.split_at(50.0))
        self.assertEqual(self.starts(0), [0.0, 50.0])
        self.assertEqual(self.starts(1), [0.0, 50.0], "the audio lane was not razored")

    def test_split_with_a_selection_razors_only_that(self) -> None:
        self.w.select_block(self.clips(1)[0])
        self.w.ungroup_blocks()  # the capture arrives grouped; take one lane alone
        self.w.select_block(self.clips(1)[0])
        self.assertTrue(self.split_at(50.0))
        self.assertEqual(self.starts(0), [0.0], "the unselected lane was razored")
        self.assertEqual(self.starts(1), [0.0, 50.0])

    def test_the_halves_keep_the_right_source_range(self) -> None:
        self.split_at(50.0)
        head, tail = self.clips(0)
        self.assertAlmostEqual(head.src_in, 0.0, places=6)
        self.assertAlmostEqual(head.src_out, 50.0, places=6)
        self.assertAlmostEqual(tail.src_in, 50.0, places=6)
        self.assertAlmostEqual(tail.src_out, 120.0, places=6)

    def test_splitting_a_moved_block_splits_where_the_playhead_is(self) -> None:
        self.split_at(60.0)
        self.w.select_block(self.clips(0)[1])
        self.w.ungroup_blocks()
        self.w.select_block(self.clips(0)[1])
        self.w.set_playhead(0.0)
        self.alt_drag(0, 90.0, 110.0)
        moved = self.clips(0)[1]
        at = moved.start + 10.0
        self.w.clear_block_selection()
        self.assertTrue(self.split_at(at))
        piece = [c for c in self.clips(0) if abs(c.start - at) < 1e-6]
        self.assertEqual(len(piece), 1)
        self.assertAlmostEqual(piece[0].src_in, moved.source_at(at), places=5)

    def test_split_makes_the_group_into_two_groups(self) -> None:
        """Both halves in one group would weld them back together and make the
        split useless; no group at all would let a picture's tail slide away
        from its own sound's tail."""
        self.split_at(50.0)
        heads = {self.clips(0)[0].group, self.clips(1)[0].group}
        tails = {self.clips(0)[1].group, self.clips(1)[1].group}
        self.assertEqual(len(heads), 1, "the heads came apart")
        self.assertEqual(len(tails), 1, "the tails came apart")
        self.assertTrue(all(heads) and all(tails))
        self.assertNotEqual(heads, tails, "the halves are still welded together")


# --------------------------------------------------------------------------- #
# groups
# --------------------------------------------------------------------------- #
class TestGroups(_Case):
    def test_a_loaded_capture_arrives_grouped(self) -> None:
        """Resume item C: the picture and its sound move as one, or a block
        drag would silently pull them out of sync."""
        self.w.set_media(video(n_audio=2))
        picture = self.clips(0)[0]
        self.assertTrue(picture.group)
        for lane in (1, 2):
            self.assertEqual(self.clips(lane)[0].group, picture.group)

    def test_a_dropped_video_brings_its_sound_grouped(self) -> None:
        lane = self.w.add_video_lane()
        before = self.w.lane_count()
        clip = self.w.add_clip(lane, video("/tmp/b.mp4", 30.0, n_audio=2), "place")
        self.assertIsNotNone(clip)
        self.assertEqual(self.w.lane_count(), before + 2, "its audio did not come")
        beds = [
            c for i in range(self.w.lane_count()) for c in self.clips(i)
            if c.path == "/tmp/b.mp4" and c.kind == "audio"
        ]
        self.assertEqual(len(beds), 2)
        self.assertTrue(all(b.group == clip.group for b in beds))
        self.assertTrue(all(abs(b.start - clip.start) < 1e-9 for b in beds), "out of sync")

    def test_the_sound_lands_where_the_picture_does(self) -> None:
        lane = self.w.add_video_lane()
        self.w.add_clip(lane, video("/tmp/b.mp4", 30.0), "place")
        clip = self.w.add_clip(lane, video("/tmp/c.mp4", 20.0), "append")
        bed = next(
            c for i in range(self.w.lane_count()) for c in self.clips(i)
            if c.path == "/tmp/c.mp4" and c.kind == "audio"
        )
        self.assertAlmostEqual(bed.start, clip.start, places=6)
        self.assertGreater(bed.start, 0.0)

    def test_selecting_one_member_selects_the_group(self) -> None:
        self.w.select_block(self.clips(1)[0])
        self.assertIn(self.clips(0)[0].cid, self.w.clip_selection())

    def test_moving_one_member_moves_the_group(self) -> None:
        self.w.select_block(self.clips(0)[0])
        self.w.set_playhead(0.0)
        self.alt_drag(0, 60.0, 80.0)
        self.assertAlmostEqual(self.clips(0)[0].start, self.clips(1)[0].start, places=6)
        self.assertGreater(self.clips(0)[0].start, 0.0)

    def test_deleting_one_member_deletes_the_group(self) -> None:
        self.w.select_block(self.clips(0)[0])
        self.w.delete_blocks()
        self.assertEqual(self.clips(0), [])
        self.assertEqual(self.clips(1), [])

    def test_ungroup_frees_them(self) -> None:
        self.w.select_block(self.clips(0)[0])
        self.assertTrue(self.w.ungroup_blocks())
        self.w.select_block(self.clips(0)[0])
        self.assertEqual(len(self.w.clip_selection()), 1)

    def test_group_needs_two(self) -> None:
        self.w.set_media(video(n_audio=0))
        self.w.select_block(self.clips(0)[0])
        self.assertFalse(self.w.group_blocks())

    def test_group_and_ungroup_are_undoable(self) -> None:
        self.w.select_block(self.clips(0)[0])
        self.w.ungroup_blocks()
        self.assertFalse(self.clips(0)[0].group)
        self.assertTrue(self.w.undo())
        self.assertTrue(self.clips(0)[0].group)


# --------------------------------------------------------------------------- #
# clipboard
# --------------------------------------------------------------------------- #
class TestClipboard(_Case):
    def test_copy_then_paste_lands_at_the_playhead(self) -> None:
        self.split_at(60.0)
        self.w.clear_block_selection()
        self.w.select_block(self.clips(0)[0])
        self.w.ungroup_blocks()
        self.w.select_block(self.clips(0)[0])
        self.assertEqual(self.w.copy_blocks(), 1)
        self.w.set_playhead(120.0)
        self.assertTrue(self.w.paste_blocks())
        self.assertEqual(self.starts(0), [0.0, 60.0, 120.0])

    def test_a_paste_that_would_overlap_is_pushed_right(self) -> None:
        self.split_at(60.0)
        self.w.clear_block_selection()
        self.w.select_block(self.clips(0)[0])
        self.w.ungroup_blocks()
        self.w.select_block(self.clips(0)[0])
        self.w.copy_blocks()
        self.w.set_playhead(30.0)  # straight on top of what is already there
        self.assertTrue(self.w.paste_blocks())
        starts = self.starts(0)
        self.assertEqual(len(starts), 3)
        for a, b in zip(self.clips(0), self.clips(0)[1:]):
            self.assertGreaterEqual(b.start + 1e-9, a.end, "a paste overlapped")

    def test_pasted_blocks_are_new_and_selected(self) -> None:
        self.w.select_block(self.clips(0)[0])
        before = {c.cid for i in range(self.w.lane_count()) for c in self.clips(i)}
        self.w.copy_blocks()
        self.w.set_playhead(120.0)
        self.w.paste_blocks()
        self.assertTrue(self.w.clip_selection())
        self.assertFalse(self.w.clip_selection() & before, "a paste reused an id")

    def test_a_pasted_group_is_its_own_group(self) -> None:
        """Keeping the copied group id would weld the copies to the originals."""
        self.w.select_block(self.clips(0)[0])  # picture + sound, grouped
        self.w.copy_blocks()
        self.w.set_playhead(120.0)
        self.w.paste_blocks()
        original = self.clips(0)[0].group
        pasted = self.clips(0)[1].group
        self.assertTrue(pasted)
        self.assertNotEqual(pasted, original)
        self.assertEqual(self.clips(1)[1].group, pasted, "the pair came apart")

    def test_paste_is_undoable(self) -> None:
        self.w.select_block(self.clips(0)[0])
        self.w.copy_blocks()
        self.w.set_playhead(120.0)
        self.w.paste_blocks()
        self.assertEqual(len(self.clips(0)), 2)
        self.assertTrue(self.w.undo())
        self.assertEqual(len(self.clips(0)), 1)

    def test_nothing_copied_means_nothing_pasted(self) -> None:
        self.assertFalse(self.w.can_paste())
        self.assertFalse(self.w.paste_blocks())


# --------------------------------------------------------------------------- #
# delete
# --------------------------------------------------------------------------- #
class TestDelete(_Case):
    def test_delete_leaves_a_gap(self) -> None:
        self.split_at(40.0)
        self.split_at(80.0)
        self.w.clear_block_selection()
        self.w.select_block(self.clips(0)[1])
        self.w.ungroup_blocks()
        self.w.select_block(self.clips(0)[1])
        self.w.delete_blocks()
        self.assertEqual(self.starts(0), [0.0, 80.0], "the survivors moved")
        self.assertAlmostEqual(self.w.edited_duration(), 120.0, places=3)

    def test_delete_is_undoable(self) -> None:
        self.w.select_block(self.clips(0)[0])
        self.w.delete_blocks()
        self.assertTrue(self.w.undo())
        self.assertEqual(len(self.clips(0)), 1)

    def test_delete_with_nothing_selected_is_refused(self) -> None:
        self.assertFalse(self.w.delete_blocks())


# --------------------------------------------------------------------------- #
# edge trimming — Vegas event trim: grab an EDGE, not the middle
# --------------------------------------------------------------------------- #
class TestEdgeTrimming(_Case):
    """The user's words: "when you hover on the left or right of a clip, and if
    it was cut previously, you could drag it out and uncut the parts you drag
    out".  A trim moves the block's own in/out points, never its neighbours,
    and it can neither overlap the next block nor ask for material the file
    does not have — both of those fail silently in the render."""

    def setUp(self) -> None:
        super().setUp()
        # 0-60 and 60-120 out of one 120 s file, the tail deleted: a survivor
        # with 60 s of its own material sitting behind its right edge.
        self.assertTrue(self.split_at(60.0))
        self.w.clear_block_selection()

    def survivor(self) -> object:
        """The head block, with the tail deleted so the right edge is free."""
        self.w.select_block(self.clips(0)[1])
        self.w.delete_blocks()
        self.w.clear_block_selection()
        return self.clips(0)[0]

    def corner(self, t_edited: float, lane: int) -> QPointF:
        """A point in the TOP strip of a lane (where the fade handles live)."""
        rect = self.w._lane_rect(lane)
        return QPointF(self.w._x_for(t_edited), rect.top() + 3.0)

    def hover(self, pos: QPointF) -> Qt.CursorShape:
        self.w.mouseMoveEvent(
            QMouseEvent(
                QMouseEvent.Type.MouseMove, pos, pos, Qt.MouseButton.NoButton,
                Qt.MouseButton.NoButton, Qt.KeyboardModifier.AltModifier,
            )
        )
        return self.w.cursor().shape()

    # -- what the pointer is offered ----------------------------------------
    def test_alt_hover_on_an_edge_offers_a_resize_cursor_not_the_hand(self) -> None:
        clip = self.clips(0)[0]
        x0, x1 = self.w.block_edges(clip)
        rect = self.w._lane_rect(0)
        mid = rect.center().y()
        self.assertEqual(
            self.hover(QPointF(x0 + 2.0, mid)), Qt.CursorShape.SizeHorCursor
        )
        self.assertEqual(
            self.hover(QPointF(x1 - 2.0, mid)), Qt.CursorShape.SizeHorCursor
        )
        self.assertEqual(
            self.hover(QPointF((x0 + x1) / 2.0, mid)), Qt.CursorShape.OpenHandCursor,
            "the middle of a block stopped being a move",
        )

    def test_a_plain_hover_never_offers_a_block_handle(self) -> None:
        """No Alt, no block gesture — the plain pointer belongs to the cut
        workflow and must not start advertising trims."""
        clip = self.clips(0)[0]
        x0, _x1 = self.w.block_edges(clip)
        pos = QPointF(x0 + 2.0, self.w._lane_rect(0).center().y())
        self.w.mouseMoveEvent(
            QMouseEvent(
                QMouseEvent.Type.MouseMove, pos, pos, Qt.MouseButton.NoButton,
                Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier,
            )
        )
        self.assertNotEqual(self.w.cursor().shape(), Qt.CursorShape.SizeHorCursor)

    def test_the_top_corners_are_fades_and_the_rest_of_the_edge_is_a_trim(self) -> None:
        clip = self.clips(0)[0]
        x0, x1 = self.w.block_edges(clip)
        self.assertEqual(
            self.hover(QPointF(x0 + 2.0, self.w._lane_rect(0).top() + 3.0)),
            Qt.CursorShape.SizeBDiagCursor,
        )
        self.assertEqual(
            self.hover(QPointF(x1 - 2.0, self.w._lane_rect(0).top() + 3.0)),
            Qt.CursorShape.SizeFDiagCursor,
        )

    def test_a_block_too_narrow_to_hold_handles_keeps_its_middle(self) -> None:
        """Zoomed out far enough, a block is a few pixels wide.  Handles there
        would cover the whole thing and it could never be dragged again."""
        self.w.select_block(self.clips(0)[1])
        self.w.delete_blocks()
        self.w.clear_block_selection()
        head = self.clips(0)[0]
        head.length = 0.4  # ~3 px across a 120 s view
        pos = self.point(0.2, 0)
        self.assertEqual(self.w.block_handle(pos)[2], "move")

    # -- what a trim does ----------------------------------------------------
    def test_dragging_the_right_edge_out_restores_what_a_split_cut_off(self) -> None:
        clip = self.survivor()
        self.assertAlmostEqual(clip.src_out, 60.0, places=6)
        applied = self.w.trim_block(clip, "right", 25.0)
        self.assertAlmostEqual(applied, 25.0, places=6)
        self.assertAlmostEqual(clip.length, 85.0, places=6)
        self.assertAlmostEqual(clip.src_out, 85.0, places=6)
        self.assertAlmostEqual(clip.start, 0.0, places=6)
        self.assertAlmostEqual(self.w.edited_duration(), 85.0, places=3)

    def test_the_right_edge_stops_at_the_end_of_the_file(self) -> None:
        """``src_in + length <= source``: past it ffmpeg trims to nothing and
        exits 0 with a file shorter than the timeline says."""
        clip = self.survivor()
        self.w.trim_block(clip, "right", 500.0)
        self.assertAlmostEqual(clip.src_out, clip.source, places=6)
        self.assertAlmostEqual(clip.length, 120.0, places=6)

    def test_the_right_edge_stops_against_the_next_block(self) -> None:
        """The no-overlap rule.  An overlap does not fail — ``visible_spans``
        takes the first clip covering an instant and the other one silently
        vanishes from the render."""
        head, tail = self.clips(0)
        self.w.trim_block(head, "right", 40.0)
        self.assertAlmostEqual(head.end, tail.start, places=6)
        self.assertLessEqual(head.end, tail.start + 1e-9)

    def test_the_right_edge_can_be_pulled_back_in(self) -> None:
        clip = self.survivor()
        self.w.trim_block(clip, "right", -20.0)
        self.assertAlmostEqual(clip.length, 40.0, places=6)
        self.assertAlmostEqual(clip.src_in, 0.0, places=6)

    def test_dragging_the_left_edge_out_does_not_move_the_picture(self) -> None:
        """The whole point of a left trim: ``start`` and ``src_in`` move by the
        SAME amount, so the material already in the block stays under the same
        timeline seconds and only new material appears in front of it."""
        self.w.select_block(self.clips(0)[0])
        self.w.delete_blocks()  # leave the TAIL, 60-120, with its head cut off
        self.w.clear_block_selection()
        tail = self.clips(0)[0]
        before = tail.source_at(90.0)
        self.w.trim_block(tail, "left", -15.0)
        self.assertAlmostEqual(tail.start, 45.0, places=6)
        self.assertAlmostEqual(tail.src_in, 45.0, places=6)
        self.assertAlmostEqual(tail.length, 75.0, places=6)
        self.assertAlmostEqual(
            tail.source_at(90.0), before, places=6,
            msg="the picture that was already in the block moved",
        )

    def test_the_left_edge_stops_at_the_head_of_the_file(self) -> None:
        self.w.select_block(self.clips(0)[0])
        self.w.delete_blocks()
        self.w.clear_block_selection()
        tail = self.clips(0)[0]
        self.w.trim_block(tail, "left", -500.0)
        self.assertAlmostEqual(tail.src_in, 0.0, places=6)
        self.assertAlmostEqual(tail.start, 0.0, places=6)
        self.assertAlmostEqual(tail.length, 120.0, places=6)

    def test_the_left_edge_stops_against_the_block_before_it(self) -> None:
        head, tail = self.clips(0)
        self.w.trim_block(head, "right", -20.0)  # 0-40, a 20 s gap, then 60-120
        self.assertAlmostEqual(self.w.trim_block(tail, "left", -35.0), -20.0, places=6)
        self.assertAlmostEqual(tail.start, head.end, places=6)

    def test_a_trim_cannot_make_a_block_vanish(self) -> None:
        clip = self.survivor()
        self.w.trim_block(clip, "right", -1000.0)
        self.assertGreater(clip.length, 0.0)
        self.assertAlmostEqual(clip.length, MIN_BLOCK_LEN, places=6)

    def test_a_grouped_picture_and_its_sound_trim_together(self) -> None:
        """They share an in-point; trimming one and not the other is exactly
        the desync a group exists to prevent."""
        self.w.select_block(self.clips(0)[1])
        self.w.delete_blocks()  # the video tail and its grouped audio tail
        self.w.clear_block_selection()
        picture, sound = self.clips(0)[0], self.clips(1)[0]
        self.assertTrue(picture.group and picture.group == sound.group)
        self.w.trim_block(picture, "right", 20.0)
        self.assertAlmostEqual(picture.length, 80.0, places=6)
        self.assertAlmostEqual(sound.length, 80.0, places=6)

    def test_a_group_trims_only_as_far_as_its_shortest_member_can(self) -> None:
        """One common limit, or the pair comes out of the drag different
        lengths — a desync that no later drag can fix."""
        self.w.select_block(self.clips(0)[1])
        self.w.delete_blocks()
        self.w.clear_block_selection()
        picture, sound = self.clips(0)[0], self.clips(1)[0]
        sound.source = 70.0  # its file simply has less material behind it
        self.w.trim_block(picture, "right", 40.0)
        self.assertAlmostEqual(picture.length, 70.0, places=6)
        self.assertAlmostEqual(sound.length, 70.0, places=6)

    def test_two_grouped_blocks_on_one_lane_do_not_trim_into_each_other(self) -> None:
        """Every member moves the SAME edge by the SAME delta, so the one
        behind never gets out of the way.  Excluding group members from the
        neighbour test walked them into an overlap — and an overlap is the
        silent one: ``visible_spans`` takes the first clip and drops the other."""
        head, tail = self.clips(0)
        self.w.select_block(head)
        self.w.select_block(tail, toggle=True)
        self.assertTrue(self.w.group_blocks())
        self.assertEqual(head.group, tail.group)
        self.w.trim_block(head, "right", 30.0)
        self.assertLessEqual(head.end, tail.start + 1e-9, "the pair overlapped")

    # -- the undo contract ---------------------------------------------------
    def test_a_trim_is_one_undo_step(self) -> None:
        clip = self.survivor()
        self.w.trim_block(clip, "right", 20.0)
        self.assertTrue(self.w.undo())
        self.assertAlmostEqual(self.clips(0)[0].length, 60.0, places=6)
        self.assertAlmostEqual(self.w.edited_duration(), 60.0, places=3)

    def test_an_alt_click_on_an_edge_pushes_nothing(self) -> None:
        """There are existing tests saying a click that moves nothing leaves no
        undo step; an edge click is the same promise."""
        clip = self.clips(0)[0]
        x0, _x1 = self.w.block_edges(clip)
        pos = QPointF(x0 + 2.0, self.w._lane_rect(0).center().y())
        before = self.w.can_undo()
        self.w.mousePressEvent(
            QMouseEvent(
                QMouseEvent.Type.MouseButtonPress, pos, pos, Qt.MouseButton.LeftButton,
                Qt.MouseButton.LeftButton, Qt.KeyboardModifier.AltModifier,
            )
        )
        self.assertEqual(self.w._drag, "trim")
        self.w.mouseReleaseEvent(
            QMouseEvent(
                QMouseEvent.Type.MouseButtonRelease, pos, pos,
                Qt.MouseButton.LeftButton, Qt.MouseButton.NoButton,
                Qt.KeyboardModifier.NoModifier,
            )
        )
        self.assertEqual(self.w.can_undo(), before, "an edge click pushed an undo step")
        self.assertTrue(self.w.clip_selection(), "it did not even select the block")

    def test_an_edge_with_nowhere_to_go_pushes_nothing(self) -> None:
        clip = self.clips(0)[0]  # at 0:00, src_in 0: its left edge is pinned
        before = self.w.can_undo()
        self.assertEqual(self.w.trim_block(clip, "left", -30.0), 0.0)
        self.assertEqual(self.w.can_undo(), before)

    # -- the same thing, through the mouse ----------------------------------
    def test_the_drag_itself_trims_and_lands_on_a_whole_undo_step(self) -> None:
        """The press/move/release path, not the model call underneath it —
        this is what the pointer actually does."""
        clip = self.survivor()
        x0, x1 = self.w.block_edges(clip)
        y = self.w._lane_rect(0).center().y()
        press = QPointF(x1 - 2.0, y)
        self.w.mousePressEvent(
            QMouseEvent(
                QMouseEvent.Type.MouseButtonPress, press, press,
                Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton,
                Qt.KeyboardModifier.AltModifier,
            )
        )
        # Past the right of the widget on purpose: the last block's edge sits on
        # the last pixel of the track (the view never shows past the end of the
        # edit), and Qt keeps delivering moves outside the widget while the
        # button is down.  That is the only way this gesture can reach.
        for dx in (60.0, 200.0):
            here = QPointF(press.x() + dx, y)
            self.w.mouseMoveEvent(
                QMouseEvent(
                    QMouseEvent.Type.MouseMove, here, here, Qt.MouseButton.LeftButton,
                    Qt.MouseButton.LeftButton, Qt.KeyboardModifier.AltModifier,
                )
            )
        end = QPointF(press.x() + 200.0, y)
        self.w.mouseReleaseEvent(
            QMouseEvent(
                QMouseEvent.Type.MouseButtonRelease, end, end,
                Qt.MouseButton.LeftButton, Qt.MouseButton.NoButton,
                Qt.KeyboardModifier.NoModifier,
            )
        )
        grown = self.clips(0)[0].length
        self.assertGreater(grown, 65.0, "the drag did not restore anything")
        self.assertLessEqual(grown, 120.0 + 1e-9)
        self.assertTrue(self.w.undo())
        self.assertAlmostEqual(self.clips(0)[0].length, 60.0, places=6)

    def test_a_trimmed_edge_snaps_to_a_neighbour(self) -> None:
        """The same snap a move gets: what a trim wants is to sit flush, not
        three milliseconds short of flush."""
        head, tail = self.clips(0)
        self.w.set_playhead(0.0)
        self.w.trim_block(head, "right", -20.0)  # 0-40, gap, 60-120
        x0, _x1 = self.w.block_edges(tail)
        y = self.w._lane_rect(0).center().y()
        press = QPointF(x0 + 2.0, y)
        self.w.mousePressEvent(
            QMouseEvent(
                QMouseEvent.Type.MouseButtonPress, press, press,
                Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton,
                Qt.KeyboardModifier.AltModifier,
            )
        )
        # A couple of pixels short of the block before it: inside SNAP_PX.
        target = QPointF(self.w._x_for(40.0) + 3.0, y)
        self.w.mouseMoveEvent(
            QMouseEvent(
                QMouseEvent.Type.MouseMove, target, target, Qt.MouseButton.LeftButton,
                Qt.MouseButton.LeftButton, Qt.KeyboardModifier.AltModifier,
            )
        )
        self.w.mouseReleaseEvent(
            QMouseEvent(
                QMouseEvent.Type.MouseButtonRelease, target, target,
                Qt.MouseButton.LeftButton, Qt.MouseButton.NoButton,
                Qt.KeyboardModifier.NoModifier,
            )
        )
        self.assertAlmostEqual(self.clips(0)[1].start, 40.0, places=6)


# --------------------------------------------------------------------------- #
# fade handles — the top corners
# --------------------------------------------------------------------------- #
class TestFadeHandles(_Case):
    """The user's words: "hover over the top left or top right of a clip for a
    fade in effect".  A fade is a property of the BLOCK; the render reads it off
    the plan, so the only thing that can go wrong here is the model carrying a
    fade the block cannot hold."""

    def fade_drag(self, lane: int, clip: object, which: str, to_t: float) -> None:
        """Press the corner handle and drag it to ``to_t``."""
        x0, x1 = self.w.block_edges(clip)
        top = self.w._lane_rect(lane).top() + 3.0
        press = QPointF((x0 + 2.0) if which == "in" else (x1 - 2.0), top)
        self.w.mousePressEvent(
            QMouseEvent(
                QMouseEvent.Type.MouseButtonPress, press, press,
                Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton,
                Qt.KeyboardModifier.AltModifier,
            )
        )
        here = QPointF(self.w._x_for(to_t), top)
        self.w.mouseMoveEvent(
            QMouseEvent(
                QMouseEvent.Type.MouseMove, here, here, Qt.MouseButton.LeftButton,
                Qt.MouseButton.LeftButton, Qt.KeyboardModifier.AltModifier,
            )
        )
        self.w.mouseReleaseEvent(
            QMouseEvent(
                QMouseEvent.Type.MouseButtonRelease, here, here,
                Qt.MouseButton.LeftButton, Qt.MouseButton.NoButton,
                Qt.KeyboardModifier.NoModifier,
            )
        )

    def test_a_block_starts_with_no_fades(self) -> None:
        clip = self.clips(0)[0]
        self.assertEqual((clip.fade_in, clip.fade_out), (0.0, 0.0))

    def test_dragging_the_top_left_corner_sets_a_fade_in(self) -> None:
        clip = self.clips(0)[0]
        self.fade_drag(0, clip, "in", 12.0)
        self.assertAlmostEqual(clip.fade_in, 12.0, delta=0.5)
        self.assertEqual(clip.fade_out, 0.0)

    def test_dragging_the_top_right_corner_sets_a_fade_out(self) -> None:
        clip = self.clips(0)[0]
        self.fade_drag(0, clip, "out", 100.0)
        self.assertAlmostEqual(clip.fade_out, 20.0, delta=0.5)
        self.assertEqual(clip.fade_in, 0.0)

    def test_a_fade_drag_takes_the_group_with_it(self) -> None:
        """Fading the picture and leaving its sound at full level is not a fade
        anybody asked for."""
        self.fade_drag(0, self.clips(0)[0], "in", 10.0)
        self.assertAlmostEqual(self.clips(1)[0].fade_in, self.clips(0)[0].fade_in, places=6)

    def test_a_fade_is_one_undo_step_and_a_click_is_none(self) -> None:
        before = self.w.can_undo()
        clip = self.clips(0)[0]
        x0, _x1 = self.w.block_edges(clip)
        top = QPointF(x0 + 2.0, self.w._lane_rect(0).top() + 3.0)
        self.w.mousePressEvent(
            QMouseEvent(
                QMouseEvent.Type.MouseButtonPress, top, top, Qt.MouseButton.LeftButton,
                Qt.MouseButton.LeftButton, Qt.KeyboardModifier.AltModifier,
            )
        )
        self.w.mouseReleaseEvent(
            QMouseEvent(
                QMouseEvent.Type.MouseButtonRelease, top, top,
                Qt.MouseButton.LeftButton, Qt.MouseButton.NoButton,
                Qt.KeyboardModifier.NoModifier,
            )
        )
        self.assertEqual(self.w.can_undo(), before, "a corner click pushed an undo step")

        self.fade_drag(0, self.clips(0)[0], "in", 10.0)
        self.assertGreater(self.clips(0)[0].fade_in, 0.0)
        self.assertTrue(self.w.undo())
        self.assertEqual(self.clips(0)[0].fade_in, 0.0, "one Ctrl+Z did not undo the fade")

    def test_a_fade_handle_already_at_zero_pulled_outward_pushes_nothing(self) -> None:
        before = self.w.can_undo()
        self.fade_drag(0, self.clips(0)[0], "in", -30.0)  # dragged off the left
        self.assertEqual(self.clips(0)[0].fade_in, 0.0)
        self.assertEqual(self.w.can_undo(), before)

    def test_a_fade_cannot_outlast_its_block(self) -> None:
        clip = self.clips(0)[0]
        self.w.set_block_fade(clip, "in", 500.0)
        self.assertAlmostEqual(clip.fade_in, clip.length, places=6)

    def test_trimming_shorter_clamps_the_fade_and_trimming_back_restores_it(self) -> None:
        """Within ONE drag: the fade is recomputed from what it was at the press
        rather than accumulated, so shrinking past it and growing back is not a
        way to silently lose it."""
        self.split_at(60.0)
        self.w.clear_block_selection()
        self.w.select_block(self.clips(0)[1])
        self.w.delete_blocks()
        self.w.clear_block_selection()
        clip = self.clips(0)[0]
        self.w.set_block_fade(clip, "in", 40.0)
        members = self.w.trim_members(clip)
        self.w._edge_kind, self.w._edge_cid = "right", clip.cid
        self.w._edge_from = {
            c.cid: (c.start, c.src_in, c.length, c.fade_in, c.fade_out)
            for _, c in members
        }
        self.w._edge_limits = self.w._trim_limits(members, "right")
        self.w.trim_blocks(-30.0)  # block is 30 s now: the 40 s fade cannot fit
        self.assertAlmostEqual(clip.fade_in, 30.0, places=6)
        self.w.trim_blocks(0.0)  # dragged back out, same drag
        self.assertAlmostEqual(clip.fade_in, 40.0, places=6, msg="the fade was lost")

    def test_a_split_gives_each_half_the_fade_at_its_own_outer_end(self) -> None:
        """The razor makes two new inner edges that did not exist; a fade there
        would dip the picture in the middle of a cut nobody asked to see."""
        clip = self.clips(0)[0]
        self.w.set_block_fade(clip, "in", 5.0)
        self.w.set_block_fade(clip, "out", 7.0)
        self.w.clear_block_selection()
        self.assertTrue(self.split_at(60.0))
        head, tail = self.clips(0)
        self.assertAlmostEqual(head.fade_in, 5.0, places=6)
        self.assertEqual(head.fade_out, 0.0)
        self.assertEqual(tail.fade_in, 0.0)
        self.assertAlmostEqual(tail.fade_out, 7.0, places=6)

    def test_a_split_clamps_a_fade_longer_than_the_half_it_lands_on(self) -> None:
        clip = self.clips(0)[0]
        self.w.set_block_fade(clip, "in", 90.0)
        self.w.clear_block_selection()
        self.split_at(60.0)
        self.assertAlmostEqual(self.clips(0)[0].fade_in, 60.0, places=6)

    def test_a_copied_block_keeps_its_fades(self) -> None:
        clip = self.clips(0)[0]
        self.w.set_block_fade(clip, "in", 4.0)
        self.assertAlmostEqual(clip.copy().fade_in, 4.0, places=6)
        lane = self.w.add_audio_lane()
        self.w.add_clip(lane, audio("/tmp/bed.mp3", 30.0), "place")
        bed = self.clips(lane)[0]
        self.w.set_block_fade(bed, "out", 3.0)
        self.w.select_block(bed)
        self.assertEqual(self.w.copy_blocks(), 1)
        self.w.set_playhead(80.0)
        self.assertTrue(self.w.paste_blocks())
        self.assertAlmostEqual(self.clips(lane)[-1].fade_out, 3.0, places=6)

    def test_the_fades_reach_the_render_plan(self) -> None:
        """The model is only half the feature; what the renderer is handed is
        the other half."""
        self.w.set_block_fade(self.clips(0)[0], "in", 3.0)
        self.w.set_block_fade(self.clips(0)[0], "out", 2.0)
        plan = self.w.render_plan()
        self.assertEqual(
            (plan.video[0].clips[0].fade_in, plan.video[0].clips[0].fade_out), (3.0, 2.0)
        )
        self.assertEqual(
            (plan.audio[0].clips[0].fade_in, plan.audio[0].clips[0].fade_out), (3.0, 2.0),
            "the grouped sound did not get the fade",
        )

    def test_painting_a_faded_block_does_not_fall_over(self) -> None:
        """The wedge is drawn through the cut mapping; a cut inside the fade is
        exactly where an off-by-one would throw."""
        self.w.set_block_fade(self.clips(0)[0], "in", 20.0)
        self.w._set_marks(5.0, 10.0)
        self.w.cut_selection()
        self.w.grab()  # paints the whole strip, wedge included


# --------------------------------------------------------------------------- #
# the render actually follows
# --------------------------------------------------------------------------- #
class TestMovedBlocksReachTheGraph(_Case):
    def graph(self) -> str:
        from ffcut.render import RenderSettings, Segment

        plan = self.w.render_plan()
        settings = RenderSettings(
            segments=[Segment(0.0, self.w.edited_duration())],
            plan=plan, fps=60, tmix_frames=2, width=None, height=None,
            out_path="/tmp/out.mp4",
        )
        return build_filter_graph(video(), settings)

    def test_a_moved_audio_block_is_delayed_in_the_graph(self) -> None:
        self.w.select_block(self.clips(1)[0])
        self.w.ungroup_blocks()
        self.w.select_block(self.clips(1)[0])
        self.w.set_playhead(0.0)
        self.alt_drag(1, 60.0, 90.0)
        start = self.clips(1)[0].start
        self.assertGreater(start, 1.0)
        self.assertIn(f"adelay={start * 1000.0:g}:all=1", self.graph())

    def test_a_gap_on_the_only_video_lane_renders_as_black(self) -> None:
        self.split_at(60.0)
        self.w.clear_block_selection()
        self.w.select_block(self.clips(0)[1])
        self.w.ungroup_blocks()
        self.w.select_block(self.clips(0)[1])
        self.w.set_playhead(0.0)
        self.alt_drag(0, 90.0, 110.0)
        self.assertIn("color=black", self.graph())

    def test_two_blocks_of_one_file_are_split_before_being_trimmed(self) -> None:
        """One decoder feeding two pieces needs a ``split`` — feeding one input
        into two trims is a hard failure, not a silent one."""
        self.split_at(60.0)
        graph = self.graph()
        self.assertIn("split=", graph)



# --------------------------------------------------------------------------- #
# things a review broke, which must stay broken-proof
# --------------------------------------------------------------------------- #
class TestReviewFindings(_Case):
    """One case per defect an adversarial review reproduced.  Each docstring
    says what actually went wrong, because the fix is only obvious once you
    know that."""

    def test_a_drag_released_after_the_selection_was_cleared_does_not_abort(self) -> None:
        """Escape (or Delete) with the button still down empties the selection
        mid-drag.  `_release_block` then read a "first block" off an empty list:
        an unhandled StopIteration inside a reimplemented virtual, which PyQt
        turns into qFatal — it took the process down with the edit in it."""
        self.split_at(60.0)
        self.w.clear_block_selection()
        self.w.select_block(self.clips(0)[1])
        lane = self.w.add_video_lane()
        self.press(90.0, 0, Qt.KeyboardModifier.AltModifier)
        self.move_to(100.0, 0, Qt.KeyboardModifier.AltModifier)
        self.w.clear_block_selection()  # Escape, button still down
        self.release(100.0, lane)  # ... and let go over ANOTHER lane
        self.assertEqual(self.w.clip_selection(), frozenset())

    def test_two_drops_on_one_empty_lane_do_not_land_on_top_of_each_other(self) -> None:
        """`drop_target` decides "place" at DROP time, but the clip is placed
        after an async probe — so two files dropped on one empty lane both
        resolved to "place" and both went to 0:00.  `visible_spans` takes the
        first clip covering an instant, so the second file silently reached
        neither the render nor the preview."""
        for lane, files, kind in (
            (self.w.add_video_lane(), ("/tmp/a.mp4", "/tmp/b.mp4"), "video"),
            (self.w.add_audio_lane(), ("/tmp/a.mp3", "/tmp/b.mp3"), "audio"),
        ):
            make = video if kind == "video" else audio
            self.w.add_clip(lane, make(files[0], 40.0), "place", with_audio=False)
            self.w.add_clip(lane, make(files[1], 30.0), "place", with_audio=False)
            first, second = self.clips(lane)
            self.assertGreaterEqual(
                second.start + 1e-9, first.end,
                f"the second {kind} drop landed on top of the first",
            )
        # Both files really reach the graph: on a video lane an overlap makes
        # visible_spans drop one, on an audio lane it makes amix play both at
        # once — neither fails, both are wrong.
        plan = self.w.render_plan()
        self.assertEqual([c.path for c in plan.video[-1].clips], ["/tmp/a.mp4", "/tmp/b.mp4"])
        self.assertEqual([c.path for c in plan.audio[-1].clips], ["/tmp/a.mp3", "/tmp/b.mp3"])
        seen = {c.path for _, _, c in visible_spans(plan) if c}
        self.assertEqual(seen, {"/tmp/main.mp4"}, "lane 0 still has priority")

    def test_the_clipboard_survives_a_lane_being_inserted_above_it(self) -> None:
        """"Add video track" inserts above the audio lanes and renumbers every
        one of them, so a clipboard holding a lane INDEX pasted onto somebody
        else's lane — silently, because the kind check still passed."""
        bed = self.w.add_audio_lane()
        self.w.add_clip(bed, audio("/tmp/bed.mp3", 20.0), "place")
        key = self.w.lanes()[bed].key
        self.w.select_block(self.clips(bed)[0])
        self.assertEqual(self.w.copy_blocks(), 1)
        self.w.add_video_lane()  # every audio lane shifts down one
        moved = next(i for i, lane in enumerate(self.w.lanes()) if lane.key == key)
        self.assertNotEqual(moved, bed, "the lane did not actually move")
        self.w.set_playhead(60.0)
        self.assertTrue(self.w.paste_blocks())
        self.assertEqual(len(self.clips(moved)), 2, "it pasted onto the wrong lane")

    def test_a_pasted_group_stays_in_sync_when_one_lane_is_occupied(self) -> None:
        """`_free_start` per clip pushed only the member whose lane was busy,
        so the pair came out welded together but seconds apart — and no drag
        can fix that without ungrouping first.  One push for the whole group."""
        self.split_at(40.0)
        self.w.clear_block_selection()
        self.w.select_block(self.clips(0)[0])  # picture + sound, grouped
        self.assertEqual(self.w.copy_blocks(), 2)
        self.w.select_block(self.clips(0)[0])
        # Free the video lane's head but leave the audio lane's in place.
        self.clips(0)[0].group = 0
        self.w._selected = {self.clips(0)[0].cid}
        self.w.delete_blocks()
        self.w.set_playhead(0.0)
        self.assertTrue(self.w.paste_blocks())
        pasted = [c for _, c in self.w.selected_blocks()]
        self.assertEqual(len(pasted), 2)
        self.assertEqual(len({c.group for c in pasted}), 1, "they came unwelded")
        self.assertAlmostEqual(
            pasted[0].start, pasted[1].start, places=6,
            msg="the picture and its sound pasted out of sync",
        )

    def test_paste_still_works_when_every_block_has_been_deleted(self) -> None:
        """Deleting everything takes the duration to 0, which used to make the
        whole widget inert while Edit ▸ Paste stayed enabled and then answered
        "Nothing copied yet" with two blocks on the clipboard."""
        self.w.select_block(self.clips(0)[0])
        self.assertEqual(self.w.copy_blocks(), 2)
        self.w.delete_blocks()
        self.assertEqual(self.w.edited_duration(), 0.0)
        self.assertFalse(self.w.has_media())
        self.assertTrue(self.w.can_paste(), "the only way back said no")
        self.w.set_playhead(0.0)
        self.assertTrue(self.w.paste_blocks())
        self.assertEqual(len(self.clips(0)), 1)
        self.assertGreater(self.w.edited_duration(), 0.0)

    def test_a_pasted_group_is_shifted_to_a_FIXED_POINT_not_one_max_push(self) -> None:
        """Taking the largest single push is not enough: shifting the group can
        walk another member into something ELSE, and the final per-lane push
        then moves that one alone.  Measured at ~2% of random pastes, spreads of
        40-55 s — one group, in sync with nothing."""
        lane = self.w.add_video_lane()
        bed = self.w.add_audio_lane()
        # A grouped 5 s pair to copy, on lanes of their own.
        head = self.w.add_clip(lane, video("/tmp/g.mp4", 5.0), "place", with_audio=False)
        tail = self.w.add_clip(bed, audio("/tmp/g.mp3", 5.0), "place")
        self.w.weld_group([head, tail])
        self.w.select_block(head)
        self.assertEqual(self.w.copy_blocks(), 2)
        self.w.delete_blocks()
        # Now block the lanes so one push walks the other member into a wall.
        self.w.add_clip(lane, video("/tmp/x.mp4", 10.0), "place", with_audio=False)
        self.w.add_clip(bed, audio("/tmp/y.mp3", 2.0), "place")
        self.w.add_clip(bed, audio("/tmp/z.mp3", 88.0), "append")
        self.w.lane_clips(bed)[1].start = 12.0  # a wall at 12-100
        self.w.set_playhead(0.0)
        self.assertTrue(self.w.paste_blocks())
        pasted = [c for _, c in self.w.selected_blocks()]
        self.assertEqual(len(pasted), 2)
        self.assertAlmostEqual(
            pasted[0].start, pasted[1].start, places=6,
            msg="the group pasted out of sync (the shift was not iterated)",
        )

    def test_loose_blocks_in_one_paste_do_not_share_a_shift(self) -> None:
        """``shift`` was keyed by group id and 0 is a key like any other, so
        every UNGROUPED block in a paste took the largest push any of them
        needed — dragging a block whose own lane was free hundreds of seconds
        from the playhead."""
        lane = self.w.add_video_lane()
        bed = self.w.add_audio_lane()
        one = self.w.add_clip(lane, video("/tmp/l.mp4", 5.0), "place", with_audio=False)
        two = self.w.add_clip(bed, audio("/tmp/l.mp3", 5.0), "place")
        self.w._selected = {one.cid, two.cid}  # two loose blocks, no group
        self.assertEqual(self.w.copy_blocks(), 2)
        self.w.delete_blocks()
        self.w.add_clip(lane, video("/tmp/wall.mp4", 500.0), "place", with_audio=False)
        self.w.set_playhead(0.0)
        self.assertTrue(self.w.paste_blocks())
        by_lane = {i: self.clips(i) for i in (lane, bed)}
        self.assertAlmostEqual(
            by_lane[bed][0].start, 0.0, places=6,
            msg="a loose block was dragged along by another block's obstacle",
        )
        self.assertGreaterEqual(by_lane[lane][-1].start, 500.0)

    def test_a_wedged_block_can_still_be_dragged_to_another_track(self) -> None:
        """Refusing the undo step for a block with nowhere to slide also
        refused the DRAG, and Alt+drag is the only way to move a block between
        tracks.  The middle block of any gapless lane is exactly this shape."""
        lane = self.w.add_video_lane()
        for name in ("a", "b", "c"):
            self.w.add_clip(
                lane, video(f"/tmp/{name}.mp4", 40.0), "append", with_audio=False
            )
        target = self.w.add_video_lane()
        middle = self.clips(lane)[1]
        self.w.select_block(middle)
        self.assertEqual(self.w._move_limits(), (0.0, 0.0), "it is not wedged")
        self.alt_drag(lane, 60.0, 62.0, to_lane=target)
        self.assertEqual(len(self.clips(lane)), 2, "the wedged block never moved")
        # An EMPTY destination keeps the block where it was: there is nothing
        # to append to (see move_blocks_to_lane).
        self.assertEqual(self.starts(target), [40.0])
        self.assertTrue(self.w.can_undo())
        self.assertTrue(self.w.undo())
        self.assertEqual(len(self.clips(lane)), 3)

    def test_can_paste_is_false_when_the_lane_it_came_from_is_gone(self) -> None:
        """It returned True and then refused, so the menu offered a paste that
        answered "nothing copied" — which is not what had happened."""
        bed = self.w.add_audio_lane()
        self.w.add_clip(bed, audio("/tmp/bed.mp3", 20.0), "place")
        self.w.select_block(self.clips(bed)[0])
        self.w.copy_blocks()
        self.assertTrue(self.w.can_paste())
        self.w.remove_lane(bed)
        self.assertFalse(self.w.can_paste(), "it offered a paste with nowhere to land")
        self.assertFalse(self.w.paste_blocks())
        self.w.undo()  # the lane is back
        self.assertTrue(self.w.can_paste())

    def test_grouping_a_selection_that_is_already_that_group_is_not_an_edit(self) -> None:
        self.w.select_block(self.clips(0)[0])
        before = self.w.can_undo()
        self.assertFalse(self.w.group_blocks())
        self.assertEqual(self.w.can_undo(), before, "it pushed an undo step")

    def test_a_block_with_no_room_to_move_pushes_no_undo_step(self) -> None:
        lane = self.w.add_video_lane()
        self.w.add_clip(lane, video("/tmp/a.mp4", 40.0), "place", with_audio=False)
        self.w.add_clip(lane, video("/tmp/b.mp4", 40.0), "append", with_audio=False)
        self.w.add_clip(lane, video("/tmp/c.mp4", 40.0), "append", with_audio=False)
        self.w.select_block(self.clips(lane)[1])  # boxed in on both sides
        before = self.w.can_undo()
        self.alt_drag(lane, 60.0, 70.0)
        self.assertEqual(self.starts(lane), [0.0, 40.0, 80.0])
        self.assertEqual(self.w.can_undo(), before, "it pushed an undo step")

    def test_a_trim_released_after_the_blocks_were_deleted_does_not_abort(self) -> None:
        """The same trap the slide had: Delete with the button still down takes
        the blocks out from under the drag.  A reimplemented virtual that then
        reads "the block I grabbed" out of a list raises inside Qt, which PyQt
        turns into qFatal — it takes the process down with the edit in it."""
        self.split_at(60.0)
        self.w.clear_block_selection()
        clip = self.clips(0)[0]
        x0, _x1 = self.w.block_edges(clip)
        pos = QPointF(x0 + 2.0, self.w._lane_rect(0).center().y())
        self.w.mousePressEvent(
            QMouseEvent(
                QMouseEvent.Type.MouseButtonPress, pos, pos, Qt.MouseButton.LeftButton,
                Qt.MouseButton.LeftButton, Qt.KeyboardModifier.AltModifier,
            )
        )
        self.w.select_block(self.clips(0)[0])
        self.w.delete_blocks()  # the grabbed block is gone, button still down
        here = QPointF(pos.x() + 40.0, pos.y())
        self.w.mouseMoveEvent(
            QMouseEvent(
                QMouseEvent.Type.MouseMove, here, here, Qt.MouseButton.LeftButton,
                Qt.MouseButton.LeftButton, Qt.KeyboardModifier.AltModifier,
            )
        )
        self.w.mouseReleaseEvent(
            QMouseEvent(
                QMouseEvent.Type.MouseButtonRelease, here, here,
                Qt.MouseButton.LeftButton, Qt.MouseButton.NoButton,
                Qt.KeyboardModifier.NoModifier,
            )
        )
        self.assertEqual(len(self.clips(0)), 1, "the survivor did not survive")

    def test_a_removed_audio_lanes_waveform_does_not_survive_onto_the_next(self) -> None:
        """`_waves` is keyed by AUDIO-LANE ordinal but was pruned against the
        TOTAL lane count, so a removed lane's trace stayed in the cache."""
        self.w.set_media(video(n_audio=3))
        for track in range(3):
            self.w._waves[track] = object()  # type: ignore[assignment]
        self.w.remove_lane(3)  # the last audio lane
        self.assertEqual(sorted(self.w._waves), [0, 1], "a stale trace survived")

    def test_a_loose_block_pasted_first_does_not_push_one_group_member(self) -> None:
        """Every group's offset was worked out before anything was placed, and
        then each member went through `_free_start` again — so a loose block
        pasted ahead of the group pushed the picture past itself and left the
        sound where it was: welded, and 2 s out of sync."""
        self.split_at(2.0)
        self.split_at(4.0)
        self.w.select_block(self.clips(0)[0])  # the 0-2 heads, grouped
        self.w.ungroup_blocks()
        self.w.select_block(self.clips(0)[0])  # the loose picture head alone
        self.w.select_block(self.clips(0)[1], toggle=True)  # + the 2-4 group
        self.assertEqual(self.w.copy_blocks(), 3)
        self.w.set_playhead(0.0)
        self.assertTrue(self.w.paste_blocks())
        grouped = [c for _, c in self.w.selected_blocks() if c.group]
        self.assertEqual(len(grouped), 2)
        self.assertAlmostEqual(
            grouped[0].start, grouped[1].start, places=6,
            msg="the pasted group came out of sync",
        )

    def test_the_armed_lane_survives_a_lane_inserted_above_it(self) -> None:
        """The armed set is lane POSITIONS, and "Add video track" renumbers
        every audio lane: the arming slid onto the neighbour and a lift then
        blanked a track nobody had picked."""
        self.w.set_media(video(n_audio=3))  # video, then streams 0, 1, 2
        self.w.set_armed_lanes([2])  # stream 1
        self.w.add_video_lane()
        (armed,) = self.w.armed_lanes()
        self.assertEqual(self.w.lane_kind(armed), "audio")
        self.assertEqual(self.clips(armed)[0].stream, 1, "the arming slid")
        self.w.remove_lane(1)  # the new, empty video lane
        (armed,) = self.w.armed_lanes()
        self.assertEqual(self.clips(armed)[0].stream, 1, "the arming slid back")

    def test_undoing_a_cut_keeps_the_mix_set_after_it(self) -> None:
        """Mute and gain are not undo steps, but every snapshot carried them:
        Ctrl+Z on a cut also put the gain back to what it was before the cut,
        and the render then used that."""
        self.w.set_playhead(10.0)
        self.w.set_mark_in()
        self.w.set_playhead(20.0)
        self.w.set_mark_out()
        self.assertTrue(self.w.cut_selection())
        header = self.w._lanes[1].header
        header.slider.setValue(6)
        header.mute.setChecked(True)
        self.assertTrue(self.w.undo())
        lane = self.w.render_plan().audio[0]
        self.assertEqual((lane.gain_db, lane.muted), (6.0, True), "the undo reset the mix")

    def test_a_block_slid_entirely_into_a_cut_is_put_back(self) -> None:
        """A drag is in TIMELINE time and jumps a whole cut at a splice, so the
        neighbour clamp parked the block wholly inside the cut: gone from the
        strip and the render, and with no pixel left to click it by."""
        self.split_at(5.0)
        self.split_at(18.0)
        self.w.select_block(self.clips(0)[1])  # the 5-18 pair
        self.assertTrue(self.w.delete_blocks())
        self.w.set_playhead(10.0)
        self.w.set_mark_in()
        self.w.set_playhead(20.0)
        self.w.set_mark_out()
        self.assertTrue(self.w.cut_selection())  # source 10-20 is gone
        hints: list[str] = []
        self.w.blockHint.connect(hints.append)
        self.w.select_block(self.clips(0)[0])
        self.alt_drag(0, 2.0, 12.0)  # the clamp stops it at 13-18: all cut
        self.assertEqual(self.starts(0), [0.0, 18.0], "the block vanished into the cut")
        self.assertTrue(hints, "nobody was told why the drag did nothing")

    def test_a_split_never_leaves_a_block_shorter_than_a_trim_may(self) -> None:
        """The split edge test used the cut list's MIN_KEEP (20 ms) while a
        trim stops at MIN_BLOCK_LEN (50 ms), so S could make a block that no
        trim would ever be allowed to."""
        self.w.set_playhead(MIN_BLOCK_LEN / 2)
        self.assertFalse(self.w.split_blocks())
        self.assertEqual(len(self.clips(0)), 1)

    def test_a_group_on_one_lane_is_not_moved_across_lanes(self) -> None:
        """Only a group spread over SEVERAL lanes was refused; one whose
        members all sat on a single lane went across, though the README says
        a grouped block has to be ungrouped first."""
        self.split_at(60.0)
        for index in (0, 1):  # break up the picture+sound pairs
            self.w.select_block(self.clips(0)[index])
            self.w.ungroup_blocks()
        self.w.select_block(self.clips(0)[0])
        self.w.select_block(self.clips(0)[1], toggle=True)
        self.assertTrue(self.w.group_blocks())
        target = self.w.add_video_lane()
        self.assertFalse(self.w.can_move_to_lane(target))

    def test_wave_request_does_not_reach_across_a_splice(self) -> None:
        """The span END went back through `edited_to_source`, which answers a
        splice with the NEXT keep's start — so a view ending on a splice asked
        for the whole cut too, and the PNG came back with half the detail."""
        self.w.set_playhead(5.0)
        self.w.set_mark_in()
        self.w.set_playhead(10.0)
        self.w.set_mark_out()
        self.assertTrue(self.w.cut_selection())
        lo, hi = self.w.wave_request(0, 0.0, 5.0)
        self.assertAlmostEqual(lo, 0.0, places=6)
        self.assertAlmostEqual(hi, 5.0, places=6, msg="it asked for the cut as well")

    def test_a_lane_never_paints_the_trace_of_the_lane_removed_before_it(self) -> None:
        """`_waves` is keyed by audio ORDINAL: removing a lane handed its trace
        to the lane that took its number, which wore another stream's waveform
        until a new fetch came back."""
        from PyQt6.QtGui import QPixmap

        folder = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, folder)
        png = os.path.join(folder, "wave.png")
        pix = QPixmap(200, 40)
        pix.fill()
        self.assertTrue(pix.save(png))
        self.w.set_media(video(n_audio=2))
        for track in (0, 1):
            self.w.set_waveform(track, png, 0.0, 120.0)
        self.assertTrue(self.w.remove_lane(1))  # stream 0; stream 1 is ordinal 0 now

        class Spy:
            draws = 0

            def drawPixmap(self, *_args: object) -> None:  # noqa: N802
                Spy.draws += 1

        self.w._paint_wave(Spy(), 1, 0, self.w._waves[0], self.w._lane_rect(1))
        self.assertEqual(Spy.draws, 0, "stream 1 was drawn with stream 0's trace")



# --------------------------------------------------------------------------- #
# the invariants, under a beating
# --------------------------------------------------------------------------- #
class TestRandomisedEdits(_Case):
    """Whatever order the verbs come in, the model's two rules must hold.

    Hand-written cases test the sequences someone thought of.  This one throws
    every verb at the widget in a random order — including undo and redo
    interleaved with the edits, which is where a stale id or a half-applied
    move would show — and re-checks after EVERY step:

    * no two blocks on a lane overlap (an overlap makes ``visible_spans`` drop
      one of them silently, exit code 0);
    * no duplicate ``cid``, no zero-length block, no negative start;
    * no block asks its file for material it does not have (a trim that walks
      ``src_in`` below 0 or ``src_out`` past the file's end renders as a short
      or empty piece, with exit code 0);
    * no fade outlasts the block carrying it;
    * ``render_plan()`` still builds.

    Seeded, so a failure is reproducible; the op list is printed with it.
    """

    OPS = (
        "split", "move", "trim", "fade", "delete", "copy", "paste", "group",
        "ungroup", "undo", "redo", "cut", "addlane", "select", "clear",
    )

    def check(self, where: str) -> None:
        seen: dict[int, int] = {}
        for lane in range(self.w.lane_count()):
            clips = self.clips(lane)
            for a, b in zip(clips, clips[1:]):
                self.assertGreaterEqual(
                    b.start + 1e-6, a.end, f"{where}: blocks overlap on lane {lane}"
                )
            for clip in clips:
                self.assertGreater(clip.length, 0.0, f"{where}: zero-length {clip}")
                self.assertGreaterEqual(clip.start, -1e-9, f"{where}: {clip} before 0:00")
                self.assertNotIn(clip.cid, seen, f"{where}: duplicate id {clip.cid}")
                self.assertGreaterEqual(clip.src_in, -1e-9, f"{where}: {clip} starts before its file")
                self.assertLessEqual(
                    clip.src_out, clip.source + 1e-6,
                    f"{where}: {clip} runs past the end of its own file",
                )
                self.assertLessEqual(
                    max(clip.fade_in, clip.fade_out), clip.length + 1e-6,
                    f"{where}: {clip} has a fade longer than itself",
                )
                seen[clip.cid] = lane
        self.w.render_plan()

    def apply(self, op: str, rng: random.Random) -> None:
        w = self.w
        if op == "split":
            w.set_playhead(rng.uniform(0.0, max(1.0, w.edited_duration())))
            w.split_blocks()
        elif op == "select":
            lanes = [i for i in range(w.lane_count()) if self.clips(i)]
            if lanes:
                lane = rng.choice(lanes)
                w.select_block(rng.choice(self.clips(lane)), toggle=rng.random() < 0.4)
        elif op == "clear":
            w.clear_block_selection()
        elif op == "move":
            blocks = w.selected_blocks()
            if blocks:
                w._block_anchor = 0.0
                w._block_from = {c.cid: c.start for _, c in blocks}
                w._block_limits = w._move_limits()
                w.move_blocks(rng.uniform(-60.0, 60.0))
                w._after_lane_change(w.all_armed())
        elif op == "trim":
            lanes = [i for i in range(w.lane_count()) if self.clips(i)]
            if lanes:
                clip = rng.choice(self.clips(rng.choice(lanes)))
                w.trim_block(clip, rng.choice(("left", "right")), rng.uniform(-40.0, 40.0))
        elif op == "fade":
            lanes = [i for i in range(w.lane_count()) if self.clips(i)]
            if lanes:
                clip = rng.choice(self.clips(rng.choice(lanes)))
                w.set_block_fade(clip, rng.choice(("in", "out")), rng.uniform(0.0, 40.0))
        elif op == "delete":
            w.delete_blocks()
        elif op == "copy":
            w.copy_blocks()
        elif op == "paste":
            w.set_playhead(rng.uniform(0.0, max(1.0, w.edited_duration())))
            w.paste_blocks()
        elif op == "group":
            w.group_blocks()
        elif op == "ungroup":
            w.ungroup_blocks()
        elif op == "undo":
            w.undo()
        elif op == "redo":
            w.redo()
        elif op == "cut":
            total = w.edited_duration()
            if total > 2.0:
                at = rng.uniform(0.0, total - 1.0)
                w._set_marks(at, min(total, at + rng.uniform(0.5, 10.0)))
                w.cut_selection()
        elif op == "addlane":
            (w.add_video_lane if rng.random() < 0.5 else w.add_audio_lane)()

    def test_the_invariants_survive_random_edit_sequences(self) -> None:
        for trial in range(40):
            rng = random.Random(20260913 + trial)
            self.w.set_media(video(n_audio=2))
            done: list[str] = []
            for step in range(20):
                op = rng.choice(self.OPS)
                done.append(op)
                self.apply(op, rng)
                self.check(f"trial {trial} step {step} ({' '.join(done)})")



# --------------------------------------------------------------------------- #
# and the file that comes out of it
# --------------------------------------------------------------------------- #
class _Measured:
    """Measurements taken off a RENDERED FILE, shared by the ffmpeg cases.

    Every graph bug this project has had exits 0 and writes a short, dark or
    picture-less file, so an edit is only verified once the output has been
    looked at.  Two calibration facts, measured: a black gap in yuv420p reads
    as luma ~0.1, not 0 (limited range), and an AAC-encoded silence reads as
    about -91 dB mean, not -inf.  Assert against those, not against zero.
    """

    LOUD = -40.0  # dB: anything above this is material, below it is silence
    LIT = 40.0  # mean luma: anything above this is a picture, below it is black

    @staticmethod
    def _run(args: list[str]) -> None:
        proc = subprocess.run(
            [FFMPEG, "-v", "error", "-y", *args], capture_output=True, text=True
        )
        if proc.returncode != 0:  # pragma: no cover - setup failure
            raise RuntimeError(proc.stderr[-800:])

    def durations(self, out: str) -> dict[str, float]:
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries",
             "stream=codec_type,duration", "-of", "json", out],
            capture_output=True, text=True,
        )
        return {
            s["codec_type"]: float(s.get("duration") or 0.0)
            for s in json.loads(probe.stdout)["streams"]
        }

    def luma(self, out: str, at: float) -> float:
        """Mean luma of the frame at ``at`` seconds."""
        proc = subprocess.run(
            # file=- puts the metadata on STDOUT; without it the print goes out
            # at log level INFO and -v error swallows it.
            [FFMPEG, "-v", "error", "-ss", f"{at:g}", "-i", out, "-frames:v", "1",
             "-vf", "signalstats,metadata=print:key=lavfi.signalstats.YAVG:file=-",
             "-f", "null", "-"],
            capture_output=True, text=True,
        )
        found = re.search(r"YAVG=([\d.]+)", proc.stdout)
        self.assertIsNotNone(found, f"no frame at {at}s: {proc.stderr[-400:]}")
        return float(found.group(1))

    def volume(self, out: str, start: float, length: float) -> float:
        """Mean volume in dB over a slice of the output's audio."""
        proc = subprocess.run(
            [FFMPEG, "-v", "info", "-ss", f"{start:g}", "-t", f"{length:g}",
             "-i", out, "-af", "volumedetect", "-f", "null", "-"],
            capture_output=True, text=True,
        )
        found = re.search(r"mean_volume:\s*(-?[\d.]+) dB", proc.stderr)
        self.assertIsNotNone(found, proc.stderr[-400:])
        return float(found.group(1))


@unittest.skipIf(FFMPEG is None, "ffmpeg not on PATH")
class TestRealFfmpeg(_Measured, unittest.TestCase):
    """Moving and deleting blocks, measured on the file that comes out."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory(prefix="ffcut-blocks-")
        cls.dir = cls._tmp.name
        cls.main = os.path.join(cls.dir, "main.mp4")
        cls._run([
            "-f", "lavfi", "-i", "testsrc2=size=160x90:rate=15:duration=12",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=12",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
            "-shortest", cls.main,
        ])

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def setUp(self) -> None:
        self.w = TimelineWidget()
        self.w.resize(WIDTH, HEIGHT)
        self.w.set_media(
            MediaInfo(self.main, 12.0, 160, 90, 15.0, "h264",
                      [AudioTrack(0, "aac", 2, 48000, None, None)])
        )
        self.addCleanup(self.w.deleteLater)

    # -- the edit, as the mouse would make it --------------------------------
    def move(self, lane: int, index: int, delta: float) -> None:
        """Exactly what an Alt+drag does, minus the pixels."""
        clip = self.w.lane_clips(lane)[index]
        self.w.select_block(clip)
        self.w._block_anchor = clip.start
        self.w._block_from = {c.cid: c.start for _, c in self.w.selected_blocks()}
        self.w._block_limits = self.w._move_limits()
        self.w.move_blocks(delta)
        self.w._after_lane_change(self.w.all_armed())

    # -- the file, as ffmpeg makes it ----------------------------------------
    def render(self, name: str) -> str:
        from ffcut.render import RenderSettings, build_command

        out = os.path.join(self.dir, name)
        info = MediaInfo(self.main, 12.0, 160, 90, 15.0, "h264",
                         [AudioTrack(0, "aac", 2, 48000, None, None)])
        argv = build_command(info, RenderSettings(
            segments=self.w.keep_segments_source(), plan=self.w.render_plan(),
            fps=15, tmix_frames=0, width=None, height=None, out_path=out,
            encoder="libx264",
        ))
        proc = subprocess.run(argv, capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stderr[-1500:])
        return out

    # -- the tests -----------------------------------------------------------
    def test_a_split_changes_not_one_frame(self) -> None:
        plain = self.durations(self.render("plain.mp4"))
        self.w.set_playhead(6.0)
        self.assertTrue(self.w.split_blocks())
        split = self.durations(self.render("split.mp4"))
        self.assertAlmostEqual(split["video"], plain["video"], delta=0.1)
        self.assertAlmostEqual(split["audio"], plain["audio"], delta=0.1)
        self.assertAlmostEqual(split["video"], 12.0, delta=0.15)

    def test_a_moved_block_lands_where_it_was_put_and_leaves_black_behind(self) -> None:
        """The headline: split at 6 s, slide the tail to 10 s.  The output has
        to GROW by 4 s, be black over the gap and carry picture either side —
        and the audio has to go silent over exactly the same stretch."""
        self.w.set_playhead(6.0)
        self.w.split_blocks()
        self.move(0, 1, 4.0)  # 0-6, gap 6-10, 10-16
        self.assertAlmostEqual(self.w.lane_clips(0)[1].start, 10.0, places=6)
        got = self.durations(self.render("moved.mp4"))
        self.assertAlmostEqual(got["video"], 16.0, delta=0.2)
        self.assertAlmostEqual(got["audio"], 16.0, delta=0.2)

        out = os.path.join(self.dir, "moved.mp4")
        self.assertGreater(self.luma(out, 3.0), self.LIT)
        for black in (6.5, 8.0, 9.5):
            self.assertLess(
                self.luma(out, black), self.LIT,
                f"the gap has a picture in it at {black}s",
            )
        self.assertGreater(self.luma(out, 12.0), self.LIT)

        self.assertGreater(self.volume(out, 0.0, 5.5), self.LOUD)
        self.assertLess(self.volume(out, 6.5, 3.0), self.LOUD, "the gap is not silent")
        self.assertGreater(self.volume(out, 10.5, 5.0), self.LOUD)

    def test_deleting_a_block_leaves_a_hole_not_a_ripple(self) -> None:
        self.w.set_playhead(4.0)
        self.w.split_blocks()
        self.w.clear_block_selection()
        self.w.set_playhead(8.0)
        self.w.split_blocks()
        self.w.clear_block_selection()
        self.w.select_block(self.w.lane_clips(0)[1])
        self.assertTrue(self.w.delete_blocks())
        out = self.render("deleted.mp4")
        got = self.durations(out)
        self.assertAlmostEqual(got["video"], 12.0, delta=0.2, msg="it rippled")
        self.assertGreater(self.luma(out, 2.0), self.LIT)
        self.assertLess(self.luma(out, 6.0), self.LIT, "the hole has a picture in it")
        self.assertGreater(self.luma(out, 10.0), self.LIT)
        self.assertLess(self.volume(out, 4.5, 3.0), self.LOUD)


@unittest.skipIf(FFMPEG is None, "ffmpeg not on PATH")
class TestRealFfmpegEdges(_Measured, unittest.TestCase):
    """Edge trims and fades, measured on the file that comes out.

    The media are built to be POSITION-IDENTIFIABLE, because "the output is 4 s
    longer" is not evidence that the right 4 seconds came back:

    * ``ramp.mp4`` — luma is ``20 x t``, so the frame at source second *t*
      measures 20t (±1, verified against the source itself below).  Its audio
      is SILENT for the first 6 s and a 440 Hz tone after, so the restored
      audio can be identified the same way.
    * ``flat.mp4`` — a constant bright picture and a continuous tone, so a
      ramp in the output can only have come from a fade.
    """

    RAMP = 20.0  # luma per source second in ramp.mp4

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory(prefix="ffcut-edges-")
        cls.dir = cls._tmp.name
        cls.ramp = os.path.join(cls.dir, "ramp.mp4")
        cls._run([
            "-f", "lavfi", "-i", "color=c=black:s=160x90:r=15:d=12",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=12",
            "-filter_complex",
            "[0:v]geq=lum='20*T':cb=128:cr=128,format=yuv420p[v];"
            "[1:a]volume=0:enable='lt(t,6)'[a]",
            "-map", "[v]", "-map", "[a]", "-c:v", "libx264", "-c:a", "aac", cls.ramp,
        ])
        cls.flat = os.path.join(cls.dir, "flat.mp4")
        cls._run([
            "-f", "lavfi", "-i", "testsrc2=size=160x90:rate=15:duration=12",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=12",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
            "-shortest", cls.flat,
        ])

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def setUp(self) -> None:
        self.w = TimelineWidget()
        self.w.resize(WIDTH, HEIGHT)
        self.addCleanup(self.w.deleteLater)
        self.info = None

    def load(self, path: str) -> None:
        self.info = MediaInfo(path, 12.0, 160, 90, 15.0, "h264",
                              [AudioTrack(0, "aac", 2, 48000, None, None)])
        self.w.set_media(self.info)

    def render(self, name: str) -> str:
        from ffcut.render import RenderSettings, build_command

        out = os.path.join(self.dir, name)
        argv = build_command(self.info, RenderSettings(
            segments=self.w.keep_segments_source(), plan=self.w.render_plan(),
            fps=15, tmix_frames=0, width=None, height=None, out_path=out,
            encoder="libx264",
        ))
        proc = subprocess.run(argv, capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stderr[-1500:])
        return out

    def source_second(self, out: str, at: float) -> float:
        """Which SOURCE second of ``ramp.mp4`` the output's frame at ``at`` is."""
        return self.luma(out, at) / self.RAMP

    # -- the media really are identifiable -----------------------------------
    def test_the_ramp_media_says_which_second_it_is(self) -> None:
        """Calibration, first: every other case in here reads a position out of
        a luma, so the mapping is measured rather than assumed."""
        for second in (1.0, 5.0, 8.0, 11.0):
            self.assertAlmostEqual(
                self.luma(self.ramp, second) / self.RAMP, second, delta=0.35
            )
        self.assertLess(self.volume(self.ramp, 0.0, 5.0), self.LOUD)
        self.assertGreater(self.volume(self.ramp, 6.5, 5.0), self.LOUD)

    # -- edge trimming --------------------------------------------------------
    def test_a_right_edge_trim_puts_the_cut_material_back_in_the_file(self) -> None:
        """The headline for feature 1.  Split at 6 s, delete the tail, then drag
        the survivor's right edge back out by 4 s: the output has to GROW by 4 s
        and the new seconds have to carry the file's own 6-10 s — both picture
        and sound — not a repeat of what was already there and not black."""
        self.load(self.ramp)
        self.w.set_playhead(6.0)
        self.assertTrue(self.w.split_blocks())
        self.w.clear_block_selection()
        self.w.select_block(self.w.lane_clips(0)[1])
        self.assertTrue(self.w.delete_blocks())
        self.w.clear_block_selection()

        before = self.render("cut.mp4")
        self.assertAlmostEqual(self.durations(before)["video"], 6.0, delta=0.2)
        self.assertLess(self.volume(before, 0.0, 5.5), self.LOUD, "the tone starts at 6 s")

        clip = self.w.lane_clips(0)[0]
        self.assertAlmostEqual(self.w.trim_block(clip, "right", 4.0), 4.0, places=6)
        self.assertAlmostEqual(clip.src_out, 10.0, places=6)

        after = self.render("restored.mp4")
        got = self.durations(after)
        self.assertAlmostEqual(got["video"], 10.0, delta=0.2, msg="it did not grow")
        self.assertAlmostEqual(got["audio"], 10.0, delta=0.2)
        # The restored seconds ARE the file's 6-10 s, second for second.
        for second in (7.0, 8.0, 9.0):
            self.assertAlmostEqual(
                self.source_second(after, second), second, delta=0.5,
                msg=f"the frame at {second}s is not the file's {second}s",
            )
        # And so is the sound: silent before 6 s, the tone after it.
        self.assertLess(self.volume(after, 0.0, 5.5), self.LOUD)
        self.assertGreater(
            self.volume(after, 6.5, 3.0), self.LOUD,
            "the restored seconds came back silent",
        )

    def test_a_left_edge_trim_restores_material_without_moving_the_picture(self) -> None:
        """The left edge moves ``start`` and ``src_in`` together.  Measured: the
        seconds that were already in the block still read as the same source
        seconds afterwards, and the black in front of them turns into the
        material that belongs there."""
        self.load(self.ramp)
        self.w.set_playhead(6.0)
        self.assertTrue(self.w.split_blocks())
        self.w.clear_block_selection()
        self.w.select_block(self.w.lane_clips(0)[0])
        self.assertTrue(self.w.delete_blocks())  # 0-6 is a gap now
        self.w.clear_block_selection()

        before = self.render("gap.mp4")
        self.assertLess(self.luma(before, 3.0), self.LIT, "0-6 should be black")
        self.assertAlmostEqual(self.source_second(before, 8.0), 8.0, delta=0.5)

        tail = self.w.lane_clips(0)[0]
        self.assertAlmostEqual(self.w.trim_block(tail, "left", -4.0), -4.0, places=6)
        self.assertAlmostEqual(tail.start, 2.0, places=6)
        self.assertAlmostEqual(tail.src_in, 2.0, places=6)

        after = self.render("pulled.mp4")
        self.assertAlmostEqual(
            self.source_second(after, 8.0), 8.0, delta=0.5,
            msg="the picture that was already in the block moved",
        )
        self.assertAlmostEqual(
            self.source_second(after, 3.0), 3.0, delta=0.5,
            msg="the restored head is not the file's own 2-6 s",
        )
        self.assertLess(self.luma(after, 1.0), self.LIT, "0-2 should still be black")

    def test_a_clip_that_starts_late_does_not_take_the_whole_render_down(self) -> None:
        """A PRE-EXISTING defect, found by trimming a left edge and reachable
        without ever touching one: delete the FIRST block and the only audio
        clip starts at 6 s, so its chain is ``atrim,asetpts,adelay`` — and an
        ``adelay`` fed by an atrim does not materialise its silence, it leaves a
        PTS hole.  The first piece's ``atrim`` then takes 0.046 s of audio out
        of that hole and ``concat`` fails the render outright (exit 183).  Not
        the silent kind of failure, but a render that simply refuses, for an
        edit as ordinary as deleting the first clip.  ``aresample=async=1``
        after the delay is the fix; this measures the file it now produces.
        """
        self.load(self.ramp)
        self.w.set_playhead(6.0)
        self.assertTrue(self.w.split_blocks())
        self.w.clear_block_selection()
        self.w.select_block(self.w.lane_clips(0)[0])
        self.assertTrue(self.w.delete_blocks())
        self.w.clear_block_selection()
        self.assertAlmostEqual(self.w.lane_clips(1)[0].start, 6.0, places=6)

        out = self.render("late.mp4")  # this alone used to exit 183
        got = self.durations(out)
        self.assertAlmostEqual(got["video"], 12.0, delta=0.2)
        self.assertAlmostEqual(got["audio"], 12.0, delta=0.2, msg="the audio came out short")
        self.assertLess(self.volume(out, 0.0, 5.0), self.LOUD, "the hole is not silent")
        self.assertGreater(
            self.volume(out, 7.0, 4.0), self.LOUD,
            "the delayed audio did not land where the picture is",
        )

    # -- fades ----------------------------------------------------------------
    def test_an_audio_fade_in_really_ramps_and_the_fade_out_comes_back_down(self) -> None:
        """Per-half-second ``volumedetect`` windows, which is the only way to
        tell a fade from a filter that ran and did nothing."""
        self.load(self.flat)
        self.w.set_block_fade(self.w.lane_clips(1)[0], "in", 4.0)
        self.w.set_block_fade(self.w.lane_clips(1)[0], "out", 3.0)
        out = self.render("afade.mp4")

        climb = [self.volume(out, t, 0.5) for t in (0.0, 0.5, 1.0, 2.0, 3.0, 5.0)]
        for a, b in zip(climb, climb[1:]):
            self.assertLess(a, b, f"the fade-in does not climb: {climb}")
        self.assertGreater(climb[-1] - climb[0], 15.0, f"barely a ramp at all: {climb}")
        self.assertGreater(climb[-1], self.LOUD, "it never reaches full level")

        fall = [self.volume(out, t, 0.5) for t in (8.0, 10.0, 11.0, 11.5)]
        for a, b in zip(fall, fall[1:]):
            self.assertGreater(a, b, f"the fade-out does not fall: {fall}")
        self.assertLess(fall[-1], self.LOUD, "the last half second is not quiet")

    def test_a_video_fade_in_climbs_out_of_black(self) -> None:
        self.load(self.flat)
        self.w.set_block_fade(self.w.lane_clips(0)[0], "in", 3.0)
        out = self.render("vfade.mp4")
        climb = [self.luma(out, t) for t in (0.1, 1.0, 2.0, 2.9)]
        for a, b in zip(climb, climb[1:]):
            self.assertLess(a, b, f"the picture does not climb: {climb}")
        self.assertLess(climb[0], self.LIT, f"it does not start near black: {climb}")
        self.assertGreater(self.luma(out, 6.0), self.LIT, "it never reaches full")

    def test_a_video_fade_out_ends_in_black(self) -> None:
        self.load(self.flat)
        self.w.set_block_fade(self.w.lane_clips(0)[0], "out", 3.0)
        out = self.render("vfadeout.mp4")
        self.assertGreater(self.luma(out, 6.0), self.LIT)
        fall = [self.luma(out, t) for t in (9.5, 10.5, 11.5)]
        for a, b in zip(fall, fall[1:]):
            self.assertGreater(a, b, f"the picture does not fall: {fall}")
        self.assertLess(fall[-1], self.LIT, f"it does not end near black: {fall}")

    def test_a_video_fade_does_not_restart_on_every_piece(self) -> None:
        """The one that matters.  ``_piece_video_parts`` cuts a clip at every
        cut and visibility boundary and ``concat``s the pieces, so a fade hung
        on the piece's LOCAL time would start again after each join — a ramp
        that stutters back to black in the middle, with exit code 0.

        The edit: two blocks from one file with a gap between them, a 4 s fade
        on the second, and a ripple cut landing INSIDE that fade.  The frame
        just after the splice must be half-lit, not black again.
        """
        self.load(self.flat)
        self.w.set_playhead(6.0)
        self.assertTrue(self.w.split_blocks())
        self.w.clear_block_selection()
        tail = self.w.lane_clips(0)[1]
        self.w.select_block(tail)
        self.w._block_anchor = tail.start
        self.w._block_from = {c.cid: c.start for _, c in self.w.selected_blocks()}
        self.w._block_limits = self.w._move_limits()
        self.w.move_blocks(2.0)  # 0-6, black 6-8, 8-14
        self.w._after_lane_change(self.w.all_armed())
        self.w.clear_block_selection()
        self.w.set_block_fade(self.w.lane_clips(0)[1], "in", 4.0)
        self.assertTrue(self.w.all_armed(), "the cut below has to be a ripple")
        self.w._set_marks(9.0, 10.0)
        self.assertTrue(self.w.cut_selection())
        self.assertAlmostEqual(self.w.edited_duration(), 13.0, delta=0.01)

        out = self.render("pieces.mp4")
        self.assertLess(self.luma(out, 7.0), self.LIT, "the gap is not black")
        self.assertLess(self.luma(out, 8.2), 25.0, "the fade did not start dark")
        after_splice = self.luma(out, 9.2)
        self.assertGreater(
            after_splice, 30.0,
            f"the fade restarted after the cut (luma {after_splice} just past it)",
        )
        self.assertLess(after_splice, 85.0, "it is already at full level")
        self.assertGreater(self.luma(out, 10.2), after_splice)
        self.assertGreater(self.luma(out, 11.5), self.LIT, "it never reaches full")

    def test_a_fade_and_a_trim_survive_each_other(self) -> None:
        """Trim the block shorter than its own fade and the fade comes down with
        it — otherwise ``fade`` would run a ramp that never finishes and leave
        the whole block dark."""
        self.load(self.flat)
        clip = self.w.lane_clips(0)[0]
        self.w.set_block_fade(clip, "in", 10.0)
        self.w.trim_block(clip, "right", -8.0)  # 4 s left, fade clamped to 4 s
        self.assertAlmostEqual(clip.fade_in, 4.0, places=6)
        out = self.render("clamped.mp4")
        self.assertAlmostEqual(self.durations(out)["video"], 4.0, delta=0.2)
        self.assertLess(self.luma(out, 0.1), self.LIT)
        self.assertGreater(
            self.luma(out, 3.8), self.LIT,
            "the fade never finished: the block is dark all the way out",
        )


class TestTextBlocks(_Case):
    def test_a_title_gets_its_own_lane_and_renders_as_text_not_footage(self) -> None:
        clip = self.w.add_title(10.0, "Hello")
        lane = next(i for i in range(self.w.lane_count()) if clip in self.w.lane_clips(i))
        self.assertGreater(lane, 0, "never on the main video lane")
        plan = self.w.render_plan()
        self.assertEqual([(t.start, t.end, t.text) for t in plan.texts], [(10.0, 15.0, "Hello")])
        self.assertEqual(len(plan.video), 1, "a text-only lane is no picture layer")
        self.assertIn(self.w.add_title(30.0, "Again"), self.w.lane_clips(lane),
                      "a second title reuses the text lane")

    def test_split_and_undo_keep_the_text(self) -> None:
        clip = self.w.add_title(10.0, "Hi")
        self.w.set_title(clip, x=0.2, y=0.3)
        self.assertEqual((clip.title.x, clip.title.y), (0.2, 0.3))
        self.w.undo()
        restored = self.w.find_clip(clip.cid)
        self.assertEqual((restored.title.x, restored.title.y), (0.5, 0.85))
        self.w.set_playhead(self.w.source_to_edited(12.0))
        self.w.select_block(restored)
        self.assertTrue(self.w.split_blocks())
        texts = self.w.render_plan().texts
        self.assertEqual([(t.start, t.end, t.text) for t in texts], [(10.0, 12.0, "Hi"), (12.0, 15.0, "Hi")])

    def test_a_title_fades_and_stretches_both_ways(self) -> None:
        clip = self.w.add_title(10.0, "Hi")
        self.w.set_fade(clip, "in", 2.0)
        self.w.set_title(clip, font="Serif", bold=False, color="#ff0000")
        text = self.w.render_plan().texts[0]
        self.assertEqual((text.fade_in, text.font, text.bold, text.color), (2.0, "Serif", False, "#ff0000"))
        self.w.trim_block(clip, "left", -8.0)
        self.w.trim_block(clip, "right", 20.0)
        self.assertEqual((clip.start, clip.end), (2.0, 35.0))

    def test_titles_at_follows_edited_time(self) -> None:
        clip = self.w.add_title(10.0, "Hi")
        self.assertEqual(self.w.titles_at(12.0), [clip])
        self.assertEqual(self.w.titles_at(16.0), [])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
