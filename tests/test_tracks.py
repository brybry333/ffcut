"""Multi-clip, multi-lane timeline: the model, the API seam and the graph.

Everything here is widget-level or pure-model; nothing imports ``ffcut.app``.
That is deliberate — the lane API is the contract the GUI is written against,
so it is tested on its own side of the seam and cannot be broken by chrome.

``TestRealFfmpeg`` is the exception and is worth its couple of seconds: the
padding bugs this file guards against are the kind that pass every string
assertion and still write a short file with exit code 0.  It generates its own
media with ``-f lavfi`` and measures the OUTPUT's stream durations.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt6.QtCore import QPointF  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

_app = QApplication.instance() or QApplication(sys.argv)

from ffcut.probe import AudioFileInfo, AudioTrack, MediaInfo  # noqa: E402
from ffcut.render import (  # noqa: E402
    AudioClip,
    AudioLane,
    RenderSettings,
    Segment,
    TimelinePlan,
    VideoClip,
    VideoLane,
    build_command,
    build_filter_graph,
    input_seeks,
    plan_inputs,
    visible_spans,
)
from ffcut.timeline import TimelineWidget  # noqa: E402

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


def graph_of(argv: list[str]) -> str:
    return argv[argv.index("-filter_complex") + 1]


class _Case(unittest.TestCase):
    def setUp(self) -> None:
        self.w = TimelineWidget()
        self.w.resize(WIDTH, HEIGHT)
        self.w.set_media(video())
        self.addCleanup(self.w.deleteLater)

    def kinds(self) -> list[str]:
        return [self.w.lane_kind(i) for i in range(self.w.lane_count())]


# --------------------------------------------------------------------------- #
# lanes
# --------------------------------------------------------------------------- #
class TestLaneStack(_Case):
    def test_loading_gives_one_video_lane_and_one_lane_per_track(self) -> None:
        self.w.set_media(video(n_audio=3))
        self.assertEqual(self.kinds(), ["video", "audio", "audio", "audio"])

    def test_add_video_lane_lands_under_the_main_video_not_over_it(self) -> None:
        """'Video track 2' goes BELOW: lane 0 keeps priority and lane 1 is what
        shows through its holes.  Audio stays underneath both."""
        lane = self.w.add_video_lane()
        self.assertEqual(lane, 1)
        self.assertEqual(self.kinds(), ["video", "video", "audio"])

    def test_add_audio_lane_lands_at_the_bottom(self) -> None:
        lane = self.w.add_audio_lane()
        self.assertEqual(lane, 2)
        self.assertEqual(self.kinds(), ["video", "audio", "audio"])
        self.assertEqual(self.w.lane_clips(lane), [])

    def test_a_new_lane_is_empty(self) -> None:
        self.assertEqual(self.w.lane_clips(self.w.add_video_lane()), [])

    def test_remove_lane_drops_it(self) -> None:
        lane = self.w.add_audio_lane()
        self.assertTrue(self.w.remove_lane(lane))
        self.assertEqual(self.kinds(), ["video", "audio"])

    def test_the_main_video_lane_cannot_be_removed(self) -> None:
        """It holds the loaded capture and the output geometry; File ▸ Open is
        how that changes."""
        self.assertFalse(self.w.remove_lane(0))
        self.assertFalse(self.w.remove_lane(99))
        self.assertEqual(self.kinds(), ["video", "audio"])

    def test_lane_kind_of_an_unknown_lane_is_empty(self) -> None:
        self.assertEqual(self.w.lane_kind(99), "")

    def test_inserting_a_video_lane_does_not_move_an_audio_lanes_holes(self) -> None:
        """Holes hang off a stable lane key, not a position."""
        self.w.set_armed_lanes([1])  # the audio lane only
        self.w._set_marks(10.0, 20.0)
        self.w.cut_selection()
        self.assertEqual(self.w.render_plan().audio[0].holes, [(10.0, 20.0)])
        self.w.add_video_lane()  # audio slides from lane 1 to lane 2
        plan = self.w.render_plan()
        self.assertEqual(plan.audio[0].holes, [(10.0, 20.0)])
        self.assertEqual([lane.holes for lane in plan.video], [[], []])


# --------------------------------------------------------------------------- #
# clips
# --------------------------------------------------------------------------- #
class TestClips(_Case):
    def test_place_puts_a_clip_at_zero_on_an_empty_lane(self) -> None:
        lane = self.w.add_video_lane()
        self.assertTrue(self.w.add_clip(lane, video("/tmp/b.mp4", 30.0), "place"))
        clip = self.w.lane_clips(lane)[0]
        self.assertEqual((clip.start, clip.length, clip.path), (0.0, 30.0, "/tmp/b.mp4"))

    def test_append_puts_it_after_the_existing_content(self) -> None:
        self.assertTrue(self.w.add_clip(0, video("/tmp/b.mp4", 30.0), "append"))
        clips = self.w.lane_clips(0)
        self.assertEqual([c.start for c in clips], [0.0, 120.0])
        self.assertEqual([c.length for c in clips], [120.0, 30.0])

    def test_appending_twice_stacks_end_to_end(self) -> None:
        self.w.add_clip(0, video("/tmp/b.mp4", 30.0), "append")
        self.w.add_clip(0, video("/tmp/c.mp4", 10.0), "append")
        self.assertEqual([c.start for c in self.w.lane_clips(0)], [0.0, 120.0, 150.0])

    def test_a_video_file_is_refused_by_an_audio_lane_and_the_other_way(self) -> None:
        self.assertFalse(self.w.add_clip(1, video("/tmp/b.mp4", 30.0), "place"))
        self.assertFalse(self.w.add_clip(0, audio(), "append"))
        self.assertEqual(len(self.w.lane_clips(0)), 1)

    def test_an_unknown_lane_and_a_zero_length_file_are_refused(self) -> None:
        self.assertFalse(self.w.add_clip(99, video(), "place"))
        self.assertFalse(self.w.add_clip(0, video("/tmp/b.mp4", 0.0), "append"))

    def test_audio_appends_to_its_own_lane(self) -> None:
        lane = self.w.add_audio_lane()
        self.w.add_clip(lane, audio("/tmp/a.mp3", 10.0), "place")
        self.w.add_clip(lane, audio("/tmp/b.mp3", 5.0), "append")
        self.assertEqual([c.start for c in self.w.lane_clips(lane)], [0.0, 10.0])


class TestDropTarget(_Case):
    def y_of(self, lane: int) -> float:
        rect = self.w._lane_rect(lane)
        return rect.top() + rect.height() / 2

    def test_a_lane_with_content_appends(self) -> None:
        self.assertEqual(
            self.w.drop_target(QPointF(400.0, self.y_of(0))), ("append", 0)
        )

    def test_an_empty_lane_places(self) -> None:
        lane = self.w.add_video_lane()
        self.assertEqual(
            self.w.drop_target(QPointF(400.0, self.y_of(lane))), ("place", lane)
        )

    def test_above_the_stack_clamps_to_the_main_video_lane(self) -> None:
        """The user's words: hover above the main track and it goes on the end
        of the first main video."""
        self.assertEqual(self.w.drop_target(QPointF(400.0, 1.0)), ("append", 0))

    def test_below_the_stack_clamps_to_the_last_lane(self) -> None:
        mode, lane = self.w.drop_target(QPointF(400.0, float(HEIGHT + 50)))
        self.assertEqual(lane, self.w.lane_count() - 1)
        self.assertEqual(mode, "append")

    def test_no_media_means_no_target(self) -> None:
        self.w.set_media(None)
        self.assertIsNone(self.w.drop_target(QPointF(400.0, 40.0)))


# --------------------------------------------------------------------------- #
# duration
# --------------------------------------------------------------------------- #
class TestTimelineLength(_Case):
    def test_a_longer_audio_lane_extends_the_timeline(self) -> None:
        """v3 trimmed the music to the video.  The user wants the opposite."""
        self.assertAlmostEqual(self.w.edited_duration(), 120.0)
        lane = self.w.add_audio_lane()
        self.w.add_clip(lane, audio("/tmp/long.mp3", 200.0), "place")
        self.assertAlmostEqual(self.w.edited_duration(), 200.0)

    def test_a_shorter_audio_lane_changes_nothing(self) -> None:
        lane = self.w.add_audio_lane()
        self.w.add_clip(lane, audio("/tmp/short.mp3", 5.0), "place")
        self.assertAlmostEqual(self.w.edited_duration(), 120.0)

    def test_an_appended_video_clip_extends_it(self) -> None:
        self.w.add_clip(0, video("/tmp/b.mp4", 40.0), "append")
        self.assertAlmostEqual(self.w.edited_duration(), 160.0)

    def test_removing_the_longest_lane_shortens_it_again(self) -> None:
        lane = self.w.add_audio_lane()
        self.w.add_clip(lane, audio("/tmp/long.mp3", 200.0), "place")
        self.w.remove_lane(lane)
        self.assertAlmostEqual(self.w.edited_duration(), 120.0)

    def test_a_cut_still_collapses_over_the_extended_length(self) -> None:
        lane = self.w.add_audio_lane()
        self.w.add_clip(lane, audio("/tmp/long.mp3", 200.0), "place")
        self.w._set_marks(150.0, 170.0)
        self.w.cut_selection()
        self.assertAlmostEqual(self.w.edited_duration(), 180.0)
        self.assertAlmostEqual(self.w.edited_to_source(155.0), 175.0)


# --------------------------------------------------------------------------- #
# undo / redo
# --------------------------------------------------------------------------- #
class TestUndoAcrossClips(_Case):
    def test_adding_a_clip_is_undoable(self) -> None:
        self.w.add_clip(0, video("/tmp/b.mp4", 30.0), "append")
        self.assertTrue(self.w.can_undo())
        self.assertTrue(self.w.undo())
        self.assertEqual(len(self.w.lane_clips(0)), 1)
        self.assertAlmostEqual(self.w.edited_duration(), 120.0)
        self.assertTrue(self.w.redo())
        self.assertEqual(len(self.w.lane_clips(0)), 2)
        self.assertAlmostEqual(self.w.edited_duration(), 150.0)

    def test_adding_a_lane_is_undoable(self) -> None:
        self.w.add_video_lane()
        self.assertEqual(self.kinds(), ["video", "video", "audio"])
        self.w.undo()
        self.assertEqual(self.kinds(), ["video", "audio"])
        self.w.redo()
        self.assertEqual(self.kinds(), ["video", "video", "audio"])

    def test_removing_a_lane_is_undoable_with_its_clips(self) -> None:
        lane = self.w.add_audio_lane()
        self.w.add_clip(lane, audio("/tmp/a.mp3", 40.0), "place")
        self.w.remove_lane(lane)
        self.assertEqual(self.w.lane_count(), 2)
        self.w.undo()
        self.assertEqual(self.w.lane_count(), 3)
        self.assertEqual([c.path for c in self.w.lane_clips(lane)], ["/tmp/a.mp3"])

    def test_clip_and_cut_edits_share_one_history(self) -> None:
        """Two stacks would make Ctrl+Z walk them in an order nobody predicts."""
        self.w._set_marks(10.0, 20.0)
        self.w.cut_selection()
        self.w.add_clip(0, video("/tmp/b.mp4", 30.0), "append")
        self.assertAlmostEqual(self.w.edited_duration(), 140.0)
        self.w.undo()  # the clip
        self.assertAlmostEqual(self.w.edited_duration(), 110.0)
        self.w.undo()  # the cut
        self.assertAlmostEqual(self.w.edited_duration(), 120.0)
        self.assertFalse(self.w.can_undo())

    def test_undoing_a_clip_restores_the_lane_mix(self) -> None:
        lane = self.w.add_audio_lane()
        self.w._lanes[lane].header.slider.setValue(-6)
        self.w.add_clip(lane, audio("/tmp/a.mp3", 10.0), "place")
        self.w.undo()
        self.assertEqual(self.w._lanes[lane].gain_db, -6.0)


# --------------------------------------------------------------------------- #
# priority / holes
# --------------------------------------------------------------------------- #
class TestLanePriority(unittest.TestCase):
    """Lane 1 shows through lane 0's holes and past its end."""

    def plan(self, holes: list[tuple[float, float]]) -> TimelinePlan:
        return TimelinePlan(
            duration=20.0,
            video=[
                VideoLane([VideoClip("/a.mp4", 0.0, 0.0, 12.0)], holes=holes),
                VideoLane([VideoClip("/b.mp4", 0.0, 0.0, 20.0)]),
            ],
            audio=[],
        )

    def test_the_top_lane_wins_where_it_has_footage(self) -> None:
        spans = visible_spans(self.plan([]))
        self.assertEqual(
            [(a, b, c.path if c else None) for a, b, c in spans],
            [(0.0, 12.0, "/a.mp4"), (12.0, 20.0, "/b.mp4")],
        )

    def test_a_hole_in_lane_0_lets_lane_1_through(self) -> None:
        spans = visible_spans(self.plan([(4.0, 6.0)]))
        self.assertEqual(
            [(a, b, c.path if c else None) for a, b, c in spans],
            [
                (0.0, 4.0, "/a.mp4"),
                (4.0, 6.0, "/b.mp4"),
                (6.0, 12.0, "/a.mp4"),
                (12.0, 20.0, "/b.mp4"),
            ],
        )

    def test_a_hole_with_nothing_underneath_is_black(self) -> None:
        plan = TimelinePlan(
            duration=10.0,
            video=[VideoLane([VideoClip("/a.mp4", 0.0, 0.0, 10.0)], holes=[(4.0, 6.0)])],
            audio=[],
        )
        self.assertEqual(
            [c.path if c else None for _, _, c in visible_spans(plan)],
            ["/a.mp4", None, "/a.mp4"],
        )

    def test_the_composite_renders_one_piece_per_span(self) -> None:
        settings = RenderSettings(
            segments=[Segment(0.0, 20.0)], plan=self.plan([(4.0, 6.0)]), fps=30,
            tmix_frames=0, width=None, height=None, out_path="/tmp/o.mp4",
        )
        graph = build_filter_graph(video("/a.mp4", 12.0, 0), settings)
        # /b.mp4 is first read at 4 s, so it opens with -ss 2 and its trims
        # are written 2 s earlier; /a.mp4 is read from 0 and is not seeked.
        self.assertEqual(input_seeks(settings.plan, settings.segments), {1: 2})
        self.assertIn("[0:v]split=2[s0_0][s0_1]", graph)
        self.assertIn("[1:v]split=2[s1_0][s1_1]", graph)
        self.assertIn("[s1_0]trim=start=2:end=4", graph)  # lane 1 through the hole
        self.assertIn("[s0_1]trim=start=6:end=12", graph)
        self.assertIn("[s1_1]trim=start=10:end=18", graph)  # lane 1 past lane 0's end
        self.assertIn("concat=n=4:v=1:a=0[vc]", graph)

    def test_a_gap_becomes_a_generated_black_piece(self) -> None:
        plan = TimelinePlan(
            duration=18.0,
            video=[VideoLane([
                VideoClip("/a.mp4", 0.0, 0.0, 6.0),
                VideoClip("/b.mp4", 10.0, 2.0, 8.0),
            ])],
            audio=[],
        )
        graph = build_filter_graph(
            video("/a.mp4", 6.0, 0),
            RenderSettings(
                segments=[Segment(0.0, 18.0)], plan=plan, fps=30, tmix_frames=0,
                width=None, height=None, out_path="/tmp/o.mp4",
            ),
        )
        self.assertIn("color=black:s=1664x936:r=60:d=4,", graph)
        self.assertIn("setsar=1,fps=60,format=yuv420p", graph)
        self.assertIn("[1:v]trim=start=2:end=8,", graph)  # src_in respected

    def test_video_and_audio_never_share_an_input_handle(self) -> None:
        """The v1 two-input fix, generalised: the audio chain must not be able
        to race the video decoder into an early EOF."""
        plan = TimelinePlan(
            duration=10.0,
            video=[VideoLane([VideoClip("/a.mp4", 0.0, 0.0, 10.0)])],
            audio=[AudioLane([AudioClip("/a.mp4", 0, 0.0, 0.0, 10.0)])],
        )
        paths, vidx, aidx = plan_inputs(plan)
        self.assertEqual(paths, ["/a.mp4", "/a.mp4"])
        self.assertEqual((vidx["/a.mp4"], aidx["/a.mp4"]), (0, 1))


# --------------------------------------------------------------------------- #
# padding
# --------------------------------------------------------------------------- #
class TestPadding(unittest.TestCase):
    """The timeline outliving the footage is the same hazard class as apad.

    A keep-segment past the end of the video makes ``trim`` yield nothing,
    ``concat`` starve, and ffmpeg exit 0 with a short (or video-less) file.
    Measured on real ffmpeg 9.0.1 with a 12 s clip on a 20 s timeline and
    segments at 1-3 s and 14-19 s: the video stream came out 2.0 s instead of
    7.0 s.  ``TestRealFfmpeg`` below re-measures it on every run.
    """

    def settings(self, **kw: object) -> RenderSettings:
        base: dict[str, object] = dict(
            segments=[Segment(0.0, 20.0)], fps=30, tmix_frames=0,
            width=None, height=None, out_path="/tmp/o.mp4",
        )
        base.update(kw)
        return RenderSettings(**base)  # type: ignore[arg-type]

    def plan(self, duration: float = 20.0) -> TimelinePlan:
        return TimelinePlan(
            duration=duration,
            video=[VideoLane([VideoClip("/a.mp4", 0.0, 0.0, 12.0)])],
            audio=[
                AudioLane([AudioClip("/a.mp4", 0, 0.0, 0.0, 12.0)]),
                AudioLane([AudioClip("/bgm.mp3", 0, 0.0, 0.0, 0.0)]),
            ],
        )

    def test_video_is_tpadded_out_to_the_timeline(self) -> None:
        graph = build_filter_graph(
            video("/a.mp4", 12.0, 1), self.settings(plan=self.plan())
        )
        self.assertIn(
            "[0:v]tpad=stop_mode=add:color=black:stop_duration=8[vp]", graph
        )
        self.assertIn("[vp]trim=start=0:end=20", graph)

    def test_embedded_audio_is_apadded_out_to_the_timeline(self) -> None:
        graph = build_filter_graph(
            video("/a.mp4", 12.0, 1), self.settings(plan=self.plan())
        )
        self.assertIn(
            "[1:a:0]atrim=start=0:end=12,asetpts=PTS-STARTPTS,volume=0dB,"
            "apad=whole_dur=20[g0]",
            graph,
        )

    def test_neither_pad_appears_when_the_video_already_fills_it(self) -> None:
        graph = build_filter_graph(
            video("/a.mp4", 12.0, 1),
            self.settings(segments=[Segment(0.0, 12.0)], plan=self.plan(12.0)),
        )
        self.assertNotIn("tpad", graph)
        self.assertNotIn("apad=whole_dur=12[g0]", graph)

    def test_tpad_comes_before_the_drawbox_so_a_lift_in_the_tail_has_frames(self) -> None:
        plan = self.plan()
        plan.video[0].holes = [(14.0, 16.0)]
        graph = build_filter_graph(video("/a.mp4", 12.0, 1), self.settings(plan=plan))
        self.assertLess(graph.index("tpad="), graph.index("drawbox"))
        self.assertIn("[vp]drawbox=", graph)
        self.assertIn("enable='between(t,14,16)'", graph)

    def test_a_delayed_audio_clip_is_shifted_and_padded(self) -> None:
        plan = TimelinePlan(
            duration=20.0,
            video=[VideoLane([VideoClip("/a.mp4", 0.0, 0.0, 20.0)])],
            audio=[AudioLane([AudioClip("/b.mp3", 0, 5.0, 1.0, 4.0)])],
        )
        graph = build_filter_graph(video("/a.mp4", 20.0, 0), self.settings(plan=plan))
        # ``aresample=async=1`` rides with the delay and only with the delay:
        # adelay after an atrim/asetpts pair leaves a PTS hole instead of real
        # silence, and a per-piece atrim landing in that hole takes the whole
        # render down (see the comment in ``_audio_heads`` and the measured
        # case in test_blocks.TestRealFfmpegEdges).
        self.assertIn(
            "[1:a:0]atrim=start=1:end=4,asetpts=PTS-STARTPTS,adelay=5000:all=1,"
            "aresample=async=1,volume=0dB,apad=whole_dur=20[am]",
            graph,
        )


# --------------------------------------------------------------------------- #
# preview plan
# --------------------------------------------------------------------------- #
class TestPreviewPlan(_Case):
    def test_a_plain_timeline_is_one_video_and_one_audio_entry(self) -> None:
        plan = self.w.preview_plan()
        self.assertAlmostEqual(plan.duration, 120.0)
        self.assertEqual(
            plan.video, [("/tmp/main.mp4", 0.0, 120.0, 0.0)]
        )
        self.assertEqual(plan.audio[0].path, "/tmp/main.mp4")
        self.assertEqual(plan.audio[0].stream, 0)
        self.assertFalse(plan.audio[0].muted)

    def test_a_cut_splits_an_entry_and_collapses_the_edited_times(self) -> None:
        self.w._set_marks(40.0, 60.0)
        self.w.cut_selection()
        plan = self.w.preview_plan()
        self.assertEqual(
            [(v.src_in, v.src_out, v.edited_start) for v in plan.video],
            [(0.0, 40.0, 0.0), (60.0, 120.0, 40.0)],
        )

    def test_an_appended_clip_gets_its_own_entry_in_edited_time(self) -> None:
        self.w.add_clip(0, video("/tmp/b.mp4", 30.0), "append")
        plan = self.w.preview_plan()
        self.assertEqual(
            [(v.path, v.edited_start) for v in plan.video],
            [("/tmp/main.mp4", 0.0), ("/tmp/b.mp4", 120.0)],
        )

    def test_mute_and_gain_ride_along(self) -> None:
        self.w._lanes[1].header.mute.setChecked(True)
        self.w._lanes[1].header.slider.setValue(-9)
        entry = self.w.preview_plan().audio[0]
        self.assertTrue(entry.muted)
        self.assertEqual(entry.gain_db, -9.0)

    def test_a_lift_removes_that_stretch_from_the_lane(self) -> None:
        self.w.set_armed_lanes([1])
        self.w._set_marks(30.0, 50.0)
        self.w.cut_selection()
        spans = [(a.edited_start, a.src_in, a.src_out) for a in self.w.preview_plan().audio]
        self.assertEqual(spans, [(0.0, 0.0, 30.0), (50.0, 50.0, 120.0)])

    def test_lane_1_shows_through_a_hole_in_lane_0(self) -> None:
        lane = self.w.add_video_lane()
        self.w.add_clip(lane, video("/tmp/b.mp4", 120.0), "place")
        self.w.set_armed_lanes([0])
        self.w._set_marks(30.0, 50.0)
        self.w.cut_selection()
        plan = self.w.preview_plan()
        self.assertEqual(
            [(v.path, v.edited_start) for v in plan.video],
            [("/tmp/main.mp4", 0.0), ("/tmp/b.mp4", 30.0), ("/tmp/main.mp4", 50.0)],
        )


# --------------------------------------------------------------------------- #
# theme
# --------------------------------------------------------------------------- #
class TestPaintTimeTheme(_Case):
    """The strip must follow the app theme, so colours are resolved per paint."""

    def tearDown(self) -> None:
        from ffcut import theme

        theme.apply(_app, "dark")

    def lane_pixel(self) -> tuple[int, int, int]:
        image = self.w.grab().toImage()
        rect = self.w._lane_rect(1)  # an audio lane's background
        colour = image.pixelColor(int(rect.left() + 40), int(rect.center().y()) - 12)
        return (colour.red(), colour.green(), colour.blue())

    def test_light_and_dark_paint_different_lanes(self) -> None:
        from ffcut import theme

        theme.apply(_app, "dark")
        dark = self.lane_pixel()
        theme.apply(_app, "light")
        light = self.lane_pixel()
        self.assertNotEqual(dark, light)
        self.assertLess(sum(dark), sum(light), "light mode must be the lighter one")

    def test_the_palette_is_not_frozen_at_import(self) -> None:
        from ffcut import theme, timeline

        theme.apply(_app, "light")
        self.w.grab()
        self.assertEqual(self.w._pal.mode, "light")
        self.assertEqual(
            self.w._pal.bg.name(), theme.TIMELINE_MODES["light"]["lane_bg"]
        )
        self.assertNotIn("C_BG", vars(timeline), "import-time colours are back")

    def test_a_theme_switch_restyles_the_scrollbar_and_headers(self) -> None:
        from ffcut import theme

        theme.apply(_app, "light")
        self.w._refresh_theme()
        before = self.w._hbar.styleSheet()
        theme.apply(_app, "dark")
        self.w._refresh_theme()
        self.assertNotEqual(before, self.w._hbar.styleSheet())


# --------------------------------------------------------------------------- #
# real ffmpeg
# --------------------------------------------------------------------------- #
@unittest.skipIf(FFMPEG is None, "ffmpeg not on PATH")
class TestRealFfmpeg(unittest.TestCase):
    """Measured, not asserted.  Both of these bugs exit 0 and write a file."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory(prefix="ffcut-lanes-")
        cls.dir = cls._tmp.name
        cls.main = os.path.join(cls.dir, "main.mp4")
        cls.second = os.path.join(cls.dir, "second.mp4")
        cls.bed = os.path.join(cls.dir, "bed.mp3")
        cls._run([
            "-f", "lavfi", "-i", "testsrc2=size=160x90:rate=15:duration=4",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=4",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
            "-shortest", cls.main,
        ])
        cls._run([
            "-f", "lavfi", "-i", "smptebars=size=160x90:rate=15:duration=8",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", cls.second,
        ])
        cls._run([
            "-f", "lavfi", "-i", "sine=frequency=220:duration=8",
            "-c:a", "libmp3lame", cls.bed,
        ])

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    @staticmethod
    def _run(args: list[str]) -> None:
        proc = subprocess.run(
            [FFMPEG, "-v", "error", "-y", *args], capture_output=True, text=True
        )
        if proc.returncode != 0:  # pragma: no cover - setup failure
            raise RuntimeError(proc.stderr[-800:])

    def render(self, plan: TimelinePlan, segments: list[Segment], name: str) -> dict:
        out = os.path.join(self.dir, name)
        info = MediaInfo(self.main, 4.0, 160, 90, 15.0, "h264",
                         [AudioTrack(0, "aac", 2, 48000, None, None)])
        argv = build_command(info, RenderSettings(
            segments=segments, plan=plan, fps=15, tmix_frames=0, width=None,
            height=None, out_path=out, encoder="libx264",
        ))
        proc = subprocess.run(argv, capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stderr[-1500:])
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries",
             "stream=codec_type,duration", "-of", "json", out],
            capture_output=True, text=True,
        )
        return {
            s["codec_type"]: float(s.get("duration") or 0.0)
            for s in json.loads(probe.stdout)["streams"]
        }

    def test_the_video_really_is_padded_past_its_own_end(self) -> None:
        """Without tpad the last piece yields no frames and the file is short —
        ffmpeg still exits 0."""
        plan = TimelinePlan(
            duration=8.0,
            video=[VideoLane([VideoClip(self.main, 0.0, 0.0, 4.0)])],
            audio=[AudioLane([AudioClip(self.bed, 0, 0.0, 0.0, 0.0)])],
        )
        got = self.render(plan, [Segment(1.0, 2.0), Segment(6.0, 8.0)], "pad.mp4")
        self.assertAlmostEqual(got["video"], 3.0, delta=0.15)
        self.assertAlmostEqual(got["audio"], 3.0, delta=0.15)

    def test_embedded_audio_really_is_padded_to_the_longer_timeline(self) -> None:
        plan = TimelinePlan(
            duration=8.0,
            video=[VideoLane([VideoClip(self.main, 0.0, 0.0, 4.0)])],
            audio=[AudioLane([AudioClip(self.main, 0, 0.0, 0.0, 4.0)])],
        )
        got = self.render(plan, [Segment(0.0, 8.0)], "apad.mp4")
        self.assertAlmostEqual(got["video"], 8.0, delta=0.15)
        self.assertAlmostEqual(got["audio"], 8.0, delta=0.15)

    def test_a_two_lane_composite_really_shows_lane_1_through_the_hole(self) -> None:
        plan = TimelinePlan(
            duration=8.0,
            video=[
                VideoLane([VideoClip(self.main, 0.0, 0.0, 4.0)], holes=[(1.0, 2.0)]),
                VideoLane([VideoClip(self.second, 0.0, 0.0, 8.0)]),
            ],
            audio=[],
        )
        got = self.render(plan, [Segment(0.0, 8.0)], "composite.mp4")
        self.assertAlmostEqual(got["video"], 8.0, delta=0.15)
        out = os.path.join(self.dir, "composite.mp4")
        # t=0.5 is the main clip, t=1.5 is lane 1 through the hole, t=6 is
        # lane 1 past the main clip's end.
        self.assertEqual(self.frame_kind(out, 0.5), self.frame_kind(self.main, 0.5))
        self.assertEqual(self.frame_kind(out, 1.5), self.frame_kind(self.second, 1.5))
        self.assertEqual(self.frame_kind(out, 6.0), self.frame_kind(self.second, 6.0))

    @staticmethod
    def frame_kind(path: str, t: float) -> tuple[int, int, int]:
        """Average RGB of one frame — enough to tell the two sources apart."""
        raw = subprocess.run(
            [FFMPEG, "-v", "error", "-ss", f"{t:.3f}", "-i", path, "-frames:v", "1",
             "-vf", "scale=8:8,format=rgb24", "-f", "rawvideo", "-"],
            capture_output=True,
        ).stdout
        n = max(1, len(raw) // 3)
        return tuple(round(sum(raw[i::3]) / n) for i in range(3))  # type: ignore[return-value]


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
