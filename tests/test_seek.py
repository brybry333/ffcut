"""Input seeking: ``-ss`` before each video ``-i`` instead of decoding up to it.

The unit cases pin the rules in :func:`ffcut.render.input_seeks`.  The measured
case is the one that matters: the same edit rendered with and without the
seeks has to come out FRAME-IDENTICAL, picture and sound, because a seek that
is off by one frame still exits 0 and still looks fine at a glance.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ffcut.render as render_mod  # noqa: E402
from ffcut.probe import AudioTrack, MediaInfo  # noqa: E402
from ffcut.render import (  # noqa: E402
    AudioClip,
    AudioLane,
    RenderSettings,
    Segment,
    TimelinePlan,
    VideoClip,
    VideoLane,
    build_command,
    input_seeks,
)

FFMPEG = shutil.which("ffmpeg")


def one_clip(duration: float = 60.0, **clip: float) -> TimelinePlan:
    return TimelinePlan(
        duration=duration,
        video=[VideoLane([VideoClip("/a.mp4", 0.0, 0.0, 60.0, **clip)])],
        audio=[AudioLane([AudioClip("/a.mp4", 0)])],
    )


class TestInputSeeks(unittest.TestCase):
    def test_a_cut_from_the_top_is_not_seeked(self) -> None:
        self.assertEqual(input_seeks(one_clip(), [Segment(0.0, 10.0)]), {})

    def test_the_seek_lands_the_margin_before_the_first_kept_frame(self) -> None:
        segs = [Segment(40.0, 45.0), Segment(10.5, 12.0)]
        self.assertEqual(input_seeks(one_clip(), segs), {0: 8})

    def test_under_a_second_is_not_worth_a_seek(self) -> None:
        self.assertEqual(input_seeks(one_clip(), [Segment(2.9, 10.0)]), {})

    def test_a_fade_in_from_the_top_pins_the_input_at_zero(self) -> None:
        """``fade`` rejects a negative ``st``, so its window counts as a read."""
        self.assertEqual(input_seeks(one_clip(fade_in=1.0), [Segment(30.0, 40.0)]), {})

    def test_a_fade_out_before_the_cut_holds_the_seek_back(self) -> None:
        plan = one_clip(fade_out=40.0)  # st = 20
        self.assertEqual(input_seeks(plan, [Segment(30.0, 40.0)]), {0: 18})

    def test_a_segment_in_the_tpad_tail_never_seeks_past_the_footage(self) -> None:
        """An audio bed longer than the video leaves keep-segments that only
        ``tpad`` covers; seeking past the last real frame would leave it
        nothing to pad."""
        plan = one_clip(duration=150.0)
        self.assertEqual(input_seeks(plan, [Segment(100.0, 150.0)]), {0: 58})

    def test_each_video_input_gets_its_own_seek(self) -> None:
        plan = TimelinePlan(
            duration=20.0,
            video=[VideoLane([
                VideoClip("/a.mp4", 0.0, 100.0, 110.0),
                VideoClip("/b.mp4", 10.0, 0.0, 10.0),
            ])],
            audio=[],
        )
        self.assertEqual(input_seeks(plan, [Segment(0.0, 20.0)]), {0: 98})

    def test_audio_inputs_are_never_seeked(self) -> None:
        """Seeking a VBR MP3 is approximate and its timestamps then come from
        the byte position, which would slide the music against the picture."""
        plan = one_clip()
        plan.audio.append(AudioLane([AudioClip("/song.mp3", 0, 0.0, 120.0, 140.0)]))
        info = MediaInfo("/a.mp4", 60.0, 160, 90, 15.0, "h264",
                         [AudioTrack(0, "aac", 2, 48000, None, None)])
        argv = build_command(info, RenderSettings(
            segments=[Segment(30.0, 40.0)], plan=plan, fps=15, tmix_frames=0,
            width=None, height=None, out_path="/tmp/o.mp4", encoder="libx264",
        ))
        self.assertEqual(argv.count("-ss"), 1)
        self.assertEqual(argv[argv.index("-ss") : argv.index("-ss") + 4],
                         ["-ss", "28", "-i", "/a.mp4"])
        self.assertIn("atrim=start=120:end=140", argv[argv.index("-filter_complex") + 1])


@unittest.skipIf(FFMPEG is None, "ffmpeg not on PATH")
class TestRealFfmpegSeek(unittest.TestCase):
    """Seeked and unseeked renders of one edit, compared frame by frame.

    ``ramp.mp4`` has luma ``4 x t`` and a tone that is silent in alternate
    seconds, so a shift of even one frame or one audio block changes the
    hashes.  The GOP is 37 frames so no cut or seek point sits on a keyframe:
    every seek has to land early and decode forward to the exact frame.
    """

    @classmethod
    def _run(cls, args: list[str]) -> None:
        proc = subprocess.run([FFMPEG, "-v", "error", "-y", *args],
                              capture_output=True, text=True)
        if proc.returncode != 0:  # pragma: no cover - setup failure
            raise RuntimeError(proc.stderr[-800:])

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory(prefix="ffcut-seek-")
        cls.dir = cls._tmp.name
        cls.ramp = os.path.join(cls.dir, "ramp.mp4")
        cls._run([
            "-f", "lavfi", "-i", "color=c=black:s=160x90:r=15:d=30",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=30",
            "-filter_complex",
            "[0:v]geq=lum='4*T':cb=128:cr=128,format=yuv420p[v];"
            "[1:a]volume=0:enable='lt(mod(t,2),1)'[a]",
            "-map", "[v]", "-map", "[a]", "-c:v", "libx264", "-g", "37",
            "-c:a", "aac", cls.ramp,
        ])
        cls.other = os.path.join(cls.dir, "other.mp4")
        cls._run([
            "-f", "lavfi", "-i", "testsrc2=size=160x90:rate=15:duration=30",
            "-f", "lavfi", "-i", "sine=frequency=660:duration=30",
            "-c:v", "libx264", "-g", "37", "-pix_fmt", "yuv420p", "-c:a", "aac",
            "-shortest", cls.other,
        ])
        cls.info = MediaInfo(cls.ramp, 30.0, 160, 90, 15.0, "h264",
                             [AudioTrack(0, "aac", 2, 48000, None, None)])

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def render(self, plan: TimelinePlan, segs: list[Segment], name: str,
               seek: bool) -> tuple[str, list[str]]:
        out = os.path.join(self.dir, name)
        # A huge margin puts every seek below the 1 s floor: the old graph.
        margin = render_mod.SEEK_MARGIN if seek else 1e9
        with mock.patch.object(render_mod, "SEEK_MARGIN", margin):
            argv = build_command(self.info, RenderSettings(
                segments=segs, plan=plan, fps=15, tmix_frames=0, width=None,
                height=None, out_path=out, encoder="libx264",
            ))
        proc = subprocess.run(argv, capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stderr[-1500:])
        return out, argv

    def frames(self, out: str, stream: str) -> list[str]:
        proc = subprocess.run(
            [FFMPEG, "-v", "error", "-i", out, "-map", f"0:{stream}",
             "-f", "framemd5", "-"],
            capture_output=True, text=True,
        )
        rows = [r for r in proc.stdout.splitlines() if r and not r.startswith("#")]
        self.assertTrue(rows, f"no {stream} frames in {out}")
        return rows

    def assert_identical(self, plan: TimelinePlan, segs: list[Segment],
                         seeks: dict[int, int]) -> None:
        self.assertEqual(input_seeks(plan, segs), seeks)
        plain, plain_argv = self.render(plan, segs, "plain.mp4", seek=False)
        seeked, seeked_argv = self.render(plan, segs, "seeked.mp4", seek=True)
        self.assertNotIn("-ss", plain_argv)
        self.assertEqual(seeked_argv.count("-ss"), len(seeks))
        for stream in ("v", "a"):
            self.assertEqual(self.frames(seeked, stream), self.frames(plain, stream),
                             f"seeking changed the {stream} stream")

    def test_the_single_clip_graph(self) -> None:
        """Split/trim off ``[0:v]``, with a lift and a fade-out riding on the
        shifted timestamps too."""
        plan = TimelinePlan(
            duration=30.0,
            video=[VideoLane([VideoClip(self.ramp, 0.0, 0.0, 30.0, fade_out=6.0)],
                             holes=[(21.0, 22.0)])],
            audio=[AudioLane([AudioClip(self.ramp, 0)])],
        )
        segs = [Segment(12.3, 14.1), Segment(20.2, 25.0)]
        self.assert_identical(plan, segs, {0: 10})

    def test_the_piece_graph_across_two_files(self) -> None:
        """Two files, both cut deep in, each with a fade, and the audio of both
        trimmed and moved — the shape the 2026-10-30 render stalled on."""
        plan = TimelinePlan(
            duration=9.0,
            video=[VideoLane([
                VideoClip(self.ramp, 0.0, 15.0, 19.0, fade_in=1.0),
                VideoClip(self.other, 4.0, 22.5, 27.5, fade_out=1.5),
            ])],
            audio=[AudioLane([
                AudioClip(self.ramp, 0, 0.0, 15.0, 19.0),
                AudioClip(self.other, 0, 4.0, 22.5, 27.5),
            ])],
        )
        self.assert_identical(plan, [Segment(0.5, 9.0)], {0: 13, 1: 20})


if __name__ == "__main__":
    unittest.main()
