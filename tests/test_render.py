"""Unit tests for ffcut.render.build_command — no ffmpeg is ever invoked."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from dataclasses import dataclass, field
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ffcut.render as render_mod  # noqa: E402
from ffcut.probe import AudioTrack, MediaInfo  # noqa: E402
from ffcut.render import (  # noqa: E402
    CONTAINERS,
    ENCODERS,
    STDERR_TAIL_LINES,
    AudioClip,
    AudioLane,
    TextClip,
    TimelinePlan,
    VideoClip,
    VideoLane,
    Progress,
    RenderJob,
    RenderSettings,
    Segment,
    _ProgressState,
    build_command,
    default_vaapi_device,
    hole_enable,
    is_crf_encoder,
    is_vaapi_encoder,
    plan_inputs,
    total_output_seconds,
)


def make_info(n_audio: int = 3) -> MediaInfo:
    """A stand-in for the user's 1664x936@120 test file."""
    return MediaInfo(
        path="/tmp/in.mp4",
        duration=2780.0,
        width=1664,
        height=936,
        fps=120.0,
        vcodec="h264",
        audio=[
            AudioTrack(
                index=i,
                codec="aac",
                channels=2,
                sample_rate=48000,
                language="eng",
                title=f"Track {i + 1}",
            )
            for i in range(n_audio)
        ],
    )


@dataclass
class Track:
    """One embedded track in the mix — shorthand for an ``AudioLane`` on it."""

    index: int
    gain_db: float = 0.0
    muted: bool = False
    holes: list = field(default_factory=list)


@dataclass
class External:
    """A dropped file in the mix: its own input, length unknown."""

    path: str
    gain_db: float = 0.0
    muted: bool = False
    holes: list = field(default_factory=list)


def mix(*indices: int) -> list[Track]:
    """Unity-gain, unmuted entries for the given track indices."""
    return [Track(i) for i in indices]


def make_plan(
    info: MediaInfo, audio: list[Track], external: list[External] = (),  # type: ignore[assignment]
    video_holes: list = (),  # type: ignore[assignment]
) -> TimelinePlan:
    """The plan a plain load builds: one video lane holding the whole file,
    one audio lane per embedded track, then one per dropped file.  A dropped
    file's clip has ``src_out == src_in`` — length unknown — so it is padded
    to the timeline instead of trimmed.  That IS the apad fix."""
    return TimelinePlan(
        duration=info.duration,
        video=[VideoLane([VideoClip(info.path, 0.0, 0.0, info.duration)], list(video_holes))],
        audio=[
            AudioLane([AudioClip(info.path, t.index, 0.0, 0.0, info.duration)],
                      t.gain_db, t.muted, list(t.holes))
            for t in audio
        ] + [
            AudioLane([AudioClip(e.path, 0, 0.0, 0.0, 0.0)], e.gain_db, e.muted, list(e.holes))
            for e in external
        ],
    )


def make_settings(
    info: MediaInfo | None = None, *, audio: list[Track] | None = None,
    external_audio: list[External] = (), video_holes: list = (),  # type: ignore[assignment]
    **kw: object,
) -> RenderSettings:
    info = info or make_info()
    base: dict[str, object] = dict(
        segments=[Segment(10.0, 20.0)],
        plan=make_plan(info, mix(0) if audio is None else audio, external_audio, video_holes),
        mbps=60,
        fps=60,
        tmix_frames=7,
        width=2560,
        height=1440,
        out_path="/tmp/out.mp4",
        encoder="libx264",
        container="mp4",
    )
    base.update(kw)
    return RenderSettings(**base)  # type: ignore[arg-type]


def graph_of(argv: list[str]) -> str:
    """Pull the single -filter_complex argv element back out."""
    return argv[argv.index("-filter_complex") + 1]


def flag_value(argv: list[str], flag: str) -> str:
    return argv[argv.index(flag) + 1]


class TestArgvStructure(unittest.TestCase):
    def test_basic_shape(self) -> None:
        argv = build_command(make_info(), make_settings())
        self.assertTrue(argv[0].endswith("ffmpeg"))
        self.assertEqual(flag_value(argv, "-i"), "/tmp/in.mp4")
        self.assertEqual(argv[-1], "/tmp/out.mp4")
        # progress plumbing is mandatory
        self.assertEqual(flag_value(argv, "-progress"), "pipe:1")
        self.assertIn("-nostats", argv)
        self.assertIn("-y", argv)
        # output flags from the user's script
        self.assertEqual(flag_value(argv, "-b:v"), "60M")
        self.assertEqual(flag_value(argv, "-maxrate"), "60M")
        self.assertEqual(flag_value(argv, "-bufsize"), "120M")
        self.assertEqual(flag_value(argv, "-pix_fmt"), "yuv420p")
        self.assertEqual(flag_value(argv, "-c:a"), "aac")
        self.assertEqual(flag_value(argv, "-b:a"), "192k")
        self.assertEqual(flag_value(argv, "-movflags"), "+faststart")

    def test_filter_complex_is_one_argv_element(self) -> None:
        argv = build_command(make_info(), make_settings(segments=[
            Segment(1, 2), Segment(5, 9)]))
        graph = graph_of(argv)
        self.assertEqual(argv.count("-filter_complex"), 1)
        self.assertIn(";", graph)
        # never pre-quoted: shell=True is not used anywhere
        self.assertFalse(graph.startswith(("'", '"')))

    def test_maps_are_the_graph_labels(self) -> None:
        argv = build_command(make_info(), make_settings())
        maps = [argv[i + 1] for i, a in enumerate(argv) if a == "-map"]
        self.assertEqual(maps, ["[vout]", "[ac]"])

    def test_no_shell_metacharacter_splitting(self) -> None:
        argv = build_command(
            make_info(), make_settings(out_path="/tmp/my clip (final).mp4")
        )
        self.assertEqual(argv[-1], "/tmp/my clip (final).mp4")


class TestSingleSegment(unittest.TestCase):
    def test_single_segment_has_no_split_or_concat(self) -> None:
        graph = graph_of(build_command(make_info(), make_settings()))
        self.assertNotIn("split=", graph)
        self.assertNotIn("asplit=", graph)
        self.assertNotIn("concat=", graph)

    def test_single_segment_still_trims(self) -> None:
        argv = build_command(make_info(), make_settings(segments=[Segment(10.0, 20.0)]))
        graph = graph_of(argv)
        # Input 0 opens at 8 s (input_seeks), so its trim reads 8 s earlier.
        self.assertEqual(argv[argv.index("-ss") + 1 : argv.index("-ss") + 4],
                         ["8", "-i", make_info().path])
        self.assertIn("[0:v]trim=start=2:end=12,setpts=PTS-STARTPTS[vc]", graph)
        self.assertIn("[am]atrim=start=10:end=20,asetpts=PTS-STARTPTS[ac]", graph)
        self.assertIn("[vc]tmix=frames=7,fps=60,scale=2560:1440[vout]", graph)

    def test_fractional_timestamps_are_not_exponential(self) -> None:
        graph = graph_of(
            build_command(
                make_info(), make_settings(segments=[Segment(0.0000005, 1234.5)])
            )
        )
        self.assertNotIn("e-", graph)
        self.assertIn("end=1234.5", graph)


class TestThreeSegments(unittest.TestCase):
    def setUp(self) -> None:
        self.segs = [Segment(0.0, 5.0), Segment(100.5, 110.25), Segment(600.0, 630.0)]
        self.graph = graph_of(
            build_command(make_info(), make_settings(segments=self.segs))
        )

    def test_video_split_then_trim(self) -> None:
        self.assertIn("[0:v]split=3[v0][v1][v2]", self.graph)
        self.assertIn("[v0]trim=start=0:end=5,setpts=PTS-STARTPTS[t0]", self.graph)
        self.assertIn(
            "[v1]trim=start=100.5:end=110.25,setpts=PTS-STARTPTS[t1]", self.graph
        )
        self.assertIn("[v2]trim=start=600:end=630,setpts=PTS-STARTPTS[t2]", self.graph)

    def test_audio_asplit_then_atrim(self) -> None:
        self.assertIn("[am]asplit=3[m0][m1][m2]", self.graph)
        self.assertIn("[m0]atrim=start=0:end=5,asetpts=PTS-STARTPTS[u0]", self.graph)
        self.assertIn(
            "[m2]atrim=start=600:end=630,asetpts=PTS-STARTPTS[u2]", self.graph
        )

    def test_concat_interleaves_video_and_audio(self) -> None:
        self.assertIn("[t0][u0][t1][u1][t2][u2]concat=n=3:v=1:a=1[vc][ac]", self.graph)

    def test_no_ss_or_to_for_multi_segment(self) -> None:
        argv = build_command(make_info(), make_settings(segments=self.segs))
        self.assertNotIn("-ss", argv)
        self.assertNotIn("-to", argv)

    def test_split_count_matches_segment_count(self) -> None:
        for n in (1, 2, 4, 7):
            segs = [Segment(i * 10.0, i * 10.0 + 5.0) for i in range(n)]
            graph = graph_of(
                build_command(make_info(), make_settings(segments=segs))
            )
            if n == 1:
                self.assertNotIn("split=", graph)
            else:
                self.assertIn(f"[0:v]split={n}", graph)
                self.assertIn(f"[am]asplit={n}", graph)
                self.assertIn(f"concat=n={n}:v=1:a=1", graph)


class TestTrimmedSingleClip(unittest.TestCase):
    def test_a_right_trimmed_clip_does_not_play_its_trimmed_footage(self) -> None:
        """One clip from 0:00, right edge trimmed to 40 s, audio still to 60 s:
        the shortcut graph would trim the WHOLE file 0..60 (audit 2026-11-15)."""
        info = make_info(n_audio=1)
        plan = TimelinePlan(
            duration=60.0,
            video=[VideoLane(clips=[VideoClip(info.path, start=0, src_in=0, src_out=40)])],
            audio=[AudioLane(clips=[AudioClip(info.path, 0, start=0, src_in=0, src_out=60)])],
        )
        s = RenderSettings(
            segments=[Segment(0.0, 60.0)], plan=plan, fps=60, tmix_frames=0,
            width=None, height=None, out_path="/tmp/o.mp4", encoder="libx264",
        )
        graph = graph_of(build_command(info, s))
        self.assertNotIn("trim=start=0:end=60", graph)
        self.assertIn("trim=start=0:end=40", graph)
        self.assertIn("color=black", graph, "the last 20 s must be black")


class TestTitles(unittest.TestCase):
    def test_a_title_across_a_cut_is_timed_in_output_time(self) -> None:
        info = make_info(n_audio=1)
        plan = TimelinePlan(
            duration=60.0,
            video=[VideoLane(clips=[VideoClip(info.path, 0, 0, 2780.0)])],
            texts=[TextClip(8.0, 25.0, "Hi")],
        )
        s = RenderSettings(
            segments=[Segment(0.0, 10.0), Segment(20.0, 60.0)], plan=plan, fps=60,
            tmix_frames=0, width=None, height=None, out_path="/tmp/o.mp4", encoder="libx264",
        )
        graph = graph_of(build_command(info, s))
        # 8-10 s lands at 8-10, 20-25 s lands at 10-15 once 10-20 is cut out
        self.assertIn("enable='between(t,8,10)+between(t,10,15)'", graph)
        self.assertTrue(graph.endswith("[vout]"))

    @unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg is required")
    def test_quoting_survives_ffmpeg(self) -> None:
        """Quotes, colons, backslashes, %{} and newlines draw exactly as a textfile does."""
        nasty = "It's 100% \"fun\": a\\b, c;d [x] {y} %{pts}\nline two = ok"
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "t.txt")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(nasty)

            def frame(option: str) -> bytes:
                vf = f"drawtext={option}:expansion=none:fontsize=40:fontcolor=white"
                return subprocess.run(
                    ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=black:s=640x360:d=0.1",
                     "-vf", vf, "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "gray", "-"],
                    capture_output=True, check=True,
                ).stdout

            drawn = frame("text=" + render_mod._quote(nasty))
            self.assertEqual(drawn, frame("textfile=" + render_mod._quote(path)))
            self.assertTrue(any(v > 128 for v in drawn))


class TestTmix(unittest.TestCase):
    def test_tmix_disabled(self) -> None:
        graph = graph_of(build_command(make_info(), make_settings(tmix_frames=0)))
        self.assertNotIn("tmix", graph)
        self.assertIn("[vc]fps=60,scale=2560:1440[vout]", graph)

    def test_tmix_runs_before_fps_at_source_rate_then_scale(self) -> None:
        """The whole point of the tool: blend at 120fps, THEN decimate to 60."""
        graph = graph_of(build_command(make_info(), make_settings(tmix_frames=3)))
        self.assertIn("tmix=frames=3,fps=60,scale=2560:1440[vout]", graph)


class TestScaling(unittest.TestCase):
    def test_no_scale_when_source_resolution(self) -> None:
        graph = graph_of(
            build_command(make_info(), make_settings(width=None, height=None))
        )
        self.assertNotIn("scale=", graph)
        self.assertIn("[vc]tmix=frames=7,fps=60[vout]", graph)

    def test_scale_1080p(self) -> None:
        graph = graph_of(
            build_command(make_info(), make_settings(width=1920, height=1080))
        )
        self.assertIn("scale=1920:1080[vout]", graph)

    def test_half_specified_scale_keeps_aspect(self) -> None:
        graph = graph_of(
            build_command(make_info(), make_settings(width=2560, height=None))
        )
        self.assertIn("scale=2560:-2[vout]", graph)


class TestAudioInputSeparation(unittest.TestCase):
    """Regression guard for the silent zero-video-frames bug.

    With video and audio on the SAME input, ``atrim`` reaches EOF long before the
    video decoder walks up to a deep trim start, and ffmpeg exits 0 having written
    an audio-only file.  Reproduced on the real 46-minute source at t=1200 and
    t=2000; fixed by opening the file a second time for audio.
    """

    def test_file_is_opened_twice_when_audio_is_used(self) -> None:
        argv = build_command(make_info(), make_settings(audio=mix(0)))
        inputs = [argv[i + 1] for i, a in enumerate(argv) if a == "-i"]
        self.assertEqual(inputs, ["/tmp/in.mp4", "/tmp/in.mp4"])

    def test_video_comes_from_input_0_audio_from_input_1(self) -> None:
        graph = graph_of(
            build_command(make_info(), make_settings(audio=mix(0, 1)))
        )
        self.assertIn("[0:v]", graph)
        self.assertNotIn("[0:a:", graph)
        self.assertIn("[1:a:0]", graph)
        self.assertIn("[1:a:1]", graph)

    def test_holds_for_multi_segment_too(self) -> None:
        graph = graph_of(
            build_command(
                make_info(),
                make_settings(segments=[Segment(0, 5), Segment(1200, 1203)]),
            )
        )
        self.assertNotIn("[0:a:", graph)
        self.assertIn("[1:a:0]volume=0dB[am]", graph)


class TestAudio(unittest.TestCase):
    def test_one_track_needs_no_amix(self) -> None:
        graph = graph_of(build_command(make_info(), make_settings(audio=mix(0))))
        self.assertIn("[1:a:0]volume=0dB[am]", graph)
        self.assertNotIn("amix", graph)

    def test_one_track_non_zero_index(self) -> None:
        graph = graph_of(build_command(make_info(), make_settings(audio=mix(2))))
        self.assertIn("[1:a:2]volume=0dB[am]", graph)

    def test_three_tracks_mixed(self) -> None:
        graph = graph_of(
            build_command(make_info(), make_settings(audio=mix(0, 1, 2)))
        )
        for index in (0, 1, 2):
            self.assertIn(f"[1:a:{index}]volume=0dB[g{index}]", graph)
        self.assertIn(
            "[g0][g1][g2]amix=inputs=3:duration=longest:normalize=0[am]", graph
        )

    def test_mix_happens_before_split_and_trim(self) -> None:
        graph = graph_of(
            build_command(
                make_info(),
                make_settings(
                    audio=mix(0, 2), segments=[Segment(0, 5), Segment(9, 12)]
                ),
            )
        )
        self.assertLess(graph.index("amix="), graph.index("asplit="))
        self.assertLess(graph.index("asplit="), graph.index("atrim="))

    def test_two_lanes_on_one_stream_both_mix_under_distinct_labels(self) -> None:
        """ffmpeg 9 lets one input stream feed two chains (measured), so a
        second lane on the same track is simply a second head in the mix."""
        graph = graph_of(
            build_command(make_info(), make_settings(audio=mix(1, 1, 0)))
        )
        self.assertIn("[1:a:1]volume=0dB[g1]", graph)
        self.assertIn("[1:a:1]volume=0dB[g1_1]", graph)
        self.assertIn("[g1][g1_1][g0]amix=inputs=3", graph)

    def test_out_of_range_track_raises_muted_or_not(self) -> None:
        for track in (Track(5), Track(9, muted=True)):
            with self.assertRaises(ValueError):
                build_command(make_info(n_audio=2), make_settings(audio=[track]))

    def test_no_audio_selected_is_video_only(self) -> None:
        argv = build_command(make_info(), make_settings(audio=[]))
        graph = graph_of(argv)
        self.assertEqual(argv.count("-i"), 1)
        self.assertNotIn("-c:a", argv)
        self.assertNotIn("[ac]", argv)
        self.assertNotIn("atrim", graph)

    def test_multi_segment_video_only_concat(self) -> None:
        graph = graph_of(
            build_command(
                make_info(),
                make_settings(
                    audio=[], segments=[Segment(0, 5), Segment(9, 12)]
                ),
            )
        )
        self.assertIn("[t0][t1]concat=n=2:v=1:a=0[vc]", graph)


AMD = "/dev/dri/renderD129"


class TestEncoders(unittest.TestCase):
    def test_default_encoder_is_vaapi(self) -> None:
        s = RenderSettings(
            segments=[Segment(0, 1)],
            plan=make_plan(make_info(), mix(0)),
            mbps=60,
            fps=60,
            tmix_frames=7,
            width=None,
            height=None,
            out_path="/tmp/o.mp4",
            vaapi_device=AMD,
        )
        self.assertEqual(s.encoder, "h264_vaapi")
        argv = build_command(make_info(), s)
        self.assertEqual(flag_value(argv, "-c:v"), "h264_vaapi")
        self.assertEqual(flag_value(argv, "-vaapi_device"), AMD)

    def test_libx264_flags(self) -> None:
        argv = build_command(make_info(), make_settings(encoder="libx264"))
        self.assertEqual(flag_value(argv, "-c:v"), "libx264")
        self.assertNotIn("-rc", argv)
        self.assertNotIn("-preset", argv)
        self.assertNotIn("-tune", argv)
        self.assertNotIn("-profile:v", argv)
        self.assertEqual(flag_value(argv, "-b:v"), "60M")
        self.assertEqual(flag_value(argv, "-maxrate"), "60M")
        self.assertEqual(flag_value(argv, "-bufsize"), "120M")

    def test_vaapi_flags(self) -> None:
        argv = build_command(
            make_info(), make_settings(encoder="h264_vaapi", vaapi_device=AMD)
        )
        self.assertEqual(flag_value(argv, "-c:v"), "h264_vaapi")
        self.assertEqual(flag_value(argv, "-rc_mode"), "CBR")
        self.assertEqual(flag_value(argv, "-profile:v"), "high")
        self.assertEqual(flag_value(argv, "-b:v"), "60M")
        self.assertEqual(flag_value(argv, "-maxrate"), "60M")
        self.assertEqual(flag_value(argv, "-bufsize"), "120M")
        # Hardware frames: a -pix_fmt would make ffmpeg refuse the encoder.
        for flag in ("-pix_fmt", "-rc", "-preset", "-tune", "-crf"):
            self.assertNotIn(flag, argv, f"{flag} must not reach h264_vaapi")

    def test_vaapi_device_comes_before_the_inputs(self) -> None:
        argv = build_command(
            make_info(), make_settings(encoder="h264_vaapi", vaapi_device=AMD)
        )
        self.assertLess(argv.index("-vaapi_device"), argv.index("-i"))
        self.assertEqual(argv.count("-vaapi_device"), 1)

    def test_av1_vaapi_flags(self) -> None:
        argv = build_command(
            make_info(), make_settings(encoder="av1_vaapi", vaapi_device=AMD)
        )
        self.assertEqual(flag_value(argv, "-c:v"), "av1_vaapi")
        self.assertEqual(flag_value(argv, "-rc_mode"), "CBR")
        self.assertEqual(flag_value(argv, "-b:v"), "60M")
        self.assertEqual(flag_value(argv, "-vaapi_device"), AMD)
        for flag in ("-pix_fmt", "-profile:v", "-crf", "-preset"):
            self.assertNotIn(flag, argv, f"{flag} must not reach av1_vaapi")

    def test_cpu_encoders_get_no_vaapi_device(self) -> None:
        for encoder in ("libx264", "libsvtav1"):
            argv = build_command(
                make_info(), make_settings(encoder=encoder, container="mkv")
            )
            self.assertNotIn("-vaapi_device", argv)
            self.assertNotIn("hwupload", graph_of(argv))

    def test_vaapi_device_is_looked_up_when_not_given(self) -> None:
        with mock.patch.object(
            render_mod, "default_vaapi_device", return_value="/dev/dri/renderD200"
        ) as lookup:
            argv = build_command(make_info(), make_settings(encoder="h264_vaapi"))
        lookup.assert_called_once_with()
        self.assertEqual(flag_value(argv, "-vaapi_device"), "/dev/dri/renderD200")

    def test_encoder_does_not_change_the_graph(self) -> None:
        a = graph_of(build_command(make_info(), make_settings(encoder="libx264")))
        c = graph_of(
            build_command(make_info(), make_settings(encoder="libsvtav1", container="mkv"))
        )
        self.assertEqual(a, c)

    def test_vaapi_only_adds_the_upload_at_the_very_end(self) -> None:
        """The graph is the CPU graph plus one upload to the GPU, nothing else."""
        a = graph_of(build_command(make_info(), make_settings(encoder="libx264")))
        self.assertTrue(a.endswith("[vout]"))
        for encoder in ("h264_vaapi", "av1_vaapi"):
            b = graph_of(
                build_command(
                    make_info(), make_settings(encoder=encoder, vaapi_device=AMD)
                )
            )
            self.assertEqual(b, a[: -len("[vout]")] + ",format=nv12,hwupload[vout]")
            self.assertIn("tmix=frames=7,fps=60,scale=2560:1440,format=nv12,hwupload", b)


class TestDefaultVaapiDevice(unittest.TestCase):
    """The render node is found by driver, never assumed to be renderD128."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = self._tmp.name
        self.drm = os.path.join(self.root, "drm")
        os.makedirs(self.drm)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def node(self, name: str, driver: str, boot_vga: str | None = "0") -> None:
        pci = os.path.join(self.root, "pci", name)
        os.makedirs(pci)
        os.symlink(os.path.join(self.root, "drivers", driver), os.path.join(pci, "driver"))
        if boot_vga is not None:
            with open(os.path.join(pci, "boot_vga"), "w", encoding="ascii") as fh:
                fh.write(boot_vga + "\n")
        os.makedirs(os.path.join(self.drm, name))
        os.symlink(pci, os.path.join(self.drm, name, "device"))

    def test_this_machine_picks_the_amd_card_over_the_intel_igpu(self) -> None:
        self.node("renderD128", "i915")
        self.node("renderD129", "amdgpu", "1")
        os.makedirs(os.path.join(self.drm, "card1"))  # not a render node
        self.assertEqual(default_vaapi_device(self.drm), "/dev/dri/renderD129")

    def test_boot_vga_wins_between_two_amd_cards(self) -> None:
        self.node("renderD128", "amdgpu", "0")
        self.node("renderD129", "amdgpu", "1")
        self.assertEqual(default_vaapi_device(self.drm), "/dev/dri/renderD129")

    def test_lowest_node_when_no_amd_card_is_boot_vga(self) -> None:
        self.node("renderD130", "amdgpu", None)
        self.node("renderD129", "amdgpu", "0")
        self.assertEqual(default_vaapi_device(self.drm), "/dev/dri/renderD129")

    def test_falls_back_to_libvas_default_without_an_amd_card(self) -> None:
        self.node("renderD128", "i915", "1")
        self.assertEqual(default_vaapi_device(self.drm), "/dev/dri/renderD128")

    def test_falls_back_when_there_is_no_drm_directory(self) -> None:
        missing = os.path.join(self.root, "nope")
        self.assertEqual(default_vaapi_device(missing), "/dev/dri/renderD128")


class TestValidation(unittest.TestCase):
    def test_no_segments(self) -> None:
        with self.assertRaises(ValueError):
            build_command(make_info(), make_settings(segments=[]))

    def test_reversed_segment(self) -> None:
        with self.assertRaises(ValueError):
            build_command(make_info(), make_settings(segments=[Segment(9.0, 3.0)]))

    def test_zero_fps(self) -> None:
        with self.assertRaises(ValueError):
            build_command(make_info(), make_settings(fps=0))

    def test_zero_bitrate(self) -> None:
        with self.assertRaises(ValueError):
            build_command(make_info(), make_settings(mbps=0))

    def test_total_output_seconds(self) -> None:
        segs = [Segment(0, 5), Segment(100.5, 110.5), Segment(600, 630)]
        self.assertAlmostEqual(total_output_seconds(segs), 45.0)


class TestGainAndMute(unittest.TestCase):
    """Per-track gain and mute (request 1, render half)."""

    def test_gain_is_applied_per_track_before_the_mix(self) -> None:
        graph = graph_of(
            build_command(
                make_info(),
                make_settings(audio=[Track(0, 6.0), Track(2, -12.5)]),
            )
        )
        self.assertIn("[1:a:0]volume=6dB[g0]", graph)
        self.assertIn("[1:a:2]volume=-12.5dB[g2]", graph)
        self.assertLess(graph.index("volume=6dB"), graph.index("amix="))
        self.assertLess(graph.index("volume=-12.5dB"), graph.index("amix="))

    def test_muted_track_is_absent_from_the_graph_entirely(self) -> None:
        """Not mixed at volume=0 — not present at all."""
        graph = graph_of(
            build_command(
                make_info(),
                make_settings(
                    audio=[Track(0, 3.0), Track(1, 0.0, muted=True), Track(2, 0.0)]
                ),
            )
        )
        self.assertNotIn("[1:a:1]", graph)
        self.assertNotIn("[g1]", graph)
        self.assertIn("amix=inputs=2", graph)
        self.assertIn("[g0][g2]amix", graph)

    def test_mute_leaving_one_track_drops_the_amix(self) -> None:
        graph = graph_of(
            build_command(
                make_info(),
                make_settings(audio=[Track(0, 0.0, True), Track(1, -6.0)]),
            )
        )
        self.assertNotIn("amix", graph)
        self.assertIn("[1:a:1]volume=-6dB[am]", graph)

    def test_all_muted_is_video_only(self) -> None:
        argv = build_command(
            make_info(), make_settings(audio=[Track(i, 4.0, True) for i in range(3)])
        )
        graph = graph_of(argv)
        self.assertEqual(argv.count("-i"), 1)
        self.assertNotIn("-c:a", argv)
        self.assertNotIn("volume=", graph)
        self.assertNotIn("atrim", graph)
        self.assertEqual([argv[i + 1] for i, a in enumerate(argv) if a == "-map"], ["[vout]"])


class TestAv1(unittest.TestCase):
    """libsvtav1 is CRF-based; av1_vaapi is the GPU and takes a bitrate."""

    def test_offered_encoders(self) -> None:
        values = [value for _, value in ENCODERS]
        self.assertEqual(values, ["h264_vaapi", "libx264", "av1_vaapi", "libsvtav1"])
        # The RTX 3070 Ti is gone: NVENC must not come back as an option.
        self.assertNotIn("h264_nvenc", values)
        self.assertNotIn("av1_nvenc", values)

    def test_av1_omits_every_bitrate_flag(self) -> None:
        argv = build_command(
            make_info(),
            make_settings(encoder="libsvtav1", container="mkv", out_path="/tmp/o.mkv"),
        )
        self.assertEqual(flag_value(argv, "-c:v"), "libsvtav1")
        for flag in ("-b:v", "-maxrate", "-bufsize", "-rc", "-tune", "-profile:v"):
            self.assertNotIn(flag, argv, f"{flag} must not reach libsvtav1")

    def test_av1_passes_crf_and_preset(self) -> None:
        argv = build_command(
            make_info(),
            make_settings(
                encoder="libsvtav1", container="mkv", crf=38, av1_preset=6,
                out_path="/tmp/o.mkv",
            ),
        )
        self.assertEqual(flag_value(argv, "-crf"), "38")
        self.assertEqual(flag_value(argv, "-preset"), "6")
        self.assertEqual(flag_value(argv, "-pix_fmt"), "yuv420p")

    def test_av1_still_gets_aac_audio(self) -> None:
        argv = build_command(
            make_info(), make_settings(encoder="libsvtav1", container="mkv")
        )
        self.assertEqual(flag_value(argv, "-c:a"), "aac")
        self.assertEqual(flag_value(argv, "-b:a"), "192k")

    def test_is_crf_encoder(self) -> None:
        self.assertTrue(is_crf_encoder("libsvtav1"))
        self.assertFalse(is_crf_encoder("h264_vaapi"))
        self.assertFalse(is_crf_encoder("av1_vaapi"))
        self.assertFalse(is_crf_encoder("libx264"))

    def test_is_vaapi_encoder(self) -> None:
        self.assertTrue(is_vaapi_encoder("h264_vaapi"))
        self.assertTrue(is_vaapi_encoder("av1_vaapi"))
        self.assertFalse(is_vaapi_encoder("libx264"))
        self.assertFalse(is_vaapi_encoder("libsvtav1"))
        self.assertFalse(is_vaapi_encoder("h264_nvenc"))

    def test_bad_crf_raises_but_bad_mbps_is_irrelevant_for_av1(self) -> None:
        with self.assertRaises(ValueError):
            build_command(
                make_info(), make_settings(encoder="libsvtav1", container="mkv", crf=99)
            )
        # mbps is meaningless for AV1, so a zero must NOT block the render
        argv = build_command(
            make_info(), make_settings(encoder="libsvtav1", container="mkv", mbps=0)
        )
        self.assertNotIn("-b:v", argv)


class TestContainers(unittest.TestCase):
    def test_mp4_keeps_faststart(self) -> None:
        argv = build_command(make_info(), make_settings(container="mp4"))
        self.assertEqual(flag_value(argv, "-movflags"), "+faststart")

    def test_mkv_omits_faststart(self) -> None:
        """-movflags +faststart is an mp4-muxer option; matroska rejects it."""
        argv = build_command(
            make_info(), make_settings(container="mkv", out_path="/tmp/out.mkv")
        )
        self.assertNotIn("-movflags", argv)
        self.assertNotIn("+faststart", argv)

    def test_both_containers_are_offered(self) -> None:
        self.assertEqual([value for _, value in CONTAINERS], ["mp4", "mkv"])

    def test_unknown_container_raises(self) -> None:
        with self.assertRaises(ValueError):
            build_command(make_info(), make_settings(container="webm"))


class TestProgressParsing(unittest.TestCase):
    def feed(self, state: _ProgressState, block: str) -> Progress | None:
        out: Progress | None = None
        for line in block.strip().splitlines():
            got = state.feed(line.strip())
            if got is not None:
                out = got
        return out

    def test_typical_block(self) -> None:
        state = _ProgressState(total=100.0)
        p = self.feed(
            state,
            """
            frame=600
            fps=240.5
            bitrate=59000.0kbits/s
            out_time_us=25000000
            out_time_ms=25000000
            out_time=00:00:25.000000
            speed=2.5x
            progress=continue
            """,
        )
        assert p is not None
        self.assertAlmostEqual(p.out_time, 25.0)
        self.assertAlmostEqual(p.fraction, 0.25)
        self.assertAlmostEqual(p.fps, 240.5)
        self.assertAlmostEqual(p.speed, 2.5)
        assert p.eta_seconds is not None
        self.assertAlmostEqual(p.eta_seconds, 30.0)

    def test_na_values_are_tolerated(self) -> None:
        state = _ProgressState(total=100.0)
        p = self.feed(
            state,
            """
            frame=N/A
            fps=N/A
            out_time_us=N/A
            out_time_ms=N/A
            out_time=N/A
            speed=N/A
            progress=continue
            """,
        )
        assert p is not None
        self.assertEqual(p.out_time, 0.0)
        self.assertEqual(p.fraction, 0.0)
        self.assertIsNone(p.eta_seconds)

    def test_timecode_fallback(self) -> None:
        state = _ProgressState(total=7200.0)
        p = self.feed(state, "out_time=01:00:00.500000\nspeed=1.0x\nprogress=continue")
        assert p is not None
        self.assertAlmostEqual(p.out_time, 3600.5)

    def test_end_forces_full_fraction(self) -> None:
        state = _ProgressState(total=6.0)
        p = self.feed(state, "out_time_us=5980000\nspeed=1.1x\nprogress=end")
        assert p is not None
        self.assertEqual(p.fraction, 1.0)
        self.assertEqual(p.eta_seconds, 0.0)

    def test_fraction_is_clamped(self) -> None:
        state = _ProgressState(total=5.0)
        p = self.feed(state, "out_time_us=9000000\nprogress=continue")
        assert p is not None
        self.assertEqual(p.fraction, 1.0)

    def test_zero_total_does_not_divide(self) -> None:
        state = _ProgressState(total=0.0)
        p = self.feed(state, "out_time_us=1000000\nprogress=continue")
        assert p is not None
        self.assertEqual(p.fraction, 0.0)
        self.assertIsNone(p.eta_seconds)

    def test_non_kv_lines_ignored(self) -> None:
        state = _ProgressState(total=10.0)
        self.assertIsNone(state.feed("this is not a key value line"))
        self.assertIsNone(state.feed(""))

    def test_eta_uses_recent_rate_not_cumulative_speed(self) -> None:
        """ffmpeg's cumulative `speed` is useless after a long decode-forward.

        Simulates a deep cut: 50s of silence, then output flowing at 2 out-seconds
        per wall-second while ffmpeg still reports a cumulative speed of 0.01x.
        """
        clock = [50.0]
        state = _ProgressState(total=10.0, clock=lambda: clock[0])
        last = None
        for i in range(1, 5):
            clock[0] = 50.0 + i * 0.5
            last = self.feed(
                state, f"out_time_us={int(i * 1_000_000)}\nspeed=0.01x\nprogress=continue"
            )
        assert last is not None
        self.assertAlmostEqual(last.speed, 0.01)  # reported verbatim, per contract
        assert last.eta_seconds is not None
        # 6 output-seconds left at ~2 out-s/wall-s => ~3s, NOT 600s
        self.assertLess(last.eta_seconds, 10.0)
        self.assertGreater(last.eta_seconds, 1.0)

    def test_eta_falls_back_to_speed_on_first_sample(self) -> None:
        state = _ProgressState(total=100.0, clock=lambda: 0.0)
        p = self.feed(state, "out_time_us=25000000\nspeed=2.5x\nprogress=continue")
        assert p is not None and p.eta_seconds is not None
        self.assertAlmostEqual(p.eta_seconds, 30.0)


class TestRenderJob(unittest.TestCase):
    """Lifecycle tests using a stand-in subprocess — ffmpeg is never invoked."""

    def _fake_ffmpeg(self, body: str) -> list[str]:
        return [sys.executable, "-c", body]

    def test_progress_and_done_fire(self) -> None:
        argv = self._fake_ffmpeg(
            "import sys\n"
            "for i in (1,2,3):\n"
            "    print(f'out_time_us={i*1000000}');print('speed=1.0x')\n"
            "    print('progress=continue');sys.stdout.flush()\n"
            "print('out_time_us=3000000');print('progress=end');sys.stdout.flush()\n"
            "sys.stderr.write('some ffmpeg noise\\n')\n"
        )
        seen: list[Progress] = []
        done: list[tuple[int, str]] = []
        ev = threading.Event()
        job = RenderJob(argv, 3.0)
        job.start(seen.append, lambda c, t: (done.append((c, t)), ev.set()))
        self.assertTrue(ev.wait(20))
        job.wait(5)
        self.assertEqual(len(seen), 4)
        self.assertEqual(seen[-1].fraction, 1.0)
        self.assertEqual(done[0][0], 0)
        self.assertIn("some ffmpeg noise", done[0][1])

    def test_done_fires_exactly_once_on_failure(self) -> None:
        argv = self._fake_ffmpeg("import sys;sys.stderr.write('boom\\n');sys.exit(3)")
        done: list[tuple[int, str]] = []
        ev = threading.Event()
        job = RenderJob(argv, 1.0)
        job.start(lambda p: None, lambda c, t: (done.append((c, t)), ev.set()))
        self.assertTrue(ev.wait(20))
        job.wait(5)
        time.sleep(0.3)
        self.assertEqual(len(done), 1)
        self.assertEqual(done[0][0], 3)
        self.assertIn("boom", done[0][1])

    def test_stderr_tail_is_capped(self) -> None:
        argv = self._fake_ffmpeg(
            "import sys\n"
            "for i in range(500): sys.stderr.write(f'line{i}\\n')\n"
            "sys.exit(1)\n"
        )
        done: list[tuple[int, str]] = []
        ev = threading.Event()
        job = RenderJob(argv, 1.0)
        job.start(lambda p: None, lambda c, t: (done.append((c, t)), ev.set()))
        self.assertTrue(ev.wait(20))
        lines = [ln for ln in done[0][1].splitlines() if ln.startswith("line")]
        self.assertEqual(len(lines), STDERR_TAIL_LINES)
        self.assertEqual(lines[-1], "line499")

    def test_cancel_terminates_and_fires_done_once(self) -> None:
        argv = self._fake_ffmpeg("import time;time.sleep(120)")
        done: list[tuple[int, str]] = []
        ev = threading.Event()
        job = RenderJob(argv, 10.0)
        job.start(lambda p: None, lambda c, t: (done.append((c, t)), ev.set()))
        for _ in range(100):  # wait for launch
            if job.is_running():
                break
            time.sleep(0.05)
        self.assertTrue(job.is_running())
        job.cancel()
        self.assertTrue(ev.wait(20))
        job.wait(5)
        time.sleep(0.3)
        self.assertEqual(len(done), 1)
        self.assertTrue(job.cancelled)
        self.assertFalse(job.is_running())
        self.assertIn("cancelled", done[0][1])

    def test_cancel_is_idempotent(self) -> None:
        job = RenderJob([sys.executable, "-c", "pass"], 1.0)
        job.cancel()  # before start
        job.cancel()
        ev = threading.Event()
        done: list[int] = []
        job.start(lambda p: None, lambda c, t: (done.append(c), ev.set()))
        self.assertTrue(ev.wait(20))
        job.cancel()  # after done
        self.assertEqual(len(done), 1)

    def test_missing_binary_still_fires_done(self) -> None:
        done: list[tuple[int, str]] = []
        ev = threading.Event()
        job = RenderJob(["/nonexistent/ffmpeg-xyz"], 1.0)
        job.start(lambda p: None, lambda c, t: (done.append((c, t)), ev.set()))
        self.assertTrue(ev.wait(20))
        self.assertEqual(len(done), 1)
        self.assertNotEqual(done[0][0], 0)
        self.assertIn("failed to launch", done[0][1])

    def test_throwing_callbacks_do_not_break_the_job(self) -> None:
        argv = self._fake_ffmpeg(
            "print('out_time_us=1000000');print('progress=end')"
        )
        ev = threading.Event()
        codes: list[int] = []

        def bad_progress(p: Progress) -> None:
            raise RuntimeError("GUI blew up")

        def on_done(c: int, t: str) -> None:
            codes.append(c)
            ev.set()

        job = RenderJob(argv, 1.0)
        job.start(bad_progress, on_done)
        self.assertTrue(ev.wait(20))
        self.assertEqual(codes, [0])

    def test_double_start_raises(self) -> None:
        job = RenderJob([sys.executable, "-c", "pass"], 1.0)
        job.start(lambda p: None, lambda c, t: None)
        with self.assertRaises(RuntimeError):
            job.start(lambda p: None, lambda c, t: None)
        job.wait(10)


class TestVaapiHint(unittest.TestCase):
    def test_hint_added_for_vaapi_encoder_failure(self) -> None:
        job = RenderJob(["ffmpeg", "-c:v", "h264_vaapi", "out.mp4"], 1.0)
        job._stderr_tail.append(
            "[h264_vaapi @ 0x55] No usable encoding entrypoint found for profile "
            "VAProfileH264High (7)."
        )
        hint = job._encoder_hint()
        self.assertIn("libx264", hint)
        self.assertIn("h264_vaapi", hint)

    def test_hint_added_when_the_device_does_not_open(self) -> None:
        job = RenderJob(
            ["ffmpeg", "-vaapi_device", AMD, "-c:v", "av1_vaapi", "out.mp4"], 1.0
        )
        job._stderr_tail.append(
            "[AVHWDeviceContext @ 0x55] Failed to initialise VAAPI connection: "
            "-1 (unknown libva error)."
        )
        self.assertIn("libx264", job._encoder_hint())

    def test_no_hint_for_libx264(self) -> None:
        job = RenderJob(["ffmpeg", "-c:v", "libx264", "out.mp4"], 1.0)
        job._stderr_tail.append("[h264_vaapi @ 0x55] whatever")
        self.assertEqual(job._encoder_hint(), "")

    def test_no_hint_for_unrelated_vaapi_failure(self) -> None:
        job = RenderJob(["ffmpeg", "-c:v", "h264_vaapi", "/nope/out.mp4"], 1.0)
        job._stderr_tail.append("/nope/out.mp4: No such file or directory")
        self.assertEqual(job._encoder_hint(), "")


class TestExternalAudio(unittest.TestCase):
    """A dropped audio file becomes its own ffmpeg input, aligned to 0:00."""

    def test_external_file_is_an_extra_input_after_the_two_handles(self) -> None:
        argv = build_command(
            make_info(),
            make_settings(audio=mix(0), external_audio=[External("/tmp/bgm.mp3")]),
        )
        inputs = [argv[i + 1] for i, a in enumerate(argv) if a == "-i"]
        self.assertEqual(inputs, ["/tmp/in.mp4", "/tmp/in.mp4", "/tmp/bgm.mp3"])
        self.assertIn("[2:a:0]", graph_of(argv))

    def test_the_two_input_video_fix_is_untouched(self) -> None:
        """External inputs must not tempt anyone into collapsing -i f -i f."""
        graph = graph_of(
            build_command(
                make_info(),
                make_settings(audio=mix(0, 1), external_audio=[External("/tmp/bgm.mp3")]),
            )
        )
        self.assertIn("[0:v]", graph)
        self.assertNotIn("[0:a:", graph)
        self.assertIn("[1:a:0]", graph)
        self.assertIn("[1:a:1]", graph)

    def test_without_embedded_audio_the_external_is_input_1(self) -> None:
        """Nothing reads input 1 when every embedded track is muted, so the
        second handle on the source is not opened and the file shifts down."""
        argv = build_command(
            make_info(),
            make_settings(
                audio=[Track(0, muted=True)], external_audio=[External("/tmp/bgm.mp3")]
            ),
        )
        inputs = [argv[i + 1] for i, a in enumerate(argv) if a == "-i"]
        self.assertEqual(inputs, ["/tmp/in.mp4", "/tmp/bgm.mp3"])
        self.assertIn("[1:a:0]", graph_of(argv))

    def test_plan_inputs_is_the_single_source_of_that_arithmetic(self) -> None:
        """One place decides input numbering: video paths first, then audio.

        A second helper doing the same sum is how the two drift apart."""
        info = make_info()
        plan = make_settings(
            audio=mix(0), external_audio=[External("/a.mp3"), External("/b.mp3")]
        ).plan
        paths, vidx, aidx = plan_inputs(plan)
        self.assertEqual(paths, [info.path, info.path, "/a.mp3", "/b.mp3"])
        self.assertEqual(vidx[info.path], 0)
        self.assertEqual(aidx[info.path], 1)
        self.assertEqual((aidx["/a.mp3"], aidx["/b.mp3"]), (2, 3))
        # ... and with no embedded audio the externals move up one.
        bare = make_settings(audio=[], external_audio=[External("/a.mp3")]).plan
        self.assertEqual(plan_inputs(bare)[2]["/a.mp3"], 1)

    def test_several_external_files_are_numbered_in_lane_order(self) -> None:
        argv = build_command(
            make_info(),
            make_settings(
                audio=mix(0), external_audio=[External("/a.mp3"), External("/b.wav")]
            ),
        )
        inputs = [argv[i + 1] for i, a in enumerate(argv) if a == "-i"]
        self.assertEqual(inputs[2:], ["/a.mp3", "/b.wav"])
        graph = graph_of(argv)
        self.assertIn("[2:a:0]", graph)
        self.assertIn("[3:a:0]", graph)
        self.assertIn("[g0][x0][x1]amix=inputs=3", graph)

    def test_external_audio_joins_the_mix_with_its_gain(self) -> None:
        graph = graph_of(
            build_command(
                make_info(),
                make_settings(
                    audio=mix(0), external_audio=[External("/tmp/bgm.mp3", gain_db=-6.0)]
                ),
            )
        )
        self.assertIn("[2:a:0]volume=-6dB,apad=whole_dur=2780[x0]", graph)
        self.assertIn("[g0][x0]amix=inputs=2:duration=longest:normalize=0[am]", graph)

    def test_a_muted_external_file_is_left_out_entirely(self) -> None:
        """Muted means EXCLUDED from amix, never mixed at zero — and it must
        not open an input either, or the remaining files would renumber."""
        argv = build_command(
            make_info(),
            make_settings(
                audio=mix(0),
                external_audio=[External("/a.mp3", muted=True), External("/b.wav")],
            ),
        )
        inputs = [argv[i + 1] for i, a in enumerate(argv) if a == "-i"]
        self.assertEqual(inputs, ["/tmp/in.mp4", "/tmp/in.mp4", "/b.wav"])
        graph = graph_of(argv)
        self.assertNotIn("/a.mp3", graph)
        self.assertIn("[2:a:0]volume=0dB,apad=whole_dur=2780[x0]", graph)
        self.assertNotIn("[x1]", graph)

    def test_external_audio_alone_still_gets_an_audio_stream(self) -> None:
        argv = build_command(
            make_info(0),  # source has no audio at all
            make_settings(audio=[], external_audio=[External("/tmp/bgm.mp3")]),
        )
        self.assertIn("[ac]", argv)
        self.assertIn("aac", argv)
        graph = graph_of(argv)
        self.assertIn("[1:a:0]volume=0dB,apad=whole_dur=2780[am]", graph)

    def test_no_audio_anywhere_is_still_video_only(self) -> None:
        argv = build_command(make_info(0), make_settings(audio=[]))
        self.assertNotIn("[ac]", argv)
        self.assertNotIn("-c:a", argv)

    def test_empty_external_path_raises(self) -> None:
        with self.assertRaises(ValueError):
            build_command(make_info(), make_settings(external_audio=[External("")]))


class TestExternalAudioPadding(unittest.TestCase):
    """The length hazard: a music bed shorter than the video.

    Without ``apad``, a keep-segment past the end of the external file produces
    NO audio for that segment, ``concat=n=N:v=1:a=1`` is starved, and ffmpeg
    exits 0 having written a short audio stream.  Measured against real ffmpeg
    9.0.1 with a 5 s mp3 over a 12 s video and segments at 1-3 s and 8-11 s: the
    audio stream came out 2.0 s long instead of 5.0 s.  The pad is what fixes it.
    """

    def test_every_external_chain_is_padded_to_the_video_duration(self) -> None:
        info = make_info()
        graph = graph_of(
            build_command(
                info,
                make_settings(
                    segments=[Segment(1.0, 3.0), Segment(2000.0, 2003.0)],
                    audio=mix(0),
                    external_audio=[External("/short.mp3")],
                ),
            )
        )
        self.assertIn(f"apad=whole_dur={int(info.duration)}", graph)

    def test_the_pad_is_the_last_filter_before_the_mix(self) -> None:
        """Gain and lift apply to the real audio; the padding is silence, so it
        has to come after them or a lift's enable window would shift."""
        graph = graph_of(
            build_command(
                make_info(),
                make_settings(
                    audio=mix(0),
                    external_audio=[External("/s.mp3", gain_db=3.0, holes=[(5.0, 9.0)])],
                ),
            )
        )
        self.assertIn(
            "[2:a:0]volume=3dB,volume=0:enable='between(t,5,9)',"
            "apad=whole_dur=2780[x0]",
            graph,
        )

    def test_the_pad_survives_a_segment_past_the_external_files_end(self) -> None:
        """The shape that breaks concat without it: every segment must get one
        atrim fed from the padded mix."""
        graph = graph_of(
            build_command(
                make_info(),
                make_settings(
                    segments=[Segment(0.0, 5.0), Segment(2700.0, 2780.0)],
                    audio=[Track(0, muted=True)],  # only the short file left
                    external_audio=[External("/short.mp3")],
                ),
            )
        )
        self.assertIn("apad=whole_dur=2780[am]", graph)
        self.assertIn("[m0]atrim=start=0:end=5", graph)
        self.assertIn("[m1]atrim=start=2700:end=2780", graph)
        self.assertIn("concat=n=2:v=1:a=1[vc][ac]", graph)

    def test_embedded_tracks_are_not_padded(self) -> None:
        """They come off the source file and are already its length; a pad
        there would only add a filter for ffmpeg to negotiate."""
        graph = graph_of(build_command(make_info(), make_settings(audio=mix(0, 1))))
        self.assertNotIn("apad", graph)


def make_graph_settings(segments: list[Segment], **kw: object) -> list[str]:
    """argv for ``segments`` with one audio track, plus whatever ``kw`` overrides."""
    base: dict[str, object] = dict(segments=segments, audio=mix(0))
    base.update(kw)
    return build_command(make_info(), make_settings(**base))


class TestHoles(unittest.TestCase):
    """LIFT: a range removed from one lane, applied at the head of its chain."""

    def test_hole_enable_ors_the_ranges(self) -> None:
        self.assertEqual(hole_enable([(1.0, 2.0)]), "between(t,1,2)")
        self.assertEqual(
            hole_enable([(1.0, 2.5), (10.0, 12.25)]),
            "between(t,1,2.5)+between(t,10,12.25)",
        )

    def test_video_hole_is_a_drawbox_before_the_split(self) -> None:
        graph = graph_of(
            build_command(
                make_info(),
                make_settings(
                    segments=[Segment(0.0, 100.0), Segment(200.0, 300.0)],
                    video_holes=[(10.0, 20.0), (250.0, 260.0)],
                ),
            )
        )
        self.assertTrue(
            graph.startswith(
                "[0:v]drawbox=x=0:y=0:w=iw:h=ih:color=black@1:t=fill:"
                "enable='between(t,10,20)+between(t,250,260)'[vh];[vh]split=2"
            ),
            graph,
        )

    def test_video_hole_on_a_single_segment_feeds_the_trim_directly(self) -> None:
        graph = graph_of(
            build_command(make_info(), make_settings(video_holes=[(12.0, 14.0)]))
        )
        # Hole and trim both read input 0's time, which its -ss 8 moved.
        self.assertIn("enable='between(t,4,6)'[vh];[vh]trim=start=2:end=12", graph)

    def test_no_holes_leaves_the_graph_exactly_as_it_was(self) -> None:
        plain = graph_of(build_command(make_info(), make_settings()))
        self.assertNotIn("drawbox", plain)
        self.assertNotIn("enable=", plain)
        self.assertTrue(plain.startswith("[0:v]trim="))

    def test_audio_hole_is_a_second_volume_after_the_gain(self) -> None:
        graph = graph_of(
            build_command(
                make_info(),
                make_settings(audio=[Track(0, gain_db=-6.0, holes=[(5.0, 9.0)])]),
            )
        )
        self.assertIn(
            "[1:a:0]volume=-6dB,volume=0:enable='between(t,5,9)'[am]", graph
        )

    def test_holes_are_per_track(self) -> None:
        graph = graph_of(
            build_command(
                make_info(),
                make_settings(audio=[Track(0, holes=[(5.0, 9.0)]), Track(1)]),
            )
        )
        self.assertIn("[1:a:0]volume=0dB,volume=0:enable='between(t,5,9)'[g0]", graph)
        self.assertIn("[1:a:1]volume=0dB[g1]", graph)

    def test_a_muted_track_with_holes_still_vanishes(self) -> None:
        graph = graph_of(
            build_command(
                make_info(),
                make_settings(audio=[Track(0, muted=True, holes=[(5.0, 9.0)]), Track(1)]),
            )
        )
        self.assertNotIn("[g0]", graph)
        self.assertNotIn("enable=", graph)

    def test_hole_ranges_are_cleaned(self) -> None:
        """Empty ranges vanish; a reversed one is put the right way round."""
        graph = graph_of(
            build_command(
                make_info(),
                make_settings(
                    video_holes=[(5.0, 5.0), (20.0, 10.0)],
                    audio=[Track(0, holes=[(9.0, 9.0000001)])],
                ),
            )
        )
        self.assertIn("enable='between(t,2,12)'", graph)  # 10-20, less the -ss 8
        self.assertNotIn("between(t,5,5)", graph)
        self.assertIn("[1:a:0]volume=0dB[am]", graph, "an empty hole must not add a filter")

    def test_holes_do_not_change_the_segment_structure(self) -> None:
        """The frozen contract: split/trim/concat and tmix->fps->scale stay put."""
        segments = [Segment(0.0, 5.0), Segment(50.0, 60.0), Segment(100.0, 110.0)]
        plain = graph_of(make_graph_settings(segments))
        holed = graph_of(
            make_graph_settings(segments, video_holes=[(2.0, 3.0)],
                                audio=[Track(0, holes=[(52.0, 53.0)])])
        )
        for marker in (
            "split=3", "concat=n=3:v=1:a=1[vc][ac]", "asplit=3",
            "[vc]tmix=frames=7,fps=60,scale=2560:1440[vout]",
        ):
            self.assertIn(marker, plain)
            self.assertIn(marker, holed)


class TestFades(unittest.TestCase):
    """FADES: a ramp at a clip's own two ends.

    Where the filter SITS is the whole game here, and these cases pin it.  The
    proof that it comes out of ffmpeg as a ramp is measured on real output in
    ``test_blocks.TestRealFfmpegEdges`` — a graph string has never been
    evidence of anything in this project.
    """

    def faded(self, **kw: object) -> TimelinePlan:
        """The default plan with fades on its one video clip and one track."""
        info = make_info()
        plan = make_plan(info, mix(0))
        for clip in (plan.video[0].clips[0], plan.audio[0].clips[0]):
            clip.fade_in = float(kw.get("fade_in", 0.0))
            clip.fade_out = float(kw.get("fade_out", 0.0))
        return plan

    def test_no_fade_leaves_the_graph_exactly_as_it_was(self) -> None:
        """The frozen v3 shape.  Every measured fact in HANDOFF was established
        on it, so a feature nobody switched on must not move a character of it."""
        plain = graph_of(build_command(make_info(), make_settings()))
        with_fields = graph_of(
            build_command(make_info(), make_settings(plan=self.faded()))
        )
        self.assertEqual(plain, with_fields)
        self.assertNotIn("fade", plain)

    def test_a_video_fade_sits_between_the_trim_and_the_setpts(self) -> None:
        """Not at the head of ``[0:v]``: ``fade=t=in`` blacks out everything
        BEFORE its start time, so a head fade would black out every other clip
        and every other piece of this one — with exit code 0."""
        graph = graph_of(
            build_command(
                make_info(),
                make_settings(plan=self.faded(fade_in=2.0, fade_out=3.0),
                              segments=[Segment(0.0, 100.0)]),
            )
        )
        self.assertIn(
            "[0:v]trim=start=0:end=100,fade=t=in:st=0:d=2,"
            "fade=t=out:st=2777:d=3,setpts=PTS-STARTPTS[vc]",
            graph,
        )

    def test_the_video_fade_is_in_the_clip_s_own_source_time(self) -> None:
        """A clip that starts 30 s into its file fades from 30 s into its file,
        not from 0 — the frames carry source timestamps at that point in the
        chain."""
        info = make_info()
        plan = TimelinePlan(
            duration=100.0,
            video=[VideoLane([
                VideoClip(info.path, 0.0, 30.0, 80.0),
                VideoClip(info.path, 50.0, 200.0, 250.0),
            ])],
            audio=[],
        )
        graph = graph_of(
            build_command(info, make_settings(plan=plan, segments=[Segment(0.0, 100.0)]))
        )
        self.assertNotIn("fade", graph, "a plan with no fades grew one")
        plan.video[0].clips[0].fade_in = 4.0
        plan.video[0].clips[1].fade_out = 5.0
        argv = build_command(info, make_settings(plan=plan, segments=[Segment(0.0, 100.0)]))
        graph = graph_of(argv)
        # The file opens 28 s in (-ss, input_seeks), so source time 30 reads as 2.
        seek = int(argv[argv.index("-ss") + 1])
        self.assertEqual(seek, 28)
        self.assertIn(f"fade=t=in:st={30 - seek}:d=4", graph)
        self.assertIn(f"fade=t=out:st={245 - seek}:d=5", graph)

    def test_every_piece_of_one_clip_carries_the_same_fade(self) -> None:
        """``_piece_video_parts`` cuts a clip at every visibility and cut
        boundary and ``concat``s the lot.  The identical filter on each piece is
        what makes the ramp continue across the joins instead of restarting —
        ffmpeg's ``fade`` reads the frame's own timestamp, so a piece that lies
        past the window comes out at full brightness."""
        info = make_info()
        plan = TimelinePlan(
            duration=100.0,
            video=[VideoLane([VideoClip(info.path, 0.0, 0.0, 100.0)])],
            audio=[],
        )
        plan.video[0].clips[0].fade_in = 6.0
        # Two keep-segments cut the clip into two pieces, and the cut lands
        # INSIDE the fade window.
        graph = graph_of(
            build_command(
                info,
                make_settings(plan=plan, segments=[Segment(0.0, 3.0), Segment(4.0, 100.0)]),
            )
        )
        self.assertEqual(graph.count("fade=t=in:st=0:d=6"), 2, graph)

    def test_an_audio_fade_is_clip_local_and_comes_before_the_delay(self) -> None:
        """``st=0`` has to mean the clip's own first sample.  After the
        ``adelay`` it would mean the top of the TIMELINE, which for a clip
        placed at 40 s is 40 s of material faded by nothing at all."""
        info = make_info()
        plan = TimelinePlan(
            duration=200.0,
            video=[VideoLane([VideoClip(info.path, 0.0, 0.0, 200.0)])],
            audio=[AudioLane([AudioClip("/tmp/bed.mp3", 0, 40.0, 5.0, 65.0)])],
        )
        plan.audio[0].clips[0].fade_in = 2.0
        plan.audio[0].clips[0].fade_out = 4.0
        graph = graph_of(
            build_command(info, make_settings(plan=plan, segments=[Segment(0.0, 200.0)]))
        )
        self.assertIn(
            "atrim=start=5:end=65,asetpts=PTS-STARTPTS,"
            "afade=t=in:st=0:d=2,afade=t=out:st=56:d=4,adelay=40000:all=1",
            graph,
        )

    def test_an_audio_fade_comes_before_the_pad(self) -> None:
        """``apad`` puts silence after the clip; fading that instead of the
        material would be a fade that does nothing, silently."""
        info = make_info()
        plan = TimelinePlan(
            duration=200.0,
            video=[VideoLane([VideoClip(info.path, 0.0, 0.0, 200.0)])],
            # A music bed that stops long before the timeline does: this is the
            # clip apad exists for.
            audio=[AudioLane([AudioClip("/tmp/bed.mp3", 0, 0.0, 0.0, 60.0)])],
        )
        plan.audio[0].clips[0].fade_out = 3.0
        graph = graph_of(
            build_command(info, make_settings(plan=plan, segments=[Segment(0.0, 200.0)]))
        )
        self.assertIn("afade=t=out:st=57:d=3", graph)
        self.assertLess(graph.index("afade=t=out"), graph.index("apad="), graph)

    def test_a_fade_longer_than_its_clip_is_clamped_not_emitted(self) -> None:
        """``fade`` with a duration past the end of the material would run the
        ramp off the end and leave the clip dark for its whole length."""
        info = make_info()
        plan = TimelinePlan(
            duration=50.0,
            video=[VideoLane([VideoClip(info.path, 0.0, 0.0, 50.0)])],
            audio=[AudioLane([AudioClip(info.path, 0, 0.0, 0.0, 50.0)])],
        )
        for clip in (plan.video[0].clips[0], plan.audio[0].clips[0]):
            clip.fade_in = 500.0
        graph = graph_of(
            build_command(info, make_settings(plan=plan, segments=[Segment(0.0, 50.0)]))
        )
        self.assertIn("fade=t=in:st=0:d=50,", graph)
        self.assertIn("afade=t=in:st=0:d=50", graph)

    def test_fades_do_not_change_the_segment_structure(self) -> None:
        """Same frozen contract the holes have to keep."""
        segments = [Segment(0.0, 5.0), Segment(50.0, 60.0), Segment(100.0, 110.0)]
        plain = graph_of(make_graph_settings(segments))
        faded = graph_of(
            make_graph_settings(segments, plan=self.faded(fade_in=1.0, fade_out=2.0))
        )
        for marker in (
            "split=3", "concat=n=3:v=1:a=1[vc][ac]", "asplit=3",
            "[vc]tmix=frames=7,fps=60,scale=2560:1440[vout]",
        ):
            self.assertIn(marker, plain)
            self.assertIn(marker, faded)


if __name__ == "__main__":
    unittest.main(verbosity=2)
