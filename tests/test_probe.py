"""Unit tests for ffcut.probe helpers; one optional test touches a real file."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ffcut import probe as probe_mod  # noqa: E402
from ffcut.probe import (  # noqa: E402
    AudioFileInfo,
    AudioTrack,
    MediaInfo,
    ProbeError,
    _spawn,
    cancel_all,
    resume,
    _parse_rate,
    _useful_title,
    probe,
    probe_audio,
    thumbnails_at,
    waveform_png,
)

#: Any real multi-track recording, to probe a file ffmpeg did not just generate.
REAL_FILE = os.environ.get("FFCUT_TEST_FILE", "")
HAVE_FFMPEG = shutil.which("ffmpeg") is not None


def png_trace_height(path: str) -> int:
    """How many rows of an RGBA PNG contain anything opaque.

    Byte size is a useless proxy for "how tall is the waveform" — PNG packs two
    solid blocks of different heights into the same number of bytes — so the
    trace is measured directly.  ffmpeg's png encoder writes 8-bit RGBA,
    non-interlaced, which is the only shape handled here.
    """
    import struct
    import zlib

    blob = open(path, "rb").read()
    width, height = struct.unpack(">II", blob[16:24])
    assert blob[24] == 8 and blob[25] == 6 and blob[28] == 0, "expected 8-bit RGBA PNG"
    data, pos = bytearray(), 8
    while pos < len(blob):
        length, kind = struct.unpack(">I", blob[pos:pos + 4])[0], blob[pos + 4:pos + 8]
        if kind == b"IDAT":
            data += blob[pos + 8:pos + 8 + length]
        pos += 12 + length
    raw = zlib.decompress(bytes(data))

    stride, bpp = width * 4, 4
    prev = bytearray(stride)
    rows = 0
    for y in range(height):
        head = y * (stride + 1)
        ftype, line = raw[head], bytearray(raw[head + 1:head + 1 + stride])
        for i in range(stride):
            a = line[i - bpp] if i >= bpp else 0
            b = prev[i]
            c = prev[i - bpp] if i >= bpp else 0
            if ftype == 1:
                line[i] = (line[i] + a) & 0xFF
            elif ftype == 2:
                line[i] = (line[i] + b) & 0xFF
            elif ftype == 3:
                line[i] = (line[i] + (a + b) // 2) & 0xFF
            elif ftype == 4:
                p = a + b - c
                pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                line[i] = (line[i] + (a if pa <= pb and pa <= pc else b if pb <= pc else c)) & 0xFF
        if any(line[3::4]):
            rows += 1
        prev = line
    return rows


class TestUsefulTitle(unittest.TestCase):
    def test_title_tag_wins(self) -> None:
        self.assertEqual(_useful_title({"title": "Mic", "name": "x", "handler_name": "y"}), "Mic")

    def test_mp4_track_name_atom_is_a_title(self) -> None:
        # ffprobe reports an MP4 per-track title (OBS track names,
        # -metadata:s:a:N title=...) as tag:name, not tag:title.
        self.assertEqual(_useful_title({"name": "Discord", "handler_name": "SoundHandler"}), "Discord")

    def test_boring_handler_is_dropped(self) -> None:
        self.assertIsNone(_useful_title({"handler_name": "SoundHandler"}))
        self.assertIsNone(_useful_title({"handler_name": "  "}))
        self.assertIsNone(_useful_title({}))

    def test_informative_handler_survives(self) -> None:
        self.assertEqual(_useful_title({"handler_name": "Game Audio"}), "Game Audio")


class TestParseRate(unittest.TestCase):
    def test_rational(self) -> None:
        self.assertEqual(_parse_rate("120/1"), 120.0)
        # 429.mp4's avg_frame_rate
        self.assertAlmostEqual(_parse_rate("55600000/463333"), 120.0000863, places=6)

    def test_bad_values(self) -> None:
        self.assertEqual(_parse_rate("0/0"), 0.0)
        self.assertEqual(_parse_rate(""), 0.0)
        self.assertEqual(_parse_rate(None), 0.0)
        self.assertEqual(_parse_rate("abc"), 0.0)


class TestAudioTrackLabel(unittest.TestCase):
    def test_label_is_one_based_with_optional_bits(self) -> None:
        self.assertEqual(AudioTrack(0, "aac", 2, 48000, None, None).label, "1: aac 2ch")
        self.assertEqual(
            AudioTrack(2, "aac", 2, 44100, "deu", "Discord").label, "3: aac 2ch deu (Discord)"
        )


@unittest.skipUnless(os.path.isfile(REAL_FILE), "real test recording not present")
class TestRealFile(unittest.TestCase):
    def test_probe_matches_known_shape(self) -> None:
        info = probe(REAL_FILE)
        self.assertEqual((info.width, info.height), (1664, 936))
        self.assertEqual(info.fps, 120.0)
        self.assertAlmostEqual(info.duration, 2780.0, delta=1.0)
        self.assertEqual(info.vcodec, "h264")
        self.assertEqual([t.index for t in info.audio], list(range(len(info.audio))))


@unittest.skipUnless(HAVE_FFMPEG, "ffmpeg not on PATH")
class TestAssetsOnARealFile(unittest.TestCase):
    """Thumbnails and waveforms against a generated 8 s / 2-track file.

    Cheap enough to build per run (~0.3 s) and it exercises the actual ffmpeg
    invocations, which is the only way these are worth testing.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.TemporaryDirectory(prefix="ffcut-probe-")
        cls.src = os.path.join(cls.tmp.name, "src.mp4")
        subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-f", "lavfi", "-i", "testsrc2=size=320x180:rate=30:duration=8",
                # track 0: quiet 1 kHz, track 1: loud 300 Hz
                "-f", "lavfi", "-i", "sine=frequency=1000:duration=8:sample_rate=48000",
                "-f", "lavfi", "-i", "sine=frequency=300:duration=8:sample_rate=48000",
                "-map", "0:v", "-map", "1:a", "-map", "2:a",
                "-filter:a:0", "volume=-20dB", "-filter:a:1", "volume=-6dB",
                "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac",
                cls.src,
            ],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()

    def test_make_proxy_keeps_timing_and_every_audio_track(self) -> None:
        from ffcut.render import default_vaapi_device

        out = os.path.join(self.tmp.name, "proxy", "p.mkv")
        probe_mod.make_proxy(self.src, out, 90, default_vaapi_device(), 60.0)
        info = probe(out)
        self.assertEqual(info.height, 90)
        self.assertEqual(len(info.audio), 2, "lane track numbers must map 1:1")
        self.assertAlmostEqual(info.duration, 8.0, delta=0.3)
        self.assertFalse(os.path.exists(out + ".part.mkv"))

    def test_probe_sees_both_tracks(self) -> None:
        info = probe(self.src)
        self.assertEqual(len(info.audio), 2)
        self.assertAlmostEqual(info.duration, 8.0, delta=0.3)

    def test_thumbnails_at_keeps_caller_order_and_grabs_distinct_frames(self) -> None:
        paths = thumbnails_at(
            self.src, os.path.join(self.tmp.name, "at"), [5.0, 1.0, 3.0], 80
        )
        self.assertEqual(
            [os.path.basename(p) for p in paths],
            ["thumb_0000.jpg", "thumb_0001.jpg", "thumb_0002.jpg"],
        )
        frames = [open(p, "rb").read() for p in paths]
        self.assertTrue(all(frames))
        self.assertEqual(len(set(frames)), 3, "three timestamps must give three frames")

    def test_fill_gaps_keeps_the_strip_length(self) -> None:
        """A failed grab must not shift every later thumbnail along the lane."""
        out = os.path.join(self.tmp.name, "gaps")
        # 200 s is past the end of an 8 s file: those grabs cannot succeed
        times = [1.0, 200.0, 3.0, 300.0]
        dropped = thumbnails_at(self.src, out, times, 80)
        filled = thumbnails_at(self.src, out, times, 80, fill_gaps=True)
        self.assertLess(len(dropped), len(times))
        self.assertEqual(len(filled), len(times))
        self.assertTrue(all(os.path.getsize(p) > 0 for p in filled))

    def test_waveform_png_per_track(self) -> None:
        from struct import unpack

        heights = []
        for track in (0, 1):
            out = os.path.join(self.tmp.name, f"wave{track}.png")
            self.assertEqual(
                waveform_png(self.src, track, out, 300, 48), out
            )
            with open(out, "rb") as fh:
                header = fh.read(24)
            self.assertEqual(header[:8], b"\x89PNG\r\n\x1a\n")
            width, height = unpack(">II", header[16:24])
            self.assertEqual((width, height), (300, 48))
            heights.append(png_trace_height(out))
        # the -6 dB track draws a visibly taller trace than the -20 dB one
        self.assertGreater(heights[1], heights[0])

    def test_waveform_scale_sqrt_lifts_a_quiet_track(self) -> None:
        """The default is sqrt, not lin: -20 dBFS must not vanish entirely.

        showwavespic's own density scaling already squashes a long file hard —
        measured here, a -20 dBFS track drawn with ``scale=lin`` lights ZERO
        rows of a 48 px strip.  That is the "flat waveform" complaint at its
        limit.  ``sqrt`` gets it back to something the lane's own peak
        normalisation can then stretch to full height.
        """
        lin = os.path.join(self.tmp.name, "lin.png")
        sqrt = os.path.join(self.tmp.name, "sqrt.png")
        waveform_png(self.src, 0, lin, 300, 48, scale="lin")
        waveform_png(self.src, 0, sqrt, 300, 48)  # default
        tall_lin, tall_sqrt = png_trace_height(lin), png_trace_height(sqrt)
        self.assertLessEqual(tall_lin, 1, "lin should be a hairline or nothing")
        self.assertGreaterEqual(tall_sqrt, 3)
        self.assertGreater(tall_sqrt, tall_lin)

    def test_waveform_png_range(self) -> None:
        out = os.path.join(self.tmp.name, "wave-range.png")
        waveform_png(self.src, 0, out, 200, 32, start=2.0, end=4.0)
        self.assertGreater(os.path.getsize(out), 0)

    def test_waveform_missing_track_raises(self) -> None:
        with self.assertRaises(ProbeError):
            waveform_png(self.src, 7, os.path.join(self.tmp.name, "no.png"), 100, 20)


@unittest.skipUnless(HAVE_FFMPEG, "ffmpeg is required")
class TestProbeAudio(unittest.TestCase):
    """Standalone audio files: what a drop onto the timeline has to survive."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.TemporaryDirectory(prefix="ffcut-audio-")
        cls.files: dict[str, str] = {}
        for name, args in {
            "mp3": ("libmp3lame", "bgm.mp3"),
            "wav": ("pcm_s16le", "bgm.wav"),
            "flac": ("flac", "bgm.flac"),
            "opus": ("libopus", "bgm.opus"),
        }.items():
            codec, filename = args
            path = os.path.join(cls.tmp.name, filename)
            subprocess.run(
                ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                 "-f", "lavfi", "-i", "sine=frequency=440:duration=6:sample_rate=48000",
                 "-ac", "2", "-c:a", codec, path],
                check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
            )
            cls.files[name] = path
        # A video with no audio at all: probe_audio must refuse it.
        cls.silent = os.path.join(cls.tmp.name, "silent.mp4")
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
             "-f", "lavfi", "-i", "testsrc2=size=64x64:rate=10:duration=2",
             "-c:v", "libx264", "-preset", "ultrafast", cls.silent],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()

    def test_every_droppable_format_probes(self) -> None:
        for name, path in self.files.items():
            info = probe_audio(path)
            self.assertIsInstance(info, AudioFileInfo, name)
            self.assertAlmostEqual(info.duration, 6.0, delta=0.4, msg=name)
            self.assertEqual(info.channels, 2, name)
            self.assertGreater(info.sample_rate, 0, name)
            self.assertNotEqual(info.codec, "?", name)
            self.assertEqual(info.path, path)

    def test_label_leads_with_the_filename(self) -> None:
        label = probe_audio(self.files["mp3"]).label
        self.assertTrue(label.startswith("bgm.mp3 · "), label)
        self.assertIn("2ch", label)

    def test_a_file_with_no_audio_stream_is_refused(self) -> None:
        with self.assertRaises(ProbeError) as caught:
            probe_audio(self.silent)
        self.assertIn("no audio stream", str(caught.exception))

    def test_missing_and_empty_paths_raise(self) -> None:
        with self.assertRaises(ProbeError):
            probe_audio(os.path.join(self.tmp.name, "nope.mp3"))
        with self.assertRaises(ProbeError):
            probe_audio("")

    def test_a_non_media_file_raises_rather_than_returning_junk(self) -> None:
        path = os.path.join(self.tmp.name, "notes.txt")
        with open(path, "w") as fh:
            fh.write("not audio")
        with self.assertRaises(ProbeError):
            probe_audio(path)

    def test_probe_audio_honours_cancellation(self) -> None:
        """It must go through the same _spawn registry as everything else, or a
        closing window would join the GUI thread against it."""
        cancel_all()
        try:
            with self.assertRaises(ProbeError):
                probe_audio(self.files["wav"])
        finally:
            resume()
        self.assertIsInstance(probe_audio(self.files["wav"]), AudioFileInfo)


SLEEPER = [sys.executable, "-c", "import time; time.sleep(30)"]


@unittest.skipUnless(HAVE_FFMPEG, "ffmpeg is required")
class TestCoverArt(unittest.TestCase):
    """An mp3 with embedded album art carries a mjpeg/png VIDEO stream flagged
    ``attached_pic``; calling that a movie would be wrong."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.TemporaryDirectory(prefix="ffcut-art-")
        song = os.path.join(cls.tmp.name, "song.mp3")
        cover = os.path.join(cls.tmp.name, "cover.png")
        cls.tagged = os.path.join(cls.tmp.name, "tagged.mp3")
        for args in (
            ("-f", "lavfi", "-i", "sine=frequency=440:duration=3", "-c:a", "libmp3lame", song),
            ("-f", "lavfi", "-i", "color=c=red:size=64x64:duration=1", "-frames:v", "1", cover),
            ("-i", song, "-i", cover, "-map", "0:a", "-map", "1:v", "-c:a", "copy",
             "-c:v", "copy", "-id3v2_version", "3", "-disposition:v", "attached_pic",
             cls.tagged),
        ):
            subprocess.run(
                ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args],
                check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
            )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()

    def test_album_art_does_not_make_an_mp3_a_video(self) -> None:
        self.assertIsInstance(probe_audio(self.tagged), AudioFileInfo)
        with self.assertRaises(ProbeError):
            probe(self.tagged)


class TestCancellation(unittest.TestCase):
    """``cancel_all`` is what keeps a closing window off a running ffmpeg."""

    def tearDown(self) -> None:
        resume()  # never leave the module latched for the next test

    def test_cancel_all_interrupts_a_child_already_running(self) -> None:
        elapsed: list[float] = []

        def work() -> None:
            t0 = time.monotonic()
            _spawn(SLEEPER, timeout=30.0)  # killed -> returns, does not raise
            elapsed.append(time.monotonic() - t0)

        thread = threading.Thread(target=work)
        thread.start()
        try:
            deadline = time.monotonic() + 5.0
            while not probe_mod._LIVE and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertTrue(probe_mod._LIVE, "child never registered as live")
            cancel_all()
            thread.join(5.0)
        finally:
            if thread.is_alive():  # don't wedge the suite on a failure
                cancel_all()
                thread.join(5.0)
        self.assertFalse(thread.is_alive(), "worker never unwound after cancel_all")
        # The point of the exercise: not the 30 s the child asked for.
        self.assertLess(elapsed[0], 5.0)
        self.assertFalse(probe_mod._LIVE, "registry was not drained")

    def test_cancelled_module_refuses_new_children_until_resume(self) -> None:
        cancel_all()
        with self.assertRaises(ProbeError):
            _spawn([sys.executable, "-c", "pass"], timeout=10.0)
        resume()
        self.assertEqual(_spawn([sys.executable, "-c", "pass"], timeout=10.0).returncode, 0)

    def test_timeout_still_kills_and_reports(self) -> None:
        with self.assertRaises(subprocess.TimeoutExpired):
            _spawn(SLEEPER, timeout=0.3)
        self.assertFalse(probe_mod._LIVE, "a timed-out child stayed in the registry")

    def test_missing_binary_still_raises_probe_error(self) -> None:
        with self.assertRaises(ProbeError):
            probe_mod._run(["ffcut-no-such-binary"], 5.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
