"""Offscreen tests for the GUI half, run against the REAL timeline widget.

Runs with QT_QPA_PLATFORM=offscreen and never launches ffmpeg: the render job
is replaced with a controllable fake, probing and asset extraction are
monkeypatched, and the modal file/option dialogs are stubbed.  The user's
~/.config/ffcut is never touched.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ffcut import probe as probe_mod  # noqa: E402
from ffcut import render as render_mod  # noqa: E402
from PyQt6 import sip  # noqa: E402
from PyQt6.QtCore import (  # noqa: E402
    QEvent,
    QMimeData,
    QObject,
    QPoint,
    QPointF,
    QSize,
    Qt,
    QTimer,
    QUrl,
)
from PyQt6.QtGui import (  # noqa: E402
    QContextMenuEvent,
    QDragEnterEvent,
    QDropEvent,
    QMouseEvent,
)
from PyQt6.QtMultimedia import QMediaPlayer  # noqa: E402
from PyQt6.QtMultimediaWidgets import QVideoWidget  # noqa: E402
from PyQt6.QtTest import QTest  # noqa: E402
from PyQt6.QtWidgets import (  # noqa: E402
    QApplication,
    QDialog,
    QFileDialog,
    QMessageBox,
    QWidget,
)

_app = QApplication.instance() or QApplication(sys.argv)

# Route Qt's own messages through Python so a qFatal (which aborts the
# process) leaves its reason in the test output instead of dying silently.
from PyQt6.QtCore import QtMsgType, qInstallMessageHandler  # noqa: E402


def _qt_message(kind: QtMsgType, _context: object, message: str) -> None:
    if kind in (QtMsgType.QtCriticalMsg, QtMsgType.QtFatalMsg):
        print(f"\n[Qt {kind.name}] {message}", file=sys.stderr, flush=True)


qInstallMessageHandler(_qt_message)


def delete(*objects: QObject) -> None:
    """Destroy Qt objects NOW.

    ``deleteLater()`` alone does nothing here: there is no event loop running
    above the test, so the deferred deletes are never dispatched and every
    window — QMediaPlayer, QVideoWidget, worker threads and all — would survive
    to interpreter teardown, where Qt objects die in arbitrary order relative
    to the QApplication.  Flushing DeferredDelete explicitly is the fix.
    """
    for obj in objects:
        obj.deleteLater()
    QApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    assert all(sip.isdeleted(obj) for obj in objects)

import ffcut.app as appmod  # noqa: E402
import ffcut.theme as theme  # noqa: E402
from ffcut.probe import AudioFileInfo, AudioTrack, MediaInfo, ProbeError  # noqa: E402
from ffcut.render import CANCEL_GRACE, Progress  # noqa: E402
from ffcut.timeline import Clip, TimelineWidget  # noqa: E402

_TMP = tempfile.TemporaryDirectory(prefix="ffcut-tests-")
appmod.CONFIG_DIR = Path(_TMP.name) / "cfg"
appmod.CONFIG_PATH = appmod.CONFIG_DIR / "config.json"
# A modal "Discard the current edit?" would block every test that edits and then
# loads or closes; TestDiscardPrompt drives the real one.
_REAL_DISCARD_OK = appmod.MainWindow._discard_ok
appmod.MainWindow._discard_ok = lambda self: True  # type: ignore[method-assign]


def pump(seconds: float) -> None:
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        _app.processEvents()
        time.sleep(0.01)


def fake_info(path: str, duration: float = 2780.0, n_audio: int = 3) -> MediaInfo:
    return MediaInfo(
        path, duration, 1664, 936, 120.0, "h264",
        [AudioTrack(i, "aac", 2, 48000, "eng", f"T{i + 1}") for i in range(n_audio)],
    )


def select(tl: TimelineWidget, a: float, b: float) -> None:
    """Mark edited ``[a, b]`` through the public seam (what I / O do)."""
    tl.set_playhead(a)
    tl.set_mark_in()
    tl.set_playhead(b)
    tl.set_mark_out()


def cut_source(tl: TimelineWidget, a: float, b: float) -> None:
    """Remove SOURCE ``[a, b]`` — the seam works in edited time, so convert."""
    select(tl, tl.source_to_edited(a), tl.source_to_edited(b))
    tl.cut_selection()


def keeps(tl: TimelineWidget) -> list[tuple[float, float]]:
    return [(s.start, s.end) for s in tl.keep_segments_source()]


#: QDropEvent does NOT take ownership of its QMimeData, and reading
#: ``event.mimeData()`` after Python has collected it is a segfault rather than
#: an exception — so every payload built here is parked for the run.
_KEEP_ALIVE: list[QMimeData] = []


def mime(*paths: str) -> QMimeData:
    data = QMimeData()
    data.setUrls([QUrl.fromLocalFile(p) for p in paths])
    _KEEP_ALIVE.append(data)
    return data


def drop_at(point: QPointF, *paths: str) -> QDropEvent:
    """A drop of ``paths`` at ``point``, in the receiving widget's coords."""
    return QDropEvent(
        point, Qt.DropAction.CopyAction, mime(*paths),
        Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
        QEvent.Type.Drop,
    )


class FakeJob:
    """Stands in for render.RenderJob; the test drives the callbacks by hand."""

    def __init__(self, argv: list[str], total: float) -> None:
        self.argv = argv
        self.total = total
        self.on_progress = None
        self.on_done = None
        self.cancel_calls = 0
        self.done_on_cancel = True

    def start(self, on_progress, on_done) -> None:  # noqa: ANN001
        self.on_progress = on_progress
        self.on_done = on_done

    def cancel(self) -> None:
        self.cancel_calls += 1
        if self.done_on_cancel and self.on_done is not None:
            # Like the real engine: from another thread, a moment later, and
            # (deliberately) twice — the dialog must only honour the first.
            def fire() -> None:
                time.sleep(0.15)
                self.on_done(255, "render cancelled")
                self.on_done(255, "render cancelled")

            threading.Thread(target=fire, daemon=True).start()

    def progress(self, out_time: float) -> None:
        assert self.on_progress is not None
        frac = out_time / self.total
        self.on_progress(Progress(out_time, self.total, frac, 60.0, 1.0, (self.total - out_time)))


class TestProgressDialogSilentWindow(unittest.TestCase):
    """ffmpeg reports nothing while decoding to the first cut; the dialog must
    stay honest and alive on its own timer."""

    def test_preparing_then_progress_then_stall_then_done(self) -> None:
        appmod._STALL_AFTER = 0.5  # keep the test short
        job = FakeJob(["ffmpeg"], 9.0)
        out = Path(_TMP.name) / "silent.mp4"
        dlg = appmod.RenderProgressDialog(job, out, None)
        seen: dict[str, object] = {}

        def phase1() -> None:  # 1.3 s in, still no callback (ticker has fired at >= 1.0 s)
            seen["bar_range"] = (dlg._bar.minimum(), dlg._bar.maximum())
            seen["status"] = dlg._status.text()
            seen["elapsed"] = dlg._elapsed.text()
            seen["cancel_enabled"] = dlg._cancel.isEnabled()
            seen["watchdog"] = dlg._watchdog.isActive()
            job.progress(0.75)
            job.progress(3.0)
            QTimer.singleShot(50, phase2)

        def phase2() -> None:
            seen["bar_after"] = (dlg._bar.minimum(), dlg._bar.maximum(), dlg._bar.value())
            seen["status_after"] = dlg._status.text()
            QTimer.singleShot(900, phase3)  # > _STALL_AFTER with no callbacks

        def phase3() -> None:
            seen["status_stalled"] = dlg._status.text()
            appmod.part_path(out).write_bytes(b"render")  # what ffmpeg wrote
            job.on_done(0, "")  # type: ignore[misc]

        QTimer.singleShot(1300, phase1)
        result = dlg.exec()

        self.assertEqual(result, QDialog.DialogCode.Accepted)
        self.assertEqual(seen["bar_range"], (0, 0), "bar must be indeterminate before progress")
        self.assertIn("Preparing", str(seen["status"]))
        self.assertEqual(seen["elapsed"], "00:01", "elapsed must tick from the dialog's own timer")
        self.assertTrue(seen["cancel_enabled"])
        self.assertFalse(seen["watchdog"], "no watchdog may be armed before Cancel")
        self.assertEqual(seen["bar_after"], (0, 1000, 333))
        self.assertEqual(seen["status_after"], "Encoded 00:03 of 00:09")
        self.assertIn("decoding ahead", str(seen["status_stalled"]))
        self.assertTrue(str(seen["status_stalled"]).startswith("Encoded 00:03 of 00:09"))
        self.assertEqual(dlg._bar.value(), 1000)
        self.assertEqual(dlg.exit_code, 0)
        delete(dlg)


class TestProgressDialogCancel(unittest.TestCase):
    def test_cancel_removes_partial_and_ignores_second_done(self) -> None:
        job = FakeJob(["ffmpeg"], 9.0)
        out = Path(_TMP.name) / "partial.mp4"
        out.write_bytes(b"yesterday's render")  # an older file the user kept
        part = appmod.part_path(out)
        part.write_bytes(b"\0" * 1024)  # what an interrupted ffmpeg leaves behind
        dlg = appmod.RenderProgressDialog(job, out, None)
        done_slots: list[int] = []
        orig = dlg._on_done

        def counting(code: int, msg: str) -> None:
            done_slots.append(code)
            orig(code, msg)

        dlg._bridge.finished_.disconnect()
        dlg._bridge.finished_.connect(counting)
        QTimer.singleShot(100, dlg._cancel.click)
        t0 = time.monotonic()
        result = dlg.exec()
        elapsed = time.monotonic() - t0

        self.assertEqual(result, QDialog.DialogCode.Rejected)
        self.assertTrue(dlg.was_cancelled)
        self.assertEqual(job.cancel_calls, 1)
        self.assertFalse(part.exists(), "partial output must be removed on cancel")
        self.assertEqual(out.read_bytes(), b"yesterday's render", "cancel must not touch the old file")
        pump(0.3)  # let the duplicate on_done arrive
        self.assertEqual(done_slots.count(255), 2, "engine fired twice in this fake")
        self.assertEqual(dlg.exit_code, 255)  # ... but the dialog only closed once
        self.assertLess(elapsed, 2.0, "closed on the engine's callback, not the watchdog")
        delete(dlg)

    def test_failed_render_keeps_old_file(self) -> None:
        job = FakeJob(["ffmpeg"], 9.0)
        out = Path(_TMP.name) / "kept.mp4"
        out.write_bytes(b"old")
        appmod.part_path(out).write_bytes(b"")  # ffmpeg truncated only its .part
        dlg = appmod.RenderProgressDialog(job, out, None)
        QTimer.singleShot(50, lambda: job.on_done(187, "width not divisible by 2"))
        self.assertEqual(dlg.exec(), QDialog.DialogCode.Rejected)
        self.assertEqual(out.read_bytes(), b"old")
        self.assertFalse(appmod.part_path(out).exists())
        delete(dlg)

    def test_cancel_that_loses_the_race_keeps_the_finished_render(self) -> None:
        job = FakeJob(["ffmpeg"], 9.0)
        job.done_on_cancel = False
        out = Path(_TMP.name) / "raced.mp4"
        out.unlink(missing_ok=True)
        appmod.part_path(out).write_bytes(b"finished")
        dlg = appmod.RenderProgressDialog(job, out, None)
        QTimer.singleShot(50, dlg._cancel.click)
        QTimer.singleShot(100, lambda: job.on_done(0, ""))  # ffmpeg had already exited 0
        self.assertEqual(dlg.exec(), QDialog.DialogCode.Accepted)
        self.assertEqual(out.read_bytes(), b"finished")
        delete(dlg)

    def test_watchdog_only_after_engine_grace(self) -> None:
        job = FakeJob(["ffmpeg"], 9.0)
        job.done_on_cancel = False  # engine never calls back
        dlg = appmod.RenderProgressDialog(job, Path(_TMP.name) / "wd.mp4", None)
        self.assertGreater(dlg.cancel_timeout_s, CANCEL_GRACE)
        dlg.cancel_timeout_s = 0.4
        QTimer.singleShot(50, dlg._cancel.click)
        t0 = time.monotonic()
        result = dlg.exec()
        self.assertEqual(result, QDialog.DialogCode.Rejected)
        self.assertGreaterEqual(time.monotonic() - t0, 0.4)
        self.assertEqual(dlg.message, "cancelled")
        delete(dlg)

    def test_escape_and_close_cancel_instead_of_closing(self) -> None:
        job = FakeJob(["ffmpeg"], 9.0)
        dlg = appmod.RenderProgressDialog(job, Path(_TMP.name) / "esc.mp4", None)
        QTimer.singleShot(50, dlg.reject)  # what Esc does
        result = dlg.exec()
        self.assertEqual(result, QDialog.DialogCode.Rejected)
        self.assertEqual(job.cancel_calls, 1)
        self.assertTrue(dlg.was_cancelled)
        delete(dlg)


# --------------------------------------------------------------------------- #
# main window
# --------------------------------------------------------------------------- #
class WindowCase(unittest.TestCase):
    """Shared scaffolding: fake probe, no ffmpeg, stubbed modal dialogs."""

    #: Dialogs whose inherited ``exec`` capture_render() overrides.  They must be
    #: *deleted* again, not assigned back: putting QDialog's unbound exec into a
    #: subclass __dict__ makes later real calls fail sip's receiver type check.
    PATCHED_DIALOGS = (appmod.RenderProgressDialog, appmod.RenderOptionsDialog)

    def setUp(self) -> None:
        self._orig = (
            appmod.probe, appmod.thumbnails_at, appmod.waveform_png, appmod.RenderJob,
            QFileDialog.getSaveFileName, QMessageBox.critical, QMessageBox.warning,
            QMessageBox.exec,
        )
        appmod.thumbnails_at = lambda *a, **k: []  # type: ignore[assignment]
        appmod.waveform_png = lambda *a, **k: ""  # type: ignore[assignment]
        appmod.probe = lambda p: fake_info(p)  # type: ignore[assignment]
        QMessageBox.critical = staticmethod(lambda *a, **k: None)  # type: ignore[method-assign]
        QMessageBox.warning = staticmethod(lambda *a, **k: None)  # type: ignore[method-assign]
        self.windows: list[appmod.MainWindow] = []

    def tearDown(self) -> None:
        for win in self.windows:
            win.close()
        pump(0.05)
        delete(*self.windows)  # never leave a QMediaPlayer for interpreter shutdown
        (
            appmod.probe, appmod.thumbnails_at, appmod.waveform_png, appmod.RenderJob,
            QFileDialog.getSaveFileName, QMessageBox.critical, QMessageBox.warning,
            QMessageBox.exec,
        ) = self._orig
        for cls in self.PATCHED_DIALOGS:
            try:
                del cls.exec
            except AttributeError:
                pass
        theme.apply(_app, "system")

    def window(self) -> appmod.MainWindow:
        win = appmod.MainWindow()
        self.windows.append(win)
        return win

    def load(self, win: appmod.MainWindow, path: str, timeout: float = 5.0) -> None:
        win.load_file(path)
        end = time.monotonic() + timeout
        while time.monotonic() < end and (win.info is None or win.info.path != path):
            pump(0.02)

    def loaded(self, path: str = os.path.abspath(__file__)) -> appmod.MainWindow:
        win = self.window()
        self.load(win, path)
        return win

    def capture_render(self) -> dict[str, object]:
        """Stub the render path and capture the argv that would have run."""
        built: dict[str, object] = {}

        class CapturingJob(FakeJob):
            def __init__(self, argv: list[str], total: float) -> None:
                super().__init__(argv, total)
                built["argv"] = argv
                built["total"] = total

        appmod.RenderJob = CapturingJob  # type: ignore[misc,assignment]
        appmod.RenderProgressDialog.exec = lambda self: (  # type: ignore[method-assign]
            setattr(self, "_cancelled", True) or QDialog.DialogCode.Rejected
        )
        appmod.RenderOptionsDialog.exec = lambda self: QDialog.DialogCode.Accepted  # type: ignore[method-assign]
        QFileDialog.getSaveFileName = staticmethod(  # type: ignore[method-assign]
            lambda *a, **k: (str(Path(_TMP.name) / "out.mp4"), "")
        )
        # VA-API is the default encoder: a GUI test must never read the real sysfs.
        device = mock.patch.object(
            render_mod, "default_vaapi_device", return_value="/dev/dri/renderD129"
        )
        device.start()
        self.addCleanup(device.stop)
        return built


def graph_of(built: dict[str, object]) -> str:
    argv = built["argv"]
    return argv[argv.index("-filter_complex") + 1]  # type: ignore[index,union-attr]


class TestLoading(WindowCase):
    def test_stale_probe_result_is_dropped(self) -> None:
        a, b = os.path.abspath(__file__), os.path.abspath(appmod.__file__)

        def slow_probe(path: str) -> MediaInfo:
            if path == a:
                time.sleep(0.6)
                return fake_info(a, duration=1111.0)
            return fake_info(b, duration=2222.0)

        appmod.probe = slow_probe
        win = self.window()
        t0 = time.monotonic()
        win.load_file(a)
        win.load_file(b)
        self.assertLess(time.monotonic() - t0, 0.3, "load_file must not block the GUI")
        seen: list[str] = []
        end = time.monotonic() + 1.5
        while time.monotonic() < end:
            pump(0.02)
            if win.info is not None and (not seen or seen[-1] != win.info.path):
                seen.append(win.info.path)
        self.assertEqual(seen, [b], "the slower, older probe must never be shown")
        self.assertEqual(win.timeline.model.duration, 2222.0)

    def test_failed_probe_keeps_previous_file_renderable(self) -> None:
        good = os.path.abspath(__file__)

        def probe_(path: str) -> MediaInfo:
            if path == good:
                return fake_info(good)
            raise ProbeError("no video stream")

        appmod.probe = probe_
        win = self.loaded(good)
        self.assertTrue(win.render_button.isEnabled())
        win.load_file(os.path.abspath(appmod.__file__))
        pump(0.3)
        self.assertEqual(win.info.path, good)  # type: ignore[union-attr]
        self.assertTrue(win.render_button.isEnabled())

    def test_recent_files_are_remembered(self) -> None:
        src = os.path.abspath(__file__)
        win = self.loaded(src)
        self.assertEqual(win.config.recent[0], src)
        self.assertTrue(win.recent_menu.isEnabled())
        self.assertEqual([a.text() for a in win.recent_menu.actions()], [Path(src).name])

    def test_media_load_builds_one_lane_per_audio_track(self) -> None:
        win = self.window()
        self.load(win, os.path.abspath(__file__))
        self.assertEqual(len(win.timeline.render_plan().audio), 3)
        self.assertEqual(win.timeline.lane_count(), 4, "one video lane + three audio")
        self.assertEqual(win.timeline.lane_kind(0), "video")
        self.assertEqual([win.timeline.lane_kind(n) for n in (1, 2, 3)], ["audio"] * 3)

    def test_the_audition_combo_is_gone(self) -> None:
        """The preview plays the lanes, so a second, preview-only mix control
        would only be a way to disagree with the render."""
        win = self.loaded()
        self.assertFalse(hasattr(win, "audition"))
        self.assertNotIn(
            "Hear", [w.text() for w in win.transport.findChildren(type(win.state_label))]
        )
        self.assertNotIn("auditionBox", theme.stylesheet("light"))


class TestAssets(WindowCase):
    def test_loading_a_file_asks_for_thumbnails_and_waveforms(self) -> None:
        thumbs: list[int] = []
        waves: list[int] = []
        appmod.thumbnails_at = lambda path, out, times, width, **kw: (  # type: ignore[assignment]
            thumbs.append(len(times)) or []
        )
        appmod.waveform_png = lambda path, track, *a, **k: (  # type: ignore[assignment]
            waves.append(track) or ""
        )
        self.loaded()
        pump(0.5)
        self.assertTrue(thumbs and thumbs[0] >= 6)  # none for the empty room past the end
        self.assertEqual(sorted(waves), [0, 1, 2], "one waveform per audio lane")

    def test_thumb_request_is_taken_at_face_value_and_handed_back(self) -> None:
        """The timeline asks in SOURCE time, per file — it is the only one that
        knows where each clip sits and which file it came from.  The app must
        pass those timestamps to ffmpeg untouched and hand the SAME ones back
        with the paths (never re-derived after the fact)."""
        win = self.loaded()
        grabbed: list[tuple[str, list[float]]] = []
        delivered: list[tuple[list[str], list[float], str]] = []
        appmod.thumbnails_at = lambda path, out, times, width, **kw: (  # type: ignore[assignment]
            grabbed.append((path, list(times))) or ["/nonexistent.jpg"] * len(times)
        )
        win.timeline.set_thumbnails = (  # type: ignore[method-assign]
            lambda paths, times, path=None: delivered.append((paths, times, path))
        )
        want = [25.0, 75.0, 1125.0, 1175.0]
        win.timeline.thumbsNeeded.emit(win.info.path, want)
        pump(0.5)
        self.assertIn((win.info.path, want), grabbed)
        self.assertIn(want, [t for _, t, _ in delivered])
        self.assertIn(win.info.path, [p for _, _, p in delivered])

    def test_a_second_file_gets_its_own_thumbnail_job(self) -> None:
        """Two files on the timeline are two jobs, into two directories — one
        shared directory would have each overwrite the other's frames."""
        win = self.loaded()
        pump(0.4)
        grabbed: list[tuple[str, str]] = []
        appmod.thumbnails_at = lambda path, out, times, width, **kw: (  # type: ignore[assignment]
            grabbed.append((path, out)) or []
        )
        other = str(Path(_TMP.name) / "other-file.mp4")
        win.timeline.thumbsNeeded.emit(win.info.path, [1.0])
        win.timeline.thumbsNeeded.emit(other, [2.0])
        pump(0.6)
        self.assertEqual({p for p, _ in grabbed}, {win.info.path, other})
        self.assertEqual(len({d for _, d in grabbed}), 2, "one directory for two files")

    def test_waveform_request_is_ranged_and_delivered_in_source_time(self) -> None:
        win = self.loaded()
        cut_source(win.timeline, 0.0, 600.0)
        calls: list[tuple[int, float, float]] = []
        delivered: list[tuple[int, str, float, float]] = []

        def fake_wave(path, track, out, w, h, start=0.0, end=None, color=""):  # noqa: ANN001
            calls.append((track, start, end))
            return out

        appmod.waveform_png = fake_wave  # type: ignore[assignment]
        win.timeline.set_waveform = lambda *a: delivered.append(a)  # type: ignore[method-assign]
        win.timeline.waveformNeeded.emit(1, 0.0, 100.0)
        pump(0.5)
        self.assertIn((1, 600.0, 700.0), [(t, round(s, 3), round(e, 3)) for t, s, e in calls])
        self.assertIn((1, 600.0, 700.0), [(t, s, e) for t, _, s, e in delivered])

    def test_requests_during_a_running_job_are_coalesced_not_dropped(self) -> None:
        """Rapid zoom: one job at a time, the newest request always runs last."""
        win = self.loaded()
        pump(0.4)  # let the load-time request finish
        calls: list[float] = []

        def slow(path, out, times, width, **kw):  # noqa: ANN001
            calls.append(times[0])
            time.sleep(0.25)
            return []

        appmod.thumbnails_at = slow  # type: ignore[assignment]
        path = win.info.path
        win.timeline.thumbsNeeded.emit(path, [50.0])  # starts now
        win.timeline.thumbsNeeded.emit(path, [250.0])  # queued...
        win.timeline.thumbsNeeded.emit(path, [450.0])  # ...and replaced by this one
        pump(1.0)
        self.assertEqual(calls, [50.0, 450.0])


class TestTransportAndPrefs(WindowCase):
    def test_skip_buttons_use_the_configured_interval(self) -> None:
        win = self.loaded()
        win.config.prefs.skip_seconds = 12
        win.timeline.set_playhead(100.0)
        win.transport.forward.click()
        self.assertAlmostEqual(win.timeline.playhead(), 112.0, places=3)
        win.transport.back.click()
        self.assertAlmostEqual(win.timeline.playhead(), 100.0, places=3)

    def test_scrubbing_suppresses_player_driven_playhead_writes(self) -> None:
        win = self.loaded()
        win.timeline.set_playhead(500.0)
        win._seek_guard_until = 0.0
        win.timeline._drag = "scrub"  # what a press on the playhead does
        win._on_position_changed(9000)  # a stale report from the player
        self.assertEqual(win.timeline.playhead(), 500.0, "a drag must win")
        win.timeline._drag = None
        win._on_position_changed(9000)
        self.assertAlmostEqual(win.timeline.playhead(), 9.0, places=3)

    def test_playback_inside_a_cut_hops_to_the_next_keep(self) -> None:
        win = self.loaded()
        cut_source(win.timeline, 100.0, 200.0)
        win._goto_edited(50.0)  # playing inside the first keep
        pump(0.1)
        self.assertEqual(win._run, 0)
        positions: list[int] = []
        win.player.setPosition = positions.append  # type: ignore[method-assign]
        win._on_position_changed(150_000)  # the demuxer ran on past the cut
        self.assertEqual(positions, [200_000], "the next run starts at source 200 s")
        self.assertEqual(win._run, 1)

    def test_skip_clamps_to_the_edit(self) -> None:
        win = self.loaded()
        win.timeline.set_playhead(2.0)
        win.transport.back.click()
        self.assertEqual(win.timeline.playhead(), 0.0)

    def test_cut_to_cut_jumps_land_on_splice_points(self) -> None:
        win = self.loaded()
        cut_source(win.timeline, 100.0, 200.0)
        cut_source(win.timeline, 500.0, 600.0)
        # splice points in edited time: 100 and 400
        win.timeline.set_playhead(0.0)
        win.transport.next_cut.click()
        self.assertAlmostEqual(win.timeline.playhead(), 100.0, places=3)
        win.transport.next_cut.click()
        self.assertAlmostEqual(win.timeline.playhead(), 400.0, places=3)
        win.transport.prev_cut.click()
        self.assertAlmostEqual(win.timeline.playhead(), 100.0, places=3)
        win.transport.prev_cut.click()
        self.assertEqual(win.timeline.playhead(), 0.0)

    def test_menu_marks_survive_a_click_on_the_timeline(self) -> None:
        """The reported bug, through the app's own Mark In / Mark Out actions."""
        win = self.loaded()
        win.show()
        pump(0.05)
        tl = win.timeline
        tl.set_playhead(100.0)
        win.act_mark_in.trigger()
        y = int(tl._lane_rect(0).center().y())
        QTest.mouseClick(tl, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
                         QPoint(int(tl._x_for(400.0)), y))
        self.assertAlmostEqual(tl.playhead(), 400.0, delta=3.0)
        win.act_mark_out.trigger()
        sel = tl.selection()
        assert sel is not None
        self.assertEqual(sel[0], 100.0)
        self.assertAlmostEqual(sel[1], 400.0, delta=3.0)
        self.assertTrue(win.act_render_sel.isEnabled())

    def test_transport_is_compact(self) -> None:
        """'The buttons are too big' was the complaint — keep the chrome small.

        The dome is the one thing WMP12 makes big, and the bar now carries a
        seek line above the buttons, so the budget is the *whole* bar: video
        editing happens below it.
        """
        win = self.loaded()
        win.show()
        pump(0.1)
        play = win.transport.play.size()
        skip = win.transport.back.size()
        self.assertLessEqual(play.height(), 38, f"play button {play}")
        self.assertLessEqual(play.width(), 38, f"play button {play}")
        self.assertLessEqual(skip.height(), 26, f"skip button {skip}")
        self.assertLess(skip.height(), play.height(), "skips must read as secondary")
        self.assertLessEqual(win.transport.height(), 62, "transport bar is too tall")
        self.assertLessEqual(win.transport.seek.height(), 16, "the seek line is a line")
        self.assertLessEqual(win.transport.play.iconSize().width(), 16)

    def test_transport_glyphs_are_painted_in_theme_ink(self) -> None:
        """Painted icons: WMP blue on the dome, slate on the pills, and both
        repainted when the theme changes."""
        win = self.window()

        def inks(button: object) -> set[str]:
            """Every fully opaque colour in the glyph — the anti-aliased rim is
            a blend, so sampling one pixel is not a test of the ink."""
            image = button.icon().pixmap(button.iconSize()).toImage()  # type: ignore[attr-defined]
            return {
                image.pixelColor(x, y).name()
                for y in range(image.height())
                for x in range(image.width())
                if image.pixelColor(x, y).alpha() == 255
            }

        for mode in ("dark", "light"):
            theme.apply(_app, mode)
            pump(0.05)
            tokens = theme.TOKENS[mode]
            self.assertEqual(inks(win.transport.play), {tokens["glyph"]}, f"{mode} dome")
            self.assertEqual(
                inks(win.transport.back), {tokens["glyph_skip"]}, f"{mode} pill"
            )
        self.assertNotEqual(
            theme.TOKENS["dark"]["glyph_skip"], theme.TOKENS["light"]["glyph_skip"]
        )

    def test_preferences_round_trip_and_live_theme(self) -> None:
        win = self.window()
        previewed: list[str] = []

        def choose(dialog: appmod.PreferencesDialog) -> QDialog.DialogCode:
            dialog.theme_box.setCurrentIndex(dialog.theme_box.findData("dark"))
            previewed.append(_app.styleSheet())  # the preview is live
            dialog.skip.setValue(9)
            return QDialog.DialogCode.Accepted

        appmod.PreferencesDialog.exec = choose  # type: ignore[method-assign]
        try:
            win.open_preferences()
        finally:
            del appmod.PreferencesDialog.exec
        self.assertIn("transportBar", previewed[0])
        self.assertEqual(win.config.prefs.theme, "dark")
        self.assertEqual(win.config.prefs.skip_seconds, 9)
        stored = json.loads(appmod.CONFIG_PATH.read_text())
        self.assertEqual(stored["prefs"], {"theme": "dark", "skip_seconds": 9, "preview": "full", "jobs": 4})
        reloaded = appmod.Config.load()
        self.assertEqual((reloaded.prefs.theme, reloaded.prefs.skip_seconds), ("dark", 9))
        self.assertIn("9s", win.transport.forward.toolTip())

    def test_cancelled_preferences_restore_the_previous_theme(self) -> None:
        theme.apply(_app, "light")
        light = _app.styleSheet()
        win = self.window()
        dialog = appmod.PreferencesDialog(appmod.Prefs("light", 5), win)
        dialog.theme_box.setCurrentIndex(dialog.theme_box.findData("dark"))
        self.assertNotEqual(_app.styleSheet(), light)
        dialog.reject()
        self.assertEqual(_app.styleSheet(), light)

    def test_stylesheet_is_valid_qss_in_every_mode(self) -> None:
        """Qt does not raise on bad QSS — it warns and silently drops the rule,
        so the only way to know the chrome is really applied is to listen."""
        messages: list[str] = []
        previous = qInstallMessageHandler(
            lambda kind, ctx, text: messages.append(str(text))
        )
        try:
            for mode in theme.MODES:
                sheet = theme.stylesheet(mode)
                self.assertGreater(len(sheet), 500, mode)
                for selector in (
                    "QMenuBar", "QStatusBar", "QDialog", "QPushButton",
                    "transportBar", "playerPanel", "playButton", "videoFrame",
                ):
                    self.assertIn(selector, sheet, f"{selector} missing in {mode}")
                # ... and now against a real widget tree, which is what parses it
                theme.apply(_app, mode)
                win = self.window()
                win.show()
                pump(0.05)
                delete(
                    appmod.PreferencesDialog(appmod.Prefs(), win),
                    appmod.RenderOptionsDialog(
                        Path(_TMP.name), "x.mp4", appmod.RenderOptions(), "s", win
                    ),
                )
        finally:
            qInstallMessageHandler(previous)
        # Deduped: one bad property produces a warning per polished widget.
        bad = sorted({m for m in messages if "parse" in m.lower() or "Unknown property" in m})
        self.assertEqual(bad, [], f"Qt rejected the stylesheet: {bad}")


class TestPlayerChrome(WindowCase):
    """The WMP12 seek line, the split, and the status strip."""

    def test_seek_line_seeks_in_edited_time(self) -> None:
        """Half way along the line is half way through the EDIT, not the file.

        With 1000 s cut out, a naive source-time scrubber would land 500 s off,
        which is exactly what the edited-time contract exists to prevent.
        """
        win = self.loaded()
        win.show()
        pump(0.1)
        cut_source(win.timeline, 100.0, 1100.0)  # 2780 -> 1780 s of edit
        bar = win.transport.seek
        total = win.timeline.edited_duration()
        self.assertAlmostEqual(total, 1780.0, places=3)

        x = int(bar.MARGIN + (bar.width() - 2 * bar.MARGIN) * 0.5)
        QTest.mouseClick(
            bar, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
            QPoint(x, bar.height() // 2),
        )
        wanted = total / 2.0
        self.assertAlmostEqual(win.timeline.playhead(), wanted, delta=total * 0.01)
        # the preview is seeked in SOURCE time: 890 edited is 1890 in the file
        self.assertIsNotNone(win._pending_seek_ms)
        self.assertAlmostEqual(win._pending_seek_ms / 1000.0, 1890.0, delta=20.0)

    def test_seek_line_follows_playback_but_not_a_drag(self) -> None:
        win = self.loaded()
        win.show()
        pump(0.1)
        bar = win.transport.seek
        win._on_position_changed(600_000)  # the player reports 600 s
        self.assertAlmostEqual(bar.position(), 600.0, places=1)
        self.assertAlmostEqual(bar.fraction(), 600.0 / 2780.0, places=3)

        bar._dragging = True  # what a press does
        win._on_position_changed(9_000)
        self.assertAlmostEqual(bar.position(), 600.0, places=1, msg="a drag must win")
        bar._dragging = False
        win._on_position_changed(9_000)
        self.assertAlmostEqual(bar.position(), 9.0, places=1)

    def test_seek_line_is_inert_without_media(self) -> None:
        win = self.window()
        win.show()
        pump(0.05)
        moved: list[float] = []
        win.transport.seeked.connect(moved.append)
        QTest.mouseClick(
            win.transport.seek, Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier, QPoint(60, 6),
        )
        self.assertEqual(moved, [])

    def test_splitter_drives_the_lanes_and_a_drag_is_remembered(self) -> None:
        win = self.loaded()
        win.show()
        pump(0.2)
        player, lanes = win.splitter.sizes()
        self.assertGreater(player, 0)
        self.assertEqual(lanes, win.timeline.height(), "the timeline IS the pane")
        self.assertGreaterEqual(lanes, win.timeline.minimumHeight(), "floored at the lane count")
        self.assertEqual(win.config.splitter, [], "nothing to persist until a drag")

        # Handing the timeline more room grows the lanes (and moves the
        # headers with them) instead of leaving a dead strip underneath.
        before = win.timeline._lane_rect(1).height()
        win.splitter.setSizes([player - 120, lanes + 120])
        win.splitter.splitterMoved.emit(player - 120, 1)  # what a drag emits
        pump(0.1)
        self.assertGreater(win.timeline._lane_rect(1).height(), before)
        header = win.timeline._headers[1]
        self.assertEqual(header.geometry().top(), int(win.timeline._lane_rect(2).top()))
        self.assertEqual(win.config.splitter, list(win.splitter.sizes()))
        win.config.save()
        self.assertEqual(appmod.Config.load().splitter, win.config.splitter)

    def test_a_pre_swap_nvenc_config_loads_as_vaapi(self) -> None:
        """The RTX 3070 Ti is gone; a saved h264_nvenc must not reach ffmpeg."""
        original = appmod.CONFIG_PATH
        appmod.CONFIG_PATH = Path(_TMP.name) / "nvenc-cfg.json"
        try:
            appmod.CONFIG_PATH.write_text(
                json.dumps({"render": {"encoder": "h264_nvenc", "mbps": 75}})
            )
            loaded = appmod.Config.load()
            self.assertEqual(loaded.render.encoder, "h264_vaapi")
            self.assertEqual(loaded.render.mbps, 75)
        finally:
            appmod.CONFIG_PATH = original

    def test_config_rejects_junk_splitter_values(self) -> None:
        original = appmod.CONFIG_PATH
        appmod.CONFIG_PATH = Path(_TMP.name) / "splitter-cfg.json"
        try:
            for stored, want in (
                ([500, 300], [500, 300]),
                ([500.0, 300.0], [500, 300]),
                ([500], []),
                ([0, 300], []),
                (["500", "300"], []),
                ([True, False], []),
                ("500,300", []),
            ):
                appmod.CONFIG_PATH.write_text(json.dumps({"splitter": stored}))
                self.assertEqual(appmod.Config.load().splitter, want, f"stored {stored!r}")
        finally:
            appmod.CONFIG_PATH = original

    def test_status_strip_carries_state_selection_and_summary(self) -> None:
        win = self.loaded()
        win.show()
        pump(0.05)
        bar = win.statusBar()
        for label in (win.state_label, win.selection_label, win.summary_label):
            self.assertIs(label.parent(), bar, "the readouts live in the strip now")
            self.assertTrue(label.isVisible())
        win._on_playback_state(QMediaPlayer.PlaybackState.PlayingState)
        self.assertIn("Playing", win.state_label.text())
        self.assertTrue(win.transport._playing, "the dome shows pause while playing")
        win._on_playback_state(QMediaPlayer.PlaybackState.PausedState)
        self.assertIn("Paused", win.state_label.text())
        self.assertFalse(win.transport._playing)
        select(win.timeline, 10.0, 20.0)
        self.assertIn("00:10", win.selection_label.text())
        self.assertNotIn("lift", win.selection_label.text(), "all armed is the quiet default")
        win.timeline.set_armed_lanes([0, 2])  # armedChanged -> the strip says what an edit hits
        self.assertIn("lift: video, audio 2", win.selection_label.text())
        win.timeline.set_armed_lanes([])  # back to everything
        self.assertNotIn("lift", win.selection_label.text())
        self.assertIn("Keeping", win.summary_label.text())


class TestPreviewAudio(WindowCase):
    """The preview mix must equal the render mix.

    The old preview decoded ONE chosen track and nothing else, so a dropped
    file was inaudible by construction and the lanes' M buttons only reached
    the renderer.  Every test here is one half of that report.
    """

    BGM = str(Path(_TMP.name) / "bgm.mp3")
    BGM_SECONDS = 150.0

    def setUp(self) -> None:
        super().setUp()
        Path(self.BGM).write_bytes(b"stub")
        self._probe_audio = appmod.probe_audio
        appmod.probe_audio = lambda p: AudioFileInfo(  # type: ignore[assignment]
            p, self.BGM_SECONDS, "mp3", 2, 44100
        )

    def tearDown(self) -> None:
        appmod.probe_audio = self._probe_audio  # type: ignore[assignment]
        super().tearDown()

    def drop_bgm(
        self, win: appmod.MainWindow, duration: float | None = None
    ) -> appmod._AuxPlayer:
        """Drop the music bed and hand back the player it grew."""
        if duration is not None:
            self.BGM_SECONDS = duration
        lanes = len(win.timeline._headers)
        win.add_audio_file(self.BGM)
        end = time.monotonic() + 3.0
        while time.monotonic() < end and len(win.timeline._headers) <= lanes:
            pump(0.02)
        return self.aux(win, self.BGM, 0)

    @staticmethod
    def aux(win: appmod.MainWindow, path: str, stream: int) -> appmod._AuxPlayer:
        """The extra player for ``path``/``stream`` (keys also carry the lane)."""
        return next(a for k, a in win._aux.items() if k[:2] == (path, stream))

    # -- the common case stays cheap -----------------------------------------
    def test_one_audio_track_spins_up_no_extra_players(self) -> None:
        """Every extra QMediaPlayer is another shot at the stop() wedge."""
        appmod.probe = lambda p: fake_info(p, n_audio=1)  # type: ignore[assignment]
        win = self.loaded()
        win._sync_preview_audio()
        self.assertEqual(win._aux, {}, "an ordinary file needs exactly one player")
        self.assertIsNotNone(win._primary)
        self.assertEqual(win._primary.stream, 0)  # type: ignore[union-attr]
        self.assertFalse(win.audio_out.isMuted())

    def test_extra_embedded_tracks_get_their_own_players(self) -> None:
        """QMediaPlayer decodes one track; three tracks render as three."""
        win = self.loaded()  # 3 embedded tracks
        win._sync_preview_audio()
        self.assertEqual(win._primary.stream, 0)  # type: ignore[union-attr]
        self.assertEqual(sorted(key[1] for key in win._aux), [1, 2])
        self.assertTrue(all(k[0] == win.info.path for k in win._aux))

    # -- the reported bug ----------------------------------------------------
    def test_a_dropped_file_becomes_audible(self) -> None:
        """'I drag the mp3 in, it shows up, but I cannot hear it.'"""
        win = self.loaded()
        aux = self.drop_bgm(win)
        self.assertEqual(aux.player.source().toLocalFile(), self.BGM)
        self.assertFalse(aux.output.isMuted())
        self.assertAlmostEqual(aux.output.volume(), 0.8, places=3)

        # ... and it is positioned with the picture, not left at zero
        asked: list[int] = []
        aux.player.setPosition = asked.append  # type: ignore[method-assign]
        win._goto_edited(100.0)
        pump(0.1)
        self.assertIn(100_000, asked, "the dropped file must follow the seek")

    def test_a_dropped_file_that_runs_out_is_parked(self) -> None:
        """A 150 s bed under a 2780 s video: silence, not free-running audio."""
        win = self.loaded()
        aux = self.drop_bgm(win)
        self.assertAlmostEqual(aux.source.source_at(100.0), 100.0, places=3)
        self.assertIsNone(aux.source.source_at(200.0))
        paused: list[int] = []
        aux.player.pause = lambda: paused.append(1)  # type: ignore[method-assign]
        aux.player.playbackState = (  # type: ignore[method-assign]
            lambda: QMediaPlayer.PlaybackState.PlayingState
        )
        aux.follow(200.0, True, force=True)
        self.assertEqual(paused, [1])

    def test_muting_a_lane_silences_the_preview(self) -> None:
        """'When I mute the audio track, it still plays when I press play.'"""
        win = self.loaded()
        aux = self.drop_bgm(win)
        win.timeline._headers[3].mute.setChecked(True)  # the dropped lane
        pump(0.05)
        self.assertTrue(aux.output.isMuted())
        self.assertEqual(aux.output.volume(), 0.0)

        # Muting the track the MAIN player carries mutes its output; no track
        # moves between players, so M never rebuilds a QMediaPlayer.
        players = dict(win._aux)
        win.timeline._headers[0].mute.setChecked(True)
        pump(0.05)
        self.assertEqual(win._primary.stream, 0)  # type: ignore[union-attr]
        self.assertTrue(win.audio_out.isMuted())
        win.timeline._headers[0].mute.setChecked(False)
        pump(0.05)
        self.assertEqual(win._aux, players, "M must not close or rebuild an aux player")
        self.assertFalse(win.audio_out.isMuted())

        # Every embedded track muted: the main player goes quiet too, which the
        # audition-based preview could never do.
        for header in win.timeline._headers[:3]:
            header.mute.setChecked(True)
        pump(0.05)
        self.assertTrue(win.audio_out.isMuted())
        self.assertEqual(win.audio_out.volume(), 0.0)

    def test_lane_gain_changes_the_preview_volume(self) -> None:
        win = self.loaded()
        win._sync_preview_audio()
        aux = self.aux(win, win.info.path, 1)
        win.timeline._headers[0].slider.setValue(-6)
        win.timeline._headers[1].slider.setValue(-12)
        pump(0.05)
        self.assertAlmostEqual(win.audio_out.volume(), 0.8 * 10 ** (-6 / 20), places=4)
        self.assertAlmostEqual(aux.output.volume(), 0.8 * 10 ** (-12 / 20), places=4)

        # The slider is the master over that mix, not a replacement for it.
        win.volume.setValue(50)
        pump(0.05)
        self.assertAlmostEqual(win.audio_out.volume(), 0.5 * 10 ** (-6 / 20), places=4)
        self.assertAlmostEqual(aux.output.volume(), 0.5 * 10 ** (-12 / 20), places=4)

    def test_a_boost_keeps_the_balance_under_qts_unity_cap(self) -> None:
        """Qt cannot play louder than 1.0.  A +6 dB lane must still sit 6 dB
        above its neighbours in the preview — the mix is scaled down as a
        whole, never one lane clamped (which is preview != render)."""
        win = self.loaded()
        win._sync_preview_audio()
        aux = self.aux(win, win.info.path, 1)
        win.timeline._headers[0].slider.setValue(6)
        pump(0.05)
        boost = 10 ** (6 / 20)
        self.assertAlmostEqual(win.audio_out.volume(), 0.8, places=4, msg="the loudest lane is at master")
        self.assertAlmostEqual(aux.output.volume(), 0.8 / boost, places=4)
        self.assertAlmostEqual(win.audio_out.volume() / aux.output.volume(), boost, places=4)
        # ... and a muted lane does not count as the peak: with the boosted
        # lane silenced, lane 1 is back at full master.
        win.timeline._headers[0].mute.setChecked(True)
        pump(0.05)
        self.assertAlmostEqual(aux.output.volume(), 0.8, places=4)

    def test_two_lanes_on_one_file_are_two_players(self) -> None:
        """Each lane has its own gain and mute, so each needs its own player."""
        win = self.loaded()
        first = self.drop_bgm(win)
        second_lane = win.timeline.add_audio_lane()
        win._on_clip_probed(
            win._media_gen, AudioFileInfo(self.BGM, 30.0, "mp3", 2, 44100), second_lane, "place"
        )
        pump(0.05)
        players = [a for k, a in win._aux.items() if k[:2] == (self.BGM, 0)]
        self.assertEqual(len(players), 2)
        win.timeline._headers[-1].mute.setChecked(True)
        pump(0.05)
        self.assertFalse(first.output.isMuted(), "the other lane keeps playing")
        self.assertTrue(next(a for a in players if a is not first).output.isMuted())

    # -- staying in sync with the picture ------------------------------------
    def test_a_cut_remaps_every_extra_track_the_way_the_render_does(self) -> None:
        """Preview and render agree: edited time maps to the same SOURCE time
        on every lane, so a cut cannot slide the extra tracks out of step."""
        win = self.loaded()
        win._sync_preview_audio()
        cut_source(win.timeline, 100.0, 1100.0)  # 1000 s removed
        pump(0.05)
        aux = self.aux(win, win.info.path, 1)
        self.assertAlmostEqual(aux.source.source_at(50.0), 50.0, places=3)
        self.assertAlmostEqual(aux.source.source_at(150.0), 1150.0, places=3)

    def test_playback_hopping_a_cut_takes_the_extra_tracks_with_it(self) -> None:
        win = self.loaded()
        win._sync_preview_audio()
        cut_source(win.timeline, 100.0, 200.0)
        win._goto_edited(50.0)  # playing inside the first keep
        pump(0.1)
        aux = self.aux(win, win.info.path, 2)
        asked: list[int] = []
        aux.player.setPosition = asked.append  # type: ignore[method-assign]
        win.player.setPosition = lambda _ms: None  # type: ignore[method-assign]
        win._on_position_changed(150_000)  # the demuxer ran on past the cut
        self.assertEqual(asked, [200_000], "the extra track must jump the cut too")

    def test_drift_is_corrected_but_a_tick_is_not_a_seek(self) -> None:
        """Re-seeking every position report would stutter the audio."""
        win = self.loaded()
        win._sync_preview_audio()
        aux = self.aux(win, win.info.path, 1)
        asked: list[int] = []
        aux.player.setPosition = asked.append  # type: ignore[method-assign]
        aux.player.position = lambda: 60_040  # type: ignore[method-assign]
        win._on_position_changed(60_000)  # 40 ms out: inside the tolerance
        self.assertEqual(asked, [])
        aux.player.position = lambda: 61_000  # type: ignore[method-assign]
        win._on_position_changed(60_000)  # 1 s out: really drifting
        self.assertEqual(asked, [60_000])

    # -- teardown ------------------------------------------------------------
    def test_close_tears_the_extra_players_down_too(self) -> None:
        win = self.loaded()
        aux = self.drop_bgm(win)
        self.assertGreaterEqual(len(win._aux), 3)  # 2 embedded + the mp3
        t0 = time.monotonic()
        win.close()
        blocked = time.monotonic() - t0
        self.assertEqual(win._aux, {}, "an aux player outlived the window")
        self.assertTrue(aux._closing)
        self.assertTrue(aux.player.source().isEmpty())
        self.assertLess(blocked, 2.0, f"close() blocked {blocked:.1f}s")

    def test_an_aux_player_is_not_re_entered_while_closing(self) -> None:
        """Same guard as _on_media_status, one per player: stop() delivers
        mediaStatusChanged synchronously, and re-priming from there is the
        teardown deadlock."""
        win = self.loaded()
        aux = self.drop_bgm(win)
        seen: list[str] = []
        aux._apply_track = lambda: seen.append("track")  # type: ignore[method-assign]
        aux._on_status(QMediaPlayer.MediaStatus.LoadedMedia)
        self.assertEqual(seen, ["track"], "it must still work while open")
        seen.clear()
        aux._closing = True
        aux._on_status(QMediaPlayer.MediaStatus.LoadedMedia)
        self.assertEqual(seen, [], "an aux player was re-entered during teardown")
        # ... and nothing moves it either
        aux.follow(10.0, True, force=True)
        aux.set_playing(True)

    def test_a_video_clip_from_another_file_takes_over_the_preview(self) -> None:
        """Resume item A, the one the last round held back.

        A block from a second file rendered correctly but PREVIEWED as the main
        track's picture.  The preview is driven from the run list now, so the
        main player swaps to that file wherever it is the visible one and swaps
        back where it is not -- and while it may have to swap, it stops
        carrying a lane's audio, because its audio would swap with it.
        """
        win = self.loaded()
        under = str(Path(_TMP.name) / "second-lane.mp4")
        Path(under).write_bytes(b"stub")
        main = win.info.path
        lane = win.timeline.add_video_lane()
        win._on_clip_probed(win._media_gen, fake_info(under, duration=600.0), lane, "place")
        pump(0.1)
        # Lane 0 covers the whole timeline, so lane 1 is invisible: no swap.
        self.assertEqual(win._video_path, main, "swapped to a hidden lane")
        self.assertTrue(win._carry_audio)

        # Lift 100..200 out of lane 0 alone: lane 1 shows through the hole.
        win.timeline.set_armed_lanes([0])
        win.timeline._set_marks(100.0, 200.0)
        self.assertTrue(win.timeline.cut_selection())
        pump(0.1)
        self.assertIn(under, [run.path for run in win._runs])
        self.assertFalse(win._carry_audio, "it must go video-only to swap")
        self.assertIsNone(win._primary)
        self.assertTrue(win._aux, "every lane needs its own player now")

        win._goto_edited(150.0)  # inside the hole
        pump(0.2)
        self.assertEqual(win._video_path, under, "the second file never came up")
        self.assertEqual(win.player.source().toLocalFile(), under)
        win._goto_edited(50.0)  # back out of it
        pump(0.2)
        self.assertEqual(win._video_path, main, "it never came back")

    def test_a_swap_does_not_spend_its_seek_on_the_file_it_is_leaving(self) -> None:
        """Measured on real media: ``setSource`` walks the OUTGOING media down
        and delivers ITS ``LoadedMedia`` synchronously, before the new file has
        begun loading.  Acting on that consumed the parked seek and the priming
        pause, so every swap away from a buffered file started the new clip at
        0:00 with a blank stage."""
        win = self.loaded()
        other = str(Path(_TMP.name) / "swap-target.mp4")
        Path(other).write_bytes(b"stub")
        win._in_swap = True
        win._video_path = other  # what we asked for...
        win._load_seek_ms = 14_000
        # ...while the player is still carrying the file we are leaving.
        self.assertNotEqual(win.player.source().toLocalFile(), other)
        win._on_media_status(QMediaPlayer.MediaStatus.LoadedMedia)
        self.assertTrue(win._in_swap, "the swap latch was cleared by the old file")
        self.assertEqual(win._load_seek_ms, 14_000, "the seek was spent on the old file")

        # The status that really belongs to the new source does the work.
        win.player.setSource(QUrl.fromLocalFile(other))
        win._on_media_status(QMediaPlayer.MediaStatus.LoadedMedia)
        self.assertFalse(win._in_swap)
        self.assertIsNone(win._load_seek_ms)

    def test_the_timeline_keeps_running_after_the_picture_stops(self) -> None:
        """An audio bed outliving the footage is a BLACK GAP: no run covers it,
        so no player can time it and the gap clock has to."""
        win = self.loaded()
        self.drop_bgm(win, duration=4000.0)  # longer than the 2780 s video
        pump(0.1)
        self.assertGreater(win.timeline.edited_duration(), 2780.0)
        self.assertEqual(win._run_at(3000.0), -1, "a run covers the gap")
        win._goto_edited(3000.0)
        pump(0.1)
        self.assertEqual(win._run, -1)
        self.assertFalse(win.video.isVisible(), "the picture must go black")
        win._set_playing(True)
        self.assertTrue(win._gap_timer.isActive(), "nothing is driving the time")
        before = win.timeline.playhead()
        pump(0.5)
        moved = win.timeline.playhead() - before
        self.assertGreater(moved, 0.2, f"the clock did not advance ({moved:.2f}s)")
        self.assertLess(moved, 1.2, f"the clock ran away ({moved:.2f}s)")
        win._set_playing(False)
        self.assertFalse(win._gap_timer.isActive())

    def test_loading_another_file_does_not_leak_its_players(self) -> None:
        win = self.loaded()
        self.drop_bgm(win)
        first = dict(win._aux)
        self.load(win, os.path.abspath(appmod.__file__))
        pump(0.2)
        self.assertFalse(
            [key for key in win._aux if key in first and key[0] == self.BGM],
            "the previous file's dropped lane is gone with it",
        )


class TestDropTargets(WindowCase):
    """Drops must land from anywhere in the window, preview included."""

    OTHER = str(Path(_TMP.name) / "dropped.mp4")

    def setUp(self) -> None:
        super().setUp()
        Path(self.OTHER).write_bytes(b"stub")

    def test_the_video_stage_is_what_was_eating_the_drop(self) -> None:
        """The diagnosis, pinned down so it cannot quietly change.

        ``QVideoWidget`` itself does not accept drops — its internal render
        surface does, and then ignores them.  Qt's ``findDnDTarget`` walks UP
        from the widget under the cursor to the first one that accepts drops,
        finds that child, and stops: the drag never reaches MainWindow.
        """
        bare = QVideoWidget()
        try:
            self.assertFalse(bare.acceptDrops())
            self.assertTrue(
                [c for c in bare.children() if getattr(c, "acceptDrops", bool)()],
                "no drop-accepting child — re-check the fix if this ever fails",
            )
        finally:
            delete(bare)

    def test_every_widget_on_the_stage_is_covered_by_the_forwarder(self) -> None:
        win = self.loaded()
        win.show()
        pump(0.1)
        for widget in [win.video_frame, win.video, *win.video.findChildren(QWidget)]:
            self.assertTrue(widget.acceptDrops(), f"{widget} would swallow a drop")
        # ... including a surface the video widget only builds later
        late = QWidget()
        late.setParent(win.video)
        self.assertTrue(late.acceptDrops(), "a child created later is a hole")

    def test_a_drop_on_the_black_preview_area_lands(self) -> None:
        """Dropping on the *preview* still means "replace the media" — the
        timeline is where clips get placed.

        Driven through the forwarder rather than ``QApplication.sendEvent``:
        Qt refuses to deliver a synthetic drop with no drag in flight (it does
        not even reach the event filters), so the real path cannot be simulated
        that way.  This is the filter's own seam, with the video widget as the
        source, which is what the mapping below has to get right.
        """
        win = self.loaded()
        win.show()
        pump(0.1)
        loaded: list[str] = []
        win.load_file = loaded.append  # type: ignore[method-assign]
        event = drop_at(QPointF(6.0, 6.0), self.OTHER)
        self.assertTrue(win._video_drops.eventFilter(win.video, event))
        self.assertTrue(event.isAccepted())
        self.assertEqual(loaded, [self.OTHER])

    def test_the_preview_area_accepts_the_drag_before_the_drop(self) -> None:
        """Without this the cursor says "no" over the whole stage."""
        win = self.loaded()
        win.show()
        pump(0.1)
        event = QDragEnterEvent(
            QPoint(6, 6), Qt.DropAction.CopyAction, mime(self.OTHER),
            Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
        )
        self.assertTrue(win._video_drops.eventFilter(win.video, event))
        self.assertTrue(event.isAccepted())

    def test_the_preview_area_refuses_what_the_window_refuses(self) -> None:
        win = self.loaded()
        win.show()
        pump(0.1)
        junk = str(Path(_TMP.name) / "notes.txt")
        Path(junk).write_bytes(b"stub")
        event = drop_at(QPointF(6.0, 6.0), junk)
        self.assertFalse(win._video_drops.eventFilter(win.video, event))
        self.assertFalse(event.isAccepted())

    def test_a_drop_over_a_lane_goes_through_the_timelines_own_placement(self) -> None:
        """The window never decides where a clip sits — it asks the timeline,
        maps the pointer into its coordinates and hands the answer back."""
        win = self.loaded()
        win.show()
        pump(0.2)
        bgm = str(Path(_TMP.name) / "placed.mp3")
        Path(bgm).write_bytes(b"stub")
        appmod.probe_audio = lambda p: AudioFileInfo(p, 30.0, "mp3", 2, 44100)  # type: ignore[assignment]

        asked: list[object] = []
        placed: list[tuple[int, str, str]] = []
        win.timeline.drop_target = lambda pos: (  # type: ignore[method-assign]
            asked.append((pos.x(), pos.y())) or ("append", 2)
        )
        # add_clip returns the Clip it placed; the window reads its start to
        # report where the drop actually LANDED.
        win.timeline.add_clip = lambda lane, info, mode: (  # type: ignore[method-assign]
            placed.append((lane, info.path, mode))
            or Clip(info.path, "audio", 12.0, 0.0, info.duration)
        )
        try:
            local = QPoint(40, 30)
            event = drop_at(QPointF(win.timeline.mapTo(win, local)), bgm)
            win.dropEvent(event)
            pump(0.5)
        finally:
            del win.timeline.drop_target, win.timeline.add_clip
        self.assertEqual(asked, [(40, 30)], "the pointer must arrive in lane coords")
        self.assertEqual(placed, [(2, bgm, "append")])

    def test_a_drop_with_no_lane_under_it_adds_its_own_lane(self) -> None:
        """A drop that is not aimed at a lane keeps the old meaning: audio adds
        a lane of its own, video replaces the media."""
        win = self.loaded()
        win.show()
        pump(0.1)
        added: list[str] = []
        win.add_audio_file = added.append  # type: ignore[method-assign]
        bgm = str(Path(_TMP.name) / "fallback.mp3")
        Path(bgm).write_bytes(b"stub")
        win.timeline.drop_target = lambda pos: None  # type: ignore[method-assign]
        try:
            win.dropEvent(drop_at(QPointF(20.0, 20.0), bgm))
        finally:
            del win.timeline.drop_target
        self.assertEqual(added, [bgm])


class TestLaneMenu(WindowCase):
    """Right-click a lane header: add and remove tracks."""

    def labels(self, menu: object) -> list[tuple[str, bool]]:
        return [
            (a.text(), a.isEnabled())
            for a in menu.actions()  # type: ignore[attr-defined]
            if not a.isSeparator()
        ]

    def test_the_menu_adds_and_removes_real_tracks(self) -> None:
        win = self.loaded()  # video lane + 3 audio lanes
        before = win.timeline.lane_count()
        menu = win.build_lane_menu(2)
        self.assertEqual(
            self.labels(menu),
            [
                ("Add video track", True),
                ("Add audio track", True),
                ("Add file to this track…", True),
                ("Remove this track", True),
            ],
        )
        next(a for a in menu.actions() if a.text() == "Add audio track").trigger()
        self.assertEqual(win.timeline.lane_count(), before + 1)
        self.assertEqual(win.timeline.lane_kind(win.timeline.lane_count() - 1), "audio")

        next(a for a in menu.actions() if a.text() == "Add video track").trigger()
        self.assertEqual(win.timeline.lane_count(), before + 2)

        menu = win.build_lane_menu(2)
        next(a for a in menu.actions() if a.text() == "Remove this track").trigger()
        self.assertEqual(win.timeline.lane_count(), before + 1)

    def test_a_new_empty_track_is_a_drop_target(self) -> None:
        """That is the whole point of adding one."""
        win = self.loaded()
        win.show()
        pump(0.1)
        lane = win.timeline.add_audio_lane()
        pump(0.1)
        centre = win.timeline._lane_rect(lane).center()
        self.assertEqual(win.timeline.drop_target(centre), ("place", lane))
        # ... while a lane that already has content appends to the end
        self.assertEqual(
            win.timeline.drop_target(win.timeline._lane_rect(1).center())[0], "append"
        )

    def test_the_menu_without_a_lane_under_it_still_adds_tracks(self) -> None:
        win = self.loaded()
        self.assertEqual(
            self.labels(win.build_lane_menu(None)),
            [
                ("Add video track", True),
                ("Add audio track", True),
                ("Add file to this track…", False),
                ("Remove this track", False),
            ],
        )

    def test_the_video_lane_cannot_be_removed(self) -> None:
        """Lane 0 holds the loaded capture and the output geometry."""
        win = self.loaded()
        menu = win.build_lane_menu(0)
        remove = next(a for a in menu.actions() if a.text() == "Remove this track")
        self.assertFalse(remove.isEnabled())
        self.assertFalse(win.timeline.remove_lane(0), "the timeline refuses it too")

    def test_the_file_picker_is_filtered_by_what_the_lane_holds(self) -> None:
        win = self.loaded()
        seen: list[str] = []
        original = QFileDialog.getOpenFileName
        QFileDialog.getOpenFileName = staticmethod(  # type: ignore[method-assign]
            lambda *a, **k: (seen.append(a[3]) or ("", ""))
        )
        try:
            win._pick_clip_for(2)  # an audio lane
            win._pick_clip_for(0)  # the video lane
        finally:
            QFileDialog.getOpenFileName = original  # type: ignore[method-assign]
        self.assertIn("*.mp3", seen[0])
        self.assertNotIn("*.mp4", seen[0])
        self.assertIn("*.mp4", seen[1])

    def test_a_right_click_on_a_lane_header_reaches_the_window(self) -> None:
        """The headers are children of the timeline, so their context-menu
        events never reach it — that is what the forwarder is for."""
        win = self.loaded()
        win.show()
        pump(0.1)
        seen: list[object] = []
        win.show_lane_menu = seen.append  # type: ignore[method-assign]
        header = win.timeline._headers[1]
        spot = QPoint(header.width() // 2, header.height() // 2)
        event = QContextMenuEvent(
            QContextMenuEvent.Reason.Mouse, spot, header.mapToGlobal(spot)
        )
        QApplication.sendEvent(header, event)
        self.assertEqual(len(seen), 1, "the right-click never reached the window")
        self.assertTrue(event.isAccepted())


class TestBlockMenu(WindowCase):
    """Right-click a BLOCK: the selection's verbs, with the track verbs under
    them.  The window owns the words; the timeline owns the edit."""

    def labels(self, menu: object) -> list[tuple[str, bool]]:
        return [
            (a.text(), a.isEnabled())
            for a in menu.actions()  # type: ignore[attr-defined]
            if not a.isSeparator()
        ]

    def block_point(self, win: appmod.MainWindow, t: float = 100.0) -> QPoint:
        rect = win.timeline._lane_rect(0)
        return QPoint(int(win.timeline._x_for(t)), int(rect.center().y()))

    def test_right_click_on_a_block_selects_it_and_offers_its_verbs(self) -> None:
        win = self.loaded()
        win.show()
        pump(0.2)
        shown: list[object] = []
        appmod.QMenu.exec = lambda self, *a: shown.append(self)  # type: ignore[method-assign]
        try:
            win.show_lane_menu(win.timeline.mapToGlobal(self.block_point(win)))
        finally:
            del appmod.QMenu.exec
        self.assertTrue(shown, "no menu was shown")
        self.assertTrue(win.timeline.clip_selection(), "the block was not selected")
        texts = [t for t, _ in self.labels(shown[0])]
        for wanted in ("Deselect", "Group", "Delete", "Copy"):
            self.assertIn(wanted, texts)
        self.assertIn("Track", texts, "the track verbs are still one click away")

    def test_a_double_click_on_a_block_reaches_the_window(self) -> None:
        """The seam for the "double-click = deselect and select this" request.
        The widget picks the block; the window's block verbs and its readout
        have to follow, off the same signal an Alt+click raises."""
        win = self.loaded()
        position = QPointF(self.block_point(win, 100.0))
        win.timeline.mouseDoubleClickEvent(
            QMouseEvent(
                QEvent.Type.MouseButtonDblClick, position, position,
                Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton,
                Qt.KeyboardModifier.NoModifier,
            )
        )
        self.assertTrue(win.timeline.clip_selection(), "no block was selected")
        state = dict(self.labels(win.build_block_menu(0)))
        self.assertTrue(state["Delete"] and state["Copy"] and state["Deselect"])
        self.assertIn("blocks selected", win.selection_label.text())
        # and the marks came with it: the capture is one block, so the whole edit
        self.assertEqual(win.timeline.selection(), (0.0, 2780.0))

    def test_the_verbs_are_enabled_by_what_is_actually_selected(self) -> None:
        win = self.loaded()
        menu = win.build_block_menu(0)
        self.assertEqual(dict(self.labels(menu))["Copy"], False)
        self.assertEqual(dict(self.labels(menu))["Paste at playhead"], False)

        win.timeline.select_block(win.timeline.lane_clips(0)[0])
        menu = win.build_block_menu(0)
        state = dict(self.labels(menu))
        self.assertTrue(state["Copy"] and state["Delete"] and state["Deselect"])
        self.assertTrue(state["Ungroup"], "the capture arrives grouped")

    def test_copy_then_paste_through_the_menu(self) -> None:
        win = self.loaded()
        win.timeline.select_block(win.timeline.lane_clips(0)[0])
        win.copy_blocks()
        self.assertTrue(win.act_paste_blocks.isEnabled())
        win.timeline.set_playhead(win.timeline.edited_duration())
        win.paste_blocks()
        self.assertEqual(len(win.timeline.lane_clips(0)), 2)

    def test_delete_through_the_menu_leaves_a_gap(self) -> None:
        win = self.loaded()
        win.timeline.set_playhead(1000.0)
        win.split_blocks()
        win.timeline.clear_block_selection()
        clip = win.timeline.lane_clips(0)[1]
        win.timeline.select_block(clip)
        win.timeline.ungroup_blocks()
        win.timeline.select_block(clip)
        win.delete_blocks()
        self.assertEqual([c.start for c in win.timeline.lane_clips(0)], [0.0])
        self.assertAlmostEqual(win.timeline.edited_duration(), 2780.0, places=3)

    def test_the_edit_menu_carries_the_same_verbs(self) -> None:
        win = self.loaded()
        self.assertTrue(win.act_split.isEnabled())
        self.assertFalse(win.act_group.isEnabled())
        win.timeline.select_block(win.timeline.lane_clips(0)[0])
        self.assertTrue(win.act_group.isEnabled())
        self.assertTrue(win.act_delete_blocks.isEnabled())

    def test_a_moved_block_re_plans_the_preview(self) -> None:
        """The runs are the preview; a block that moved and did not move them
        would play the old arrangement."""
        win = self.loaded()
        win.timeline.set_playhead(1000.0)
        win.split_blocks()
        pump(0.1)
        self.assertEqual(len(win._runs), 2)
        clip = win.timeline.lane_clips(0)[1]
        win.timeline.select_block(clip)
        win.timeline.ungroup_blocks()
        win.timeline.select_block(clip)
        tl = win.timeline
        tl._block_anchor = 1000.0
        tl._block_from = {c.cid: c.start for _, c in tl.selected_blocks()}
        tl._block_limits = tl._move_limits()
        tl.move_blocks(500.0)
        tl._after_lane_change(tl.all_armed())
        pump(0.1)
        self.assertAlmostEqual(win._runs[-1].edited_start, 1500.0, delta=1.0)
        self.assertEqual(win._run_at(1200.0), -1, "the gap is not a gap in the preview")

    @staticmethod
    def _decodes(win: appmod.MainWindow) -> None:
        """Pretend the fake media loads.

        The suite's "video" is a text file, so the real player sits in an error
        state and ``_resume`` rightly steps over every run it is asked to play.
        A test about the end-of-file signals is not a test about that.
        """
        win.player.error = lambda: QMediaPlayer.Error.NoError  # type: ignore[method-assign]

    def _two_runs(self, win: appmod.MainWindow) -> None:
        """Split the capture so the preview has two runs, parked on the first."""
        self._decodes(win)
        win.timeline.set_playhead(1000.0)
        win.split_blocks()
        win.timeline.clear_block_selection()
        win._goto_edited(10.0)
        pump(0.1)
        self.assertEqual(len(win._runs), 2)
        self.assertEqual(win._run, 0)
        win._playing = True

    @staticmethod
    def _played(win: appmod.MainWindow, seconds: float) -> None:
        """Pretend the player has decoded ``seconds`` past where it was seeked."""
        at = win._run_entered_ms + int(seconds * 1000)
        win.player.position = lambda: at  # type: ignore[method-assign]

    def test_end_of_file_advances_the_run_exactly_once(self) -> None:
        """Qt emits StoppedState AND EndOfMedia at the end of a file.  Both used
        to advance, so the second skipped the run the first had just entered —
        measured on real media as three blocks playing 0, 2, never 1, and in a
        gap as playback ending while a music bed still had seconds to go."""
        win = self.loaded()
        self._two_runs(win)
        self._played(win, 900.0)

        # Whichever of the pair lands first does the work; the other arrives
        # with the player sitting exactly where that advance put it.
        win._on_playback_state(QMediaPlayer.PlaybackState.StoppedState)
        self.assertEqual(win._run, 1, "Stopped did not advance")
        win._on_media_status(QMediaPlayer.MediaStatus.EndOfMedia)
        self.assertEqual(win._run, 1, "the second signal skipped a run")

        # ... and the other order.
        win._goto_edited(10.0)
        pump(0.1)
        self.assertEqual(win._run, 0)
        win._playing = True
        self._played(win, 900.0)
        win._on_media_status(QMediaPlayer.MediaStatus.EndOfMedia)
        self.assertEqual(win._run, 1, "EndOfMedia did not advance")
        win._on_playback_state(QMediaPlayer.PlaybackState.StoppedState)
        self.assertEqual(win._run, 1, "the second signal skipped a run")

    def test_stopping_without_an_end_of_media_still_advances(self) -> None:
        """Deferring to EndOfMedia alone would hang the transport on "Playing"
        forever if a backend ever stopped without one."""
        win = self.loaded()
        self._two_runs(win)
        self._played(win, 900.0)
        win._on_playback_state(QMediaPlayer.PlaybackState.StoppedState)
        self.assertEqual(win._run, 1, "Stopped alone left the transport hanging")

    def test_an_end_of_file_just_after_another_advance_is_not_swallowed(self) -> None:
        """A wall-clock debounce was tried first and was WRONG: it also swallowed
        a genuine end-of-file landing a few ms after a position-driven advance,
        freezing the transport — worse than the skip it fixed.  What decides is
        whether the run has actually PLAYED, not how long ago the last advance
        was."""
        win = self.loaded()
        self._decodes(win)
        win.timeline.set_playhead(500.0)
        win.split_blocks()
        win.timeline.clear_block_selection()
        win.timeline.set_playhead(1000.0)
        win.split_blocks()
        win.timeline.clear_block_selection()
        win._goto_edited(10.0)
        pump(0.1)
        self.assertEqual(len(win._runs), 3)
        win._playing = True

        self._played(win, 400.0)
        win._advance_run()  # a position report crossed a boundary a moment ago
        self.assertEqual(win._run, 1)
        self._played(win, 300.0)  # ... and THIS run has now played too
        win._on_media_status(QMediaPlayer.MediaStatus.EndOfMedia)
        self.assertEqual(win._run, 2, "a real end-of-file was swallowed")

    def test_an_end_of_file_on_a_run_entered_near_its_own_end_is_believed(self) -> None:
        """Progress alone was not enough either: a run ENTERED within the window
        of where its file really ends had its first, genuine end-of-file
        swallowed — 5/5 hangs on real media with an optimistic container
        duration.  What also has to be true is that an end-of-file advance is
        what entered this run."""
        win = self.loaded()
        self._decodes(win)
        win.timeline.set_playhead(1000.0)
        win.split_blocks()
        win.timeline.clear_block_selection()
        # Seek to a hair before the end of run 0, then play: the player reports
        # nothing before the file ends.
        win._goto_edited(999.99)
        pump(0.1)
        self.assertEqual(win._run, 0)
        self.assertFalse(win._run_from_eof, "a seek is not an end-of-file advance")
        win._playing = True
        win._on_media_status(QMediaPlayer.MediaStatus.EndOfMedia)
        self.assertEqual(win._run, 1, "a genuine end-of-file was swallowed")

    def test_two_dead_runs_of_one_file_do_not_freeze_the_transport(self) -> None:
        """Entering another run of the SAME undecodable file issues no
        setSource, so no fresh error is emitted and nothing else can move.  Two
        blocks of one file the backend refuses is a single split away."""
        win = self.loaded()
        win.timeline.set_playhead(1000.0)
        win.split_blocks()
        win.timeline.clear_block_selection()
        win._goto_edited(10.0)
        pump(0.1)
        self.assertEqual(win._run, 0)
        self.assertNotEqual(
            win.player.error(), QMediaPlayer.Error.NoError, "the fake media loaded?"
        )
        win._set_playing(True)
        pump(0.3)  # the step-over is deferred so the stack can unwind
        self.assertFalse(win._playing, "it never walked off the dead runs")
        self.assertAlmostEqual(
            win.timeline.playhead(), win.timeline.edited_duration(), delta=1.0
        )

    def test_a_gap_is_not_ended_early_by_a_second_end_of_file(self) -> None:
        win = self.loaded()
        bed = str(Path(_TMP.name) / "gapbed.mp3")
        Path(bed).write_bytes(b"stub")
        win.timeline.add_clip(None, AudioFileInfo(bed, 4000.0, "mp3", 2, 44100), "place")
        pump(0.1)
        win._goto_edited(3000.0)
        win._set_playing(True)
        self.assertEqual(win._run, -1)
        win._on_media_status(QMediaPlayer.MediaStatus.EndOfMedia)
        self.assertTrue(win._playing, "the gap clock was stopped by a stale EOF")
        self.assertTrue(win._gap_timer.isActive())
        win._set_playing(False)

    def test_a_re_seat_during_a_load_re_parks_the_seek(self) -> None:
        """setPosition is dropped on media that has not loaded, so the PARKED
        target is the only one that lands.  A second seek during the load left
        the first one parked and the picture arrived where the playhead had
        already been and left."""
        win = self.loaded()
        other = str(Path(_TMP.name) / "reseat.mp4")
        Path(other).write_bytes(b"stub")
        win._runs = [appmod.PreviewVideo(other, 0.0, 600.0, 0.0)]
        win._video_path = win.info.path
        win._enter_run(0, 100.0)  # swap starts, target parked
        self.assertEqual(win._load_seek_ms, 100_000)
        self.assertTrue(win._in_swap)
        win._enter_run(0, 190.0)  # re-seat while it is still loading
        self.assertEqual(win._load_seek_ms, 190_000, "the first seek won")

    def test_an_undecodable_run_does_not_freeze_the_transport(self) -> None:
        """No position reports come off a run that will not decode, and the gap
        clock does not run while a run is current — the transport sat on
        Playing with a frozen playhead."""
        win = self.loaded()
        win.timeline.set_playhead(1000.0)
        win.split_blocks()
        pump(0.1)
        win._goto_edited(10.0)
        pump(0.1)
        win._playing = True
        self.assertEqual(win._run, 0)
        win._on_player_error(QMediaPlayer.Error.ResourceError, "Could not open file")
        self.assertNotEqual(win._run, 0, "it stayed on the dead run")
        # ... and a stale end-of-file arriving behind it must not skip another.
        landed = win._run
        win._on_media_status(QMediaPlayer.MediaStatus.EndOfMedia)
        self.assertEqual(win._run, landed, "a stale EOF skipped the run it landed on")

    def test_paste_is_offered_only_when_it_would_work(self) -> None:
        win = self.loaded()
        self.assertFalse(win.act_paste_blocks.isEnabled())
        win.timeline.select_block(win.timeline.lane_clips(0)[0])
        win.copy_blocks()
        self.assertTrue(win.act_paste_blocks.isEnabled())
        win.delete_blocks()  # every block gone: duration 0
        pump(0.05)
        self.assertEqual(win.timeline.edited_duration(), 0.0)
        self.assertTrue(
            win.act_paste_blocks.isEnabled(), "the only way back was greyed out"
        )
        win.paste_blocks()
        self.assertGreater(win.timeline.edited_duration(), 0.0)

    def test_the_block_selection_shows_in_the_status_strip(self) -> None:
        win = self.loaded()
        win.timeline.select_block(win.timeline.lane_clips(0)[0])
        pump(0.05)
        self.assertIn("block", win.selection_label.text())


class TestTeardown(WindowCase):
    """Closing must not join the GUI thread against a running ffmpeg.

    The original hang: closeEvent waited on the waveform worker, which sat in
    subprocess.communicate().  The bounded wait that replaced it only masked it
    for children that happened to finish inside the budget.
    """

    def test_close_kills_assets_instead_of_waiting_them_out(self) -> None:
        started = threading.Event()

        def blocking_wave(*_a: object, **_k: object) -> str:
            started.set()
            probe_mod._spawn([sys.executable, "-c", "import time; time.sleep(30)"], 30.0)
            return ""

        appmod.waveform_png = blocking_wave  # type: ignore[assignment]
        win = self.loaded()
        win._on_waveform_needed(0, 0.0, 5.0)
        self.assertTrue(started.wait(5.0), "the waveform worker never started")
        end = time.monotonic() + 5.0
        while not probe_mod._LIVE and time.monotonic() < end:
            pump(0.01)
        self.assertTrue(probe_mod._LIVE, "the child never registered as live")

        t0 = time.monotonic()
        win.close()
        blocked = time.monotonic() - t0

        self.assertLess(blocked, 2.0, f"close() blocked {blocked:.1f}s on a live child")
        self.assertFalse(
            [w for w in win._workers if w.isRunning()], "a worker was still running"
        )
        self.assertFalse(probe_mod._LIVE, "the child outlived the window")
        self.assertFalse(probe_mod._CANCELLED, "the module was left latched")

    def test_media_status_does_not_re_prime_the_player_while_closing(self) -> None:
        """The real deadlock: stop() delivers LoadedMedia synchronously.

        Priming from that callback calls pause() re-entrantly inside stop(),
        which wedges against the QFFmpeg audio renderer thread.
        """
        win = self.loaded()
        calls: list[str] = []
        win.player.pause = lambda: calls.append("pause")  # type: ignore[method-assign]

        win._closing = False
        win._on_media_status(QMediaPlayer.MediaStatus.LoadedMedia)
        self.assertEqual(calls, ["pause"], "priming must still work while open")

        calls.clear()
        win._closing = True
        win._on_media_status(QMediaPlayer.MediaStatus.LoadedMedia)
        self.assertEqual(calls, [], "the player was re-primed during teardown")


class TestRenderSeam(WindowCase):
    def test_render_full_builds_the_two_input_command(self) -> None:
        """The GUI's cuts must reach ffmpeg as the engine's fixed graph: video
        from input 0, audio from a SECOND -i of the same file."""
        src = os.path.abspath(__file__)
        built = self.capture_render()
        win = self.loaded(src)
        win.options = appmod.RenderOptions()  # 60 Mbps / 60 fps / 1440p / tmix 7 / VA-API
        for a, b in [(0, 1200), (1203, 2000), (2003, 2500), (2503, 2780)]:
            cut_source(win.timeline, float(a), float(b))
        self.assertEqual(keeps(win.timeline), [(1200.0, 1203.0), (2000.0, 2003.0), (2500.0, 2503.0)])
        win.timeline._headers[1].mute.setChecked(True)
        win.render_full()
        pump(0.1)

        argv = built["argv"]
        graph = graph_of(built)
        self.assertEqual(argv.count("-i"), 2)  # type: ignore[union-attr]
        self.assertEqual([a for a in argv if a == src], [src, src])  # type: ignore[union-attr]
        self.assertIn("[0:v]split=3", graph)
        # The mix is asserted through the ffmpeg stream refs, not the engine's
        # internal chain labels, which are render.py's business to name.
        self.assertIn("amix=inputs=2", graph)
        self.assertIn("[1:a:0]", graph)
        self.assertIn("[1:a:2]", graph)
        self.assertNotIn("[1:a:1]", graph, "a muted track must not be in the graph")
        self.assertNotIn("[0:a", graph)
        for seg in ("trim=start=1200:end=1203", "trim=start=2000:end=2003",
                    "trim=start=2500:end=2503"):
            self.assertIn(seg, graph)
        self.assertIn("tmix=frames=7,fps=60,scale=2560:1440,format=nv12,hwupload", graph)
        self.assertIn("h264_vaapi", argv)  # type: ignore[arg-type]
        self.assertEqual(argv[argv.index("-vaapi_device") + 1], "/dev/dri/renderD129")  # type: ignore[union-attr,index]
        self.assertAlmostEqual(built["total"], 9.0)  # type: ignore[arg-type]

    def test_a_second_video_lane_reaches_ffmpeg(self) -> None:
        """The whole point of ``RenderSettings.plan``.

        The legacy ``audio``/``external_audio``/``video_holes`` fields can only
        describe a single-clip timeline, so without the plan a second video
        lane is silently dropped: the render runs, exits 0, and the composite
        the user built is simply not in the file.
        """
        built = self.capture_render()
        win = self.loaded()
        under = str(Path(_TMP.name) / "under.mp4")
        Path(under).write_bytes(b"stub")
        lane = win.timeline.add_video_lane()
        self.assertTrue(
            win.timeline.add_clip(lane, fake_info(under, duration=90.0), "place")
        )
        # A lift on lane 0 is what the lane underneath shows through.
        win.timeline.set_armed_lanes([0])
        select(win.timeline, 10.0, 40.0)
        win.timeline.cut_selection()
        win.timeline.set_armed_lanes([])  # back to everything

        self.assertIsNotNone(win.timeline.render_plan())
        win.render_full()
        pump(0.1)
        argv = built["argv"]
        graph = graph_of(built)
        self.assertIn(under, argv, "the second lane's file never became an input")
        # The composite is resolved per span, not overlaid: the lifted window
        # on lane 0 is filled from the lane underneath, and the embedded audio
        # has moved along to input 2 because the second file took input 1.
        # under.mp4 opens with -ss 8 (input_seeks), so 10-40 reads as 2-32.
        self.assertEqual(argv[argv.index(under) - 3 : argv.index(under)], ["-ss", "8", "-i"])
        self.assertIn("[1:v]trim=start=2:end=32", graph, "lane 2 does not fill the lift")
        self.assertIn("[0:v]split=2", graph, "lane 1 must still supply the rest")
        self.assertIn("[2:a:0]", graph, "the audio input index did not follow the plan")
        self.assertIn("concat=n=3", graph)

    def test_the_render_summary_counts_the_extra_video_tracks(self) -> None:
        built = self.capture_render()
        win = self.loaded()
        seen: list[str] = []
        appmod.RenderOptionsDialog.exec = lambda self: (  # type: ignore[method-assign]
            seen.append(self.findChildren(type(win.state_label))[0].text())
            or QDialog.DialogCode.Accepted
        )
        under = str(Path(_TMP.name) / "under2.mp4")
        Path(under).write_bytes(b"stub")
        lane = win.timeline.add_video_lane()
        win.timeline.add_clip(lane, fake_info(under, duration=90.0), "place")
        win.render_full()
        pump(0.1)
        self.assertIn("1 extra video track", seen[0])
        self.assertIn("argv", built)

    def test_gain_and_mute_from_the_lanes_reach_the_graph(self) -> None:
        built = self.capture_render()
        win = self.loaded()
        win.timeline._headers[0].slider.setValue(-7)
        win.timeline._headers[2].slider.setValue(4)
        win.timeline._headers[1].mute.click()
        win.render_full()
        pump(0.1)
        graph = graph_of(built)
        self.assertIn("[1:a:0]volume=-7dB", graph)
        self.assertIn("[1:a:2]volume=4dB", graph)
        self.assertNotIn("[1:a:1]", graph)

    def test_render_selection_is_enabled_only_with_a_selection(self) -> None:
        win = self.loaded()
        self.assertFalse(win.act_render_sel.isEnabled())
        select(win.timeline, 10.0, 20.0)
        self.assertTrue(win.act_render_sel.isEnabled())
        win.timeline.clear_selection()
        self.assertFalse(win.act_render_sel.isEnabled())

    def test_render_selection_renders_only_the_selection_honouring_cuts(self) -> None:
        built = self.capture_render()
        win = self.loaded()
        cut_source(win.timeline, 100.0, 200.0)  # edited 100 == source 200
        select(win.timeline, 50.0, 150.0)  # source 50..100 + 200..250
        win.render_selection()
        pump(0.1)
        graph = graph_of(built)
        self.assertIn("trim=start=50:end=100", graph)
        self.assertIn("trim=start=200:end=250", graph)
        self.assertIn("split=2", graph)
        self.assertAlmostEqual(built["total"], 100.0)  # type: ignore[arg-type]

    def test_render_selection_without_a_selection_does_nothing(self) -> None:
        built = self.capture_render()
        win = self.loaded()
        win.render_selection()
        pump(0.05)
        self.assertNotIn("argv", built)

    def test_av1_mkv_through_the_gui_omits_bitrate_and_movflags(self) -> None:
        built = self.capture_render()
        out = str(Path(_TMP.name) / "av1.mkv")
        QFileDialog.getSaveFileName = staticmethod(lambda *a, **k: (out, ""))  # type: ignore[method-assign]
        win = self.loaded()
        win.options = appmod.RenderOptions(
            encoder="libsvtav1", container="mkv", crf=34, av1_preset=7
        )
        win.render_full()
        pump(0.1)
        argv = built["argv"]
        self.assertIn("libsvtav1", argv)  # type: ignore[arg-type]
        self.assertEqual(argv[argv.index("-crf") + 1], "34")  # type: ignore[index,union-attr]
        self.assertEqual(argv[argv.index("-preset") + 1], "7")  # type: ignore[index,union-attr]
        for flag in ("-b:v", "-maxrate", "-bufsize", "-movflags"):
            self.assertNotIn(flag, argv)  # type: ignore[arg-type]
        self.assertEqual(argv[-1], str(appmod.part_path(Path(out))))  # type: ignore[index]

    def test_options_dialog_swaps_bitrate_for_crf(self) -> None:
        dialog = appmod.RenderOptionsDialog(
            Path(_TMP.name), "x.mp4", appmod.RenderOptions(), "summary", None
        )
        widget = dialog.options_widget
        form = widget._form
        self.assertTrue(form.isRowVisible(widget.mbps))
        self.assertFalse(form.isRowVisible(widget.quality_row))
        self.assertFalse(form.isRowVisible(widget.note))
        widget.encoder.setCurrentIndex(widget.encoder.findData("libsvtav1"))
        self.assertFalse(form.isRowVisible(widget.mbps), "bitrate must be hidden for AV1")
        self.assertTrue(form.isRowVisible(widget.quality_row))
        self.assertTrue(form.isRowVisible(widget.note))
        self.assertIn("CPU", widget.note.text())
        widget.encoder.setCurrentIndex(widget.encoder.findData("libx264"))
        self.assertTrue(form.isRowVisible(widget.mbps))
        delete(dialog)

    def test_container_choice_fixes_the_output_extension(self) -> None:
        dialog = appmod.RenderOptionsDialog(
            Path(_TMP.name), "clip-cut.mp4", appmod.RenderOptions(), "summary", None
        )
        widget = dialog.options_widget
        self.assertEqual(widget.options().container, "mp4")
        widget.container.setCurrentIndex(widget.container.findData("mkv"))
        self.assertEqual(widget.output_name(), "clip-cut.mkv")
        self.assertEqual(dialog.out_path().suffix, ".mkv")
        delete(dialog)

    def test_save_dialog_suffix_selects_the_container(self) -> None:
        """Picking clip.mkv in the file dialog must preselect the MKV container."""
        dialog = appmod.RenderOptionsDialog(
            Path(_TMP.name), "clip.mkv", appmod.RenderOptions(), "summary", None
        )
        self.assertEqual(dialog.options().container, "mkv")
        self.assertEqual(dialog.out_path().suffix, ".mkv")
        delete(dialog)

    def test_all_tracks_muted_renders_video_only(self) -> None:
        built = self.capture_render()
        win = self.loaded()
        for header in win.timeline._headers:
            header.mute.setChecked(True)
        win.render_full()
        pump(0.1)
        argv = built["argv"]
        self.assertEqual(argv.count("-i"), 1)  # type: ignore[union-attr]
        self.assertNotIn("-c:a", argv)  # type: ignore[arg-type]

    def test_delete_key_cuts_once_and_never_warns(self) -> None:
        win = self.loaded()
        messages: list[str] = []
        win.statusBar().messageChanged.connect(messages.append)
        select(win.timeline, 10.0, 20.0)
        QTest.keyClick(win.timeline, Qt.Key.Key_Delete)
        pump(0.05)
        self.assertEqual(keeps(win.timeline), [(0.0, 10.0), (20.0, 2780.0)], "exactly one cut")
        self.assertFalse(
            [m for m in messages if "Select a range" in m],
            f"Delete must not nag about a selection: {messages}",
        )
        # the menu path still warns when there really is no selection
        win.cut_selection()
        pump(0.05)
        self.assertTrue([m for m in messages if "Select a range" in m])

    def test_everything_cut_blocks_the_render_and_undo_restores_it(self) -> None:
        win = self.loaded()
        win.show()
        win.timeline.select_all()
        win.timeline.cut_selection()
        pump(0.05)
        self.assertFalse(win.render_button.isEnabled())
        self.assertFalse(win.act_render_full.isEnabled())
        self.assertTrue(win.warning_label.isVisible())
        win.act_undo.trigger()
        pump(0.05)
        self.assertTrue(win.render_button.isEnabled())
        self.assertFalse(win.warning_label.isVisible())


class _OwnConfig:
    """Mixin: these tests write recent files and prefs; keep them off the shared config."""

    def setUp(self) -> None:
        super().setUp()  # type: ignore[misc]
        path = Path(tempfile.mkdtemp(dir=_TMP.name)) / "config.json"
        patcher = mock.patch.object(appmod, "CONFIG_PATH", path)
        patcher.start()
        self.addCleanup(patcher.stop)  # type: ignore[attr-defined]


class TestAuditFixes(_OwnConfig, WindowCase):
    """Regression tests for the 2026-11-15 audit (file safety first)."""

    def test_render_refuses_to_write_over_the_source_through_a_symlink(self) -> None:
        root = Path(tempfile.mkdtemp(dir=_TMP.name))
        (root / "real").mkdir()
        src = root / "real" / "clip.mp4"
        src.write_bytes(b"the only copy")
        (root / "alias").symlink_to(root / "real")
        win = self.loaded(str(src))
        built = self.capture_render()
        QFileDialog.getSaveFileName = staticmethod(  # type: ignore[method-assign]
            lambda *a, **k: (str(root / "alias" / "clip.mp4"), "")
        )
        errors: list[object] = []
        QMessageBox.critical = staticmethod(lambda *a, **k: errors.append(a))  # type: ignore[method-assign]
        win.render_full()
        self.assertNotIn("argv", built, "no ffmpeg may be started over the source")
        self.assertEqual(len(errors), 1)
        self.assertEqual(src.read_bytes(), b"the only copy")

    def test_relative_paths_are_stored_absolute(self) -> None:
        here = Path(__file__).resolve()
        cwd = os.getcwd()
        os.chdir(here.parent)
        try:
            win = self.loaded(here.name)
        finally:
            os.chdir(cwd)
        self.assertEqual(win.info.path, str(here))  # type: ignore[union-attr]

    def test_second_overwrite_prompt_guards_a_name_the_options_changed(self) -> None:
        win = self.loaded()
        built = self.capture_render()
        existing = Path(_TMP.name) / "out.mkv"
        existing.write_bytes(b"keep me")
        def pick_mkv(dialog: appmod.RenderOptionsDialog) -> QDialog.DialogCode:
            box = dialog.options_widget.container  # the user switches MP4 -> MKV in the dialog
            box.setCurrentIndex(box.findData("mkv"))
            return QDialog.DialogCode.Accepted

        appmod.RenderOptionsDialog.exec = pick_mkv  # type: ignore[method-assign]
        asked: list[object] = []
        answer = [QMessageBox.StandardButton.No]
        orig = QMessageBox.question
        QMessageBox.question = staticmethod(lambda *a, **k: asked.append(a) or answer[0])  # type: ignore[method-assign]
        self.addCleanup(setattr, QMessageBox, "question", orig)
        win.render_full()
        self.assertEqual(len(asked), 1)
        self.assertNotIn("argv", built)
        answer[0] = QMessageBox.StandardButton.Yes
        win.render_full()
        self.assertIn("argv", built)

    def test_discard_prompt_only_when_there_is_an_edit(self) -> None:
        win = self.loaded()
        asked: list[object] = []
        orig = QMessageBox.question
        QMessageBox.question = staticmethod(  # type: ignore[method-assign]
            lambda *a, **k: asked.append(a) or QMessageBox.StandardButton.Cancel
        )
        self.addCleanup(setattr, QMessageBox, "question", orig)
        self.assertTrue(_REAL_DISCARD_OK(win), "nothing edited: no question")
        self.assertEqual(asked, [])
        cut_source(win.timeline, 100.0, 900.0)
        self.assertFalse(_REAL_DISCARD_OK(win), "Cancel keeps the edit")
        self.assertEqual(len(asked), 1)

    def test_render_shortcuts_are_off_while_a_file_probes(self) -> None:
        win = self.loaded()
        appmod.probe = lambda p: (time.sleep(0.4), fake_info(p))[1]  # type: ignore[assignment]
        win.load_file(os.path.abspath(appmod.__file__))
        self.assertFalse(win.act_render_full.isEnabled())
        self.assertFalse(win.act_render_sel.isEnabled())

    def test_config_with_nan_and_huge_numbers_still_loads(self) -> None:
        original = appmod.CONFIG_PATH
        appmod.CONFIG_PATH = Path(_TMP.name) / "junk-cfg.json"
        self.addCleanup(setattr, appmod, "CONFIG_PATH", original)
        appmod.CONFIG_PATH.write_text(
            '{"splitter": [NaN, 300], "render": {"mbps": 99999999999}}', encoding="utf-8"
        )
        cfg = appmod.Config.load()
        self.assertEqual(cfg.splitter, [])
        dialog = appmod.RenderOptionsDialog(Path(_TMP.name), "x.mp4", cfg.render, "", None)
        self.assertEqual(dialog.options().mbps, 500, "clamped to the spin box, no OverflowError")
        delete(dialog)

    def test_typed_odd_size_is_rounded_to_even(self) -> None:
        dialog = appmod.RenderOptionsDialog(Path(_TMP.name), "x.mp4", appmod.RenderOptions(), "", None)
        dialog.options_widget.custom_w.setValue(1919)
        dialog.options_widget.custom_h.setValue(1081)
        opts = dialog.options()
        self.assertEqual((opts.custom_w, opts.custom_h), (1918, 1080))
        delete(dialog)


class TestImport(_OwnConfig, WindowCase):
    def test_import_appends_a_video_to_track_one(self) -> None:
        win = self.loaded()
        other = os.path.abspath(appmod.__file__)
        appmod.probe = lambda p: fake_info(p, duration=60.0, n_audio=1)  # type: ignore[assignment]
        win.import_file(other)
        deadline = time.monotonic() + 3.0
        while len(win.timeline.lane_clips(0)) < 2 and time.monotonic() < deadline:
            pump(0.02)
        clips = win.timeline.lane_clips(0)
        self.assertEqual([c.path for c in clips][-1], other)
        self.assertAlmostEqual(clips[-1].start, 2780.0, places=3)
        self.assertEqual(win.info.path, os.path.abspath(__file__), "import must not replace the media")  # type: ignore[union-attr]

    def test_import_with_nothing_loaded_opens_the_video(self) -> None:
        win = self.window()
        win.import_file(os.path.abspath(__file__))
        deadline = time.monotonic() + 3.0
        while win.info is None and time.monotonic() < deadline:
            pump(0.02)
        self.assertIsNotNone(win.info)


class TestPreviewCopies(_OwnConfig, WindowCase):
    def test_the_preview_swaps_onto_a_copy_once_built_and_back_on_full(self) -> None:
        appmod.PROXY_DIR = Path(_TMP.name) / "proxies"
        built: list[str] = []

        def fake_make_proxy(path: str, out: str, height: int, device: str, timeout: float) -> str:
            Path(out).parent.mkdir(parents=True, exist_ok=True)
            Path(out).write_bytes(b"copy")
            built.append(out)
            return out

        orig = appmod.make_proxy
        appmod.make_proxy = fake_make_proxy  # type: ignore[assignment]
        self.addCleanup(setattr, appmod, "make_proxy", orig)
        win = self.loaded()
        source = win.info.path  # type: ignore[union-attr]
        self.assertEqual(win._video_path, source, "Full: the original plays")
        win.config.prefs.preview = "540"
        win._sync_preview()
        deadline = time.monotonic() + 3.0
        while not built and time.monotonic() < deadline:
            pump(0.02)
        pump(0.1)
        self.assertEqual(len(built), 1)
        self.assertEqual(win._video_path, built[0], "the player moved onto the copy")
        win._sync_preview()
        pump(0.1)
        self.assertEqual(len(built), 1, "a built copy is reused, never rebuilt")
        win.config.prefs.preview = "full"
        win._sync_preview()
        self.assertEqual(win._video_path, source)

    def test_thumbnail_jobs_follow_the_preference(self) -> None:
        seen: list[int] = []
        appmod.thumbnails_at = lambda *a, **k: seen.append(k.get("workers")) or []  # type: ignore[assignment]
        win = self.window()
        win.config.prefs.jobs = 2
        self.load(win, os.path.abspath(__file__))
        pump(0.5)
        self.assertTrue(seen and set(seen) == {2})


class TestText(_OwnConfig, WindowCase):
    def test_add_text_shows_on_the_paused_stage_and_drags_in_one_undo_step(self) -> None:
        win = self.loaded()
        answers = {"text": "Hello", "font": "Serif", "bold": False, "color": "#ffcc00",
                   "size": 0.1, "fade_in": 1.0, "fade_out": 0.5}
        win._text_dialog = lambda *a: dict(answers)  # type: ignore[method-assign]
        win.act_add_text.trigger()
        text = win.timeline.render_plan().texts[0]
        self.assertEqual((text.text, text.font, text.bold, text.color, text.size, text.fade_in, text.fade_out),
                         ("Hello", "Serif", False, "#ffcc00", 0.1, 1.0, 0.5))
        win.timeline.undo()
        self.assertEqual(win.timeline.render_plan().texts, [], "add + its settings = one undo step")
        win.timeline.redo()
        self.assertFalse(win.stage.isHidden(), "paused on a text block: the still + text")
        self.assertTrue(win.video.isHidden())

        win.stage.resize(640, 360)
        win.stage.frame_size = QSize(640, 360)
        clip = win.stage.titles[0]
        win.timeline.set_fade(clip, "in", 0.0)  # full opacity for the drag
        before = win.timeline.model.can_undo
        centre = win.stage.text_rect(clip).center()
        QTest.mousePress(win.stage, Qt.MouseButton.LeftButton, pos=centre.toPoint())
        for step in (10, 30, 64):  # three moves, one gesture
            QTest.mouseMove(win.stage, (centre + QPointF(step, 0)).toPoint())
        QTest.mouseRelease(win.stage, Qt.MouseButton.LeftButton, pos=(centre + QPointF(64, 0)).toPoint())
        self.assertAlmostEqual(clip.title.x, 0.6, places=2)
        self.assertTrue(before)
        win.timeline.undo()
        self.assertAlmostEqual(win.timeline.find_clip(clip.cid).title.x, 0.5, places=6,
                               msg="the whole drag is one undo step")

        win._set_playing(True)
        self.assertTrue(win.stage.isHidden(), "playing: the real video, not the still")
        win._set_playing(False)


class TestProjects(_OwnConfig, WindowCase):
    def save_dialog(self, path: Path) -> None:
        orig = QFileDialog.getSaveFileName
        QFileDialog.getSaveFileName = staticmethod(lambda *a, **k: (str(path), ""))  # type: ignore[method-assign]
        self.addCleanup(setattr, QFileDialog, "getSaveFileName", orig)

    def wait_info(self, win: appmod.MainWindow) -> None:
        deadline = time.monotonic() + 3.0
        while (win.info is None or win._pending_project) and time.monotonic() < deadline:
            pump(0.02)
        pump(0.05)

    def test_save_and_reopen_restores_the_edit(self) -> None:
        win = self.loaded()
        cut_source(win.timeline, 100.0, 900.0)
        win.timeline.add_title(50.0, "Kept")
        path = Path(tempfile.mkdtemp(dir=_TMP.name)) / "edit"  # no suffix typed
        self.save_dialog(path)
        self.assertTrue(win.save_project())
        saved = path.with_name("edit.ffcut")
        self.assertTrue(saved.is_file(), "the .ffcut suffix is added")
        self.assertEqual(json.loads(saved.read_text())["media"], win.info.path)  # type: ignore[union-attr]

        other = self.window()
        other.load_file(str(saved))  # what File > Recent, a drop and argv all do
        self.wait_info(other)
        self.assertEqual(other.timeline.render_plan(), win.timeline.render_plan())
        self.assertEqual(keeps(other.timeline), keeps(win.timeline))
        self.assertEqual(other.windowTitle(), "ffcut — edit.ffcut")
        self.assertIn(str(saved), other.config.recent)

    def test_the_prompt_only_asks_about_unsaved_changes(self) -> None:
        win = self.loaded()
        asked: list[object] = []
        orig = QMessageBox.question
        answer = [QMessageBox.StandardButton.Cancel]
        QMessageBox.question = staticmethod(lambda *a, **k: asked.append(a) or answer[0])  # type: ignore[method-assign]
        self.addCleanup(setattr, QMessageBox, "question", orig)
        cut_source(win.timeline, 100.0, 900.0)
        self.save_dialog(Path(tempfile.mkdtemp(dir=_TMP.name)) / "p.ffcut")
        answer[0] = QMessageBox.StandardButton.Save  # Save from the prompt
        self.assertTrue(_REAL_DISCARD_OK(win))
        self.assertEqual(len(asked), 1)
        self.assertIsNotNone(win._project_path)
        win.timeline.set_playhead(500.0)
        self.assertTrue(_REAL_DISCARD_OK(win), "saved and unchanged: no question")
        self.assertEqual(len(asked), 1)
        cut_source(win.timeline, 1000.0, 1100.0)
        answer[0] = QMessageBox.StandardButton.Cancel
        self.assertFalse(_REAL_DISCARD_OK(win), "an unsaved change asks again")
        self.assertEqual(len(asked), 2)

    def test_a_project_whose_video_is_gone_is_refused(self) -> None:
        win = self.loaded()
        bad = Path(tempfile.mkdtemp(dir=_TMP.name)) / "gone.ffcut"
        bad.write_text(json.dumps({"ffcut_project": 1, "media": "/nonexistent/x.mp4",
                                   "timeline": {"model": {}, "lanes": []}}))
        warned: list[object] = []
        QMessageBox.warning = staticmethod(lambda *a, **k: warned.append(a))  # type: ignore[method-assign]
        win.load_file(str(bad))
        self.assertEqual(len(warned), 1)
        self.assertEqual(win.info.path, os.path.abspath(__file__), "the open edit is untouched")  # type: ignore[union-attr]


if __name__ == "__main__":
    unittest.main(verbosity=2)
