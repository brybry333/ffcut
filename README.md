# ffcut

A PyQt6 front end for the `ffcut`/`ffca` ffmpeg workflow: load a recording, snip
the dead parts out of the timeline, balance the audio tracks, and render with
tmix motion blending at the usual 60 Mbps / 60 fps / 1440p settings.

Cutting is non-destructive — the source file is never touched. What survives on
the timeline is handed to ffmpeg as `trim`/`atrim` keep-ranges and concatenated
in one pass. Cuts **collapse**: the timeline is drawn in *edited* time, so what
you see is the edit, not the source with holes in it.

## Requirements

Arch, Python 3.14, PyQt6 (`python-pyqt6`), `qt6-multimedia` with the ffmpeg
backend, and `ffmpeg`/`ffprobe` on `PATH`. Nothing from pip.

## Run

```sh
./ffcut-gui                      # empty, then drop a file on the window
./ffcut-gui ~/Videos/clip.mp4    # or open one straight away
```

## Using it

1. **Load** — drag a video onto the window, or File ▸ Open… (Ctrl+O). File ▸
   Recent remembers the last eight. File ▸ Import… (Ctrl+I) *adds* a file
   instead: a video goes on the end of video track 1 (its sound grouped to it),
   an audio file gets a lane of its own at 0:00. Opening another file over an
   edit asks before it throws the edit away.
2. **Scrub** — drag the playhead, or click anywhere on the timeline. The
   transport's outer buttons jump cut-to-cut; the inner ones skip by the
   interval set in Preferences (default 5 s).
3. **Select** — click-drag on the timeline, or Mark In / Mark Out (I / O) at the
   playhead. Setting one never clears the other, and a plain click never clears
   the selection. Double-click resets the selection to whatever is under the
   cursor: over a block it selects that block (and its group) and marks the
   section of it you clicked; over a gap or an empty lane it marks the segment
   under the cursor, as it always did.
4. **Cut** — Delete or Ctrl+S removes the selection; the timeline closes the gap
   and leaves a thin splice marker where the cut was. Ctrl+Z / Ctrl+Shift+Z.
5. **Audio** — every audio track gets its own waveform lane with an **M** mute
   toggle and a −30…+12 dB gain slider. That is the render mix *and* the preview
   mix: muted tracks are left out of `amix` entirely and go silent in the
   player, gains become per-track `volume` filters and per-player volumes. The
   slider by the Render button is the master over that mix. (A boost above 0 dB
   previews as balance, not level: Qt cannot play louder than unity, so the
   other lanes come down instead.)
6. **More tracks** — drop an audio file anywhere (the preview included) and it
   gets a lane of its own; drop it on an empty lane to place it at 0:00, on a
   lane with content to append it at the end. Drop a **video** and its embedded
   audio comes with it, one lane per track, grouped to the picture. Right-click
   the strip for Add video track / Add audio track / Add file to this track… /
   Remove this track. A second video lane sits *underneath* the first and shows
   through wherever the first has a lift hole or has run out. The edit runs to
   the end of the longest lane; the render pads the video with black and short
   audio with silence to match.
7. **Blocks** — every clip on a lane is a block you can move, like an event in
   Vegas.

   | | |
   | --- | --- |
   | Alt+click | select the block under the pointer (and its group) |
   | Ctrl+click (with a block already selected) | add it to / remove it from the selection — Ctrl+click with nothing selected is still a plain seek |
   | Ctrl+Alt+click | the same, and it always means the block |
   | Alt+drag | slide it along its lane — it snaps to neighbours, 0:00 and the playhead, and stops against a neighbour rather than overlapping it |
   | Alt+drag to another track | appended to the END of that track's content; refused across kinds, and for a grouped block (Ungroup first) |
   | Double-click | reset the selection to the block under the cursor, and mark the section of it you clicked |
   | Alt+drag its LEFT or RIGHT edge | **trim** it — dragging out restores material a split or an earlier trim cut off, dragging in gives it back. Bounded by the file's own ends, by the neighbouring block and by a 50 ms minimum. The left edge moves the in-point and the start together, so the picture already in the block does not shift |
   | Alt+drag its TOP-LEFT or TOP-RIGHT corner | **fade in / fade out**, 0 to the block's whole length, painted as a shaded wedge |
   | S | split every block the playhead crosses in two |
   | Ctrl+C / Ctrl+V | copy the selection, paste at the playhead |
   | Ctrl+G / Ctrl+Shift+G | group / ungroup |
   | Delete | remove the selected blocks, **leaving a gap** — video black, audio silence |

   Right-click a block for the same verbs (Deselect / Group / Ungroup / Split /
   Copy / Paste / Delete), with the track menu under them. A selected block is
   outlined; a grouped one carries an amber bar along its bottom edge. Groups
   select, move and delete as one — a capture's picture and its sound arrive
   grouped, so they cannot drift apart, and a split leaves the two heads in one
   group and the two tails in another. Plain click and plain drag are unchanged:
   they still seek and mark a range.

   A moved block is a rendered block: the graph places it with `adelay` (audio)
   or a per-piece `trim` (video), and a gap becomes a generated black source.
   The preview follows too — the picture, the waveform and the mix are all
   re-planned on release. A trimmed block is a rendered block for the same
   reason: the graph already reads `src_in`/`src_out` off every clip.

   A **fade** renders as `fade` (video) / `afade` (audio) in the clip's own time,
   and survives a cut placed inside it rather than restarting. The preview does
   **not** play fades — Qt cannot ramp the picture and the preview mix is Qt
   volumes, not an ffmpeg graph — so the wedge on the strip is what tells you one
   is there.
8. **Text** — Edit ▸ Add Text… (Ctrl+T) puts a text block at the playhead on a
   video track of its own (5 s; Alt+drag its edges to stretch it, Alt+drag it
   to move it, S to split it, right-click ▸ Edit text…). Paused over it, the
   preview shows that frame as a still with the text on it: drag the text to
   move it, drag the square at its corner to resize it, double-click it to
   edit it. Position and size are fractions of the frame, so the render
   (`drawtext`, Noto Sans Bold, after scaling) puts it in the same place at any
   resolution. While playing, the preview shows the video without the text
   (Qt's video widget is a native window nothing can be painted over); the
   render always has it. Text blocks do not fade.
9. **Lift vs ripple** — with every lane armed (the default) Delete ripples the
   whole edit. Click a lane header to arm that lane alone, Ctrl+click to add or
   remove one, drag inside an unarmed lane to re-arm to it; Delete then *lifts*
   the range from those lanes only (black / silence) and the timeline keeps its
   length so the lanes stay in sync. `A` arms everything again.
10. **Render** — Render ▸ Render Full… (Ctrl+R) renders the whole edit. Render ▸
   Render Selection… (Ctrl+Shift+R) is enabled only when a selection exists and
   renders just that range, with any cuts inside it still honoured.

The timeline view runs a fifth past the end of the edit (at least 10 s), as in
Vegas or Premiere, so there is always room to drag a block or drop a file
beyond the last one.

The preview plays the whole timeline, not just the loaded file: the timeline
hands it the picture as a list of RUNS (contiguous stretches of one file, in
edited time), and the player swaps sources at a run boundary. Where no run
covers the playhead — a lift on the only video lane, a gap between two blocks,
an audio bed outliving the footage — the stage goes black and a clock advances
edited time until the next run starts. While it may have to swap or park, the
main player goes video-only and every lane gets its own audio player, because
its audio would otherwise swap and park with it.

A boost above 0 dB previews as *balance*, not level: `QAudioOutput` is capped at
unity, so the loudest lane sits at master and the rest keep their distance below
it. The render will be louder overall than the preview suggests. Fixing that
properly means running the mix through ffmpeg itself, which is a different
architecture; the gain slider's tooltip says so.

## Render options

| Option | Notes |
| --- | --- |
| Container | MP4 or MKV. MP4 gets `-movflags +faststart`; MKV does not (it is an mp4-muxer option). |
| Encoder | `h264_vaapi` (GPU), `libx264` (CPU), `av1_vaapi` (GPU, AV1), `libsvtav1` (CPU, AV1). |
| Bitrate / Quality | H.264 and VA-API AV1 use Mbps (CBR); SVT-AV1 swaps that control for CRF + preset — it must not be given `-b:v`. |
| Frame rate, Resolution, TMIX | As in v1: tmix blends at the **source** rate, then fps decimates, then scale. |

**The GPU encoders are VA-API** on the RX 9070 XT (RDNA4, since 2026-10-28; it
replaced an RTX 3070 Ti, so NVENC is gone). It encodes H.264 and AV1 in hardware.
VA-API needs a device and hardware frames: the render gets `-vaapi_device` (the
amdgpu render node, looked up by driver because libva's default `renderD128` is
the Intel iGPU here), the graph ends in `format=nv12,hwupload`, and there is no
`-pix_fmt`. A saved `h264_nvenc` setting loads as `h264_vaapi`.

**SVT-AV1 is the CPU AV1**, quality-targeted (CRF): slower, dramatically smaller
files (measured on the old box, 6 s 1440p60 test render: NVENC 3.2 s / 44.5 MB,
SVT-AV1 3.7 s / 3.6 MB).

## Preferences

Edit ▸ Preferences… (Ctrl+,), stored in `~/.config/ffcut/config.json` next to
the last-used directories, the recent list and the render options.

* **Theme** — System / Dark / Light, applied immediately. System keeps Qt's own
  palette (your desktop theme) and only decides whether the transport gloss is
  drawn light or dark.
* **Skip interval** — seconds for the transport's skip-back / skip-forward.
* **Preview quality** — Full plays the original. 720p / 540p build a light
  preview copy of each file once, in the background on the GPU (VA-API, CPU
  fallback), cache it in `~/.cache/ffcut/proxies` (unused copies go after 14
  days) and switch the preview to it when it is ready. Measured on a 1440p120
  172 Mbps replay: playback ~100 % of a core on the original, ~14 % on the 720p
  copy, ~9 % on 540p; building the copy ran at ~2.7× real time (reading the
  file off the HDD was the limit). Renders always use the original.
* **Background jobs** — how many thumbnail extractions run at once while a file
  loads (default 4; each decodes a full-size frame).

The transport is styled after Windows 7 Media Player: a round play/pause flanked
by pill-shaped skip buttons with a vertical gloss, drawn entirely with QSS
gradients and painted glyphs (no image assets, no extra dependencies).

## What a render looks like

The cut is done with ffmpeg `trim` filters, which means ffmpeg decodes the
source from the very start and simply discards the cut footage. Two visible
consequences:

- Every video input is opened with `-ss` a couple of seconds before the first
  frame the edit reads from it, so ffmpeg jumps there instead of decoding the
  file from 0:00 (2026-10-30: a range 1:41 into a 1440p120 replay used to
  stall the bar for 25 s at the piece boundary; now 1 s, whole render 60 s → 28 s). Output is
  frame-identical (`tests/test_seek.py`). Audio inputs are not seeked: audio
  decode is nearly free and a VBR MP3 seeks inexactly.
- Between two kept ranges of the SAME file ffmpeg still decodes forward
  linearly, and the dialog says so ("decoding ahead to the next kept range…").

The source file is opened **twice** (`-i f -i f`): video comes off input 0 and
audio off input 1. This is load-bearing — with both on one input, a keep-range
past roughly t=600 s silently produces a file with zero video frames and exit
code 0.

ffmpeg writes to `NAME.part.mp4` and only a clean exit renames it onto the name
you chose, so a failed or cancelled render never touches an older file of that
name. Cancel sends ffmpeg SIGTERM (SIGKILL after 3 s) and deletes the `.part`.
A render whose output would land on one of the files being edited (same file
through a symlink, a hard link or another spelling) is refused.

## Tests

```sh
HOME=/tmp/ffcut-tests python -m pytest -q     # 587 tests, ~50 s; HOME keeps it off your config
# needs python-pytest; without it: QT_QPA_PLATFORM=offscreen python -m unittest discover tests
```

`test_render.py` covers the command builder, the audio mix and the job lifecycle
without ffmpeg; `test_app.py` drives the real window and the real timeline
offscreen against a fake render job and fake probe/asset extraction;
`test_timeline.py` covers the edit model and the widget; `test_tracks.py` covers
the multi-lane model and *measures* the padding and composite graphs against a
real ffmpeg; `test_blocks.py` covers select/move/split/group/copy/delete and
pins the two rules the model rests on (blocks on a lane never overlap, and a
moved block reaches the graph); `test_probe.py` builds a small two-track clip and runs the real
thumbnail and waveform extraction.

Every window and dialog a test creates is destroyed explicitly
(`QApplication.sendPostedEvents(None, DeferredDelete)`): `deleteLater()` on its
own is never dispatched outside a running event loop, and a QMediaPlayer left
for interpreter teardown is how a green run ends in a segfault.

## Keys

| Key | Action |
| --- | --- |
| Ctrl+O | open a file |
| Ctrl+I | import a file into the edit |
| Ctrl+T | add text at the playhead |
| Space | play / pause |
| ← → | nudge playhead (Shift finer, Ctrl coarser) |
| Ctrl + wheel | zoom the timeline · plain wheel pans |
| Home / End | jump to start / end |
| I / O | mark selection in / out at the playhead |
| Delete or Ctrl+S | cut the selection |
| Ctrl+Z / Ctrl+Shift+Z | undo / redo |
| Ctrl+A | select the whole edit |
| A | arm every lane again (back to ripple) |
| click / Ctrl+click a lane header | arm that lane only / toggle it |
| Esc | clear the selection |
| Ctrl+R | render the whole edit |
| Ctrl+Shift+R | render the selection |
| Ctrl+, | preferences |

## Layout

```
ffcut/
  probe.py     ffprobe metadata, thumbnail strips, showwavespic waveforms
  render.py    filter-graph construction + the ffmpeg job/progress parser
  timeline.py  the multi-lane timeline widget, the cut model and the
               source<->edited time mapping
  theme.py     palette, QSS and the painted transport glyphs
  app.py       main window, menus, preferences, render dialogs, wiring
  main.py      def main()
```

## License

GPL-3.0-only (see `LICENSE`). ffcut links PyQt6, which is GPL.
