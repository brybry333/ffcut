"""Palette + stylesheet for ffcut — Windows 7 Media Player (WMP12) chrome.

Three modes, switchable live (``apply`` is idempotent and cheap):

``system``
    Leave Qt's own palette alone (the desktop's theme) and only decide whether
    the WMP chrome should be drawn light or dark by looking at it.
``dark`` / ``light``
    Install one of the two palettes below instead.

The look is Windows 7 Media Player, not XP: light silver-blue glass panels,
1px ``#a3b7c9``-family borders with a white inner top highlight, and *restrained*
vertical gradients — a soft sheen across the top half and a barely-there step at
the midpoint, never the 50/50 candy split of the XP era.  The focal point is the
round play button: a white-to-pale-blue dome with a blue rim and a blue glyph,
lit from the top with a reflected sliver of light along the bottom.

Everything is ``qlineargradient`` / ``qradialgradient`` and painted polygons.
No image assets and no extra dependencies.

Metrics live in one place (:data:`METRICS`) because "the buttons are too big"
was an explicit complaint: every padding and icon size in the chrome is derived
from these numbers, so there is a single knob to turn.
"""

from __future__ import annotations

from PyQt6.QtCore import QPointF, Qt
from PyQt6.QtGui import QColor, QIcon, QPainter, QPalette, QPixmap, QPolygonF

__all__ = [
    "METRICS",
    "MODES",
    "TIMELINE_MODES",
    "TOKENS",
    "apply",
    "media_icon",
    "stylesheet",
    "variant",
]

#: Valid values for the Preferences theme setting.
MODES: tuple[str, ...] = ("system", "dark", "light")

#: Chrome metrics, in pixels.  Deliberately small — except the play button,
#: which is the one thing WMP12 makes big on purpose.
METRICS: dict[str, int] = {
    "play": 34,  # round play/pause diameter
    "skip": 22,  # pill button height
    "icon": 11,  # icon edge inside a skip button
    "icon_play": 15,
    "pad": 3,  # generic control padding
    "bar_pad": 3,  # transport bar inner padding
    "seek": 13,  # height of the thin seek strip above the buttons
    "radius": 4,  # panel corner radius
}


# --------------------------------------------------------------------------- #
# palettes
# --------------------------------------------------------------------------- #
#: WMP12 dark skin: blue-tinted charcoal, not neutral black.  The gloss stops
#: (g1..g4) are deliberately close together — the sheen is a hint, not a stripe.
_DARK: dict[str, str] = {
    "window": "#242c34",
    "base": "#161c22",
    "text": "#e7eef5",
    "dim": "#93a7b8",
    "border": "#3d4a58",
    "accent": "#3b9ce0",
    "accent_text": "#06121c",
    # chrome gloss, top -> bottom
    "g1": "#4d5b68",
    "g2": "#3c4854",
    "g3": "#333e48",
    "g4": "#232b33",
    "gloss_edge": "#151b21",
    "gloss_top": "#7f93a4",
    "bar": "#1f262d",
    "warn_fg": "#f0a0a0",
    "warn_bg": "#402326",
    "warn_border": "#7a3b3f",
    # --- WMP12 specifics ---
    "play_rim": "#2f6f9f",
    "play_face1": "#e9f3fb",
    "play_face2": "#9dc4e2",
    "glyph": "#0d2b44",
    "glyph_skip": "#d3e2ee",
    "seek_track": "#131a20",
    "seek_fill": "#3b9ce0",
    # --- window / menu / hover grounds ---
    "panel1": "#2b343d",
    "panel2": "#1d242b",
    "menu1": "#37424e",
    "menu2": "#28313a",
    "hover": "#4a6c88",
    "sel": "#2f6f9f",
    "sunken": "#141a20",
    "hilite": "#61717f",
}

#: WMP12 light skin: the real thing.  Silver-blue, white sheen, blue accents.
_LIGHT: dict[str, str] = {
    "window": "#eef3f8",
    "base": "#ffffff",
    "text": "#1a2530",
    "dim": "#5d7085",
    "border": "#a3b7c9",
    "accent": "#2f7fd0",
    "accent_text": "#ffffff",
    "g1": "#ffffff",
    "g2": "#f4f8fc",
    "g3": "#e9f1f8",
    "g4": "#d6e3ef",
    "gloss_edge": "#9db2c6",
    "gloss_top": "#ffffff",
    "bar": "#e7eff7",
    "warn_fg": "#7a2b2b",
    "warn_bg": "#fbe3e3",
    "warn_border": "#d9a2a2",
    # --- WMP12 specifics ---
    "play_rim": "#5387b3",
    "play_face1": "#ffffff",
    "play_face2": "#a6cdec",
    "glyph": "#14527f",
    "glyph_skip": "#2c4a66",
    "seek_track": "#c6d6e4",
    "seek_fill": "#3f9ad8",
    # --- window / menu / hover grounds ---
    "panel1": "#f2f7fb",
    "panel2": "#dce7f1",
    "menu1": "#f8fbfe",
    "menu2": "#dfeaf3",
    "hover": "#cfe4f7",
    "sel": "#9dcbf0",
    "sunken": "#dbe5ee",
    "hilite": "#ffffff",
}

#: Both skins by name.  ``timeline.py`` reads this too; do not rename or drop keys.
TOKENS: dict[str, dict[str, str]] = {"dark": _DARK, "light": _LIGHT}

#: Lane colours per app variant — the source of truth for the strip.  The strip
#: must look these up at PAINT time, not at import time, or it can never follow
#: a live theme switch (which is exactly why the strip stayed dark in light
#: mode).
#:
#: The light set is tuned against the transforms ``timeline.py`` applies to it,
#: not just read on its own.  The gutter is ``lane_bg.lighter(135)`` and the
#: ruler ``lane_bg.lighter(125)``: both clip to white on any pale ``lane_bg``,
#: so a near-white lane makes gutter, ruler and lanes one flat sheet (measured
#: — that was the first cut of this palette).  ``lane_bg`` is therefore a clear
#: step *below* white, which reads the way the dark set does: gutter and ruler
#: are chrome, the lanes are wells cut into it.  ``wave_bg`` keeps its dark-set
#: relationship too — a shade LIGHTER than the lane, so the waveform strip is a
#: panel inside the well rather than a second colour of well.
TIMELINE_MODES: dict[str, dict[str, str]] = {
    "dark": {
        "lane_bg": "#12171b",
        "lane_edge": "#3a4650",
        "wave": "#5fc3f0",
        "wave_bg": "#17232c",
        "sel": "#2f7fd0",
        "playhead": "#ffffff",
        "splice": "#f0a848",
        "ruler_text": "#9fb0bd",
        "armed": "#2f7fd0",
        "hole": "#2a3138",
        # Block selection: a near-white cyan reads over both a dark thumbnail
        # and the blue waveform, which the plain accent does not.
        "block_sel": "#8fe3ff",
        "group": "#f0a848",
    },
    "light": {
        "lane_bg": "#c6d3e0",
        "lane_edge": "#8fa6ba",
        "wave": "#2f7fd0",
        "wave_bg": "#d6e1ec",
        "sel": "#2f7fd0",
        "playhead": "#16212c",
        "splice": "#c2761a",
        "ruler_text": "#4a5c6e",
        "armed": "#2f7fd0",
        "hole": "#aab8c6",
        "block_sel": "#0b3f6e",
        "group": "#c2761a",
    },
}


def _shade(colour: str, percent: int) -> str:
    """``percent`` > 100 lightens, < 100 darkens (QColor's own semantics)."""
    c = QColor(colour)
    out = c.lighter(percent) if percent >= 100 else c.darker(round(10000 / percent))
    return out.name()


def _rgba(colour: str, alpha: float) -> str:
    """``colour`` as a QSS ``rgba()`` string — QSS has no opacity property."""
    c = QColor(colour)
    return f"rgba({c.red()}, {c.green()}, {c.blue()}, {alpha:.3f})"


# --------------------------------------------------------------------------- #
# transport glyphs
# --------------------------------------------------------------------------- #
#: Each glyph as unit-square polygons (x, y in 0..1), painted at any size.
_GLYPHS: dict[str, tuple[tuple[tuple[float, float], ...], ...]] = {
    "play": (((0.26, 0.12), (0.84, 0.50), (0.26, 0.88)),),
    "pause": (
        ((0.28, 0.14), (0.44, 0.14), (0.44, 0.86), (0.28, 0.86)),
        ((0.56, 0.14), (0.72, 0.14), (0.72, 0.86), (0.56, 0.86)),
    ),
    "back": (
        ((0.52, 0.14), (0.52, 0.86), (0.06, 0.50)),
        ((0.96, 0.14), (0.96, 0.86), (0.50, 0.50)),
    ),
    "forward": (
        ((0.48, 0.14), (0.48, 0.86), (0.94, 0.50)),
        ((0.04, 0.14), (0.04, 0.86), (0.50, 0.50)),
    ),
    "prev": (
        ((0.06, 0.12), (0.20, 0.12), (0.20, 0.88), (0.06, 0.88)),
        ((0.94, 0.14), (0.94, 0.86), (0.26, 0.50)),
    ),
    "next": (
        ((0.80, 0.12), (0.94, 0.12), (0.94, 0.88), (0.80, 0.88)),
        ((0.06, 0.14), (0.06, 0.86), (0.74, 0.50)),
    ),
}


def media_icon(kind: str, colour: str, px: int = 14) -> QIcon:
    """A transport glyph drawn at ``px`` in ``colour``.

    Painted rather than loaded so the ink is ours: WMP12 wants a *blue* play
    triangle on the glass dome and slate skip glyphs on the pills, which no
    platform icon theme would give us.  Code, not assets.
    """
    size = max(8, int(px))
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QColor(colour))
    for polygon in _GLYPHS[kind]:
        painter.drawPolygon(QPolygonF([QPointF(x * size, y * size) for x, y in polygon]))
    painter.end()
    return QIcon(pixmap)


#: The palette Qt handed us before we touched anything; "system" restores it.
_system_palette: QPalette | None = None


def _palette(variant: str) -> QPalette:
    """Build a full QPalette from the ``variant`` token set."""
    t = TOKENS[variant]
    window, base, text = QColor(t["window"]), QColor(t["base"]), QColor(t["text"])
    dim, accent = QColor(t["dim"]), QColor(t["accent"])
    pal = QPalette()
    role = QPalette.ColorRole
    group = QPalette.ColorGroup
    for r, c in (
        (role.Window, window),
        (role.WindowText, text),
        (role.Base, base),
        (role.AlternateBase, window),
        (role.Text, text),
        (role.Button, window),
        (role.ButtonText, text),
        (role.ToolTipBase, base),
        (role.ToolTipText, text),
        (role.BrightText, QColor("#ff5555")),
        (role.Link, accent),
        (role.Highlight, accent),
        (role.HighlightedText, QColor(t["accent_text"])),
        (role.PlaceholderText, dim),
    ):
        pal.setColor(r, c)
    for r in (role.WindowText, role.Text, role.ButtonText):
        pal.setColor(group.Disabled, r, dim)
    return pal


def variant(app: object | None) -> str:
    """``"dark"`` or ``"light"``: which chrome reads on the *current* palette.

    In System mode that is the desktop's own palette, so the chrome follows it
    instead of guessing.
    """
    pal = app.palette() if app is not None else QPalette()  # type: ignore[attr-defined]
    return "dark" if pal.color(QPalette.ColorRole.Window).lightness() < 128 else "light"


# --------------------------------------------------------------------------- #
# stylesheet
# --------------------------------------------------------------------------- #
def stylesheet(variant: str) -> str:
    """QSS for ``"dark"`` or ``"light"`` (anything else falls back to dark).

    One sheet dresses the whole application — window, menu strip, transport,
    dialogs and status strip — so a dialog cannot drift out of the skin.  The
    timeline widget paints itself from :data:`TIMELINE_MODES` and is
    deliberately untouched here, as are the per-lane gain sliders in its gutter.
    """
    t = TOKENS[variant if variant in TOKENS else "dark"]
    m = METRICS
    play, skip, rad = m["play"], m["skip"], m["radius"]

    # -- reusable fragments --------------------------------------------------
    # WMP12 gloss: bright top, a *soft* step just past the middle, darker foot.
    glass = (
        f"qlineargradient(x1:0, y1:0, x2:0, y2:1,"
        f" stop:0 {t['g1']}, stop:0.49 {t['g2']},"
        f" stop:0.51 {t['g3']}, stop:1 {t['g4']})"
    )
    glass_hover = (
        f"qlineargradient(x1:0, y1:0, x2:0, y2:1,"
        f" stop:0 {_shade(t['g1'], 106)}, stop:0.49 {_shade(t['g2'], 106)},"
        f" stop:0.51 {_shade(t['hover'], 108)}, stop:1 {t['hover']})"
    )
    glass_press = (
        f"qlineargradient(x1:0, y1:0, x2:0, y2:1,"
        f" stop:0 {t['g4']}, stop:0.5 {t['g3']}, stop:1 {t['g2']})"
    )
    ground = (
        f"qlineargradient(x1:0, y1:0, x2:0, y2:1,"
        f" stop:0 {t['panel1']}, stop:1 {t['panel2']})"
    )
    strip = (
        f"qlineargradient(x1:0, y1:0, x2:0, y2:1,"
        f" stop:0 {t['menu1']}, stop:1 {t['menu2']})"
    )
    field = (
        f"qlineargradient(x1:0, y1:0, x2:0, y2:1,"
        f" stop:0 {_shade(t['base'], 94)}, stop:0.16 {t['base']}, stop:1 {t['base']})"
    )
    # The dome: white crown, a soft step to pale blue, then a sliver of
    # reflected light along the very bottom.  That last stop is the whole trick.
    dome = (
        f"qlineargradient(x1:0, y1:0, x2:0, y2:1,"
        f" stop:0 {t['play_face1']},"
        f" stop:0.46 {_shade(t['play_face1'], 97)},"
        f" stop:0.50 {_shade(t['play_face2'], 118)},"
        f" stop:0.88 {_shade(t['play_face2'], 86)},"
        f" stop:1 {_shade(t['play_face2'], 122)})"
    )
    dome_hover = (
        f"qlineargradient(x1:0, y1:0, x2:0, y2:1,"
        f" stop:0 #ffffff,"
        f" stop:0.46 {t['play_face1']},"
        f" stop:0.50 {_shade(t['play_face2'], 132)},"
        f" stop:0.88 {_shade(t['play_face2'], 96)},"
        f" stop:1 {_shade(t['play_face2'], 140)})"
    )
    dome_press = (
        f"qlineargradient(x1:0, y1:0, x2:0, y2:1,"
        f" stop:0 {_shade(t['play_face2'], 82)},"
        f" stop:0.5 {_shade(t['play_face2'], 100)},"
        f" stop:1 {_shade(t['play_face1'], 96)})"
    )

    return f"""
/* ---- window ground ------------------------------------------------------ */
QMainWindow, QWidget#central {{ background: {ground}; }}
QMainWindow::separator {{ background: {t['border']}; width: 1px; height: 1px; }}
QSplitter::handle:vertical {{ height: 7px; background: transparent; }}
QSplitter::handle:vertical:hover {{ background: {_rgba(t['accent'], 0.22)}; }}
QToolTip {{
    background: {t['base']}; color: {t['text']};
    border: 1px solid {t['border']}; padding: 3px 5px;
}}

/* ---- menu strip: flat, light, one hairline underneath ------------------- */
QMenuBar {{
    background: {strip};
    border-bottom: 1px solid {t['border']};
    padding: 1px 4px; spacing: 1px; color: {t['text']};
}}
QMenuBar::item {{
    background: transparent; padding: 3px 9px; border-radius: 3px;
    color: {t['text']};
}}
QMenuBar::item:selected {{
    background: {glass_hover};
    border: 1px solid {_rgba(t['accent'], 0.55)};
    padding: 2px 8px;                 /* the border eats a pixel each side */
}}
QMenuBar::item:pressed {{
    background: {t['sel']}; border: 1px solid {t['accent']}; padding: 2px 8px;
}}
QMenu {{
    background: {strip}; border: 1px solid {t['border']};
    padding: 3px; color: {t['text']};
}}
QMenu::item {{ padding: 4px 26px 4px 22px; border-radius: 3px; }}
QMenu::item:selected {{
    background: {glass_hover}; border: 1px solid {_rgba(t['accent'], 0.55)};
    padding: 3px 25px 3px 21px;
}}
QMenu::item:disabled {{ color: {t['dim']}; }}
QMenu::separator {{ height: 1px; background: {t['border']}; margin: 3px 6px; }}

/* ---- status strip: thin, lit, its own hairline on top ------------------- */
QStatusBar {{
    background: {strip};
    border-top: 1px solid {t['border']};
    color: {t['dim']}; min-height: 19px;
}}
QStatusBar::item {{ border: 0px; }}
QStatusBar QLabel {{ background: transparent; color: {t['dim']}; padding: 0 6px; }}
QLabel#statusState {{ color: {t['text']}; font-weight: 600; padding-left: 7px; }}
QFrame#statusSep {{ background: {t['border']}; max-width: 1px; }}

/* ---- generic controls, kept tight -------------------------------------- */
QGroupBox {{
    border: 1px solid {t['border']}; border-radius: {rad}px;
    margin-top: 10px; padding: 4px 2px 2px 2px; font-weight: 600;
}}
QGroupBox::title {{
    subcontrol-origin: margin; subcontrol-position: top left;
    left: 7px; padding: 0 3px; color: {t['dim']};
}}
QLabel {{ background: transparent; color: {t['text']}; }}
QLabel#hintLabel {{ color: {t['dim']}; }}
QLabel#sectionLabel {{ font-weight: 600; color: {t['text']}; }}
QLabel#warnLabel {{
    color: {t['warn_fg']}; background: {t['warn_bg']};
    border: 1px solid {t['warn_border']}; border-radius: {rad}px; padding: 3px 6px;
}}
QLabel#timecode {{
    font-family: monospace; padding: 0 6px; color: {t['text']};
}}
QDialog {{ background: {ground}; }}

/* Win7 button: hairline border, white inner crown, blue glow on hover. */
QPushButton {{
    color: {t['text']};
    border: 1px solid {t['border']};
    border-top-color: {t['hilite']};
    border-radius: {rad}px;
    padding: {m['pad']}px 11px; min-height: 18px;
    background: {glass};
}}
QPushButton:hover {{
    background: {glass_hover}; border-color: {_rgba(t['accent'], 0.75)};
}}
QPushButton:pressed {{
    background: {glass_press}; border-top-color: {t['gloss_edge']};
}}
QPushButton:default {{ border-color: {t['accent']}; }}
QPushButton:disabled {{
    color: {t['dim']}; background: {t['bar']};
    border-color: {t['border']}; border-top-color: {t['border']};
}}
QPushButton#primary {{ font-weight: 600; padding: {m['pad']}px 13px; }}

/* QSpinBox is deliberately left to the platform style: ANY rule matching it
   switches Qt to the stylesheet painter for its subcontrols too, and with no
   arrow image to hand that turns the up/down buttons into a pair of stray
   lines (measured on Fusion).  A native spin box next to a styled combo is a
   far smaller mismatch than that. */
QComboBox, QLineEdit {{
    color: {t['text']}; background: {field};
    border: 1px solid {t['border']}; border-radius: 3px;
    padding: 2px 5px; min-height: 17px;
    selection-background-color: {t['accent']};
    selection-color: {t['accent_text']};
}}
QComboBox:hover, QLineEdit:hover {{ border-color: {_rgba(t['accent'], 0.7)}; }}
QComboBox:focus, QLineEdit:focus {{ border-color: {t['accent']}; }}
QComboBox:disabled, QLineEdit:disabled {{
    color: {t['dim']}; background: {t['bar']};
}}
/* NOTE: ::drop-down / ::down-arrow are deliberately NOT styled.  Qt6 renders a
   QSS border-triangle as a filled rectangle (measured on Fusion), and a styled
   drop-down with no arrow image loses the arrow entirely — so the frame is ours
   and the arrow stays the platform style's. */
QComboBox QAbstractItemView {{
    background: {t['base']}; color: {t['text']};
    border: 1px solid {t['border']};
    selection-background-color: {t['sel']};
    selection-color: {t['text']};
}}
QProgressBar {{
    border: 1px solid {t['border']}; border-radius: 3px;
    background: {t['sunken']}; color: {t['text']}; text-align: center;
}}
QProgressBar::chunk {{
    border-radius: 2px; margin: 1px;
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 {_shade(t['accent'], 145)}, stop:0.48 {_shade(t['accent'], 112)},
        stop:0.52 {t['accent']}, stop:1 {_shade(t['accent'], 82)});
}}

/* ---- the player: video stage + glass control panel ---------------------- */
QFrame#playerPanel {{
    background: {glass};
    border: 1px solid {t['gloss_edge']};
    border-radius: {rad}px;
}}
QFrame#videoFrame {{
    background: #000;
    border: 0px; border-bottom: 1px solid {t['gloss_edge']};
    border-top-left-radius: {rad - 1}px; border-top-right-radius: {rad - 1}px;
}}
QFrame#transportBar {{
    background: {glass};
    border: 0px; border-top: 1px solid {t['hilite']};
    border-bottom-left-radius: {rad - 1}px; border-bottom-right-radius: {rad - 1}px;
}}
QFrame#transportBar QLabel {{ background: transparent; }}

/* Skip buttons: small flat pills.  Nearly invisible at rest, a soft blue
   glow under the pointer — WMP12 keeps the emphasis on the dome. */
QToolButton[transport="skip"] {{
    color: {t['text']};
    min-height: {skip}px; max-height: {skip}px;
    min-width: {skip + 6}px; max-width: {skip + 6}px;
    padding: 0;
    border: 1px solid {_rgba(t['gloss_edge'], 0.45)};
    border-top-color: {_rgba(t['hilite'], 0.6)};
    border-radius: {skip // 2 + 1}px;   /* the 1px border rounds the box, not the content */
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 {_rgba(t['g1'], 0.55)}, stop:0.5 {_rgba(t['g2'], 0.35)},
        stop:1 {_rgba(t['g4'], 0.30)});
}}
QToolButton[transport="skip"]:hover {{
    background: {glass_hover}; border-color: {_rgba(t['accent'], 0.8)};
}}
QToolButton[transport="skip"]:pressed {{
    background: {glass_press}; border-top-color: {t['gloss_edge']};
}}
QToolButton[transport="skip"]:disabled {{
    background: transparent; border-color: {_rgba(t['border'], 0.5)};
}}

/* The dome: the focal point of the whole window.  A blue rim, a white crown
   and a sliver of reflected light along the bottom edge. */
QToolButton#playButton {{
    min-width: {play}px; max-width: {play}px;
    min-height: {play}px; max-height: {play}px;
    border-radius: {play // 2 + 1}px;
    border: 1px solid {t['play_rim']};
    background: {dome};
}}
QToolButton#playButton:hover {{
    background: {dome_hover}; border-color: {_shade(t['play_rim'], 118)};
}}
QToolButton#playButton:pressed {{
    background: {dome_press}; border-color: {_shade(t['play_rim'], 84)};
}}
QToolButton#playButton:disabled {{
    background: {t['bar']}; border-color: {t['border']};
}}

QSlider#volumeSlider::groove:horizontal {{
    height: 4px; border-radius: 2px;
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 {_shade(t['seek_track'], 88)}, stop:1 {_shade(t['seek_track'], 112)});
}}
QSlider#volumeSlider::sub-page:horizontal {{
    height: 4px; border-radius: 2px;
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 {_shade(t['seek_fill'], 135)}, stop:1 {t['seek_fill']});
}}
QSlider#volumeSlider::handle:horizontal {{
    width: 9px; margin: -4px 0; border-radius: 5px;
    border: 1px solid {t['play_rim']};
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 {t['play_face1']}, stop:1 {t['play_face2']});
}}
QSlider#volumeSlider::handle:horizontal:hover {{ border-color: {t['accent']}; }}
"""


def apply(app: object, mode: str) -> str:
    """Apply theme ``mode`` to ``app`` live.  Returns the effective variant."""
    global _system_palette
    if _system_palette is None:
        _system_palette = QPalette(app.palette())  # type: ignore[attr-defined]
    if mode in TOKENS:
        app.setPalette(_palette(mode))  # type: ignore[attr-defined]
    else:  # "system", or anything unknown
        app.setPalette(QPalette(_system_palette))  # type: ignore[attr-defined]
    look = variant(app)
    app.setStyleSheet(stylesheet(look))  # type: ignore[attr-defined]
    return look
