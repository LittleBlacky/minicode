"""CSS styles for MiniCode TUI - Dark professional theme."""
from __future__ import annotations

from minicode.tui.themes.dark import dark_theme as theme

# Dark theme CSS
DARK_CSS = f"""
/* ============================================
   MiniCode TUI - Dark Professional Theme
   Inspired by GitHub Dark and Claude Code
   ============================================ */

/* Screen / Root */
Screen {{
    background: {theme.BACKGROUND};
    color: {theme.TEXT};
}}

/* ============================================
   Header
   ============================================ */
#header {{
    background: {theme.HEADER_BG};
    color: {theme.HEADER_TEXT};
    dock: top;
    height: auto;
    padding: 0 1;
    border-bottom: solid {theme.HEADER_BORDER};
}}

/* Title bar area */
#title-bar {{
    height: 1;
    layout: horizontal;
    background: {theme.HEADER_BG};
}}

/* Logo and title */
#logo {{
    width: auto;
    color: {theme.SUCCESS};
    text-style: bold;
    padding: 0 1;
}}

#model-name {{
    width: auto;
    color: {theme.TEXT_MUTED};
    padding: 0 1;
}}

#status-indicator {{
    width: auto;
    color: {theme.STATUS_ONLINE};
    padding: 0 1;
}}

/* Window controls */
#window-controls {{
    width: auto;
    layout: horizontal;
    align: right;
}}

.window-btn {{
    width: 3;
    height: 1;
    background: transparent;
    color: {theme.TEXT_MUTED};
}}

.window-btn:hover {{
    background: {theme.SURFACE_HOVER};
    color: {theme.TEXT};
}}

.window-btn.close:hover {{
    background: {theme.ERROR};
    color: white;
}}

/* ============================================
   Main Container (Horizontal Layout)
   ============================================ */
#main-container {{
    layout: horizontal;
    height: 1fr;
    width: 100%;
}}

/* ============================================
   Message Area
   ============================================ */
#message-area {{
    width: 1fr;
    height: 100%;
    background: {theme.BACKGROUND};
    border: solid {theme.BORDER};
    border-title-color: {theme.TEXT_MUTED};
    margin: 1 1 1 2;
    padding: 0 1;
}}

#message-scroll {{
    height: 1fr;
    scrollbar-color: {theme.SCROLLBAR_COLOR} {theme.SCROLLBAR_BG};
}}

/* Message bubbles */
.message-bubble {{
    width: 100%;
    height: auto;
    margin: 1 0;
    padding: 1 2;
    border: solid {theme.BORDER};
    background: {theme.SURFACE};
}}

.message-bubble.user {{
    border-title-color: {theme.USER_BORDER};
    border-title-style: bold;
}}

.message-bubble.agent {{
    border-title-color: {theme.AGENT_BORDER};
    border-title-style: bold;
}}

.message-bubble.tool {{
    border-title-color: {theme.TOOL_BORDER};
    background: {theme.TOOL_BG};
}}

.message-bubble.error {{
    border-title-color: {theme.ERROR};
    background: {theme.ERROR_BG};
}}

/* Message animations - handled via Textual API */
/* Note: CSS @keyframes not supported in Textual */

/* Code blocks */
.code-block {{
    background: {theme.CODE_BG};
    border: solid {theme.CODE_BORDER};
    margin: 1 0;
    padding: 1;
}}

/* ============================================
   Sidebar / Tool Panel
   ============================================ */
#sidebar {{
    width: 28;
    max-width: 40;
    min-width: 20;
    height: 100%;
    background: {theme.SIDEBAR_BG};
    border: solid {theme.SIDEBAR_BORDER};
    margin: 1 2 1 1;
}}

#sidebar-header {{
    height: auto;
    background: {theme.SURFACE};
    color: {theme.TEXT};
    text-style: bold;
    padding: 1 2;
    border-bottom: solid {theme.BORDER};
}}

#sidebar-tabs {{
    height: 3;
    layout: horizontal;
    background: {theme.SURFACE};
    border-bottom: solid {theme.BORDER};
}}

.sidebar-tab {{
    width: 1fr;
    height: 100%;
    background: {theme.SURFACE};
    color: {theme.TEXT_MUTED};
    content-align: center middle;
}}

.sidebar-tab.active {{
    background: {theme.SIDEBAR_BG};
    color: {theme.TEXT};
    border-bottom: tall {theme.PRIMARY};
}}

.sidebar-tab:hover {{
    background: {theme.SURFACE_HOVER};
}}

#sidebar-content {{
    height: 1fr;
    padding: 1;
}}

/* Tool items */
.tool-item {{
    height: auto;
    width: 100%;
    margin: 0 0 1 0;
    padding: 0 1;
    background: {theme.SURFACE};
    border: solid {theme.BORDER};
}}

.tool-item:hover {{
    background: {theme.SURFACE_HOVER};
}}

.tool-item.running {{
    border-left: solid 3 {theme.WARNING};
}}

.tool-item.success {{
    border-left: solid 3 {theme.SUCCESS};
}}

.tool-item.error {{
    border-left: solid 3 {theme.ERROR};
}}

.tool-name {{
    color: {theme.PRIMARY};
    text-style: bold;
}}

/* Tool status - pulse animation not supported in Textual CSS */

.tool-status {{
    color: {theme.TEXT_MUTED};
}}

.tool-status.running::before {{
    content: "●";
    /* Animation handled via Textual API */
}}

/* Command history items */
.history-item {{
    height: auto;
    width: 100%;
    margin: 0 0 1 0;
    padding: 0 1;
    color: {theme.TEXT_MUTED};
}}

.history-item:hover {{
    background: {theme.SURFACE_HOVER};
    color: {theme.TEXT};
}}

.history-item.selected {{
    background: {theme.PRIMARY_HOVER}40;
    color: {theme.TEXT};
}}

/* ============================================
   Input Area
   ============================================ */
#input-container {{
    height: auto;
    min-height: 3;
    max-height: 10;
    background: {theme.SURFACE};
    border-top: solid {theme.BORDER};
    padding: 1 2;
}}

#input-row {{
    layout: horizontal;
    height: auto;
}}

#prompt-indicator {{
    width: auto;
    padding: 0 1;
    color: {theme.PROMPT_TEXT};
    text-style: bold;
}}

#prompt-indicator.user {{ color: {theme.PROMPT_USER}; }}
#prompt-indicator.tool {{ color: {theme.PROMPT_TOOL}; }}

Input {{
    width: 1fr;
    height: auto;
    background: {theme.BACKGROUND};
    color: {theme.TEXT};
    border: solid {theme.BORDER};
    margin: 0 1;
}}

Input:focus {{
    border: solid {theme.FOCUS};
}}

Input::placeholder {{
    color: {theme.TEXT_DIM};
}}

/* ============================================
   Status Bar
   ============================================ */
#status-bar {{
    height: auto;
    background: {theme.SURFACE};
    color: {theme.FOOTER_TEXT};
    dock: bottom;
    padding: 0 2;
    border-top: solid {theme.BORDER};
}}

#status-left {{
    width: auto;
}}

#status-center {{
    width: 1fr;
    align: center middle;
}}

#status-right {{
    width: auto;
}}

.status-hint {{
    color: {theme.TEXT_DIM};
}}

.status-value {{
    color: {theme.TEXT};
}}

/* ============================================
   Command Palette (Overlay)
   ============================================ */
#command-palette {{
    width: 60;
    height: auto;
    max-height: 20;
    background: {theme.SURFACE};
    border: solid {theme.BORDER};
    box-shadow: 0 4 8 {theme.BACKGROUND}80;
}}

#command-palette-input {{
    background: {theme.BACKGROUND};
    border: none;
    border-bottom: solid {theme.BORDER};
    padding: 1 2;
}}

#command-list {{
    height: 1fr;
    background: {theme.SURFACE};
}}

.command-item {{
    height: auto;
    padding: 0 2;
}}

.command-item:hover {{
    background: {theme.SURFACE_HOVER};
}}

.command-item.selected {{
    background: {theme.PRIMARY_HOVER}40;
}}

.command-name {{
    color: {theme.PRIMARY};
    text-style: bold;
}}

.command-desc {{
    color: {theme.TEXT_MUTED};
}}

.command-key {{
    color: {theme.TEXT_DIM};
    align: right;
}}

/* ============================================
   Completion Popup
   ============================================ */
#completion-popup {{
    width: auto;
    max-width: 40;
    height: auto;
    max-height: 15;
    background: {theme.SURFACE};
    border: solid {theme.BORDER};
    box-shadow: 0 4 8 {theme.BACKGROUND}80;
}}

.completion-item {{
    height: auto;
    padding: 0 1;
}}

.completion-item:hover,
.completion-item.selected {{
    background: {theme.PRIMARY_HOVER}40;
}}

.completion-type {{
    color: {theme.TEXT_DIM};
    width: 6;
}}

.completion-value {{
    color: {theme.TEXT};
}}

/* ============================================
   Footer / Help Bar
   ============================================ */
#footer {{
    dock: bottom;
    height: 1;
    background: {theme.FOOTER_BG};
    color: {theme.FOOTER_TEXT};
    padding: 0 2;
}}

.footer-hint {{
    color: {theme.TEXT_DIM};
}}

.footer-key {{
    color: {theme.PRIMARY};
    text-style: bold;
}}

/* ============================================
   Scrollbar styling
   ============================================ */
ScrollableContainer > .scrollbar-grip {{
    background: {theme.SCROLLBAR_COLOR};
}}

ScrollableContainer > .scrollbar-grip:hover {{
    background: {theme.PRIMARY};
}}

/* ============================================
   Loading / Thinking indicator
   ============================================ */
#thinking-indicator {{
    color: {theme.WARNING};
    text-style: italic;
}}

/* Thinking dots animation - handled via Textual API */

.thinking {{
    color: {theme.WARNING};
}}

.thinking::after {{
    content: "...";
    /* Animation handled via Textual API */
}}

/* ============================================
   ASCII Art Container
   ============================================ */
#ascii-art {{
    width: 100%;
    height: auto;
    content-align: center middle;
    color: {theme.SUCCESS};
    text-style: bold;
    padding: 0;
    margin: 0;
}}

#ascii-art.thinking {{
    color: {theme.WARNING};
}}

/* ============================================
   Notification / Toast
   ============================================ */
.notification {{
    width: auto;
    max-width: 50;
    height: auto;
    background: {theme.SURFACE};
    border: solid {theme.BORDER};
    padding: 1 2;
    box-shadow: 0 4 8 {theme.BACKGROUND}80;
}}

.notification.success {{
    border-left: solid 3 {theme.SUCCESS};
}}

.notification.error {{
    border-left: solid 3 {theme.ERROR};
}}

.notification.warning {{
    border-left: solid 3 {theme.WARNING};
}}

.notification.info {{
    border-left: solid 3 {theme.PRIMARY};
}}

/* ============================================
   Dialog / Modal
   ============================================ */
Dialog {{
    background: {theme.SURFACE};
    border: solid {theme.BORDER};
    box-shadow: 0 8 16 {theme.BACKGROUND}80;
}}

Dialog > .dialog-window {{
    background: {theme.SURFACE};
    border: solid {theme.BORDER};
    padding: 2;
}}

#dialog-title {{
    text-style: bold;
    color: {theme.TEXT};
    padding: 0 0 1 0;
}}

#dialog-body {{
    color: {theme.TEXT};
    padding: 1 0;
}}

#dialog-buttons {{
    layout: horizontal;
    height: auto;
    align: right;
    padding: 1 0 0 0;
}}

/* ============================================
   Progress indicators
   ============================================ */
ProgressBar {{
    color: {theme.PRIMARY};
    background: {theme.SURFACE_HOVER};
}}

ProgressBar > .progress-bar {{
    color: {theme.SUCCESS};
}}

/* ============================================
   Tooltip
   ============================================ */
Tooltip {{
    background: {theme.SURFACE};
    color: {theme.TEXT};
    border: solid {theme.BORDER};
    padding: 0 1;
}}
"""


def get_theme_css(theme_name: str = "dark") -> str:
    """Get CSS for the specified theme.

    Args:
        theme_name: Name of the theme ('dark' currently supported)

    Returns:
        CSS string for the theme
    """
    if theme_name == "dark":
        return DARK_CSS
    # Future themes can be added here
    return DARK_CSS


def get_inline_styles(element: str) -> dict:
    """Get inline styles for a specific element.

    Args:
        element: Element name (e.g., 'header', 'sidebar')

    Returns:
        Dict of CSS property-value pairs
    """
    styles = {
        "header": {
            "background": theme.HEADER_BG,
            "color": theme.HEADER_TEXT,
            "border_bottom": f"solid 1px {theme.HEADER_BORDER}",
        },
        "sidebar": {
            "background": theme.SIDEBAR_BG,
            "border": f"solid 1px {theme.SIDEBAR_BORDER}",
        },
        "input": {
            "background": theme.BACKGROUND,
            "color": theme.TEXT,
            "border": f"solid 1px {theme.BORDER}",
        },
        "button": {
            "background": theme.SURFACE_HOVER,
            "color": theme.TEXT,
            "border": f"solid 1px {theme.BORDER}",
        },
        "button:hover": {
            "background": theme.SURFACE_ACTIVE,
        },
        "button.primary": {
            "background": theme.PRIMARY,
            "color": theme.TEXT_INVERSE,
        },
        "link": {
            "color": theme.LINK,
        },
        "link:hover": {
            "color": theme.LINK_HOVER,
        },
    }
    return styles.get(element, {})
