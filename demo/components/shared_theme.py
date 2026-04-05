"""Shared visual identity for frontend demos."""

THEME = {
    "primary_color": "#1E3A5F",
    "accent_color": "#00B4D8",
    "danger_color": "#E63946",
    "warning_color": "#F4A261",
    "success_color": "#2A9D8F",
    "background": "#0A1628",
    "text_color": "#E2E8F0",
}

RISK_COLORS = {
    "CRITICAL": "#E63946",
    "HIGH": "#E63946",
    "MEDIUM": "#F4A261",
    "LOW": "#2A9D8F",
}

RISK_EMOJI = {
    "CRITICAL": "🔴",
    "HIGH": "🔴",
    "MEDIUM": "🟡",
    "LOW": "🟢",
}


def risk_color(level: str) -> str:
    return RISK_COLORS.get(str(level).upper(), "#94A3B8")


def risk_emoji(level: str) -> str:
    return RISK_EMOJI.get(str(level).upper(), "⚪")
