"""予測ビューア HTML 生成ツール."""

from tools.generate_viewer.builder import (
    build_viewer_html,
    load_metadata,
    load_predictions,
    resolve_batter,
    select_samples,
)

__all__ = [
    "build_player_names",
    "build_viewer_html",
    "load_metadata",
    "load_predictions",
    "resolve_batter",
    "select_samples",
]


def __getattr__(name: str):
    if name == "build_player_names":
        from tools.generate_viewer.metadata import build_player_names

        return build_player_names
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
