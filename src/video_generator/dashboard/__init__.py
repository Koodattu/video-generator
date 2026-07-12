"""Local dashboard for inspecting and supervising durable video-generation Runs."""

from __future__ import annotations

from .app import create_dashboard_app, run_dashboard

__all__ = ["create_dashboard_app", "run_dashboard"]
