"""Smoke tests for live scanner (best-effort; skips if OS APIs fail)."""

from __future__ import annotations

from ai_model_detector.scanner import scan_system


def test_scan_system_returns_profile():
    profile = scan_system()
    assert profile.os_name
    assert profile.os_arch
    assert profile.cpu.brand
    assert profile.cpu.cores_physical >= 1
    assert profile.ram.total_gb > 0
    assert profile.ram.available_gb >= 0
    assert profile.source in ("live_scan", "spx_import")
