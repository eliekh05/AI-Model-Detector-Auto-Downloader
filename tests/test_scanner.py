"""Tests for the system scanner module."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ai_model_detector.scanner import scan_system, SystemProfile, CPUProfile, RAMProfile


def test_live_scan_returns_profile():
    profile = scan_system()
    assert isinstance(profile, SystemProfile)
    assert profile.cpu.cores_logical >= 1
    assert profile.ram.total_gb > 0
    assert profile.os_name != ""
    assert profile.source == "live_scan"


def test_cpu_profile_fields():
    profile = scan_system()
    cpu = profile.cpu
    assert isinstance(cpu, CPUProfile)
    assert isinstance(cpu.supports_avx, bool)
    assert isinstance(cpu.supports_avx2, bool)
    assert cpu.cores_physical >= 1


def test_ram_profile_fields():
    profile = scan_system()
    ram = profile.ram
    assert isinstance(ram, RAMProfile)
    assert ram.total_gb > 0
    assert ram.available_gb >= 0


def test_disk_profile():
    profile = scan_system()
    assert profile.disk is not None
    assert profile.disk.free_gb >= 0


def test_to_dict():
    profile = scan_system()
    d = profile.to_dict()
    assert "cpu" in d
    assert "ram" in d
    assert "os_name" in d
