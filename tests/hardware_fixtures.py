"""Shared hardware fixtures for compatibility regression tests."""

from __future__ import annotations

from ai_model_detector.registry import ModelInfo
from ai_model_detector.scanner import (
    CPUProfile,
    DiskProfile,
    GPUDevice,
    RAMProfile,
    SystemProfile,
)


def profile_intel_mac_8gb(
    available_gb: float = 2.2,
    total_gb: float = 8.0,
) -> SystemProfile:
    """
    Intel MacBook-class profile matching the reported test machine:
      - 8 GB total RAM / ~2.2 GB available
      - Intel i5-8257U, AVX2
      - Intel Iris Plus Graphics 645 (integrated)
    """
    return SystemProfile(
        os_name="Darwin",
        os_version="14.0",
        os_arch="x86_64",
        cpu=CPUProfile(
            brand="Intel(R) Core(TM) i5-8257U CPU @ 1.40GHz",
            cores_physical=4,
            cores_logical=8,
            frequency_max_mhz=3400.0,
            architecture="x86_64",
            supports_avx=True,
            supports_avx2=True,
            supports_avx512=False,
            supports_f16c=True,
        ),
        ram=RAMProfile(total_gb=total_gb, available_gb=available_gb, speed_mhz=2133),
        gpus=[
            GPUDevice(
                name="Intel Iris Plus Graphics 645",
                vram_gb=0.0,
                driver_version=None,
                metal_support=True,
                cuda_version=None,
                rocm_version=None,
                vulkan_support=False,
                is_integrated=True,
            )
        ],
        disk=DiskProfile(free_gb=120.0, total_gb=256.0, mount="/"),
        ollama_installed=True,
        ollama_version="0.5.0",
        source="test_fixture",
    )


def make_model(
    name: str,
    tag: str,
    *,
    full_tag: str | None = None,
    size_gb: float = 0.0,
    ram_required_gb: float = 0.0,
    vram_required_gb: float = 0.0,
    quantization: str = "unknown",
    ollama_pullable: bool = True,
    ollama_pull_count: int = 1000,
    known_issues: list[str] | None = None,
    categories: list[str] | None = None,
    description: str = "test model",
) -> ModelInfo:
    return ModelInfo(
        name=name,
        tag=tag,
        full_tag=full_tag or f"{name}:{tag}",
        size_gb=size_gb,
        ram_required_gb=ram_required_gb,
        vram_required_gb=vram_required_gb,
        quantization=quantization,
        description=description,
        categories=categories or ["chat"],
        known_issues=known_issues or [],
        ollama_pullable=ollama_pullable,
        ollama_pull_count=ollama_pull_count,
        source="ollama" if ollama_pullable else "huggingface",
    )
