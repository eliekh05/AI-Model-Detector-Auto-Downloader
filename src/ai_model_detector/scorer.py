def _classify_gpu(gpus: list[GPUDevice], os_name: str, os_arch: str) -> tuple[GPUTier, float]:
    """
    Classify the GPU tier and return (GPUTier, usable_vram_gb).

    Intel/AMD integrated GPUs share system RAM — they do NOT have
    dedicated VRAM usable for model weights independently.
    Apple Silicon uses unified memory but Metal acceleration is real.
    """
    if not gpus:
        return GPUTier.NONE, 0.0

    # Apple Silicon detection — architecture is arm64 + Darwin.
    is_apple_silicon = (
        os_name == "Darwin"
        and "arm" in os_arch.lower()
    )

    # Explicit GPU metadata should win over the host-platform heuristic.
    # A simulated Intel Iris iGPU on ARM macOS must not be reclassified as
    # Apple Silicon, and must not be rewarded with GPU acceleration.
    for gpu in gpus:
        name_lower = gpu.name.lower()
        if gpu.is_integrated:
            is_apple_gpu = any(
                keyword in name_lower
                for keyword in ("apple", "m1", "m2", "m3", "m4")
            )
            if not (is_apple_silicon and is_apple_gpu):
                return GPUTier.INTEGRATED, 0.0

    if is_apple_silicon:
        # Unified memory — the whole RAM pool is usable via Metal.
        # We report 0.0 dedicated VRAM but the tier enables GPU scoring.
        return GPUTier.APPLE_SILICON, 0.0

    # Check for discrete NVIDIA / AMD
    for gpu in gpus:
        if gpu.cuda_version and gpu.vram_gb and gpu.vram_gb > 0:
            return GPUTier.DISCRETE_CUDA, gpu.vram_gb
        if gpu.rocm_version and gpu.vram_gb and gpu.vram_gb > 0:
            return GPUTier.DISCRETE_ROCM, gpu.vram_gb

    # Integrated GPU detection — Intel/AMD iGPU keywords
    _INTEGRATED_KEYWORDS = (
        "iris", "uhd", "hd graphics", "radeon vega", "radeon rx vega",
        "amd radeon(tm)", "intel(r) hd", "intel(r) uhd", "intel(r) iris",
        "vega 8", "vega 11", "llano", "trinity", "kaveri", "renoir",
    )
    for gpu in gpus:
        name_lower = gpu.name.lower()
        if any(kw in name_lower for kw in _INTEGRATED_KEYWORDS):
            return GPUTier.INTEGRATED, 0.0
        # macOS Intel GPU — Metal is available but it's an iGPU
        if gpu.metal_support and (gpu.vram_gb is None or gpu.vram_gb == 0.0):
            return GPUTier.INTEGRATED, 0.0

    # Unknown GPU with no VRAM info — treat as integrated
    for gpu in gpus:
        if gpu.vram_gb is None or gpu.vram_gb == 0.0:
            return GPUTier.INTEGRATED, 0.0

    # Discrete but unknown backend
    best_vram = max((g.vram_gb or 0.0) for g in gpus)
    return (
    GPUTier.INTEGRATED if best_vram == 0 else GPUTier.DISCRETE_CUDA,
    best_vram,
  )