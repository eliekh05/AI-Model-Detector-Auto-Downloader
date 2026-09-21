"""
scorer.py — Hardware-aware model scoring engine.

Architecture: two-layer evaluation
  Layer 1 — Hardware Compatibility: can this model physically run?
  Layer 2 — Quality Ranking: among runnable models, which is best?

A model that fails Layer 1 is not recommended, only listed as incompatible.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum

from .registry import ModelInfo
from .scanner import GPUDevice, SystemProfile

logger = logging.getLogger(__name__)


# ── Enums ──────────────────────────────────────────────────────────────────────

class RAMFit(Enum):
    FIT       = "FIT"         # comfortably fits with headroom
    TIGHT     = "TIGHT"       # fits but barely; expect swapping under load
    RISKY     = "RISKY"       # may fit depending on context length / overhead
    OVER      = "DOES_NOT_FIT"# will not fit
    UNKNOWN   = "UNKNOWN"     # no size data available

class GPUTier(Enum):
    DISCRETE_CUDA   = "discrete_cuda"    # NVIDIA with dedicated VRAM
    DISCRETE_ROCM   = "discrete_rocm"    # AMD with dedicated VRAM
    APPLE_SILICON   = "apple_silicon"    # M-series unified memory
    INTEGRATED      = "integrated"       # Intel/AMD iGPU — shared system RAM
    NONE            = "none"             # no GPU detected

class Confidence(Enum):
    HIGH    = "High"
    MEDIUM  = "Medium"
    LOW     = "Low"
    UNKNOWN = "Unknown"


# ── RAM budget calculator ──────────────────────────────────────────────────────

# Overhead constants (all in GB) — based on empirical llama.cpp / Ollama data
_OLLAMA_BASE_OVERHEAD_GB   = 0.25   # Ollama server + runtime base
_KV_CACHE_PER_GB_MODEL     = 0.12   # typical KV cache as fraction of model size
_ACTIVATION_OVERHEAD_GB    = 0.10   # activation buffers
_OS_RESERVED_GB            = 0.50   # OS keeps ~512 MB for itself under load
_SAFETY_HEADROOM_GB        = 0.25   # conservative safety margin

# Quantization RAM multipliers relative to fp16 (≈ 2 bytes/param)
# These express (disk_bytes × multiplier) ≈ RAM_needed for inference
_QUANT_RAM_MULTIPLIER: dict[str, float] = {
    "f32":    2.00,   # unlikely to see in practice
    "f16":    1.00,   # baseline
    "bf16":   1.00,
    "q8_0":   0.52,
    "q6_k":   0.42,
    "q5_k_m": 0.36,
    "q5_k_s": 0.35,
    "q5_0":   0.34,
    "q4_k_m": 0.30,
    "q4_k_s": 0.29,
    "q4_0":   0.28,
    "q3_k_m": 0.22,
    "q3_k_s": 0.21,
    "q2_k":   0.16,
    "iq4_xs": 0.27,
    "gguf":   0.30,   # generic GGUF unknown quant — assume q4-ish
    "unknown":0.30,
}


def _ram_budget(model: ModelInfo, profile: SystemProfile) -> tuple[RAMFit, float, float, str]:
    """
    Calculate realistic RAM budget for running this model.

    Returns:
        (RAMFit, estimated_model_ram_gb, total_required_gb, calculation_note)
    """
    size_gb = model.size_gb

    if size_gb <= 0:
        # No size data — cannot estimate
        return RAMFit.UNKNOWN, 0.0, 0.0, "Model size unknown — cannot estimate RAM requirement."

    quant = model.quantization.lower()
    multiplier = _QUANT_RAM_MULTIPLIER.get(quant, _QUANT_RAM_MULTIPLIER["unknown"])

    # Model weights in RAM (quantized models load at their quantized size)
    model_ram = size_gb * multiplier

    # KV cache scales with model size
    kv_cache = size_gb * _KV_CACHE_PER_GB_MODEL

    # Total inference requirement
    total_required = (
        model_ram
        + kv_cache
        + _ACTIVATION_OVERHEAD_GB
        + _OLLAMA_BASE_OVERHEAD_GB
        + _OS_RESERVED_GB
        + _SAFETY_HEADROOM_GB
    )

    total_ram   = profile.ram.total_gb
    avail_ram   = profile.ram.available_gb

    note = (
        f"model weights ~{model_ram:.1f} GB "
        f"+ KV cache ~{kv_cache:.1f} GB "
        f"+ overhead ~{_ACTIVATION_OVERHEAD_GB + _OLLAMA_BASE_OVERHEAD_GB:.1f} GB "
        f"+ OS/safety ~{_OS_RESERVED_GB + _SAFETY_HEADROOM_GB:.1f} GB "
        f"= ~{total_required:.1f} GB total"
    )

    if total_required <= avail_ram:
        return RAMFit.FIT, model_ram, total_required, note
    elif total_required <= avail_ram * 1.25:
        return RAMFit.TIGHT, model_ram, total_required, note
    elif total_required <= total_ram * 0.90:
        # Fits in total RAM but not in currently available — might work with page-out
        return RAMFit.RISKY, model_ram, total_required, note
    else:
        return RAMFit.OVER, model_ram, total_required, note


# ── GPU tier classifier ────────────────────────────────────────────────────────

def _classify_gpu(gpus: list[GPUDevice], os_name: str, os_arch: str) -> tuple[GPUTier, float]:
    """
    Classify the GPU tier and return (GPUTier, usable_vram_gb).

    Intel/AMD integrated GPUs share system RAM — they do NOT have
    dedicated VRAM usable for model weights independently.
    Apple Silicon uses unified memory but Metal acceleration is real.
    """
    if not gpus:
        return GPUTier.NONE, 0.0

    # Apple Silicon detection — architecture is arm64 + Darwin
    is_apple_silicon = (
        os_name == "Darwin"
        and "arm" in os_arch.lower()
    )
    if is_apple_silicon:
        # Unified memory — the whole RAM pool is usable via Metal
        # We report 0.0 dedicated VRAM but the tier enables GPU scoring
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
    return GPUTier.INTEGRATED if best_vram == 0 else GPUTier.DISCRETE_CUDA, best_vram


# ── Token speed estimator ──────────────────────────────────────────────────────

def _estimate_tokens_per_sec(
    model: ModelInfo,
    profile: SystemProfile,
    gpu_tier: GPUTier,
    ram_fit: RAMFit,
) -> tuple[float, Confidence]:
    """
    Rough token/sec estimate based on CPU speed, model size, and RAM fit.
    Returns (tokens_per_sec, confidence).

    Formula grounded in llama.cpp benchmarks on similar hardware:
      - Intel i5-8257U (4C/8T, 1.4 GHz base, ~3.8 GHz boost) with AVX2
      - ~8-12 tok/s for 3B Q4, ~4-6 tok/s for 7B Q4 (CPU only)
      - Integrated GPU (Metal on Intel) gives very marginal speedup for llama models
    """
    if model.size_gb <= 0:
        return 0.0, Confidence.UNKNOWN

    if ram_fit == RAMFit.OVER:
        return 0.0, Confidence.HIGH  # won't run at all

    # Baseline: tokens/sec for a 1B parameter Q4_K_M model on this CPU
    # Derived from: clock speed × cores × AVX2 throughput factor
    freq_ghz = max(profile.cpu.frequency_max_mhz / 1000.0, 1.0) if profile.cpu.frequency_max_mhz > 0 else 2.0
    cores    = profile.cpu.cores_physical or 4

    # Base throughput in "compute units" — empirically calibrated
    if profile.cpu.supports_avx2:
        avx_factor = 1.0
    elif profile.cpu.supports_avx:
        avx_factor = 0.7
    else:
        avx_factor = 0.4

    cpu_base = freq_ghz * min(cores, 8) * avx_factor * 2.5  # tok/s for 1B Q4

    # Adjust for model size (larger models are slower)
    # Approximate: tok/s ∝ 1 / (param_count)^0.9
    # Derive approximate param count from size_gb at Q4 (~0.5 bytes/param)
    approx_params_b = model.size_gb / 0.5  # billions
    if approx_params_b <= 0:
        approx_params_b = 7.0  # default assumption

    size_penalty = max(0.05, 1.0 / (approx_params_b ** 0.85))
    base_tps = cpu_base * size_penalty

    # RAM pressure penalty
    if ram_fit == RAMFit.RISKY:
        base_tps *= 0.40   # heavy paging
    elif ram_fit == RAMFit.TIGHT:
        base_tps *= 0.75   # some pressure

    # GPU tier adjustments
    if gpu_tier == GPUTier.APPLE_SILICON:
        # Metal on Apple Silicon is significant — ~3-5x vs CPU
        base_tps *= 3.5
        confidence = Confidence.MEDIUM
    elif gpu_tier in (GPUTier.DISCRETE_CUDA, GPUTier.DISCRETE_ROCM):
        base_tps *= 8.0
        confidence = Confidence.MEDIUM
    elif gpu_tier == GPUTier.INTEGRATED:
        # Intel/AMD iGPU with Metal/Vulkan: marginal gain for most models
        # llama.cpp with Metal on Intel Iris: ~10-20% speedup at best
        base_tps *= 1.10
        confidence = Confidence.LOW
    else:
        confidence = Confidence.LOW

    # Quantization speedup
    quant = model.quantization.lower()
    if quant in ("q4_k_m", "q4_0", "q4_k_s"):
        pass  # baseline
    elif quant in ("q8_0", "f16", "f32"):
        base_tps *= 0.55   # heavier
    elif quant in ("q2_k", "q3_k_m"):
        base_tps *= 1.20   # lighter but lower quality

    return round(base_tps, 1), confidence


# ── Main scoring function ──────────────────────────────────────────────────────

@dataclass
class ScoredModel:
    model: ModelInfo
    score: float                    # 0..100 — composite score
    ram_fit: RAMFit                 # FIT / TIGHT / RISKY / OVER / UNKNOWN
    fits_vram: bool
    fits_disk: bool
    gpu_tier: GPUTier
    will_use_gpu: bool
    estimated_tps: float            # tokens per second (0 = unknown)
    tps_confidence: Confidence
    estimated_total_ram_gb: float   # total RAM this model needs
    ram_budget_note: str            # human-readable calculation
    confidence: Confidence          # overall recommendation confidence
    explanation: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    disqualified: bool = False      # True = hardware cannot run this model

    # Legacy compat
    @property
    def fits_ram(self) -> bool:
        return self.ram_fit in (RAMFit.FIT, RAMFit.TIGHT)

    @property
    def estimated_speed(self) -> str:
        if self.estimated_tps <= 0:
            return "unknown"
        if self.estimated_tps >= 10:
            return "fast"
        if self.estimated_tps >= 4:
            return "medium"
        if self.estimated_tps >= 1:
            return "slow"
        return "very slow"


def score_model(model: ModelInfo, profile: SystemProfile) -> ScoredModel:
    """
    Two-layer evaluation:
      Layer 1 — Hardware Compatibility Gate
      Layer 2 — Quality/Usefulness Score
    """
    disk_free = profile.disk.free_gb if profile.disk else 999.0
    gpu_tier, usable_vram = _classify_gpu(
        profile.gpus, profile.os_name, profile.os_arch
    )

    explanation: list[str] = []
    warnings:    list[str] = []
    score = 50.0
    disqualified = False

    # ── Layer 1A: Disk ────────────────────────────────────────────────────────
    fits_disk = (disk_free >= model.size_gb * 1.1) if model.size_gb > 0 else True
    if not fits_disk:
        warnings.append(
            f"Disk: only {disk_free:.1f} GB free, "
            f"model requires ~{model.size_gb:.1f} GB — cannot download."
        )
        disqualified = True
        score -= 40

    # ── Layer 1B: RAM budget ──────────────────────────────────────────────────
    ram_fit, _model_ram_gb, total_req_gb, ram_note = _ram_budget(model, profile)

    if ram_fit == RAMFit.UNKNOWN:
        # Unknown size — penalise, do NOT treat as compatible
        score -= 15
        warnings.append(
            "Model size unknown — RAM compatibility cannot be verified. "
            "This model is deprioritised."
        )
    elif ram_fit == RAMFit.FIT:
        headroom = profile.ram.available_gb - total_req_gb
        score += min(25, headroom * 6)
        explanation.append(
            f"RAM: estimated {total_req_gb:.1f} GB needed, "
            f"{profile.ram.available_gb:.1f} GB available — fits comfortably "
            f"({headroom:.1f} GB headroom)."
        )
    elif ram_fit == RAMFit.TIGHT:
        score += 5
        explanation.append(
            f"RAM: estimated {total_req_gb:.1f} GB needed, "
            f"{profile.ram.available_gb:.1f} GB available — tight fit."
        )
        warnings.append("May be slow or unstable under memory pressure.")
    elif ram_fit == RAMFit.RISKY:
        score -= 15
        warnings.append(
            f"RAM: estimated {total_req_gb:.1f} GB needed but only "
            f"{profile.ram.available_gb:.1f} GB currently free "
            f"({profile.ram.total_gb:.1f} GB total). "
            "May require closing other apps — expect paging/swapping."
        )
    elif ram_fit == RAMFit.OVER:
        score -= 45
        disqualified = True
        warnings.append(
            f"RAM: estimated {total_req_gb:.1f} GB needed, "
            f"only {profile.ram.total_gb:.1f} GB installed — WILL NOT FIT."
        )

    # ── Layer 1C: GPU tier / acceleration reality check ───────────────────────
    fits_vram   = False
    will_use_gpu = False

    if gpu_tier == GPUTier.NONE:
        explanation.append("No GPU detected — CPU-only inference.")
        score -= 3

    elif gpu_tier == GPUTier.INTEGRATED:
        # Intel Iris / AMD Vega iGPU: Metal/Vulkan available but iGPU
        # shares system RAM. No dedicated VRAM. Marginal speedup only.
        will_use_gpu = False
        fits_vram    = True   # no dedicated VRAM requirement
        warnings.append(
            f"GPU detected ({profile.gpus[0].name if profile.gpus else 'iGPU'}) "
            "but it is an integrated GPU sharing system RAM. "
            "GPU acceleration is unverified — treating as CPU-only for scoring."
        )
        score -= 3   # same penalty as no GPU

    elif gpu_tier == GPUTier.APPLE_SILICON:
        will_use_gpu = True
        fits_vram    = True
        score += 15
        explanation.append(
            "Apple Silicon with Metal — unified memory allows full GPU acceleration."
        )

    elif gpu_tier in (GPUTier.DISCRETE_CUDA, GPUTier.DISCRETE_ROCM):
        backend = "CUDA" if gpu_tier == GPUTier.DISCRETE_CUDA else "ROCm"
        if model.vram_required_gb > 0 and usable_vram >= model.vram_required_gb:
            fits_vram    = True
            will_use_gpu = True
            score += min(25, (usable_vram - model.vram_required_gb) * 3)
            explanation.append(
                f"{backend}: {usable_vram:.1f} GB VRAM ≥ "
                f"{model.vram_required_gb:.1f} GB required — full GPU acceleration."
            )
        elif model.vram_required_gb == 0 and model.size_gb > 0:
            # We don't know exact VRAM requirement — estimate from model size
            if usable_vram >= model.size_gb * 0.9:
                fits_vram    = True
                will_use_gpu = True
                score += 15
                explanation.append(
                    f"{backend}: {usable_vram:.1f} GB VRAM — likely fits model "
                    f"({model.size_gb:.1f} GB)."
                )
            else:
                fits_vram    = False
                will_use_gpu = False
                warnings.append(
                    f"{backend}: {usable_vram:.1f} GB VRAM may be insufficient "
                    f"for {model.size_gb:.1f} GB model — partial CPU offload."
                )
                score += 5
        else:
            fits_vram    = True
            will_use_gpu = True
            score += 10
            explanation.append(f"{backend}: GPU available — acceleration enabled.")

    # ── Layer 2A: CPU instruction sets ───────────────────────────────────────
    cpu = profile.cpu
    if cpu.supports_avx2:
        score += 6
        explanation.append("AVX2 supported — optimized quantized inference.")
    elif cpu.supports_avx:
        score += 3
        explanation.append("AVX supported (no AVX2) — basic SIMD acceleration.")
    else:
        score -= 8
        warnings.append("No AVX support — inference will be very slow.")

    # ── Layer 2B: Model size vs available hardware ────────────────────────────
    if model.size_gb > 0:
        if fits_disk:
            explanation.append(
                f"Disk: {disk_free:.1f} GB free — fits model ({model.size_gb:.1f} GB)."
            )
        # Give a bonus proportional to how WELL it fits (not just that it fits)
        # Larger models that still fit score slightly better than tiny ones
        if ram_fit in (RAMFit.FIT, RAMFit.TIGHT):
            # Prefer models that use a healthy fraction of available RAM
            # (too small = underutilising hardware; too large = risky)
            utilisation = total_req_gb / max(profile.ram.total_gb, 1.0)
            if 0.40 <= utilisation <= 0.75:
                score += 8
                explanation.append(
                    f"Good RAM utilisation ({utilisation*100:.0f}% of total) — "
                    "large enough to be capable, small enough to be safe."
                )
            elif utilisation < 0.25:
                score += 2  # very small model — runs but may lack capability

    # ── Layer 2C: Quantization quality ───────────────────────────────────────
    quant = model.quantization.lower()
    if quant in ("q4_k_m", "q4_k_s"):
        score += 8
        explanation.append(f"{model.quantization} — best quality/speed balance.")
    elif quant == "q5_k_m":
        score += 7
        explanation.append(f"{model.quantization} — high quality, moderate size.")
    elif quant == "q8_0":
        if profile.ram.total_gb >= 16:
            score += 5
            explanation.append(f"{model.quantization} — near-lossless, hardware can handle it.")
        else:
            score -= 5
            warnings.append(f"{model.quantization} — high quality but heavy; tight on 8 GB.")
    elif quant in ("q2_k", "q3_k_m", "q3_k_s"):
        score += 2
        warnings.append(f"{model.quantization} — very compressed; quality trade-offs expected.")
    elif quant == "unknown":
        score -= 5
        warnings.append("Quantization unknown — quality and speed cannot be assessed.")

    # ── Layer 2D: Token speed estimate ───────────────────────────────────────
    est_tps, tps_conf = _estimate_tokens_per_sec(model, profile, gpu_tier, ram_fit)

    if est_tps > 0:
        # Reward for usable speed
        if est_tps >= 8:
            score += 10
        elif est_tps >= 4:
            score += 6
        elif est_tps >= 2:
            score += 2
        else:
            score -= 5
            warnings.append(
                f"Estimated speed ~{est_tps:.1f} tok/s — may feel slow for interactive use."
            )

    # ── Layer 2E: Known issues ────────────────────────────────────────────────
    if model.known_issues:
        score -= len(model.known_issues) * 3
        warnings.append(f"{len(model.known_issues)} open community bug report(s).")

    # ── Layer 2F: Popularity ──────────────────────────────────────────────────
    if model.ollama_pull_count > 1_000_000:
        score += 5
        explanation.append("Very popular (1M+ pulls) — well-tested in production.")
    elif model.ollama_pull_count > 100_000:
        score += 2

    # ── Layer 2G: Installability ──────────────────────────────────────────────
    if not model.ollama_pullable:
        score -= 25
        warnings.append(
            "HuggingFace-only — cannot install via `ollama pull`. Manual download required."
        )
    else:
        score += 5
        explanation.append("Installable with `ollama pull`.")

    # ── Overall confidence ────────────────────────────────────────────────────
    data_gaps = sum([
        model.size_gb == 0,
        model.quantization == "unknown",
        model.ram_required_gb == 0,
    ])
    if disqualified:
        confidence = Confidence.HIGH   # highly confident it won't work
    elif data_gaps == 0 and ram_fit in (RAMFit.FIT, RAMFit.TIGHT):
        confidence = Confidence.HIGH
    elif data_gaps <= 1:
        confidence = Confidence.MEDIUM
    else:
        confidence = Confidence.LOW

    score = max(0.0, min(100.0, round(score, 1)))

    return ScoredModel(
        model=model,
        score=score,
        ram_fit=ram_fit,
        fits_vram=fits_vram,
        fits_disk=fits_disk,
        gpu_tier=gpu_tier,
        will_use_gpu=will_use_gpu,
        estimated_tps=est_tps,
        tps_confidence=tps_conf,
        estimated_total_ram_gb=total_req_gb,
        ram_budget_note=ram_note,
        confidence=confidence,
        explanation=explanation,
        warnings=warnings,
        disqualified=disqualified,
    )


def rank_models(
    models: list[ModelInfo],
    profile: SystemProfile,
    top_n: int = 10,
    category_filter: str | None = None,
) -> list[ScoredModel]:
    """
    Score and rank all models.
    Disqualified models (hardware can't run them) are pushed to the end.
    """
    if category_filter:
        models = [m for m in models if category_filter in m.categories]

    disk_free = profile.disk.free_gb if profile.disk else 999.0
    # Hard pre-filter: only models that can physically download
    candidates = [m for m in models if m.size_gb == 0 or m.size_gb <= disk_free * 0.95]

    logger.info("Scoring %d models (%d pre-filtered by disk)", len(candidates), len(models))

    scored = [score_model(m, profile) for m in candidates]

    # Sort: non-disqualified first, then by score descending
    scored.sort(key=lambda s: (int(s.disqualified), -s.score))
    return scored[:top_n]
