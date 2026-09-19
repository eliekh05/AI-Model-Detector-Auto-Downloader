"""
scorer.py — Hardware-aware model scoring engine.

Scores every model from the live registry against the user's hardware
profile to produce a ranked list with explanations. No scores or
thresholds are hardcoded — everything is computed from the system profile.
"""

import logging
from dataclasses import dataclass, field

from .registry import ModelInfo
from .scanner import SystemProfile

logger = logging.getLogger(__name__)


@dataclass
class ScoredModel:
    model: ModelInfo
    score: float                  # 0..100
    fits_ram: bool
    fits_vram: bool
    fits_disk: bool
    estimated_speed: str          # "fast" | "medium" | "slow" | "very slow"
    will_use_gpu: bool
    explanation: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _best_gpu(profile: SystemProfile) -> tuple[float, bool, bool, bool]:
    """
    Returns (max_vram_gb, has_cuda, has_metal, has_rocm).
    """
    max_vram = 0.0
    has_cuda  = False
    has_metal = False
    has_rocm  = False
    for gpu in profile.gpus:
        vram = gpu.vram_gb or 0.0
        max_vram = max(max_vram, vram)
        if gpu.cuda_version:
            has_cuda = True
        if gpu.metal_support:
            has_metal = True
        if gpu.rocm_version:
            has_rocm = True
    return max_vram, has_cuda, has_metal, has_rocm


def _speed_label(
    model: ModelInfo,
    ram_gb: float,
    vram_gb: float,
    has_gpu: bool,
) -> str:
    """Estimate inference speed given the model size and hardware."""
    if model.size_gb == 0:
        return "unknown"

    if has_gpu and vram_gb >= model.size_gb:
        ratio = vram_gb / max(model.size_gb, 0.1)
        if ratio >= 2.0:
            return "fast"
        if ratio >= 1.0:
            return "medium"
        return "slow"

    # CPU-only / partial offload
    ratio = ram_gb / max(model.size_gb, 0.1)
    if ratio >= 4.0:
        return "medium"
    if ratio >= 2.0:
        return "slow"
    return "very slow"


def score_model(model: ModelInfo, profile: SystemProfile) -> ScoredModel:
    """
    Produce a ScoredModel for a single ModelInfo against the given system profile.
    """
    ram_gb = profile.ram.total_gb
    avail_ram = profile.ram.available_gb
    disk_free = profile.disk.free_gb if profile.disk else 999.0
    max_vram, has_cuda, has_metal, has_rocm = _best_gpu(profile)
    has_gpu = has_cuda or has_metal or has_rocm

    explanation: list[str] = []
    warnings: list[str] = []
    score = 50.0   # neutral baseline

    # ── RAM fit ───────────────────────────────────────────────────────────────
    req_ram = model.ram_required_gb
    fits_ram = avail_ram >= req_ram if req_ram > 0 else True

    if req_ram == 0:
        explanation.append("RAM requirement unknown; treating as fits.")
    elif fits_ram:
        headroom = avail_ram - req_ram
        score += min(20, headroom * 3)
        explanation.append(
            f"RAM: {avail_ram:.1f} GB available ≥ {req_ram:.1f} GB required "
            f"({headroom:.1f} GB headroom)"
        )
    else:
        deficit = req_ram - avail_ram
        score -= min(40, deficit * 10)
        warnings.append(
            f"RAM: only {avail_ram:.1f} GB available, "
            f"model needs {req_ram:.1f} GB ({deficit:.1f} GB short)"
        )

    # ── VRAM fit ──────────────────────────────────────────────────────────────
    req_vram = model.vram_required_gb
    fits_vram = (max_vram >= req_vram) if (has_gpu and req_vram > 0) else True
    will_use_gpu = has_gpu and fits_vram and req_vram > 0

    if not has_gpu:
        explanation.append("No GPU detected — model will run on CPU only.")
        score -= 5  # minor penalty for CPU-only inference
    elif fits_vram:
        if req_vram > 0:
            score += min(20, (max_vram - req_vram) * 2)
            explanation.append(
                f"GPU VRAM: {max_vram:.1f} GB ≥ {req_vram:.1f} GB required — GPU acceleration enabled."
            )
        else:
            explanation.append("GPU available — partial GPU offload possible.")
            score += 5
    else:
        warnings.append(
            f"VRAM: only {max_vram:.1f} GB, model may need {req_vram:.1f} GB — "
            "will fall back to CPU for some layers."
        )
        score -= 10

    # ── Disk fit ──────────────────────────────────────────────────────────────
    fits_disk = (disk_free >= model.size_gb * 1.1) if model.size_gb > 0 else True
    if not fits_disk:
        warnings.append(
            f"Disk: only {disk_free:.1f} GB free, model requires ~{model.size_gb:.1f} GB."
        )
        score -= 30
    elif model.size_gb > 0:
        explanation.append(f"Disk: {disk_free:.1f} GB free — fits model ({model.size_gb:.1f} GB).")

    # ── CPU instruction sets ─────────────────────────────────────────────────
    cpu = profile.cpu
    if cpu.supports_avx2:
        score += 5
        explanation.append("CPU supports AVX2 — optimized quantized inference.")
    elif cpu.supports_avx:
        score += 2
        explanation.append("CPU supports AVX (not AVX2) — basic SIMD acceleration.")
    else:
        score -= 5
        warnings.append("CPU has no AVX support — inference may be very slow.")

    # ── Architecture-specific boosts ─────────────────────────────────────────
    arch = profile.os_arch.lower()
    is_apple_silicon = profile.os_name == "Darwin" and "arm" in arch

    if is_apple_silicon and has_metal:
        score += 10
        explanation.append("Apple Silicon + Metal detected — excellent unified memory bandwidth.")

    # ── Quantization suitability ─────────────────────────────────────────────
    quant = model.quantization.lower()
    if quant in ("q4_k_m", "q4_0", "q4_k_s"):
        score += 8
        explanation.append(f"Quantization {model.quantization} is the sweet spot for quality/speed.")
    elif quant in ("q8_0", "f16"):
        if ram_gb >= 16 or max_vram >= 8:
            score += 4
            explanation.append(f"High-precision {model.quantization} suits your hardware.")
        else:
            score -= 8
            warnings.append(f"{model.quantization} needs more RAM/VRAM for smooth inference.")
    elif quant in ("q2_k", "q3_k_m", "q3_k_s"):
        score += 3
        explanation.append(f"{model.quantization} is very compressed — expect quality trade-offs.")

    # ── Known issues ─────────────────────────────────────────────────────────
    if model.known_issues:
        score -= len(model.known_issues) * 3
        warnings.append(f"{len(model.known_issues)} open bug reports found for this model.")

    # ── Popularity bonus ──────────────────────────────────────────────────────
    if model.ollama_pull_count > 1_000_000:
        score += 5
        explanation.append("Very popular model (1M+ pulls) — well-tested.")
    elif model.ollama_pull_count > 100_000:
        score += 2

    # ── Installability — Ollama-pullable models get a strong bonus ────────────
    # HuggingFace-only models cannot be installed with `ollama pull` and require
    # manual download steps, so we rank them well below Ollama library models.
    if not model.ollama_pullable:
        score -= 25
        warnings.append(
            "HuggingFace-only model — cannot be installed via `ollama pull`. "
            "Manual download required (see https://huggingface.co)."
        )
    else:
        score += 8
        explanation.append("Installable with a single `ollama pull` command.")

    score = max(0.0, min(100.0, round(score, 1)))
    speed = _speed_label(model, avail_ram, max_vram, has_gpu)

    return ScoredModel(
        model=model,
        score=score,
        fits_ram=fits_ram,
        fits_vram=fits_vram,
        fits_disk=fits_disk,
        estimated_speed=speed,
        will_use_gpu=will_use_gpu,
        explanation=explanation,
        warnings=warnings,
    )


def rank_models(
    models: list[ModelInfo],
    profile: SystemProfile,
    top_n: int = 10,
    category_filter: str | None = None,
) -> list[ScoredModel]:
    """
    Score and rank all models against the hardware profile.
    Optionally filter by category (e.g. 'code', 'vision', 'chat').
    Returns top_n results sorted by score descending.
    """
    if category_filter:
        models = [m for m in models if category_filter in m.categories]

    # Skip models we know won't fit disk
    disk_free = profile.disk.free_gb if profile.disk else 999.0
    candidates = [m for m in models if m.size_gb == 0 or m.size_gb <= disk_free * 0.9]

    logger.info(
        "Scoring %d models (filtered from %d by disk constraint)",
        len(candidates), len(models)
    )

    scored = [score_model(m, profile) for m in candidates]
    scored.sort(key=lambda s: s.score, reverse=True)
    return scored[:top_n]
