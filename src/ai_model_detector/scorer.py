"""
scorer.py — Hardware-aware model compatibility evaluator.

Architecture: factual multi-signal evaluation (no universal numeric score)
  1. Installability — can Ollama pull the model? (pullable ≠ runnable)
  2. Runtime compatibility — does the detected runtime support it?
  3. Memory fit — FITS / TIGHT / RISKY / DOES_NOT_FIT / UNKNOWN
  4. Acceleration — usable LLM accel established (not merely GPU present)?
  5. Performance — estimated / inferred / unknown (never claimed measured unless measured)
  6. Recommendations — explainable categories based on explicit evidence

Unknown size/RAM metadata never becomes a confirmed FITS. Unknown models stay
discoverable but are labeled UNVERIFIED and ranked below verified fits.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum

from .registry import ModelInfo
from .scanner import GPUDevice, SystemProfile

logger = logging.getLogger(__name__)


# ── Enums ──────────────────────────────────────────────────────────────────────


class RAMFit(Enum):
    FITS = "FITS"  # comfortably fits with headroom
    TIGHT = "TIGHT"  # fits but barely; expect pressure under load
    RISKY = "RISKY"  # may fit depending on context length / closing apps
    OVER = "DOES_NOT_FIT"  # will not fit
    UNKNOWN = "UNKNOWN"  # insufficient metadata — not a confirmed fit


class GPUTier(Enum):
    DISCRETE_CUDA = "discrete_cuda"  # NVIDIA with dedicated VRAM
    DISCRETE_ROCM = "discrete_rocm"  # AMD with dedicated VRAM
    APPLE_SILICON = "apple_silicon"  # M-series unified memory
    INTEGRATED = "integrated"  # Intel/AMD iGPU — shared system RAM
    NONE = "none"  # no GPU detected


class AccelerationStatus(Enum):
    """Usable LLM acceleration — distinct from mere GPU detection."""

    CUDA = "cuda"
    ROCM = "rocm"
    METAL_APPLE = "metal_apple"  # Apple Silicon — Metal established
    CPU_ONLY = "cpu_only"
    UNVERIFIED = "unverified"  # GPU present but backend use not established


class PerformanceBasis(Enum):
    MEASURED = "measured"
    ESTIMATED = "estimated"
    INFERRED = "inferred"
    UNKNOWN = "unknown"


class Confidence(Enum):
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"
    UNKNOWN = "Unknown"


class RecommendationLabel(Enum):
    """Explainable recommendation categories — evidence-based, not numeric scores."""

    BEST_FIT = "Best fit"
    LOWEST_MEMORY = "Lowest memory requirement"
    FASTEST_ESTIMATED = "Fastest estimated"
    BEST_QUALITY_THAT_FITS = "Best quality among models that fit"
    CODING = "Coding"
    REASONING = "Reasoning"
    GENERAL_CHAT = "General chat"
    EXPERIMENTAL = "Experimental"
    NOT_RECOMMENDED = "Not recommended"


# ── Constants ──────────────────────────────────────────────────────────────────

_OLLAMA_BASE_OVERHEAD_GB = 0.25
_ACTIVATION_OVERHEAD_GB = 0.15
# OS reserve: modern macOS/Linux manage memory dynamically; 0.30 GB is a
# conservative floor rather than a hard reservation.
_OS_RESERVED_GB = 0.30
_SAFETY_HEADROOM_GB = 0.25
_DEFAULT_CONTEXT_TOKENS = 4096
# Integrated GPUs (and Apple unified) share system RAM. The reserve accounts
# for GPU driver overhead and framebuffer — 0.50 GB is conservative for most
# iGPUs; Apple Silicon uses unified memory so overhead is lower in practice.
_IGPU_SHARED_RESERVE_GB = 0.50

# Bytes-per-parameter on disk for common quants (GGUF-ish averages).
# Used only to *infer* size from a parameter-count tag when size_gb is missing.
# Never presented as a confirmed measured size.
_BYTES_PER_PARAM: dict[str, float] = {
    "f32": 4.00,
    "f16": 2.00,
    "bf16": 2.00,
    "q8_0": 1.05,
    "q6_k": 0.80,
    "q5_k_m": 0.70,
    "q5_k_s": 0.68,
    "q5_0": 0.65,
    "q4_k_m": 0.55,
    "q4_k_s": 0.53,
    "q4_0": 0.50,
    "q3_k_m": 0.40,
    "q3_k_s": 0.38,
    "q2_k": 0.30,
    "iq4_xs": 0.48,
    "gguf": 0.55,
    "unknown": 0.55,  # assume Q4-class only for inference heuristics
}


# ── Metadata helpers ───────────────────────────────────────────────────────────


def parse_param_count_b(model: ModelInfo) -> float | None:
    """
    Extract approximate parameter count (billions) from tag/name when present.
    Examples: '30b', '3.2b', '7b-instruct', 'llama3.2:1b'.
    Returns None when no reliable signal exists — never invents a count.
    """
    text = f"{model.tag} {model.name} {model.full_tag}".lower()
    match = re.search(r"(?<![a-z0-9])(\d+(?:\.\d+)?)\s*b(?:illion)?(?![a-z0-9])", text)
    if not match:
        # Also accept patterns glued to quant: 30b-q4_k_m
        match = re.search(r"(?<![a-z0-9])(\d+(?:\.\d+)?)b(?=[-_]|$)", text)
    if not match:
        return None
    value = float(match.group(1))
    # Guard against nonsense (e.g. matching years); LLMs today are < 1000B
    if value <= 0 or value > 1000:
        return None
    return value


def _bytes_per_param(quant: str) -> float:
    return _BYTES_PER_PARAM.get(quant.lower(), _BYTES_PER_PARAM["unknown"])


def _infer_size_gb_from_params(params_b: float, quant: str) -> float:
    """Heuristic disk/weights size from params × bytes/param. Low-confidence only."""
    return round(params_b * _bytes_per_param(quant), 2)


def _kv_cache_gb(params_b: float | None, context_tokens: int = _DEFAULT_CONTEXT_TOKENS) -> float:
    """
    Rough KV-cache estimate.
    Empirically ~0.05 GB per billion params at 2k context for typical decoder models;
    scales roughly linearly with context.
    """
    if params_b is None or params_b <= 0:
        return 0.25  # minimal placeholder when params unknown but size known
    return round(params_b * 0.05 * (context_tokens / 2048.0), 2)


def _shared_gpu_reserve(gpu_tier: GPUTier) -> float:
    if gpu_tier in (GPUTier.INTEGRATED, GPUTier.APPLE_SILICON):
        return _IGPU_SHARED_RESERVE_GB
    return 0.0


# ── RAM budget calculator ──────────────────────────────────────────────────────


@dataclass
class MemoryEstimate:
    ram_fit: RAMFit
    model_weights_gb: float
    total_required_gb: float
    note: str
    confidence: Confidence
    source: str  # "size_metadata" | "param_inference" | "none"
    missing: list[str] = field(default_factory=list)


def estimate_memory(model: ModelInfo, profile: SystemProfile, gpu_tier: GPUTier) -> MemoryEstimate:
    """
    Estimate memory requirement from available metadata.

    - size_gb > 0: treat as quantized on-disk weights (do NOT re-apply quant shrink).
    - size_gb == 0 but params known: infer a *heuristic* size for warnings only;
      fit category remains UNKNOWN (never a confirmed FITS).
    - neither: UNKNOWN with no numeric estimate.
    """
    params_b = parse_param_count_b(model)
    missing: list[str] = []
    if model.size_gb <= 0:
        missing.append("disk/weight size")
    if model.ram_required_gb <= 0:
        missing.append("stated RAM requirement")
    if params_b is None:
        missing.append("parameter count")
    if model.quantization.lower() in ("unknown",):
        missing.append("quantization")

    overhead = (
        _ACTIVATION_OVERHEAD_GB
        + _OLLAMA_BASE_OVERHEAD_GB
        + _OS_RESERVED_GB
        + _SAFETY_HEADROOM_GB
        + _shared_gpu_reserve(gpu_tier)
    )

    # ── Path A: known size metadata ─────────────────────────────────────────
    if model.size_gb > 0:
        weights = model.size_gb  # already quantized GGUF/disk size ≈ weights in RAM
        if params_b is None and model.size_gb > 0:
            # Derive rough params for KV estimate only (Q4-ish fallback)
            params_b = model.size_gb / _bytes_per_param(model.quantization)
        kv = _kv_cache_gb(params_b)
        total = weights + kv + overhead
        note = (
            f"weights ~{weights:.1f} GB (from size metadata) "
            f"+ KV cache ~{kv:.1f} GB (@{_DEFAULT_CONTEXT_TOKENS} ctx) "
            f"+ overhead/safety ~{overhead:.1f} GB "
            f"= ~{total:.1f} GB total"
        )
        fit = _classify_fit(total, profile)
        return MemoryEstimate(
            ram_fit=fit,
            model_weights_gb=weights,
            total_required_gb=round(total, 2),
            note=note,
            confidence=Confidence.MEDIUM if model.quantization.lower() != "unknown" else Confidence.LOW,
            source="size_metadata",
            missing=missing,
        )

    # ── Path B: size unknown, params known — infer for warnings, keep UNKNOWN ─
    if params_b is not None:
        inferred_weights = _infer_size_gb_from_params(params_b, model.quantization)
        kv = _kv_cache_gb(params_b)
        total = inferred_weights + kv + overhead
        note = (
            f"size metadata missing; inferred weights ~{inferred_weights:.1f} GB "
            f"from ~{params_b:g}B params × {model.quantization} heuristic "
            f"+ KV ~{kv:.1f} GB + overhead ~{overhead:.1f} GB "
            f"≈ ~{total:.1f} GB (UNVERIFIED — not a confirmed requirement)"
        )
        return MemoryEstimate(
            ram_fit=RAMFit.UNKNOWN,
            model_weights_gb=inferred_weights,
            total_required_gb=round(total, 2),
            note=note,
            confidence=Confidence.LOW,
            source="param_inference",
            missing=missing,
        )

    # ── Path C: nothing usable ──────────────────────────────────────────────
    return MemoryEstimate(
        ram_fit=RAMFit.UNKNOWN,
        model_weights_gb=0.0,
        total_required_gb=0.0,
        note="Model size and parameter count unknown — cannot estimate RAM requirement.",
        confidence=Confidence.UNKNOWN,
        source="none",
        missing=missing or ["disk/weight size", "parameter count"],
    )


def _classify_fit(total_required: float, profile: SystemProfile) -> RAMFit:
    """Map required GB onto FITS / TIGHT / RISKY / DOES_NOT_FIT using total vs available RAM."""
    total_ram = profile.ram.total_gb
    avail_ram = profile.ram.available_gb

    if total_required <= avail_ram:
        return RAMFit.FITS
    if total_required <= avail_ram * 1.25:
        return RAMFit.TIGHT
    if total_required <= total_ram * 0.90:
        # Fits in installed RAM but not currently free — may work after freeing memory
        return RAMFit.RISKY
    return RAMFit.OVER


def _ram_budget(model: ModelInfo, profile: SystemProfile) -> tuple[RAMFit, float, float, str]:
    """Legacy tuple API used by existing tests."""
    gpu_tier, _ = _classify_gpu(profile.gpus, profile.os_name, profile.os_arch)
    est = estimate_memory(model, profile, gpu_tier)
    return est.ram_fit, est.model_weights_gb, est.total_required_gb, est.note


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

    is_apple_silicon = os_name == "Darwin" and "arm" in os_arch.lower()

    # Explicit GPU metadata should win over the host-platform heuristic.
    for gpu in gpus:
        name_lower = gpu.name.lower()
        if gpu.is_integrated:
            is_apple_gpu = any(keyword in name_lower for keyword in ("apple", "m1", "m2", "m3", "m4"))
            if not (is_apple_silicon and is_apple_gpu):
                return GPUTier.INTEGRATED, 0.0

    if is_apple_silicon:
        return GPUTier.APPLE_SILICON, 0.0

    for gpu in gpus:
        if gpu.cuda_version and gpu.vram_gb and gpu.vram_gb > 0:
            return GPUTier.DISCRETE_CUDA, gpu.vram_gb
        if gpu.rocm_version and gpu.vram_gb and gpu.vram_gb > 0:
            return GPUTier.DISCRETE_ROCM, gpu.vram_gb

    _INTEGRATED_KEYWORDS = (
        "iris",
        "uhd",
        "hd graphics",
        "radeon vega",
        "radeon rx vega",
        "amd radeon(tm)",
        "intel(r) hd",
        "intel(r) uhd",
        "intel(r) iris",
        "vega 8",
        "vega 11",
        "llano",
        "trinity",
        "kaveri",
        "renoir",
    )
    for gpu in gpus:
        name_lower = gpu.name.lower()
        if any(kw in name_lower for kw in _INTEGRATED_KEYWORDS):
            return GPUTier.INTEGRATED, 0.0
        if gpu.metal_support and (gpu.vram_gb is None or gpu.vram_gb == 0.0):
            return GPUTier.INTEGRATED, 0.0

    for gpu in gpus:
        if gpu.vram_gb is None or gpu.vram_gb == 0.0:
            return GPUTier.INTEGRATED, 0.0

    best_vram = max((g.vram_gb or 0.0) for g in gpus)
    return (
        GPUTier.INTEGRATED if best_vram == 0 else GPUTier.DISCRETE_CUDA,
        best_vram,
    )


def acceleration_for_tier(
    gpu_tier: GPUTier,
    os_name: str = "",
    os_arch: str = "",
    metal_available: bool = False,
) -> AccelerationStatus:
    # Parameters kept for API compatibility; Metal on Intel is not established as LLM backend.
    """Map detection tier → whether LLM acceleration is actually established.

    Metal API availability on Intel Macs does NOT mean Ollama/llama.cpp uses
    the iGPU for inference. Ollama only establishes Metal acceleration on
    Apple Silicon. Intel iGPUs are detected but not confirmed as backends.
    """
    if gpu_tier == GPUTier.APPLE_SILICON:
        return AccelerationStatus.METAL_APPLE
    if gpu_tier == GPUTier.DISCRETE_CUDA:
        return AccelerationStatus.CUDA
    if gpu_tier == GPUTier.DISCRETE_ROCM:
        return AccelerationStatus.ROCM
    if gpu_tier == GPUTier.INTEGRATED:
        # GPU detected but Ollama/llama.cpp backend use is not established.
        # Metal API presence on Intel does not mean the backend uses it.
        return AccelerationStatus.UNVERIFIED
    return AccelerationStatus.CPU_ONLY


# ── Token speed estimator ──────────────────────────────────────────────────────


def _estimate_tokens_per_sec(
    model: ModelInfo,
    profile: SystemProfile,
    gpu_tier: GPUTier,
    ram_fit: RAMFit,
    params_b: float | None,
    effective_size_gb: float,
) -> tuple[float, Confidence, PerformanceBasis]:
    if ram_fit == RAMFit.OVER:
        return 0.0, Confidence.HIGH, PerformanceBasis.ESTIMATED

    if effective_size_gb <= 0 and params_b is None:
        return 0.0, Confidence.UNKNOWN, PerformanceBasis.UNKNOWN

    freq_ghz = max(profile.cpu.frequency_max_mhz / 1000.0, 1.0) if profile.cpu.frequency_max_mhz > 0 else 2.0
    cores = profile.cpu.cores_physical or 4

    if profile.cpu.supports_avx2:
        avx_factor = 1.0
    elif profile.cpu.supports_avx:
        avx_factor = 0.7
    else:
        avx_factor = 0.4

    cpu_base = freq_ghz * min(cores, 8) * avx_factor * 2.5  # tok/s for ~1B Q4

    if params_b and params_b > 0:
        approx_params_b = params_b
        basis = PerformanceBasis.INFERRED if effective_size_gb <= 0 else PerformanceBasis.ESTIMATED
    else:
        approx_params_b = max(effective_size_gb / 0.55, 0.5)
        basis = PerformanceBasis.ESTIMATED

    size_penalty = max(0.05, 1.0 / (approx_params_b**0.85))
    base_tps = cpu_base * size_penalty

    if ram_fit == RAMFit.RISKY:
        base_tps *= 0.40
    elif ram_fit == RAMFit.TIGHT:
        base_tps *= 0.75
    elif ram_fit == RAMFit.UNKNOWN:
        base_tps *= 0.85  # uncertain — don't over-claim speed

    accel = acceleration_for_tier(gpu_tier, profile.os_name, profile.os_arch, profile.metal_available)
    if accel == AccelerationStatus.METAL_APPLE:
        base_tps *= 3.5
        confidence = Confidence.MEDIUM
    elif accel in (AccelerationStatus.CUDA, AccelerationStatus.ROCM):
        base_tps *= 8.0
        confidence = Confidence.MEDIUM
    elif accel == AccelerationStatus.UNVERIFIED:
        # Detected iGPU — treat as CPU-only; do not claim acceleration speedup
        confidence = Confidence.LOW
        basis = PerformanceBasis.INFERRED if basis == PerformanceBasis.UNKNOWN else basis
    else:
        confidence = Confidence.LOW

    quant = model.quantization.lower()
    if quant in ("q8_0", "f16", "f32", "bf16"):
        base_tps *= 0.55
    elif quant in ("q2_k", "q3_k_m", "q3_k_s"):
        base_tps *= 1.20

    if ram_fit == RAMFit.UNKNOWN and params_b is None:
        return 0.0, Confidence.UNKNOWN, PerformanceBasis.UNKNOWN

    return round(base_tps, 1), confidence, basis


# ── Compatibility evaluation ───────────────────────────────────────────────────


@dataclass
class EvaluatedModel:
    """Compatibility assessment for one model on one hardware profile."""

    model: ModelInfo
    ram_fit: RAMFit
    fits_vram: bool
    fits_disk: bool
    gpu_tier: GPUTier
    will_use_gpu: bool
    estimated_tps: float
    tps_confidence: Confidence
    estimated_total_ram_gb: float
    ram_budget_note: str
    confidence: Confidence
    explanation: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    disqualified: bool = False

    # Separated concerns
    installable: bool = True  # pullable via ollama
    runtime_compatible: bool = True
    acceleration: AccelerationStatus = AccelerationStatus.CPU_ONLY
    gpu_detected: bool = False
    performance_basis: PerformanceBasis = PerformanceBasis.UNKNOWN
    verified: bool = False  # True only when size metadata supports a real fit assessment
    unverified: bool = True
    missing_metadata: list[str] = field(default_factory=list)
    params_b: float | None = None
    memory_confidence: Confidence = Confidence.UNKNOWN
    rank_reason: str = ""
    labels: list[RecommendationLabel] = field(default_factory=list)

    @property
    def fits_ram(self) -> bool:
        return self.ram_fit in (RAMFit.FITS, RAMFit.TIGHT)

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

    @property
    def is_safe_install_default(self) -> bool:
        """Verified FITS/TIGHT — preferred default install (no override needed)."""
        return (
            self.installable and self.verified and not self.disqualified and self.ram_fit in (RAMFit.FITS, RAMFit.TIGHT)
        )

    @property
    def is_acceptable_install_candidate(self) -> bool:
        """
        Verified and not OVER/disqualified. Includes RISKY — still preferable to UNVERIFIED,
        but not a silent automatic default (see select_install_candidate).
        """
        return (
            self.installable
            and self.verified
            and not self.disqualified
            and self.ram_fit in (RAMFit.FITS, RAMFit.TIGHT, RAMFit.RISKY)
        )

    @property
    def label_names(self) -> list[str]:
        return [label.value for label in self.labels]


# Backward-compatible alias (no numeric score field).
ScoredModel = EvaluatedModel


def evaluate_model(model: ModelInfo, profile: SystemProfile) -> EvaluatedModel:
    """Evaluate one model against a hardware profile (facts + classifications, no score)."""
    disk_free = profile.disk.free_gb if profile.disk else 999.0
    gpu_tier, usable_vram = _classify_gpu(profile.gpus, profile.os_name, profile.os_arch)
    accel = acceleration_for_tier(gpu_tier, profile.os_name, profile.os_arch, profile.metal_available)
    gpu_detected = bool(profile.gpus)
    params_b = parse_param_count_b(model)

    explanation: list[str] = []
    warnings: list[str] = []
    disqualified = False

    mem = estimate_memory(model, profile, gpu_tier)
    ram_fit = mem.ram_fit
    total_req_gb = mem.total_required_gb
    ram_note = mem.note
    missing = list(mem.missing)
    verified = mem.source == "size_metadata"
    unverified = not verified

    # Effective size for disk / speed: prefer real metadata, else inferred heuristic
    effective_size = model.size_gb if model.size_gb > 0 else mem.model_weights_gb

    # ── Installability (pullable ≠ runnable / suitable) ─────────────────────
    installable = bool(model.ollama_pullable)
    if installable:
        explanation.append("Pullable via `ollama pull` (pullable ≠ compatible or likely to fit).")
    else:
        warnings.append("HuggingFace-only — cannot install via `ollama pull`.")

    # ── Runtime compatibility (theoretical: format is GGUF which backends support) ─
    runtime_compatible = True
    if installable:
        explanation.append("Runtime: Ollama/llama.cpp-class backend supports GGUF format.")
    else:
        explanation.append(
            "Runtime: GGUF format is theoretically supported by llama.cpp-class backends, "
            "but this model is not installed and has not been verified to run."
        )

    # ── Disk ────────────────────────────────────────────────────────────────
    if model.size_gb > 0:
        fits_disk = disk_free >= model.size_gb * 1.1
        if not fits_disk:
            warnings.append(
                f"Disk: only {disk_free:.1f} GB free, model requires ~{model.size_gb:.1f} GB — cannot download."
            )
            disqualified = True
        else:
            explanation.append(f"Disk: {disk_free:.1f} GB free — fits download ({model.size_gb:.1f} GB).")
    else:
        # Unknown size — do not claim disk fit; soft-check inferred size if any
        fits_disk = True
        if effective_size > 0 and disk_free < effective_size * 1.1:
            warnings.append(
                f"Disk: {disk_free:.1f} GB free may be insufficient for inferred ~{effective_size:.1f} GB download."
            )

    # ── Memory fit ──────────────────────────────────────────────────────────
    if ram_fit == RAMFit.UNKNOWN:
        warnings.append(
            "UNVERIFIED: size/RAM metadata incomplete — "
            + (", ".join(missing) if missing else "key fields missing")
            + ". Not a confirmed fit (UNKNOWN is never treated as FITS)."
        )
        if mem.source == "param_inference" and total_req_gb > 0:
            explanation.append(
                f"Param-based heuristic suggests ~{total_req_gb:.1f} GB may be needed "
                f"({profile.ram.available_gb:.1f} GB available / {profile.ram.total_gb:.1f} GB total) "
                f"— confidence {mem.confidence.value}."
            )
            if total_req_gb > profile.ram.total_gb:
                warnings.append(
                    f"Inferred requirement (~{total_req_gb:.1f} GB) exceeds installed RAM "
                    f"({profile.ram.total_gb:.1f} GB) — likely DOES_NOT_FIT if size were known."
                )
            elif total_req_gb > profile.ram.available_gb:
                warnings.append(
                    f"Inferred requirement (~{total_req_gb:.1f} GB) exceeds currently available RAM "
                    f"({profile.ram.available_gb:.1f} GB)."
                )
    elif ram_fit == RAMFit.FITS:
        headroom = profile.ram.available_gb - total_req_gb
        explanation.append(
            f"RAM: estimated {total_req_gb:.1f} GB needed, "
            f"{profile.ram.available_gb:.1f} GB available / {profile.ram.total_gb:.1f} GB total — "
            f"FITS ({headroom:.1f} GB headroom). Confidence: {mem.confidence.value}."
        )
    elif ram_fit == RAMFit.TIGHT:
        explanation.append(
            f"RAM: estimated {total_req_gb:.1f} GB needed, {profile.ram.available_gb:.1f} GB available — TIGHT fit."
        )
        warnings.append("May be slow or unstable under memory pressure.")
    elif ram_fit == RAMFit.RISKY:
        warnings.append(
            f"RAM: estimated {total_req_gb:.1f} GB needed but only "
            f"{profile.ram.available_gb:.1f} GB currently free "
            f"({profile.ram.total_gb:.1f} GB total). RISKY — expect paging."
        )
    elif ram_fit == RAMFit.OVER:
        disqualified = True
        warnings.append(
            f"RAM: estimated {total_req_gb:.1f} GB needed, only {profile.ram.total_gb:.1f} GB installed — DOES_NOT_FIT."
        )

    if verified and ram_fit in (RAMFit.FITS, RAMFit.TIGHT) and total_req_gb > 0:
        utilisation = total_req_gb / max(profile.ram.total_gb, 1.0)
        if 0.35 <= utilisation <= 0.70:
            explanation.append(
                f"RAM utilisation ~{utilisation * 100:.0f}% of installed — capable without exhausting the machine."
            )
        elif utilisation < 0.20:
            explanation.append("Small footprint — safe but limited capability on this hardware.")

    # ── Acceleration (detection ≠ usable LLM accel) ─────────────────────────
    fits_vram = False
    will_use_gpu = False

    if gpu_tier == GPUTier.NONE:
        explanation.append("GPU detected: no. LLM acceleration: CPU-only.")
    elif gpu_tier == GPUTier.INTEGRATED:
        gpu_name = profile.gpus[0].name if profile.gpus else "integrated GPU"
        explanation.append(
            f"GPU detected: yes ({gpu_name}) — integrated, shared system memory. "
            "LLM acceleration: not confirmed."
        )
        warnings.append(
            "Integrated GPU detected but Ollama/llama.cpp backend acceleration "
            "on this GPU is not established. The tool reports CPU-only performance."
        )
        fits_vram = True  # no discrete VRAM gate
    elif gpu_tier == GPUTier.APPLE_SILICON:
        will_use_gpu = True
        fits_vram = True
        explanation.append("GPU detected: yes (Apple Silicon). LLM acceleration: Metal (established).")
    elif gpu_tier in (GPUTier.DISCRETE_CUDA, GPUTier.DISCRETE_ROCM):
        backend = "CUDA" if gpu_tier == GPUTier.DISCRETE_CUDA else "ROCm"
        if model.vram_required_gb > 0 and usable_vram >= model.vram_required_gb:
            fits_vram = True
            will_use_gpu = True
            explanation.append(
                f"GPU detected: yes. LLM acceleration: {backend} "
                f"({usable_vram:.1f} GB VRAM ≥ {model.vram_required_gb:.1f} GB required)."
            )
        elif model.vram_required_gb == 0 and effective_size > 0:
            if usable_vram >= effective_size * 0.9:
                fits_vram = True
                will_use_gpu = True
                explanation.append(
                    f"GPU detected: yes. LLM acceleration: {backend} likely "
                    f"({usable_vram:.1f} GB VRAM vs ~{effective_size:.1f} GB weights)."
                )
            else:
                fits_vram = False
                will_use_gpu = False
                warnings.append(
                    f"{backend}: {usable_vram:.1f} GB VRAM may be insufficient "
                    f"for ~{effective_size:.1f} GB weights — expect CPU offload."
                )
        else:
            fits_vram = True
            will_use_gpu = True
            explanation.append(f"GPU detected: yes. LLM acceleration: {backend} available.")

    # ── CPU instruction sets ────────────────────────────────────────────────
    cpu = profile.cpu
    if cpu.supports_avx2:
        explanation.append("AVX2 supported — optimized quantized CPU inference.")
    elif cpu.supports_avx:
        explanation.append("AVX supported (no AVX2) — basic SIMD acceleration.")
    else:
        warnings.append("No AVX support — inference will be very slow.")

    # ── Quantization notes (factual, not scored) ────────────────────────────
    quant = model.quantization.lower()
    if quant in ("q4_k_m", "q4_k_s"):
        explanation.append(f"{model.quantization} — commonly used 4-bit GGUF quant.")
    elif quant == "q5_k_m":
        explanation.append(f"{model.quantization} — higher-bit quant; larger footprint.")
    elif quant == "q8_0":
        if profile.ram.total_gb < 16:
            warnings.append(f"{model.quantization} — heavy for ≤8 GB systems.")
        else:
            explanation.append(f"{model.quantization} — near-lossless; needs more RAM.")
    elif quant in ("q2_k", "q3_k_m", "q3_k_s"):
        warnings.append(f"{model.quantization} — aggressive compression; quality trade-offs likely.")
    elif quant == "unknown":
        warnings.append("Quantization unknown — quality and memory density cannot be assessed.")

    # ── Speed estimate (LLM tok/s — not applicable to ASR/audio/embeddings) ─
    task_cats = {c.lower() for c in model.categories}
    non_llm_tasks = task_cats & {"asr", "audio", "embeddings", "embedding", "translation"}
    if non_llm_tasks and "chat" not in task_cats and "coding" not in task_cats and "code" not in task_cats:
        est_tps, tps_conf, perf_basis = 0.0, Confidence.UNKNOWN, PerformanceBasis.UNKNOWN
        explanation.append(
            f"Performance: not estimated as chat tok/s — task categorized as "
            f"{', '.join(sorted(non_llm_tasks))}."
        )
    else:
        est_tps, tps_conf, perf_basis = _estimate_tokens_per_sec(
            model, profile, gpu_tier, ram_fit, params_b, effective_size
        )
        if est_tps > 0:
            explanation.append(
                f"Performance {perf_basis.value}: ~{est_tps:.1f} tok/s "
                f"(confidence {tps_conf.value}); not a measured benchmark."
            )
            if est_tps < 2:
                warnings.append(
                    f"Estimated speed ~{est_tps:.1f} tok/s ({perf_basis.value}) — may feel slow interactively."
                )
        else:
            warnings.append("Performance: unknown — insufficient metadata for a speed estimate.")

    # ── Known issues / popularity (weak signals, not scores) ────────────────
    if model.known_issues:
        warnings.append(f"{len(model.known_issues)} open community bug report(s).")

    if model.ollama_pull_count > 1_000_000:
        explanation.append("High pull count (1M+) — widely exercised, not a fit guarantee.")

    # ── Overall confidence ──────────────────────────────────────────────────
    if disqualified or (verified and ram_fit in (RAMFit.FITS, RAMFit.TIGHT) and not missing):
        confidence = Confidence.HIGH
    elif verified:
        confidence = Confidence.MEDIUM
    elif mem.source == "param_inference":
        confidence = Confidence.LOW
    else:
        confidence = Confidence.UNKNOWN

    rank_reason = _build_rank_reason(
        ram_fit=ram_fit,
        verified=verified,
        total_req_gb=total_req_gb,
        params_b=params_b,
        accel=accel,
        perf_basis=perf_basis,
        missing=missing,
    )

    return EvaluatedModel(
        model=model,
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
        installable=installable,
        runtime_compatible=runtime_compatible,
        acceleration=accel,
        gpu_detected=gpu_detected,
        performance_basis=perf_basis,
        verified=verified,
        unverified=unverified,
        missing_metadata=missing,
        params_b=params_b,
        memory_confidence=mem.confidence,
        rank_reason=rank_reason,
    )


# Public alias kept for call sites that still say "score" in the verb sense of evaluate.
score_model = evaluate_model


def _build_rank_reason(
    *,
    ram_fit: RAMFit,
    verified: bool,
    total_req_gb: float,
    params_b: float | None,
    accel: AccelerationStatus,
    perf_basis: PerformanceBasis,
    missing: list[str],
) -> str:
    parts: list[str] = []
    if not verified:
        parts.append("UNVERIFIED (incomplete size/RAM metadata)")
        if missing:
            parts.append("missing: " + ", ".join(missing))
    else:
        parts.append(f"memory fit {ram_fit.value}")
        if total_req_gb > 0:
            parts.append(f"est. ~{total_req_gb:.1f} GB RAM")
    if params_b is not None:
        parts.append(f"~{params_b:g}B params")
    parts.append(f"accel={accel.value}")
    parts.append(f"perf={perf_basis.value}")
    return "; ".join(parts)


def _sort_key(sm: EvaluatedModel) -> tuple:
    """
    Sort key for recommendations — classification tiers only, no numeric score.

    Verified FITS/TIGHT first; UNKNOWN/RISKY never win ties over verified fits;
    disqualified last. Within a tier: better fit class, then capability / memory evidence.
    """
    if sm.disqualified or sm.ram_fit == RAMFit.OVER:
        tier = 3
    elif sm.verified and sm.ram_fit in (RAMFit.FITS, RAMFit.TIGHT):
        tier = 0
    elif sm.verified and sm.ram_fit == RAMFit.RISKY:
        tier = 1
    else:
        # UNKNOWN / unverified — never promoted above verified fits
        tier = 2

    fit_rank = {
        RAMFit.FITS: 0,
        RAMFit.TIGHT: 1,
        RAMFit.RISKY: 2,
        RAMFit.UNKNOWN: 3,
        RAMFit.OVER: 4,
    }[sm.ram_fit]

    # Among verified fits: prefer higher capability (params), then lower RAM need as tiebreak.
    # Among risky/unknown: prefer lower estimated requirement when known.
    if tier == 0:
        capability = -(sm.params_b if sm.params_b is not None else sm.estimated_total_ram_gb or 0.0)
        memory_key = sm.estimated_total_ram_gb or 0.0
    elif tier == 1:
        capability = 0.0
        memory_key = sm.estimated_total_ram_gb or 1e9
    elif tier == 2:
        capability = 0.0
        # Prefer having an inferred estimate over none; then lower pressure
        memory_key = sm.estimated_total_ram_gb if sm.estimated_total_ram_gb > 0 else 1e9
        if sm.params_b is None and sm.estimated_total_ram_gb <= 0:
            memory_key = 2e9
    else:
        capability = 0.0
        memory_key = sm.estimated_total_ram_gb or 1e9

    installable_rank = 0 if sm.installable else 1
    return (tier, fit_rank, capability, memory_key, installable_rank, sm.model.full_tag)


def _assign_recommendation_labels(ranked: list[EvaluatedModel]) -> None:
    """Attach explainable category labels from explicit evidence (not numeric scores)."""
    for sm in ranked:
        sm.labels = []

    fitting = [
        sm
        for sm in ranked
        if sm.verified and sm.ram_fit in (RAMFit.FITS, RAMFit.TIGHT) and not sm.disqualified
    ]

    if fitting:
        # Best fit: prefer FITS over TIGHT, then highest params among that class
        fits_only = [sm for sm in fitting if sm.ram_fit == RAMFit.FITS]
        best_pool = fits_only or fitting
        best_fit = max(
            best_pool,
            key=lambda s: (s.params_b or 0.0, -(s.estimated_total_ram_gb or 0.0)),
        )
        best_fit.labels.append(RecommendationLabel.BEST_FIT)

        with_ram = [sm for sm in fitting if sm.estimated_total_ram_gb > 0]
        if with_ram:
            lowest = min(with_ram, key=lambda s: s.estimated_total_ram_gb)
            if RecommendationLabel.LOWEST_MEMORY not in lowest.labels:
                lowest.labels.append(RecommendationLabel.LOWEST_MEMORY)

        with_tps = [sm for sm in fitting if sm.estimated_tps > 0]
        if with_tps:
            fastest = max(with_tps, key=lambda s: s.estimated_tps)
            if RecommendationLabel.FASTEST_ESTIMATED not in fastest.labels:
                fastest.labels.append(RecommendationLabel.FASTEST_ESTIMATED)

        best_quality = max(fitting, key=lambda s: (s.params_b or 0.0, s.estimated_total_ram_gb or 0.0))
        if RecommendationLabel.BEST_QUALITY_THAT_FITS not in best_quality.labels:
            best_quality.labels.append(RecommendationLabel.BEST_QUALITY_THAT_FITS)

    for sm in ranked:
        cats = {c.lower() for c in sm.model.categories}
        if "code" in cats or "coding" in cats:
            sm.labels.append(RecommendationLabel.CODING)
        if "reasoning" in cats or "math" in cats:
            sm.labels.append(RecommendationLabel.REASONING)
        if "chat" in cats or "general" in cats:
            sm.labels.append(RecommendationLabel.GENERAL_CHAT)
        # Task-specialized models should not pick up a chat label from empty defaults
        if cats & {"asr", "audio", "embeddings", "embedding", "translation"}:
            sm.labels = [lb for lb in sm.labels if lb != RecommendationLabel.GENERAL_CHAT]

        if sm.disqualified or sm.ram_fit == RAMFit.OVER:
            if RecommendationLabel.NOT_RECOMMENDED not in sm.labels:
                sm.labels.append(RecommendationLabel.NOT_RECOMMENDED)
        elif (sm.unverified or sm.ram_fit == RAMFit.UNKNOWN) and (
            RecommendationLabel.EXPERIMENTAL not in sm.labels
        ):
            sm.labels.append(RecommendationLabel.EXPERIMENTAL)


def partition_recommendations(
    ranked: list[EvaluatedModel],
) -> tuple[list[EvaluatedModel], list[EvaluatedModel], str | None]:
    """
    Split evaluated models into verified recommendations vs potential candidates.

    Recommended: verified FITS / TIGHT only (never UNKNOWN-as-FITS).
    Potential: RISKY / UNKNOWN / unverified — discoverable but not claimed fits.
    DOES_NOT_FIT / disqualified are omitted from both lists.

    Returns (recommended, potential, advisory_message).
    """
    recommended = [
        sm
        for sm in ranked
        if sm.verified and not sm.disqualified and sm.ram_fit in (RAMFit.FITS, RAMFit.TIGHT)
    ]
    rec_tags = {sm.model.full_tag for sm in recommended}
    potential = [
        sm
        for sm in ranked
        if sm.model.full_tag not in rec_tags
        and not sm.disqualified
        and sm.ram_fit != RAMFit.OVER
        and (sm.unverified or sm.ram_fit in (RAMFit.UNKNOWN, RAMFit.RISKY) or not sm.verified)
    ]

    advisory: str | None = None
    if not recommended:
        advisory = (
            "No verified model currently fits the available memory. "
            "Nothing below is a confirmed recommendation — "
            "potential candidates need more evidence or free RAM."
        )
    return recommended, potential, advisory


def rank_models(
    models: list[ModelInfo],
    profile: SystemProfile,
    top_n: int = 10,
    category_filter: str | None = None,
) -> list[EvaluatedModel]:
    """Evaluate and order models by compatibility classifications (no numeric score)."""
    if category_filter:
        from .registry import normalize_category

        want = normalize_category(category_filter)
        models = [
            m
            for m in models
            if want in {normalize_category(c) for c in m.categories}
            or (want == "unknown" and (not m.categories or m.categories == ["unknown"]))
        ]

    disk_free = profile.disk.free_gb if profile.disk else 999.0
    candidates = [m for m in models if m.size_gb == 0 or m.size_gb <= disk_free * 0.95]

    logger.info("Evaluating %d models (%d pre-filtered by disk)", len(candidates), len(models))

    evaluated = [evaluate_model(m, profile) for m in candidates]
    evaluated.sort(key=_sort_key)
    top = evaluated[:top_n]
    _assign_recommendation_labels(top)
    return top


_NO_VERIFIED_FIT_MSG = (
    "No verified model currently fits the available memory. "
    "Automatic installation is disabled. You can inspect candidates "
    "or explicitly override the safety check."
)


def select_install_candidate(
    ranked: list[EvaluatedModel],
    available_ram_gb: float = 0.0,
) -> tuple[EvaluatedModel | None, EvaluatedModel | None, str]:
    """
    Choose the default install candidate.

    Returns:
        (default_candidate, override_candidate, advisory_message)

    Preference:
      1. Verified FITS/TIGHT only as automatic default
      2. Otherwise no default — RISKY/UNKNOWN require explicit override
      3. OVER/disqualified models are never offered for install

    Unknown-size / UNVERIFIED models are never the silent default.
    Models whose estimated RAM exceeds available RAM are flagged but may
    still appear as override candidates (user must explicitly confirm).
    """
    pullable = [sm for sm in ranked if sm.installable]
    if not pullable:
        return None, None, "No Ollama-pullable models in the recommendation list."

    # Exclude OVER/disqualified from all install paths
    installable_fit = [sm for sm in pullable if sm.ram_fit != RAMFit.OVER and not sm.disqualified]
    if not installable_fit:
        return None, None, "All pullable models exceed installed RAM — no install candidates."

    top = installable_fit[0]
    safe = next((sm for sm in installable_fit if sm.is_safe_install_default), None)

    if safe is not None:
        return (
            safe,
            None if safe.model.full_tag == top.model.full_tag else top,
            (
                f"Recommended install {safe.model.full_tag} is a verified "
                f"{safe.ram_fit.value} fit — suitable as the default install."
            ),
        )

    # No verified FITS/TIGHT — disable automatic install.
    # Prefer override candidates where estimated RAM ≤ available RAM.
    def _ram_shortfall(sm: EvaluatedModel) -> float | None:
        """Return estimated shortfall vs available RAM, or None if unknown."""
        if sm.estimated_total_ram_gb > 0 and available_ram_gb > 0:
            return round(sm.estimated_total_ram_gb - available_ram_gb, 2)
        return None

    risky = next(
        (sm for sm in installable_fit if sm.verified and sm.ram_fit == RAMFit.RISKY),
        None,
    )
    if risky is not None:
        shortfall = _ram_shortfall(risky)
        shortfall_str = ""
        if shortfall is not None and shortfall > 0:
            shortfall_str = f" — estimated {shortfall:.1f} GB shortfall vs available RAM"
        elif shortfall is not None and shortfall <= 0:
            shortfall_str = f" — estimated {abs(shortfall):.1f} GB headroom"
        return (
            None,
            risky,
            (
                f"{_NO_VERIFIED_FIT_MSG} "
                f"Closest verified candidate: {risky.model.full_tag} "
                f"({risky.ram_fit.value}, est. ~{risky.estimated_total_ram_gb:.1f} GB)"
                f"{shortfall_str}."
            ),
        )

    # No verified RISKY either — offer top pullable as override with clear warnings
    shortfall = _ram_shortfall(top)
    shortfall_str = ""
    if shortfall is not None and shortfall > 0:
        shortfall_str = f" — estimated {shortfall:.1f} GB shortfall vs available RAM"

    reasons = []
    if top.unverified or top.ram_fit == RAMFit.UNKNOWN:
        reasons.append("UNVERIFIED / UNKNOWN memory fit (incomplete metadata)")
    if top.ram_fit == RAMFit.OVER or top.disqualified:
        reasons.append("DOES_NOT_FIT / incompatible")
    reason_txt = "; ".join(reasons) or "not a verified FITS/TIGHT install"

    return (
        None,
        top,
        f"{_NO_VERIFIED_FIT_MSG} Top-listed {top.model.full_tag} is {reason_txt}{shortfall_str}.",
    )
