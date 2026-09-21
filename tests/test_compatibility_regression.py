"""
Regression tests: classification-first recommendations (no numeric suitability score).

Primary profile: 8 GB Intel MacBook Pro (Iris Plus 645, ~2.2 GB available RAM).
"""

from __future__ import annotations

import pytest

from ai_model_detector.scorer import (
    AccelerationStatus,
    EvaluatedModel,
    GPUTier,
    RAMFit,
    RecommendationLabel,
    _classify_gpu,
    acceleration_for_tier,
    estimate_memory,
    evaluate_model,
    rank_models,
    select_install_candidate,
)
from tests.hardware_fixtures import make_model, profile_intel_mac_8gb


@pytest.fixture
def intel_8gb():
    return profile_intel_mac_8gb()


def _assert_no_numeric_score(sm: EvaluatedModel) -> None:
    assert not hasattr(sm, "score") or getattr(sm, "score", None) is None
    # rank_reason must not resurrect a disguised 0–100 score
    assert "/100" not in sm.rank_reason
    assert "Score " not in sm.rank_reason


def test_no_numeric_suitability_score_field(intel_8gb):
    """Materially different models must not be collapsed via a shared numeric score — scores are gone."""
    models = [
        make_model(name="granite4.1", tag="3b", quantization="q4_k_m"),
        make_model(name="granite4.2", tag="3b", quantization="q4_k_m"),
        make_model(name="granite3.3", tag="3b", size_gb=2.0, quantization="q4_k_m"),
        make_model(name="granite3.3", tag="3b-q4_k_s", size_gb=1.9, quantization="q4_k_s"),
        make_model(name="granite3.3", tag="3b-q5_k_m", size_gb=2.4, quantization="q5_k_m"),
    ]
    evaluated = [evaluate_model(m, intel_8gb) for m in models]
    for sm in evaluated:
        _assert_no_numeric_score(sm)
        assert "score" not in sm.__dataclass_fields__

    # Distinct footprints still produce distinct classifications / estimates
    known = [sm for sm in evaluated if sm.verified]
    assert len({round(sm.estimated_total_ram_gb, 1) for sm in known}) >= 2
    # Identical incomplete metadata → same UNKNOWN class (not a false FITS)
    unknown = [sm for sm in evaluated if not sm.verified]
    assert all(sm.ram_fit == RAMFit.UNKNOWN for sm in unknown)
    assert all(not sm.verified for sm in unknown)


def test_param_scale_3b_vs_9b_vs_30b_differ(intel_8gb):
    """3B / 9B / 30B must not be treated identically on an 8 GB machine."""
    small = evaluate_model(
        make_model(name="llama", tag="3b", size_gb=2.0, quantization="q4_k_m"),
        intel_8gb,
    )
    mid = evaluate_model(
        make_model(name="gemma2", tag="9b", size_gb=5.4, quantization="q4_k_m"),
        intel_8gb,
    )
    huge = evaluate_model(
        make_model(name="llama", tag="30b", size_gb=19.0, quantization="q4_k_m"),
        intel_8gb,
    )
    assert small.ram_fit != mid.ram_fit or small.estimated_total_ram_gb != mid.estimated_total_ram_gb
    assert mid.estimated_total_ram_gb < huge.estimated_total_ram_gb
    assert huge.ram_fit == RAMFit.OVER
    assert huge.disqualified
    assert mid.ram_fit in (RAMFit.RISKY, RAMFit.OVER)
    for sm in (small, mid, huge):
        _assert_no_numeric_score(sm)


def test_q4_vs_q5_differ_when_memory_constrained(intel_8gb):
    """Q4 vs Q5 weight deltas must change estimated RAM pressure (not a shared score)."""
    q4 = evaluate_model(
        make_model(name="granite3.3", tag="3b", size_gb=2.0, quantization="q4_k_m"),
        intel_8gb,
    )
    q5 = evaluate_model(
        make_model(name="granite3.3", tag="3b-q5_k_m", size_gb=2.4, quantization="q5_k_m"),
        intel_8gb,
    )
    assert q5.estimated_total_ram_gb > q4.estimated_total_ram_gb
    _assert_no_numeric_score(q4)
    _assert_no_numeric_score(q5)


def test_known_size_vs_unknown_size_differ(intel_8gb):
    known = evaluate_model(
        make_model(name="granite3.3", tag="3b", size_gb=2.0, quantization="q4_k_m"),
        intel_8gb,
    )
    unknown = evaluate_model(
        make_model(name="granite4.1", tag="3b", quantization="q4_k_m"),
        intel_8gb,
    )
    assert known.verified
    assert not unknown.verified
    assert unknown.ram_fit == RAMFit.UNKNOWN
    assert known.ram_fit != RAMFit.UNKNOWN


def test_unknown_memory_metadata_is_not_treated_as_fit(intel_8gb):
    """No size and no params → UNKNOWN, not FITS."""
    sm = evaluate_model(make_model(name="mystery", tag="latest"), intel_8gb)
    assert sm.ram_fit == RAMFit.UNKNOWN
    assert not sm.verified
    assert sm.unverified
    assert sm.ram_fit != RAMFit.FITS
    assert not sm.fits_ram
    ranked = rank_models([sm.model], intel_8gb, top_n=1)
    assert ranked[0].ram_fit == RAMFit.UNKNOWN
    assert RecommendationLabel.EXPERIMENTAL in ranked[0].labels


def test_exceeding_available_ram_is_not_high_compatibility(intel_8gb):
    """~3.9+ GB needed with 2.2 GB free must not be classified as FITS."""
    sm = evaluate_model(
        make_model(name="granite3.3", tag="3b", size_gb=2.0, quantization="q4_k_m"),
        intel_8gb,
    )
    assert sm.estimated_total_ram_gb > intel_8gb.ram.available_gb
    assert sm.ram_fit in (RAMFit.TIGHT, RAMFit.RISKY, RAMFit.OVER)
    assert sm.ram_fit != RAMFit.FITS


def test_provisional_fit_from_inference_still_unverified(intel_8gb):
    """Param inference may estimate GB but must stay UNKNOWN / unverified."""
    sm = evaluate_model(
        make_model(name="granite4.1", tag="3b", quantization="q4_k_m"),
        intel_8gb,
    )
    assert sm.ram_fit == RAMFit.UNKNOWN
    assert not sm.verified
    assert sm.estimated_total_ram_gb > 0  # heuristic present
    assert sm.memory_confidence.value in ("Low", "Unknown")


def test_igpu_detection_is_not_treated_as_acceleration(intel_8gb):
    tier, vram = _classify_gpu(intel_8gb.gpus, intel_8gb.os_name, intel_8gb.os_arch)
    assert tier == GPUTier.INTEGRATED
    assert vram == 0.0
    accel = acceleration_for_tier(tier)
    assert accel == AccelerationStatus.UNVERIFIED

    sm = evaluate_model(
        make_model(name="tinyllama", tag="1.1b", size_gb=0.6, quantization="q4_0"),
        intel_8gb,
    )
    assert sm.gpu_detected is True
    assert sm.acceleration == AccelerationStatus.UNVERIFIED
    assert sm.will_use_gpu is False
    assert any("Iris Plus" in line or "integrated" in line.lower() for line in sm.explanation)
    assert any("unverified" in w.lower() or "does not prove" in w.lower() for w in sm.warnings + sm.explanation)


def test_ollama_pullable_is_not_guaranteed_runnable(intel_8gb):
    """Pullable 30B still DOES_NOT_FIT — installable ≠ runnable on this machine."""
    sm = evaluate_model(
        make_model(name="llama3.1", tag="30b", size_gb=19.0, quantization="q4_k_m"),
        intel_8gb,
    )
    assert sm.installable is True
    assert sm.ram_fit == RAMFit.OVER
    assert sm.disqualified
    assert RecommendationLabel.NOT_RECOMMENDED in rank_models([sm.model], intel_8gb)[0].labels


def test_unverified_top_is_not_silent_install_default(intel_8gb):
    models = [
        make_model(name="granite4.1", tag="3b", quantization="q4_k_m"),
        make_model(name="tinyllama", tag="1.1b", size_gb=0.6, quantization="q4_0"),
    ]
    ranked = rank_models(models, intel_8gb, top_n=5)
    default, _override, msg = select_install_candidate(ranked)
    assert default is not None
    assert default.verified
    assert default.model.full_tag == "tinyllama:1.1b"
    assert "verified" in msg.lower()


def test_no_verified_candidate_says_so(intel_8gb):
    models = [
        make_model(name="granite4.1", tag="3b", quantization="q4_k_m"),
        make_model(name="granite4.2", tag="3b", quantization="q4_k_m"),
        make_model(name="llama3.1", tag="30b", size_gb=19.0, quantization="q4_k_m"),
    ]
    ranked = rank_models(models, intel_8gb, top_n=5)
    default, override, msg = select_install_candidate(ranked)
    assert default is None
    assert override is not None
    assert "No verified model currently fits the available memory" in msg
    assert "Automatic installation is disabled" in msg


def test_recommendations_include_explanation_and_evidence(intel_8gb):
    models = [
        make_model(name="tinyllama", tag="1.1b", size_gb=0.6, quantization="q4_0", categories=["chat"]),
        make_model(
            name="qwen2.5-coder",
            tag="3b",
            size_gb=1.9,
            quantization="q4_k_m",
            categories=["code", "chat"],
        ),
        make_model(name="mystery", tag="latest"),
    ]
    ranked = rank_models(models, intel_8gb, top_n=5)
    assert ranked
    for sm in ranked:
        _assert_no_numeric_score(sm)
        assert sm.rank_reason
        assert sm.explanation or sm.warnings
        assert sm.labels  # at least one category label
    tiny = next(sm for sm in ranked if sm.model.full_tag == "tinyllama:1.1b")
    assert any(
        label in tiny.labels
        for label in (
            RecommendationLabel.BEST_FIT,
            RecommendationLabel.LOWEST_MEMORY,
            RecommendationLabel.FASTEST_ESTIMATED,
            RecommendationLabel.GENERAL_CHAT,
        )
    )
    coder = next(sm for sm in ranked if "coder" in sm.model.full_tag)
    assert RecommendationLabel.CODING in coder.labels


def test_memory_estimate_matches_reported_ballpark(intel_8gb):
    """User-reported ballparks: Q4_K_M ~3.9, Q4_K_S ~3.8, Q5_K_M ~4.3 GB total."""
    tier, _ = _classify_gpu(intel_8gb.gpus, intel_8gb.os_name, intel_8gb.os_arch)
    q4m = estimate_memory(
        make_model(name="granite3.3", tag="3b", size_gb=2.0, quantization="q4_k_m"),
        intel_8gb,
        tier,
    )
    q4s = estimate_memory(
        make_model(name="granite3.3", tag="3b-q4_k_s", size_gb=1.9, quantization="q4_k_s"),
        intel_8gb,
        tier,
    )
    q5 = estimate_memory(
        make_model(name="granite3.3", tag="3b-q5_k_m", size_gb=2.4, quantization="q5_k_m"),
        intel_8gb,
        tier,
    )
    assert 3.5 <= q4m.total_required_gb <= 4.5
    assert 3.4 <= q4s.total_required_gb <= 4.4
    assert 4.0 <= q5.total_required_gb <= 5.0
    assert q5.total_required_gb > q4m.total_required_gb >= q4s.total_required_gb


def test_insufficient_ram_detected_for_oversize_model(intel_8gb):
    sm = evaluate_model(
        make_model(name="gemma2", tag="9b", size_gb=5.4, quantization="q4_k_m"),
        intel_8gb,
    )
    assert sm.estimated_total_ram_gb > intel_8gb.ram.available_gb
    assert sm.ram_fit in (RAMFit.RISKY, RAMFit.OVER)
    assert any("RAM" in w for w in sm.warnings)


def test_rank_models_never_promotes_unknown_over_verified_fit(intel_8gb):
    models = [
        make_model(name="mystery", tag="latest"),
        make_model(name="tinyllama", tag="1.1b", size_gb=0.6, quantization="q4_0"),
        make_model(name="granite4.1", tag="3b", quantization="q4_k_m"),
    ]
    ranked = rank_models(models, intel_8gb, top_n=5)
    assert ranked[0].verified
    assert ranked[0].ram_fit in (RAMFit.FITS, RAMFit.TIGHT)
    unknown_idx = next(i for i, sm in enumerate(ranked) if sm.ram_fit == RAMFit.UNKNOWN)
    assert unknown_idx > 0
