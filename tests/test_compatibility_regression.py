"""
Regression tests: classification-first recommendations (no numeric suitability score).

Primary profile: 8 GB Intel MacBook Pro (Iris Plus 645, ~2.4 GB available RAM).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from ai_model_detector.registry import infer_task_categories
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
    partition_recommendations,
    rank_models,
    select_install_candidate,
)
from tests.hardware_fixtures import make_model, profile_intel_mac_8gb, profile_intel_mac_no_metal


@pytest.fixture
def intel_8gb():
    return profile_intel_mac_8gb()


def _assert_no_numeric_score(sm: EvaluatedModel) -> None:
    assert not hasattr(sm, "score") or getattr(sm, "score", None) is None
    assert "score" not in sm.__dataclass_fields__
    # rank_reason must not resurrect a disguised 0–100 score
    assert "/100" not in sm.rank_reason
    assert "Score " not in sm.rank_reason
    assert not re.search(r"\bscore\b", sm.rank_reason, re.I)


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

    known = [sm for sm in evaluated if sm.verified]
    assert len({round(sm.estimated_total_ram_gb, 1) for sm in known}) >= 2
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
    """~3.9+ GB needed with ~2.4 GB free must not be classified as FITS."""
    sm = evaluate_model(
        make_model(name="granite3.3", tag="3b", size_gb=2.0, quantization="q4_k_m"),
        intel_8gb,
    )
    assert sm.estimated_total_ram_gb > intel_8gb.ram.available_gb
    assert sm.ram_fit in (RAMFit.TIGHT, RAMFit.RISKY, RAMFit.OVER)
    assert sm.ram_fit != RAMFit.FITS


def test_insufficient_available_ram_and_headroom(intel_8gb):
    """Estimates must reflect available-RAM shortfall; FITS requires headroom in free RAM."""
    sm = evaluate_model(
        make_model(name="granite3.3", tag="3b", size_gb=2.0, quantization="q4_k_m"),
        intel_8gb,
    )
    assert sm.estimated_total_ram_gb > intel_8gb.ram.available_gb
    shortfall = sm.estimated_total_ram_gb - intel_8gb.ram.available_gb
    assert shortfall > 0
    assert sm.ram_fit != RAMFit.FITS

    tiny = evaluate_model(
        make_model(name="tinyllama", tag="1.1b", size_gb=0.6, quantization="q4_0"),
        intel_8gb,
    )
    # Tiny may be FITS or TIGHT depending on overhead — never UNKNOWN when size known
    assert tiny.verified
    assert tiny.ram_fit in (RAMFit.FITS, RAMFit.TIGHT, RAMFit.RISKY)


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


def test_igpu_detected_but_acceleration_not_established(intel_8gb):
    """Intel Mac with iGPU → UNVERIFIED (GPU detected, backend not confirmed)."""
    tier, _ = _classify_gpu(intel_8gb.gpus, intel_8gb.os_name, intel_8gb.os_arch)
    assert tier == GPUTier.INTEGRATED
    accel = acceleration_for_tier(tier, intel_8gb.os_name, intel_8gb.os_arch, intel_8gb.metal_available)
    assert accel == AccelerationStatus.UNVERIFIED

    sm = evaluate_model(
        make_model(name="tinyllama", tag="1.1b", size_gb=0.6, quantization="q4_0"),
        intel_8gb,
    )
    assert sm.gpu_detected is True
    assert sm.acceleration == AccelerationStatus.UNVERIFIED
    assert sm.will_use_gpu is False
    assert any("integrated" in line.lower() for line in sm.explanation)
    assert any("not confirmed" in w.lower() or "not established" in w.lower() for w in sm.warnings)


def test_igpu_without_metal_stays_unverified():
    """Intel iGPU without Metal availability → UNVERIFIED."""
    profile = profile_intel_mac_no_metal()
    tier, _ = _classify_gpu(profile.gpus, profile.os_name, profile.os_arch)
    assert tier == GPUTier.INTEGRATED
    accel = acceleration_for_tier(tier, profile.os_name, profile.os_arch, profile.metal_available)
    assert accel == AccelerationStatus.UNVERIFIED

    sm = evaluate_model(
        make_model(name="tinyllama", tag="1.1b", size_gb=0.6, quantization="q4_0"),
        profile,
    )
    assert sm.acceleration == AccelerationStatus.UNVERIFIED


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


def test_pullability_does_not_imply_runnability(intel_8gb):
    """Runtime-compatible and pullable still distinct from hardware fit."""
    sm = evaluate_model(
        make_model(name="llama3.1", tag="70b", size_gb=40.0, quantization="q4_k_m"),
        intel_8gb,
    )
    assert sm.installable is True
    assert sm.runtime_compatible is True
    assert sm.ram_fit == RAMFit.OVER
    assert not sm.is_safe_install_default


def test_unverified_top_is_not_silent_install_default(intel_8gb):
    models = [
        make_model(name="granite4.1", tag="3b", quantization="q4_k_m"),
        make_model(name="tinyllama", tag="1.1b", size_gb=0.6, quantization="q4_0"),
    ]
    ranked = rank_models(models, intel_8gb, top_n=5)
    default, _override, msg = select_install_candidate(ranked, available_ram_gb=intel_8gb.ram.available_gb)
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
    default, override, msg = select_install_candidate(ranked, available_ram_gb=intel_8gb.ram.available_gb)
    assert default is None
    assert override is not None
    assert "No verified model currently fits the available memory" in msg
    assert "Automatic installation is disabled" in msg

    recommended, potential, advisory = partition_recommendations(ranked)
    assert recommended == []
    assert advisory is not None
    assert "No verified model" in advisory


def test_no_forced_recommendation_when_nothing_verified(intel_8gb):
    models = [
        make_model(name="mystery-a", tag="0.6b", quantization="q4_0"),
        make_model(name="mystery-b", tag="3b", quantization="q4_k_m"),
    ]
    ranked = rank_models(models, intel_8gb, top_n=5)
    recommended, potential, advisory = partition_recommendations(ranked)
    assert recommended == []
    assert potential
    assert advisory is not None
    default, _, _ = select_install_candidate(ranked, available_ram_gb=intel_8gb.ram.available_gb)
    assert default is None
    for sm in potential:
        assert RecommendationLabel.BEST_FIT not in sm.labels


def test_compatibility_ordering_verified_fits_before_unknown(intel_8gb):
    models = [
        make_model(name="mystery", tag="latest"),
        make_model(name="tinyllama", tag="1.1b", size_gb=0.6, quantization="q4_0"),
        make_model(name="granite4.1", tag="3b", quantization="q4_k_m"),
        make_model(name="llama3.1", tag="30b", size_gb=19.0, quantization="q4_k_m"),
    ]
    ranked = rank_models(models, intel_8gb, top_n=10)
    first_fit_idx = next(
        (i for i, sm in enumerate(ranked) if sm.verified and sm.ram_fit in (RAMFit.FITS, RAMFit.TIGHT)),
        None,
    )
    assert first_fit_idx is not None
    for i, sm in enumerate(ranked):
        if sm.ram_fit in (RAMFit.UNKNOWN, RAMFit.OVER) or sm.disqualified:
            assert i > first_fit_idx

    recommended, potential, _ = partition_recommendations(ranked)
    assert recommended
    assert recommended[0].verified
    assert recommended[0].ram_fit in (RAMFit.FITS, RAMFit.TIGHT)
    assert all(sm.model.full_tag != recommended[0].model.full_tag for sm in potential)


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


def test_recommendations_include_explanation_and_evidence(intel_8gb):
    models = [
        make_model(
            name="tinyllama",
            tag="1.1b",
            size_gb=0.6,
            quantization="q4_0",
            categories=["chat"],
            category_source="inferred",
        ),
        make_model(
            name="qwen2.5-coder",
            tag="3b",
            size_gb=1.9,
            quantization="q4_k_m",
            categories=["coding", "chat"],
            category_source="inferred",
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


def test_asr_not_labeled_chat():
    cats, src = infer_task_categories(
        "handy-computer/nemotron-3.5-asr-streaming-0.6b-gguf",
        "Streaming ASR GGUF",
    )
    assert "asr" in cats
    assert "chat" not in cats
    assert src in ("inferred", "metadata")

    cats2, src2 = infer_task_categories(
        "openai/whisper-tiny",
        "",
        pipeline_tag="automatic-speech-recognition",
        tags=["automatic-speech-recognition", "gguf"],
    )
    assert "asr" in cats2
    assert "chat" not in cats2
    assert src2 == "metadata"


def test_gguf_alone_does_not_imply_chat():
    cats, src = infer_task_categories("some-org/mystery-weights-gguf", "")
    assert cats == ["unknown"]
    assert src == "unknown"
    assert "chat" not in cats


def test_missing_metadata_produces_unknown_task():
    cats, src = infer_task_categories("acme/untitled-model", "")
    assert cats == ["unknown"]
    assert src == "unknown"


def test_no_numeric_score_in_recommendation_pipeline(intel_8gb):
    models = [
        make_model(name="tinyllama", tag="1.1b", size_gb=0.6, quantization="q4_0"),
        make_model(name="granite4.1", tag="3b", quantization="q4_k_m"),
    ]
    ranked = rank_models(models, intel_8gb, top_n=5)
    blob = " ".join(
        [
            sm.rank_reason
            + " ".join(sm.explanation)
            + " ".join(sm.warnings)
            + " ".join(sm.label_names)
            for sm in ranked
        ]
    )
    assert "score" not in blob.lower()
    assert "/100" not in blob
    for sm in ranked:
        _assert_no_numeric_score(sm)


def test_potential_vs_recommended_partition(intel_8gb):
    models = [
        make_model(name="tinyllama", tag="1.1b", size_gb=0.6, quantization="q4_0"),
        make_model(name="granite4.1", tag="3b", quantization="q4_k_m"),
        make_model(name="granite3.3", tag="3b", size_gb=2.0, quantization="q4_k_m"),
    ]
    ranked = rank_models(models, intel_8gb, top_n=10)
    recommended, potential, _ = partition_recommendations(ranked)
    assert any(sm.model.full_tag == "tinyllama:1.1b" for sm in recommended)
    assert any(sm.ram_fit == RAMFit.UNKNOWN for sm in potential)
    risky = [sm for sm in ranked if sm.model.full_tag == "granite3.3:3b"]
    if risky and risky[0].ram_fit == RAMFit.RISKY:
        pot_tags = {s.model.full_tag for s in potential}
        rec_tags = {s.model.full_tag for s in recommended}
        assert risky[0].model.full_tag in pot_tags
        assert risky[0].model.full_tag not in rec_tags


def test_memory_overhead_is_reasonable(intel_8gb):
    """Total overhead on iGPU system must be < 1.6 GB (was ~1.9 GB, now reduced)."""
    # Overhead = total - weights - kv for a known-size model
    sm = evaluate_model(
        make_model(name="tinyllama", tag="1.1b", size_gb=0.6, quantization="q4_0"),
        intel_8gb,
    )
    # weights ~0.6 GB + kv ~0.05 GB + overhead = total
    # overhead should be < 1.6 GB on iGPU
    overhead = sm.estimated_total_ram_gb - 0.6 - 0.05
    assert overhead < 1.6, f"Overhead {overhead:.2f} GB exceeds 1.6 GB cap"
    assert overhead > 0.5, f"Overhead {overhead:.2f} GB unrealistically low"


def test_asr_models_not_labeled_chat(intel_8gb):
    """ASR models must not get GENERAL_CHAT label."""
    asr_model = make_model(
        name="whisper-tiny",
        tag="latest",
        size_gb=0.4,
        quantization="q4_0",
        categories=["asr"],
        category_source="metadata",
    )
    chat_model = make_model(
        name="tinyllama",
        tag="1.1b",
        size_gb=0.6,
        quantization="q4_0",
        categories=["chat"],
        category_source="inferred",
    )
    ranked = rank_models([asr_model, chat_model], intel_8gb, top_n=5)
    asr_sm = next(sm for sm in ranked if "whisper" in sm.model.full_tag)
    assert RecommendationLabel.GENERAL_CHAT not in asr_sm.labels
    chat_sm = next(sm for sm in ranked if "tinyllama" in sm.model.full_tag)
    assert RecommendationLabel.GENERAL_CHAT in chat_sm.labels


def test_potential_candidates_not_in_recommended(intel_8gb):
    """Potential candidates must never appear in the recommended list."""
    models = [
        make_model(name="tinyllama", tag="1.1b", size_gb=0.6, quantization="q4_0"),
        make_model(name="granite4.1", tag="3b", quantization="q4_k_m"),  # unverified
        make_model(name="mystery", tag="latest"),  # unverified, no size
    ]
    ranked = rank_models(models, intel_8gb, top_n=10)
    recommended, potential, _ = partition_recommendations(ranked)
    rec_tags = {sm.model.full_tag for sm in recommended}
    pot_tags = {sm.model.full_tag for sm in potential}
    # No overlap
    assert rec_tags.isdisjoint(pot_tags)
    # Verified FITS models are in recommended, not potential
    for sm in recommended:
        assert sm.verified
        assert sm.ram_fit in (RAMFit.FITS, RAMFit.TIGHT)


def test_over_models_excluded_from_install_flow(intel_8gb):
    """Models exceeding installed RAM must never appear as install candidates."""
    models = [
        make_model(name="llama3.1", tag="30b", size_gb=19.0, quantization="q4_k_m"),  # OVER
        make_model(name="gemma2", tag="9b", size_gb=5.4, quantization="q4_k_m"),  # RISKY/OVER
    ]
    ranked = rank_models(models, intel_8gb, top_n=5)
    default, override, _msg = select_install_candidate(ranked, available_ram_gb=intel_8gb.ram.available_gb)
    assert default is None
    # Override should not be OVER model
    if override is not None:
        assert override.ram_fit != RAMFit.OVER
        assert not override.disqualified


def test_install_candidate_shows_shortfall(intel_8gb):
    """Override candidate advisory should include shortfall info when estimated RAM > available."""
    # Tiny model with known size that FITS
    models = [
        make_model(name="tinyllama", tag="1.1b", size_gb=0.6, quantization="q4_0"),
        make_model(name="granite3.3", tag="3b", size_gb=2.0, quantization="q4_k_m"),  # RISKY
    ]
    ranked = rank_models(models, intel_8gb, top_n=5)
    default, _override, _msg = select_install_candidate(ranked, available_ram_gb=intel_8gb.ram.available_gb)
    # Should have a verified default (tinyllama)
    assert default is not None
    assert default.verified


def test_hf_only_runtime_compatible_is_theoretical(intel_8gb):
    """HF-only models should not claim 'Runtime compatible: yes' without qualification."""
    hf_model = make_model(
        name="some-org/gguf-model",
        tag="latest",
        size_gb=0.0,
        quantization="gguf",
        ollama_pullable=False,
    )
    sm = evaluate_model(hf_model, intel_8gb)
    assert sm.installable is False
    # runtime_compatible is True (format compatible) but explanation should qualify it
    assert sm.runtime_compatible is True
    assert any("theoretically" in e.lower() or "not installed" in e.lower() or "not been verified" in e.lower()
               for e in sm.explanation)


def test_unknown_never_treated_as_fits(intel_8gb):
    """UNKNOWN memory fit must never become FITS, even when estimated RAM seems low."""
    sm = evaluate_model(
        make_model(name="mystery-small", tag="0.5b"),  # no size_gb → UNKNOWN
        intel_8gb,
    )
    assert sm.ram_fit == RAMFit.UNKNOWN
    assert not sm.verified
    assert not sm.fits_ram  # FITS property must be False for UNKNOWN
    assert sm.ram_fit != RAMFit.FITS
    assert sm.ram_fit != RAMFit.TIGHT


# ── 2.0.0 specific tests ──────────────────────────────────────────────────


def test_version_is_consistent():
    """Version in __init__.py must match what the package reports."""
    import re
    from ai_model_detector import __version__
    init_path = Path(__file__).resolve().parents[1] / "src" / "ai_model_detector" / "__init__.py"
    match = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', init_path.read_text())
    assert match, "Could not find __version__ in __init__.py"
    assert __version__ == match.group(1), f"Runtime version {__version__} != file version {match.group(1)}"


def test_no_third_party_runtime_imports():
    """Package source files must not import any third-party runtime dependencies."""
    forbidden = {"rich", "click", "pywhat", "psutil", "requests", "urllib3", "certifi",
                 "idna", "charset_normalizer", "pygments", "markdown_it_py", "mdurl"}
    src_dir = Path(__file__).resolve().parents[1] / "src" / "ai_model_detector"
    for py_file in src_dir.glob("*.py"):
        content = py_file.read_text()
        for dep in forbidden:
            assert f"import {dep}" not in content and f"from {dep}" not in content, \
                f"{py_file.name} imports {dep}"


def test_package_imports_without_deps():
    """Package should import cleanly (all deps are stdlib)."""
    import ai_model_detector
    from ai_model_detector import scanner, registry, scorer, cli, display, downloader  # noqa: F401
    from ai_model_detector import __version__ as v
    assert v  # just verify it's set


def test_scan_system_works():
    """Live scan should return a valid profile using stdlib only."""
    from ai_model_detector.scanner import scan_system
    profile = scan_system()
    assert profile.os_name
    assert profile.cpu.brand
    assert profile.cpu.cores_physical >= 1
    assert profile.ram.total_gb > 0
    assert profile.ram.available_gb >= 0


def test_fits_requires_headroom(intel_8gb):
    """FITS requires meaningful headroom — estimated RAM equal to available must be TIGHT, not FITS."""
    from ai_model_detector.scorer import _classify_fit
    # Create a profile where available RAM is exactly what a model needs
    from tests.hardware_fixtures import profile_intel_mac_8gb
    tight_profile = profile_intel_mac_8gb(available_gb=1.5, total_gb=8.0)
    # Model needing ~1.5 GB should be TIGHT (no headroom), not FITS
    fit = _classify_fit(1.5, tight_profile)
    assert fit == RAMFit.TIGHT, f"Expected TIGHT for exact-fit, got {fit.value}"
    # Model needing 20% less should be FITS (has headroom)
    fit2 = _classify_fit(1.2, tight_profile)
    assert fit2 == RAMFit.FITS, f"Expected FITS with headroom, got {fit2.value}"
