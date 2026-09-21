"""Tests for the two-layer scoring engine."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ai_model_detector.scanner import scan_system
from ai_model_detector.registry import ModelInfo
from ai_model_detector.scorer import (
    score_model, rank_models, ScoredModel,
    RAMFit, GPUTier, Confidence,
    _ram_budget, _classify_gpu, _estimate_tokens_per_sec,
)


def _make_model(
    name="testmodel", tag="3b", size_gb=2.0,
    ram_req=3.0, vram_req=0.0, quant="q4_k_m",
    pullable=True,
) -> ModelInfo:
    return ModelInfo(
        name=name, tag=tag, full_tag=f"{name}:{tag}",
        size_gb=size_gb, ram_required_gb=ram_req, vram_required_gb=vram_req,
        quantization=quant, description="Test model",
        categories=["chat"], ollama_pullable=pullable,
    )


def test_score_model_returns_scored():
    profile = scan_system()
    model   = _make_model()
    result  = score_model(model, profile)
    assert isinstance(result, ScoredModel)
    assert 0 <= result.score <= 100


def test_rank_models_sorted():
    profile = scan_system()
    models  = [
        _make_model("small", "3b",  size_gb=2.0,  ram_req=3.0),
        _make_model("huge",  "70b", size_gb=40.0, ram_req=64.0),
        _make_model("medium","7b",  size_gb=4.5,  ram_req=6.0),
    ]
    ranked = rank_models(models, profile, top_n=3)
    # Non-disqualified models must come before disqualified ones
    disq_flags = [r.disqualified for r in ranked]
    # Once we see True, all subsequent must also be True
    seen_disq = False
    for d in disq_flags:
        if d:
            seen_disq = True
        if seen_disq:
            assert d, "Disqualified models must be grouped at the end"


def test_category_filter():
    profile = scan_system()
    models  = [
        _make_model("chatbot", "7b"),
        ModelInfo(
            name="coder", tag="7b", full_tag="coder:7b",
            size_gb=4.5, ram_required_gb=6.0, vram_required_gb=0.0,
            quantization="q4_k_m", description="code model",
            categories=["code"], ollama_pullable=True,
        ),
    ]
    ranked = rank_models(models, profile, top_n=5, category_filter="code")
    assert all("code" in r.model.categories for r in ranked)


def test_ram_budget_unknown_size():
    """Models with no size data must return UNKNOWN, not FIT."""
    profile = scan_system()
    model = _make_model(size_gb=0.0)
    fit, _, _, _ = _ram_budget(model, profile)
    assert fit == RAMFit.UNKNOWN, "Zero size must be UNKNOWN, never FIT"


def test_ram_budget_huge_model_does_not_fit():
    """A 400 GB model should NEVER fit on an 8 GB machine."""
    profile = scan_system()
    model = _make_model(size_gb=400.0, ram_req=400.0)
    fit, _, _, _ = _ram_budget(model, profile)
    assert fit == RAMFit.OVER


def test_ram_budget_small_model_fits():
    """A 2 GB model should fit on any reasonable machine."""
    profile = scan_system()
    model = _make_model(size_gb=2.0, ram_req=3.0)
    fit, _, _, _ = _ram_budget(model, profile)
    assert fit in (RAMFit.FIT, RAMFit.TIGHT, RAMFit.RISKY)


def test_unknown_size_penalised_not_rewarded():
    """A model with unknown size must score lower than a known-fitting model."""
    profile = scan_system()
    unknown_model = _make_model("unknown", "?", size_gb=0.0)
    known_model   = _make_model("known",   "3b", size_gb=2.0)
    sm_unknown = score_model(unknown_model, profile)
    sm_known   = score_model(known_model,   profile)
    assert sm_known.score > sm_unknown.score, (
        "Known-fitting model must outscore unknown-size model"
    )


def test_huge_model_disqualified():
    """A 397B model must be disqualified on an 8 GB machine."""
    profile = scan_system()
    huge = _make_model("huge", "397b", size_gb=230.0, ram_req=230.0)
    sm   = score_model(huge, profile)
    assert sm.disqualified, "397B model must be disqualified on 8 GB machine"
    assert sm.ram_fit == RAMFit.OVER


def test_scores_are_not_all_identical():
    """The 76/100 identical-score bug must be fixed."""
    profile = scan_system()
    models = [
        _make_model("s3b",  "3b",   size_gb=2.0,   quant="q4_k_m"),
        _make_model("s9b",  "9b",   size_gb=5.5,   quant="q4_k_m"),
        _make_model("s35b", "35b",  size_gb=20.0,  quant="q4_k_m"),
        _make_model("s70b", "70b",  size_gb=40.0,  quant="q4_k_m"),
        _make_model("s397b","397b", size_gb=230.0, quant="q4_k_m"),
    ]
    scored = [score_model(m, profile) for m in models]
    unique_scores = {s.score for s in scored}
    assert len(unique_scores) > 1, (
        f"All models scored identically ({unique_scores}) — scoring bug not fixed"
    )


def test_integrated_gpu_not_awarded_gpu_bonus():
    """Intel Iris should not give GPU acceleration bonus."""
    from ai_model_detector.scanner import GPUDevice
    profile = scan_system()
    # Simulate Intel Iris iGPU
    profile.gpus = [GPUDevice(
        name="Intel Iris Plus Graphics 645",
        vram_gb=None, driver_version=None,
        metal_support=True, cuda_version=None,
        rocm_version=None, vulkan_support=False,
        is_integrated=True,
    )]
    model  = _make_model(size_gb=2.0)
    sm     = score_model(model, profile)
    assert not sm.will_use_gpu, "Integrated GPU must not claim GPU acceleration"
    assert sm.gpu_tier == GPUTier.INTEGRATED


def test_ram_fit_values_are_distinct():
    """FIT/TIGHT/RISKY/OVER must produce distinct scores for same model."""
    profile = scan_system()

    # A 2 GB model at Q4 needs ~1.6 GB — should fit on any CI machine
    fit_model  = _make_model(size_gb=2.0, quant="q4_k_m")
    # A 500 GB model will never fit on any real machine
    over_model = _make_model(size_gb=500.0, quant="q4_k_m")

    sm_fit  = score_model(fit_model,  profile)
    sm_over = score_model(over_model, profile)

    assert sm_fit.score > sm_over.score, "Fitting model must score higher than over-RAM model"
    assert sm_over.disqualified, "500 GB model must always be disqualified"
