"""Tests for the model scoring engine."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ai_model_detector.scanner import scan_system
from ai_model_detector.registry import ModelInfo
from ai_model_detector.scorer import score_model, rank_models, ScoredModel


def _make_model(name="testmodel", tag="7b", size_gb=4.5, ram_req=6.0, vram_req=4.5) -> ModelInfo:
    return ModelInfo(
        name=name,
        tag=tag,
        full_tag=f"{name}:{tag}",
        size_gb=size_gb,
        ram_required_gb=ram_req,
        vram_required_gb=vram_req,
        quantization="q4_k_m",
        description="Test model",
        categories=["chat"],
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
        _make_model("small", "3b", size_gb=2.0, ram_req=3.0, vram_req=2.0),
        _make_model("huge",  "70b", size_gb=40.0, ram_req=64.0, vram_req=40.0),
        _make_model("medium","7b",  size_gb=4.5, ram_req=6.0, vram_req=4.5),
    ]
    ranked = rank_models(models, profile, top_n=3)
    scores = [r.score for r in ranked]
    assert scores == sorted(scores, reverse=True)


def test_category_filter():
    profile = scan_system()
    models  = [
        _make_model("chatbot", "7b"),
        ModelInfo(
            name="coder", tag="7b", full_tag="coder:7b",
            size_gb=4.5, ram_required_gb=6.0, vram_required_gb=4.5,
            quantization="q4_k_m", description="code model",
            categories=["code"],
        ),
    ]
    ranked = rank_models(models, profile, top_n=5, category_filter="code")
    assert all("code" in r.model.categories for r in ranked)
