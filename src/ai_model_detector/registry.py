"""
registry.py — Live AI model registry fetcher (stdlib only, zero dependencies).

Queries Ollama's public library, Hugging Face API, and community
sources at runtime to build a fresh model list. Nothing is hardcoded —
the registry is always fetched fresh so new models appear automatically.
"""

import json
import logging
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

OLLAMA_LIBRARY_URL = "https://ollama.com/library"
OLLAMA_API_SHOW_URL = "https://ollama.com/api/show"
OLLAMA_SEARCH_URL = "https://ollama.com/search"
OLLAMA_MODEL_API = "https://ollama.com/api/models"

HF_API_URL = "https://huggingface.co/api/models"

GITHUB_ISSUES_URL = "https://api.github.com/repos/ollama/ollama/issues?state=open&labels=bug&per_page=20"

REQUEST_TIMEOUT = 15
USER_AGENT = "AI-Model-Detector/2.0 (https://github.com/eliekh05/AI-Model-Detector-Auto-Downloader)"


# Canonical task categories. Never invent "chat" merely because a file is GGUF.
TASK_CATEGORIES = (
    "asr",
    "audio",
    "chat",
    "coding",
    "reasoning",
    "embeddings",
    "vision",
    "translation",
    "multimodal",
    "unknown",
)

# CLI / filter aliases → canonical name
_CATEGORY_ALIASES = {
    "code": "coding",
    "codes": "coding",
    "embedding": "embeddings",
    "embed": "embeddings",
    "speech": "asr",
    "speech-to-text": "asr",
    "stt": "asr",
    "math": "reasoning",
    "general": "chat",
}

# Hugging Face pipeline_tag → canonical category (metadata, not a guess)
_PIPELINE_TAG_TO_CATEGORY = {
    "automatic-speech-recognition": "asr",
    "audio-classification": "audio",
    "audio-to-audio": "audio",
    "text-to-speech": "audio",
    "text-to-audio": "audio",
    "translation": "translation",
    "summarization": "chat",
    "text-generation": "chat",
    "text2text-generation": "chat",
    "conversational": "chat",
    "feature-extraction": "embeddings",
    "sentence-similarity": "embeddings",
    "image-classification": "vision",
    "object-detection": "vision",
    "image-segmentation": "vision",
    "image-to-text": "vision",
    "image-text-to-text": "multimodal",
    "visual-question-answering": "multimodal",
    "document-question-answering": "multimodal",
    "any-to-any": "multimodal",
    "zero-shot-image-classification": "vision",
}


def normalize_category(category: str) -> str:
    """Map aliases to canonical task categories."""
    key = (category or "").strip().lower()
    if not key:
        return "unknown"
    return _CATEGORY_ALIASES.get(key, key)


@dataclass
class ModelInfo:
    name: str  # e.g. "llama3.2"
    tag: str  # e.g. "3b", "7b", "70b"
    full_tag: str  # e.g. "llama3.2:3b"
    size_gb: float  # approximate disk size
    ram_required_gb: float  # minimum RAM to run comfortably
    vram_required_gb: float  # 0 if CPU-only is fine
    quantization: str  # e.g. "q4_0", "q4_K_M", "f16"
    description: str
    categories: list[str] = field(default_factory=list)  # canonical task categories
    # "metadata" = HF pipeline/tags; "inferred" = name/description heuristics; "unknown" = none
    category_source: str = "unknown"
    known_issues: list[str] = field(default_factory=list)  # from community sources
    hf_downloads: int = 0  # Hugging Face download count (0 if N/A)
    ollama_pull_count: int = 0  # from Ollama library page
    source: str = "ollama"  # "ollama" | "huggingface"
    ollama_pullable: bool = True  # False for HF-only models that need manual download


# ── helpers ───────────────────────────────────────────────────────────


def _http_get(url: str, params: dict | None = None, timeout: int = REQUEST_TIMEOUT) -> str | None:
    """Fetch URL with urllib.request. Returns response text or None on failure."""
    try:
        if params:
            url = url + "?" + urllib.parse.urlencode(params)
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, urllib.error.HTTPError, OSError, TimeoutError) as exc:
        logger.warning("HTTP request failed: %s — %s", url, exc)
        return None


# ── Ollama library scraper ────────────────────────────────────────────────────


def _parse_size_to_gb(size_str: str) -> float:
    """Convert strings like '3.8GB', '2.0 GB', '500MB' to float GB."""
    size_str = size_str.strip().upper()
    match = re.search(r"([\d.]+)\s*(GB|MB|TB)", size_str)
    if not match:
        return 0.0
    value = float(match.group(1))
    unit = match.group(2)
    if unit == "MB":
        return round(value / 1024, 3)
    if unit == "TB":
        return round(value * 1024, 1)
    return round(value, 2)


def _ram_from_size(size_gb: float, has_gpu: bool) -> tuple[float, float]:
    """Estimate RAM / VRAM needed given model size."""
    ram = round(size_gb * 1.25, 1)
    vram = round(size_gb * 1.1, 1) if has_gpu else 0.0
    return ram, vram


def _fetch_ollama_library() -> list[ModelInfo]:
    """
    Fetch the Ollama library and get real tags for each model.

    Strategy:
    1. Get the list of model slugs from ollama.com/search HTML
    2. For each slug, fetch ollama.com/library/<slug> to get real tags + sizes
    3. Only emit tag variants that actually exist — never append :latest blindly
    """
    models: list[ModelInfo] = []

    # ── Step 1: get slug list ─────────────────────────────────────────────────
    html = _http_get(OLLAMA_SEARCH_URL, params={"q": "", "c": "", "o": "popular"})
    if html is None:
        logger.warning("Could not reach Ollama search endpoint")
        return []

    slugs: list[str] = []
    seen_slugs: set[str] = set()
    for m in re.finditer(r'href="/library/([a-zA-Z0-9_.-]+)"', html):
        slug = m.group(1)
        if slug not in seen_slugs:
            seen_slugs.add(slug)
            slugs.append(slug)

    # Also try embedded JSON
    for block in re.findall(
        r'<script[^>]*type=["\']application/json["\'][^>]*>(.*?)</script>',
        html,
        re.DOTALL,
    ):
        try:
            data = json.loads(block)
            items = data if isinstance(data, list) else data.get("models", [])
            for item in items:
                slug = item.get("name", "") or item.get("model", "")
                if slug and slug not in seen_slugs:
                    seen_slugs.add(slug)
                    slugs.append(slug)
        except json.JSONDecodeError:
            pass

    logger.info("Found %d model slugs on Ollama library", len(slugs))

    # ── Step 2: fetch real tags for each model ────────────────────────────────
    for slug in slugs[:60]:  # cap at 60 to avoid hammering the server
        page_html = _http_get(f"https://ollama.com/library/{slug}/tags")
        if page_html is None:
            page_html = _http_get(f"https://ollama.com/library/{slug}")

        if page_html is None:
            page_html = ""
        description = ""
        pull_count = 0

        # Extract description
        desc_m = re.search(
            r'<p[^>]*class="[^"]*(?:description|subtitle)[^"]*"[^>]*>(.*?)</p>',
            page_html,
            re.DOTALL,
        )
        if desc_m:
            description = re.sub(r"<[^>]+>", "", desc_m.group(1)).strip()

        # Extract pull count
        pull_m = re.search(r"([\d.]+[KMB]?)\s*[Pp]ulls?", page_html)
        if pull_m:
            raw = pull_m.group(1).upper()
            try:
                if raw.endswith("B"):
                    pull_count = int(float(raw[:-1]) * 1_000_000_000)
                elif raw.endswith("M"):
                    pull_count = int(float(raw[:-1]) * 1_000_000)
                elif raw.endswith("K"):
                    pull_count = int(float(raw[:-1]) * 1_000)
                else:
                    pull_count = int(raw)
            except ValueError:
                pass

        tag_entries: list[tuple[str, float]] = []  # (tag_name, size_gb)

        # href pattern: /library/slug:tag
        for tm in re.finditer(
            rf'href="/library/{re.escape(slug)}:([a-zA-Z0-9_.\-]+)"',
            page_html,
        ):
            tag_name = tm.group(1)
            if tag_name and tag_name not in {t for t, _ in tag_entries}:
                tag_entries.append((tag_name, 0.0))

        # Size pattern near each tag
        size_blocks = re.findall(
            r"([a-zA-Z0-9_.\-]+)\s*[^<]*?([\d.]+\s*(?:GB|MB))",
            page_html,
        )
        size_map: dict[str, float] = {}
        for tag_candidate, size_str in size_blocks:
            gb = _parse_size_to_gb(size_str)
            if gb > 0 and len(tag_candidate) <= 30:
                size_map[tag_candidate] = gb

        if not tag_entries:
            for tm in re.finditer(
                r"<(?:span|code|td)[^>]*>\s*([a-zA-Z0-9][a-zA-Z0-9_.\-]{0,25})\s*</(?:span|code|td)>",
                page_html,
            ):
                candidate = tm.group(1)
                if (
                    re.match(r"^[a-zA-Z0-9][a-zA-Z0-9_.\-]*$", candidate)
                    and candidate != slug
                    and candidate not in {t for t, _ in tag_entries}
                ):
                    tag_entries.append((candidate, size_map.get(candidate, 0.0)))

        tag_entries = [(tag, size_map.get(tag, size)) for tag, size in tag_entries]

        categories, category_source = infer_task_categories(slug, description)

        if not tag_entries:
            logger.debug("No tags found for %s — skipping", slug)
            continue

        for tag_name, size_gb in tag_entries:
            ram_gb, vram_gb = _ram_from_size(size_gb, has_gpu=True)
            quant = _infer_quantization(tag_name)

            models.append(
                ModelInfo(
                    name=slug,
                    tag=tag_name,
                    full_tag=f"{slug}:{tag_name}",
                    size_gb=size_gb,
                    ram_required_gb=ram_gb,
                    vram_required_gb=vram_gb,
                    quantization=quant,
                    description=description,
                    categories=categories,
                    category_source=category_source,
                    ollama_pull_count=pull_count,
                    source="ollama",
                    ollama_pullable=True,
                )
            )

    return models


def infer_task_categories(
    name: str,
    description: str = "",
    *,
    tags: list[str] | None = None,
    pipeline_tag: str | None = None,
) -> tuple[list[str], str]:
    """
    Assign task categories from reliable metadata first, then name/description cues.

    Returns (categories, source) where source is:
      - "metadata" — Hugging Face pipeline_tag / explicit tags
      - "inferred" — name or description heuristics
      - "unknown"  — insufficient evidence (never invent "chat" for GGUF alone)
    """
    tags = tags or []
    found: list[str] = []
    source = "unknown"

    # ── 1. Explicit HF pipeline_tag (strongest) ─────────────────────────────
    if pipeline_tag:
        mapped = _PIPELINE_TAG_TO_CATEGORY.get(pipeline_tag.strip().lower())
        if mapped:
            found.append(mapped)
            source = "metadata"

    # ── 2. Explicit HF / library tags ───────────────────────────────────────
    tag_set = {t.strip().lower() for t in tags if isinstance(t, str)}
    tag_hits: list[str] = []
    if tag_set & {"asr", "automatic-speech-recognition", "speech-to-text", "whisper"}:
        tag_hits.append("asr")
    if tag_set & {"audio", "text-to-speech", "tts"}:
        tag_hits.append("audio")
    if tag_set & {"translation", "translate"}:
        tag_hits.append("translation")
    if tag_set & {"embedding", "embeddings", "feature-extraction", "sentence-similarity"}:
        tag_hits.append("embeddings")
    if tag_set & {"vision", "image", "image-text-to-text"}:
        tag_hits.append("vision")
    if tag_set & {"multimodal", "any-to-any", "vlm"}:
        tag_hits.append("multimodal")
    if tag_set & {"code", "coding", "codellama", "starcoder"}:
        tag_hits.append("coding")
    if tag_set & {"reasoning", "math"}:
        tag_hits.append("reasoning")
    if tag_set & {"conversational", "chat", "text-generation"}:
        tag_hits.append("chat")

    for cat in tag_hits:
        if cat not in found:
            found.append(cat)
    if tag_hits and source != "metadata":
        source = "metadata"

    # ── 3. Name / description heuristics (weaker) ───────────────────────────
    text = f"{name} {description}".lower()
    text_for_task = re.sub(r"\b(gguf|ggml|safetensors|bin|q\d[_k0-9]*|f16|f32|bf16)\b", " ", text)

    inferred: list[str] = []
    if any(
        kw in text_for_task
        for kw in (
            "asr",
            "speech-to-text",
            "speech recognition",
            "speech_recognition",
            "automatic-speech",
            "transcri",
            "whisper",
            "streaming-asr",
            "nemotron-3.5-asr",
            "nemotron-asr",
        )
    ) or re.search(r"(?<![a-z])asr(?![a-z])", text_for_task) or "-asr-" in text_for_task or text_for_task.endswith("-asr"):
        inferred.append("asr")
    if (
        any(kw in text_for_task for kw in ("text-to-speech", "tts", "audio generation", "voice clone"))
        and "asr" not in inferred
    ) or (
        "audio" in text_for_task
        and not inferred
        and any(kw in text_for_task for kw in ("speech", "sound", "voice", "waveform"))
    ):
        inferred.append("audio")
    if any(kw in text_for_task for kw in ("translat", "nllb", "marianmt", "opus-mt")):
        inferred.append("translation")
    if any(
        kw in text_for_task
        for kw in ("embed", "embedding", "nomic-embed", "mxbai", "bge-", "e5-", "gte-")
    ):
        inferred.append("embeddings")
    if any(
        kw in text_for_task
        for kw in ("vision", "llava", "bakllava", "minicpm-v", "image-to-text", "clip-")
    ):
        inferred.append("vision")
    if any(
        kw in text_for_task
        for kw in ("multimodal", "any-to-any", "vision-language", "vlm", "image-text")
    ):
        inferred.append("multimodal")
    if any(
        kw in text_for_task
        for kw in ("coder", "starcoder", "deepseek-coder", "codellama", "code llama", "coding")
    ) or re.search(r"(?<![a-z])code(?![a-z])", text_for_task):
        inferred.append("coding")
    if any(
        kw in text_for_task
        for kw in ("reasoning", "mathstral", "qwen2-math", "deepseek-r1", "qwq")
    ) or re.search(r"(?<![a-z])math(?![a-z])", text_for_task):
        inferred.append("reasoning")
    if any(
        kw in text_for_task
        for kw in (
            "chat",
            "instruct",
            "assistant",
            "conversational",
            "dialogue",
            "instruction-tuned",
            "instruction tuned",
        )
    ):
        inferred.append("chat")

    for cat in inferred:
        if cat not in found:
            found.append(cat)
    if inferred and source == "unknown":
        source = "inferred"

    if not found:
        return ["unknown"], "unknown"

    # Prefer specific non-chat labels when ASR/audio/etc. are present
    if "asr" in found and "chat" in found:
        found = [c for c in found if c != "chat"]
    if "embeddings" in found and "chat" in found:
        found = [c for c in found if c != "chat"]

    return found, source


def _infer_quantization(tag: str) -> str:
    tag_lower = tag.lower()
    for quant in [
        "q8_0",
        "q6_k",
        "q5_k_m",
        "q5_k_s",
        "q5_0",
        "q4_k_m",
        "q4_k_s",
        "q4_0",
        "q3_k_m",
        "q3_k_s",
        "q2_k",
        "f16",
        "f32",
        "bf16",
        "iq4_xs",
    ]:
        if quant in tag_lower:
            return quant
    if re.search(r"\d+b", tag_lower):
        return "q4_0"  # Ollama default quantization
    return "unknown"


# ── Hugging Face supplemental data ───────────────────────────────────────────


def _fetch_hf_popular_models(limit: int = 30) -> list[ModelInfo]:
    """Fetch popular GGUF models from Hugging Face for cross-referencing."""
    params = {
        "search": "gguf",
        "sort": "downloads",
        "direction": -1,
        "limit": limit,
        "full": True,
    }
    text = _http_get(HF_API_URL, params=params)
    if text is None:
        return []

    models = []
    try:
        data = json.loads(text)
        for item in data:
            model_id = item.get("modelId", "") or item.get("id", "")
            downloads = item.get("downloads", 0)
            card = item.get("cardData", {}) or {}
            description = ""
            if isinstance(card.get("language"), list) and card["language"]:
                description = str(card["language"][0])
            elif isinstance(card.get("language"), str):
                description = card["language"]
            tags = item.get("tags", []) or []
            pipeline_tag = item.get("pipeline_tag") or card.get("pipeline_tag")

            categories, category_source = infer_task_categories(
                model_id,
                description,
                tags=[t for t in tags if isinstance(t, str)],
                pipeline_tag=pipeline_tag if isinstance(pipeline_tag, str) else None,
            )

            models.append(
                ModelInfo(
                    name=model_id,
                    tag="latest",
                    full_tag=model_id,
                    size_gb=0.0,
                    ram_required_gb=0.0,
                    vram_required_gb=0.0,
                    quantization="gguf",
                    description=description or f"HuggingFace GGUF (manual download): {model_id}",
                    categories=categories,
                    category_source=category_source,
                    hf_downloads=downloads,
                    source="huggingface",
                    ollama_pullable=False,
                )
            )
    except Exception as exc:
        logger.warning("HuggingFace parse error: %s", exc)

    return models


# ── Community issue tracker ───────────────────────────────────────────────────


def _fetch_known_issues() -> dict[str, list[str]]:
    """
    Pull open bug reports from the Ollama GitHub repo.
    Returns a dict mapping model name keywords to issue titles.
    """
    issues: dict[str, list[str]] = {}
    text = _http_get(GITHUB_ISSUES_URL)
    if text is None:
        return issues

    try:
        data = json.loads(text)
        for issue in data:
            title = issue.get("title", "")
            body = issue.get("body", "") or ""
            for word in re.findall(r"\b[a-z][a-z0-9._-]+\b", (title + " " + body).lower()):
                if len(word) > 3 and word not in {"with", "from", "when", "this", "that", "have"}:
                    issues.setdefault(word, []).append(title[:120])
    except Exception as exc:
        logger.warning("GitHub issues parse error: %s", exc)

    return issues


# ── Public API ──────────────────────────────────────────────────────────


def _fetch_local_ollama_models() -> list[ModelInfo]:
    """
    Query the local Ollama instance (localhost:11434) for installed models.
    These are guaranteed pullable and have accurate size data.
    """
    try:
        text = _http_get("http://localhost:11434/api/tags", timeout=3)
        if text is None:
            return []
        data = json.loads(text)
        models = []
        for item in data.get("models", []):
            full = item.get("name", "")
            if ":" in full:
                name, tag = full.rsplit(":", 1)
            else:
                name, tag = full, "latest"
            size_bytes = item.get("size", 0)
            size_gb = round(size_bytes / (1024**3), 2) if size_bytes else 0.0
            ram_gb, vram_gb = _ram_from_size(size_gb, has_gpu=True)
            categories, category_source = infer_task_categories(name, "")
            models.append(
                ModelInfo(
                    name=name,
                    tag=tag,
                    full_tag=full,
                    size_gb=size_gb,
                    ram_required_gb=ram_gb,
                    vram_required_gb=vram_gb,
                    quantization=_infer_quantization(tag),
                    description="Already installed locally",
                    categories=categories,
                    category_source=category_source,
                    ollama_pull_count=0,
                    source="ollama",
                    ollama_pullable=True,
                )
            )
        return models
    except Exception:
        return []


def fetch_registry(include_hf: bool = True) -> list[ModelInfo]:
    """
    Fetch the complete, live model registry from Ollama + optional HF.
    Returns a list of ModelInfo objects sorted by popularity.
    No model data is hardcoded — everything is fetched at runtime.
    """
    logger.info("Fetching Ollama model registry…")
    ollama_models = _fetch_ollama_library()
    logger.info("Found %d Ollama model variants", len(ollama_models))

    local_models = _fetch_local_ollama_models()
    logger.info("Found %d locally installed models", len(local_models))

    local_tags = {m.full_tag for m in local_models}
    ollama_models = [m for m in ollama_models if m.full_tag not in local_tags]
    ollama_models = local_models + ollama_models

    hf_models: list[ModelInfo] = []
    if include_hf:
        logger.info("Fetching HuggingFace supplemental data…")
        hf_models = _fetch_hf_popular_models()
        logger.info("Found %d HuggingFace models", len(hf_models))

    known_issues = _fetch_known_issues()
    logger.info("Loaded %d known issue keywords", len(known_issues))

    all_models = ollama_models + hf_models
    for m in all_models:
        matched = []
        for keyword, titles in known_issues.items():
            if keyword in m.name.lower():
                matched.extend(titles[:2])
        m.known_issues = matched[:4]

    all_models.sort(
        key=lambda m: (
            int(m.ollama_pullable),
            int(m.size_gb > 0),
            m.ollama_pull_count + m.hf_downloads,
        ),
        reverse=True,
    )
    return all_models
