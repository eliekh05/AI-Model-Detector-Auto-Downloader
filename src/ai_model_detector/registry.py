"""
registry.py — Live AI model registry fetcher.

Queries Ollama's public library, Hugging Face API, and community
sources at runtime to build a fresh model list. Nothing is hardcoded —
the registry is always fetched fresh so new models appear automatically.
"""

import json
import logging
import re
from dataclasses import dataclass, field

import requests

logger = logging.getLogger(__name__)

OLLAMA_LIBRARY_URL = "https://ollama.com/library"
OLLAMA_API_SHOW_URL = "https://ollama.com/api/show"
OLLAMA_SEARCH_URL   = "https://ollama.com/search"

HF_API_URL = "https://huggingface.co/api/models"

# Community issue trackers (Reddit / GitHub) parsed via simple JSON API
GITHUB_ISSUES_URL = (
    "https://api.github.com/repos/ollama/ollama/issues"
    "?state=open&labels=bug&per_page=20"
)

REQUEST_TIMEOUT = 15
USER_AGENT = "AI-Model-Detector/1.0 (https://github.com/eliekh05/AI-Model-Detector-Auto-Downloader)"


@dataclass
class ModelInfo:
    name: str                     # e.g. "llama3.2"
    tag: str                      # e.g. "3b", "7b", "70b"
    full_tag: str                 # e.g. "llama3.2:3b"
    size_gb: float                # approximate disk size
    ram_required_gb: float        # minimum RAM to run comfortably
    vram_required_gb: float       # 0 if CPU-only is fine
    quantization: str             # e.g. "q4_0", "q4_K_M", "f16"
    description: str
    categories: list[str] = field(default_factory=list)   # ["chat", "code", "vision"]
    known_issues: list[str] = field(default_factory=list) # from community sources
    hf_downloads: int = 0         # Hugging Face download count (0 if N/A)
    ollama_pull_count: int = 0    # from Ollama library page
    source: str = "ollama"        # "ollama" | "huggingface"
    ollama_pullable: bool = True   # False for HF-only models that need manual download


# ── helpers ───────────────────────────────────────────────────────────────────

def _http_get(url: str, params: dict | None = None, timeout: int = REQUEST_TIMEOUT) -> requests.Response | None:
    try:
        headers = {"User-Agent": USER_AGENT}
        resp = requests.get(url, params=params, headers=headers, timeout=timeout)
        resp.raise_for_status()
        return resp
    except requests.RequestException as exc:
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
    unit  = match.group(2)
    if unit == "MB":
        return round(value / 1024, 3)
    if unit == "TB":
        return round(value * 1024, 1)
    return round(value, 2)


def _ram_from_size(size_gb: float, has_gpu: bool) -> tuple[float, float]:
    """Estimate RAM / VRAM needed given model size."""
    # Typical rule: model size × 1.2 for RAM overhead, full size for VRAM
    ram  = round(size_gb * 1.25, 1)
    vram = round(size_gb * 1.1, 1) if has_gpu else 0.0
    return ram, vram


def _fetch_ollama_library() -> list[ModelInfo]:
    """
    Fetch the Ollama library page and parse the JSON-LD or embedded
    model data. Falls back to the /search API if scraping fails.
    """
    models: list[ModelInfo] = []

    # Try the Ollama search JSON API (undocumented but stable)
    resp = _http_get(
        OLLAMA_SEARCH_URL,
        params={"q": "", "c": "", "o": "popular"},
    )
    if resp is None:
        logger.warning("Could not reach Ollama search endpoint")
        return []

    # The page returns HTML; extract embedded JSON data if available
    html = resp.text

    # Try to parse JSON blocks embedded in <script type="application/json">
    json_blocks = re.findall(
        r'<script[^>]*type=["\']application/json["\'][^>]*>(.*?)</script>',
        html,
        re.DOTALL,
    )
    raw_models = []
    for block in json_blocks:
        try:
            data = json.loads(block)
            if isinstance(data, list):
                raw_models.extend(data)
            elif isinstance(data, dict) and "models" in data:
                raw_models.extend(data["models"])
        except json.JSONDecodeError:
            pass

    # Fallback: parse basic model cards from the HTML
    if not raw_models:
        # Find model name + description in the HTML structure
        # e.g. <h2 ...>llama3.2</h2>
        name_pattern = re.compile(
            r'href="/library/([a-zA-Z0-9_.-]+)"[^>]*>[^<]*<h2[^>]*>\s*([^<]+)\s*</h2>',
            re.DOTALL,
        )
        re.compile(r"([\d.]+[KMB]?)\s*[Pp]ulls?")
        re.compile(r'<p[^>]*class="[^"]*description[^"]*"[^>]*>(.*?)</p>', re.DOTALL)

        for m in name_pattern.finditer(html):
            slug = m.group(1)
            raw_models.append({"name": slug, "description": ""})

    # Try the official API endpoint as another source
    api_resp = _http_get("https://ollama.com/api/tags")
    if api_resp:
        try:
            api_data = api_resp.json()
            raw_models.extend(api_data.get("models", []))
        except Exception:
            pass

    seen: set[str] = set()
    for item in raw_models:
        name = item.get("name", "") or item.get("model", "")
        if not name or name in seen:
            continue
        seen.add(name)

        description = item.get("description", "") or item.get("readme", "")[:200]
        categories  = _infer_categories(name, description)

        # If item has tags/sizes, expand each variant
        tags = item.get("tags", []) or item.get("sizes", [])
        if not tags:
            # Build a single generic entry
            models.append(ModelInfo(
                name=name,
                tag="latest",
                full_tag=f"{name}:latest",
                size_gb=0.0,
                ram_required_gb=0.0,
                vram_required_gb=0.0,
                quantization="unknown",
                description=description,
                categories=categories,
                source="ollama",
            ))
            continue

        for tag in tags:
            if isinstance(tag, dict):
                tag_name  = tag.get("name", "latest")
                size_str  = tag.get("size", "0")
            else:
                tag_name  = str(tag)
                size_str  = "0"

            size_gb = _parse_size_to_gb(str(size_str))
            ram_gb, vram_gb = _ram_from_size(size_gb, has_gpu=True)
            quant = _infer_quantization(tag_name)

            models.append(ModelInfo(
                name=name,
                tag=tag_name,
                full_tag=f"{name}:{tag_name}",
                size_gb=size_gb,
                ram_required_gb=ram_gb,
                vram_required_gb=vram_gb,
                quantization=quant,
                description=description,
                categories=categories,
                ollama_pull_count=item.get("pull_count", 0),
                source="ollama",
            ))

    return models


def _infer_categories(name: str, description: str) -> list[str]:
    text = (name + " " + description).lower()
    cats = []
    if any(kw in text for kw in ["code", "coder", "starcoder", "deepseek-coder", "codellama"]):
        cats.append("code")
    if any(kw in text for kw in ["vision", "image", "visual", "llava", "bakllava", "minicpm-v"]):
        cats.append("vision")
    if any(kw in text for kw in ["embed", "embedding", "nomic-embed", "mxbai"]):
        cats.append("embedding")
    if any(kw in text for kw in ["math", "mathstral", "qwen2-math"]):
        cats.append("math")
    if any(kw in text for kw in ["reasoning", "think", "reason", "o1", "qwq"]):
        cats.append("reasoning")
    if not cats:
        cats.append("chat")
    return cats


def _infer_quantization(tag: str) -> str:
    tag_lower = tag.lower()
    for quant in ["q8_0", "q6_k", "q5_k_m", "q5_k_s", "q5_0",
                  "q4_k_m", "q4_k_s", "q4_0", "q3_k_m", "q3_k_s",
                  "q2_k", "f16", "f32", "bf16", "iq4_xs"]:
        if quant in tag_lower:
            return quant
    # Numeric size hints
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
    resp = _http_get(HF_API_URL, params=params)
    if resp is None:
        return []

    models = []
    try:
        data = resp.json()
        for item in data:
            model_id = item.get("modelId", "") or item.get("id", "")
            downloads = item.get("downloads", 0)
            description = (item.get("cardData", {}) or {}).get("language", [""])[0]
            tags = item.get("tags", [])

            categories = []
            if "code" in tags or "code" in model_id.lower():
                categories.append("code")
            if "vision" in tags:
                categories.append("vision")
            if not categories:
                categories.append("chat")

            models.append(ModelInfo(
                name=model_id,
                tag="latest",
                full_tag=model_id,
                size_gb=0.0,
                ram_required_gb=0.0,
                vram_required_gb=0.0,
                quantization="gguf",
                description=description or f"HuggingFace GGUF (manual download): {model_id}",
                categories=categories,
                hf_downloads=downloads,
                source="huggingface",
                ollama_pullable=False,   # must be downloaded manually, not via ollama pull
            ))
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
    resp = _http_get(GITHUB_ISSUES_URL)
    if resp is None:
        return issues

    try:
        data = resp.json()
        for issue in data:
            title = issue.get("title", "")
            body  = issue.get("body", "") or ""
            # Look for model names mentioned in the issue
            for word in re.findall(r"\b[a-z][a-z0-9._-]+\b", (title + " " + body).lower()):
                if len(word) > 3 and word not in {"with", "from", "when", "this", "that", "have"}:
                    issues.setdefault(word, []).append(title[:120])
    except Exception as exc:
        logger.warning("GitHub issues parse error: %s", exc)

    return issues


# ── Public API ────────────────────────────────────────────────────────────────

def fetch_registry(include_hf: bool = True) -> list[ModelInfo]:
    """
    Fetch the complete, live model registry from Ollama + optional HF.
    Returns a list of ModelInfo objects sorted by popularity.
    No model data is hardcoded — everything is fetched at runtime.
    """
    logger.info("Fetching Ollama model registry…")
    ollama_models = _fetch_ollama_library()
    logger.info("Found %d Ollama model variants", len(ollama_models))

    hf_models: list[ModelInfo] = []
    if include_hf:
        logger.info("Fetching HuggingFace supplemental data…")
        hf_models = _fetch_hf_popular_models()
        logger.info("Found %d HuggingFace models", len(hf_models))

    known_issues = _fetch_known_issues()
    logger.info("Loaded %d known issue keywords", len(known_issues))

    # Attach known issues to models
    all_models = ollama_models + hf_models
    for m in all_models:
        matched = []
        for keyword, titles in known_issues.items():
            if keyword in m.name.lower():
                matched.extend(titles[:2])
        m.known_issues = matched[:4]

    # Sort: Ollama-pullable models first (can be installed with one command),
    # then by popularity, with HF-only models at the end.
    all_models.sort(
        key=lambda m: (
            int(m.ollama_pullable),          # 1 for Ollama, 0 for HF-only → descending puts Ollama first
            m.ollama_pull_count + m.hf_downloads,
        ),
        reverse=True,
    )
    return all_models
