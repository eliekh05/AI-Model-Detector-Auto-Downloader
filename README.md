# AI Model Detector & Auto Downloader

> Zero third-party dependencies · Deep hardware scanning · Live model registry · Evidence-based recommendations

A precise, transparent tool that scans your entire system — from CPU instruction sets to GPU drivers and available RAM — then queries the **live** Ollama library, Hugging Face, and community issue trackers to recommend and install the best local AI model for your hardware.

**Version 2.0.0** — Zero third-party runtime dependencies. Uses only Python standard library.

---

## Install

```bash
# Recommended: uvx (zero-install, no cache, runs directly)
uvx --no-cache ai-model-detector

# Alternative: pip3 (system-wide)
pip3 install --break-system-packages --no-cache-dir ai-model-detector

# From source (development)
git clone https://github.com/eliekh05/AI-Model-Detector-Auto-Downloader
cd AI-Model-Detector-Auto-Downloader
pip3 install --break-system-packages --no-cache-dir --no-deps -e .
```

**Requirements:** Python ≥ 3.11, internet connection (for live registry fetch)

---

## Quick Start

```bash
# Scan hardware, fetch live registry, recommend + optionally install
ai-model-detector

# Filter by use-case
ai-model-detector --category code
ai-model-detector --category vision
ai-model-detector --category asr

# Show more recommendations
ai-model-detector --top 10

# Import a macOS system profile instead of live scan
ai-model-detector --import ~/Desktop/MyMac.spx

# Pull a specific model directly
ai-model-detector --pull llama3.2:3b

# Output full JSON (pipe to other tools)
ai-model-detector --json > results.json

# List already-installed models
ai-model-detector --installed

# Verbose output with full diagnostics
ai-model-detector -v
```

---

## How It Works

### 1 — System Scan (zero dependencies)

Reads hardware directly from the OS using only the Python standard library:

| Source | Data collected |
|--------|---------------|
| `/proc/cpuinfo` · `sysctl` · `wmic` | CPU brand, cores, AVX / AVX2 / AVX-512 / F16C flags |
| `os.sysconf` · `vm_stat` · `/proc/meminfo` · `GlobalMemoryStatusEx` | RAM total, RAM available |
| `dmidecode` · `system_profiler` · `wmic` | RAM speed |
| `nvidia-smi` | NVIDIA GPU name, VRAM, CUDA version |
| `rocm-smi` · `rocminfo` | AMD GPU name, VRAM, ROCm version |
| `system_profiler SPDisplaysDataType` | Apple Silicon GPU, Metal support |
| `shutil.disk_usage` | Free disk space |
| `ollama --version` | Ollama presence and version |

On macOS, `machdep.cpu.features` and `machdep.cpu.leaf7_features` are both queried so AVX2 is correctly detected on Intel Macs.

On macOS, pass `--import file.spx` to read a `system_profiler` export instead of scanning live hardware.

### 2 — Live Registry Fetch (no caching)

Every run fetches fresh data — no model list is stored in the source code, no cache is created:

- **Ollama library** — all available models with tags, sizes, and pull counts
- **Hugging Face API** — top GGUF models by download count (shown for reference; flagged as manual-download only)
- **Ollama GitHub issues** — open bug reports mapped to model names

If live data cannot be reached, the tool reports the failure clearly. It never silently falls back to stale data.

### 3 — Compatibility Evaluation

Each model is assessed with **factual classifications** — there is no universal 0–100 suitability score:

| Signal | What you see |
|--------|----------------|
| Memory fit | `FITS` · `TIGHT` · `RISKY` · `DOES_NOT_FIT` · `UNKNOWN` |
| Metadata confidence | Verified (known size) vs UNVERIFIED (incomplete metadata) |
| Pullability | Ollama-pullable vs HuggingFace-only (pullable ≠ runnable) |
| GPU vs acceleration | GPU name reported separately from LLM acceleration status |
| Performance | Estimated / inferred / unknown tok/s — never claimed measured unless measured |
| Recommendation labels | Best fit · Lowest memory · Fastest estimated · Coding · Reasoning · Experimental · Not recommended |

**`UNKNOWN` is never treated as `FITS`.** If no verified model fits available memory, automatic installation is disabled and you must explicitly override.

### 4 — Install

Runs `ollama pull <model>` with streaming output. Only models from the Ollama library are offered for auto-install — HuggingFace-only GGUF models are shown in the list but flagged as manual-download. If Ollama isn't installed, platform-specific install instructions are provided.

**Confirmation is always required.** The default answer is always `n` (no download without explicit approval).

---

## Evidence Model

The detector distinguishes between what it knows and what it infers:

### Hardware detection layers

| Layer | What it means | Confidence |
|-------|--------------|------------|
| GPU hardware detected | A GPU device was found by the OS | High |
| Compute API available | Metal/CUDA/ROCm driver is present | High |
| Backend supports that API | Ollama/llama.cpp can use the detected GPU | Medium (inferred from API presence) |
| Backend initialized GPU | The backend confirmed GPU use at startup | Unknown (not verified externally) |
| Inference uses GPU | An actual model run confirmed GPU acceleration | Unknown (not verified externally) |

**Important:** Detecting an integrated GPU (e.g. Intel Iris Plus Graphics) does **not** mean Ollama can use it for LLM acceleration. The tool reports this honestly: "GPU detected but backend acceleration is not established."

### Memory estimation

Memory estimates combine:
- **Model weights** (from size metadata or parameter-count heuristics)
- **KV cache** (estimated from parameter count × context length)
- **Runtime overhead** (Ollama base + activation + OS reserve + safety headroom)
- **GPU shared memory reserve** (for integrated GPUs)

These are estimates, not measurements. Available RAM is a snapshot at scan time, not a guarantee at load time.

---

## CLI Reference

```
usage: ai-model-detector [options]

options:
  --import FILE        Import a macOS .spx system profile
  --category CAT       Filter: asr, audio, chat, coding, reasoning,
                       embeddings, vision, translation, multimodal, unknown
  --top N              Number of recommendations to show (default: 5)
  --json               Output full results as JSON
  --installed          List already-installed Ollama models
  --pull MODEL         Pull a specific model (e.g. llama3.2:3b)
  --no-hf              Skip Hugging Face supplemental data
  --verbose / -v       Enable verbose logging
  --version            Show version and exit
```

---

## Project Structure

```
src/ai_model_detector/
├── __init__.py      — version, author
├── __main__.py      — python -m support
├── scanner.py       — hardware profiler (stdlib only)
├── registry.py      — live model registry fetcher (stdlib only)
├── scorer.py        — compatibility evaluation + recommendations
├── downloader.py    — ollama pull wrapper
├── display.py       — terminal UI (ANSI, no rich)
└── cli.py           — CLI entry point (argparse, no click)
```

---

## Supported Platforms

- **macOS** — Intel and Apple Silicon, Metal detection, .spx import
- **Linux** — NVIDIA (CUDA), AMD (ROCm), Vulkan detection
- **Windows** — NVIDIA (CUDA), AMD detection
- Python ≥ 3.11

---

## Development

```bash
# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest

# Lint
ruff check src/

# Bump version
python scripts/bump_version.py 2.1.0
```

---

## What Changed in 2.0.0

- **Zero third-party runtime dependencies** — removed psutil, requests, rich, click, pywhat
- **No application caches** — live data only; failures reported clearly
- **Hardware detection via stdlib** — subprocess calls, ctypes, /proc/cpuinfo, sysctl
- **HTTP via urllib.request** — no requests library
- **Terminal UI via ANSI codes** — no rich library
- **CLI via argparse** — no click
- **Clearer acceleration reporting** — GPU detection ≠ LLM acceleration
- **UNKNOWN never treated as FITS** — evidence-based compatibility states only
- **No numerical scores** — factual classifications only

---

## License

MIT — see [LICENSE](LICENSE)
