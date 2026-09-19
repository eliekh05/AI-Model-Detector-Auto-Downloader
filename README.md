# AI Model Detector & Auto Downloader

> Deep hardware scanning · Live model registry · Smart recommendations · Auto install

A precise, transparent tool that scans your entire system — from OS and CPU instruction sets to GPU drivers and available VRAM — then queries the **live** Ollama library, Hugging Face, and community issue trackers to recommend and install the best local AI model for your hardware.

Unlike tools that rely solely on GPU data or maintain a hardcoded model list, this tool fetches its registry fresh every run, so newly released models appear automatically without requiring a software update.

---

## Features

- **Full system scan** — OS, CPU (AVX/AVX2/AVX-512/F16C), RAM, GPU (CUDA/Metal/ROCm), disk
- **Live registry** — queries Ollama library + Hugging Face at runtime; no hardcoded model list
- **Hardware-aware scoring** — ranks every model variant against your actual available RAM, VRAM, and disk
- **Community signals** — pulls open bug reports from the Ollama GitHub to flag known issues
- **Auto-download** — runs `ollama pull` with live progress output
- **Fallback guidance** — if Ollama isn't installed, provides platform-specific install steps and GGUF alternatives
- **macOS `.spx` import** — parse a `system_profiler` XML export instead of scanning live hardware
- **JSON output** — pipe the full scored profile to other tools with `--json`
- **Cross-platform** — macOS, Linux, Windows

---

## Quick Start

```bash
# Install
pip install ai-model-detector

# Run (scans hardware, fetches live registry, recommends + optionally installs)
ai-model-detector

# Filter by use-case
ai-model-detector --category code
ai-model-detector --category vision

# Show more recommendations
ai-model-detector --top 10

# Import a macOS system profile instead of live scan
ai-model-detector --import ~/Desktop/MyMac.spx

# Pull a specific model directly
ai-model-detector --pull llama3.2:3b

# Output full JSON
ai-model-detector --json > results.json

# List already-installed models
ai-model-detector --installed
```

---

## How It Works

### 1 — System Scan

Reads hardware directly from the OS without relying on any config file:

| Source | Data collected |
|--------|---------------|
| `/proc/cpuinfo` · `sysctl` · `wmic` | CPU brand, cores, AVX/AVX2/AVX-512 flags |
| `psutil` | RAM total, RAM available |
| `dmidecode` · `system_profiler` · `wmic` | RAM speed |
| `nvidia-smi` | NVIDIA GPU name, VRAM, CUDA version |
| `rocm-smi` · `rocminfo` | AMD GPU name, VRAM, ROCm version |
| `system_profiler SPDisplaysDataType` | Apple Silicon GPU, Metal support |
| `psutil.disk_usage` | Free disk space |
| `ollama --version` | Ollama presence and version |

On macOS, pass `--import file.spx` to read a `system_profiler` XML export (`.spx` are ZIP archives containing plist files).

### 2 — Live Registry Fetch

Every run fetches fresh data — nothing about models is stored or hardcoded in the source code:

- **Ollama library** (`ollama.com/search`) — all available models with tags, sizes, and pull counts
- **Hugging Face API** — top GGUF models by download count
- **Ollama GitHub issues** — open bug reports tagged `bug`, mapped to model names

### 3 — Hardware-Aware Scoring

Each model variant is scored 0–100 against your specific hardware:

| Factor | Effect |
|--------|--------|
| Available RAM vs model RAM requirement | ±20 pts |
| GPU VRAM vs model VRAM requirement | ±20 pts |
| Free disk space | ±30 pts (hard penalty if insufficient) |
| CPU instruction sets (AVX2, AVX-512) | ±5 pts |
| Apple Silicon + Metal | +10 pts |
| Quantization suitability (q4_K_M sweet spot) | ±8 pts |
| Community bug reports | −3 pts per issue |
| Popularity (pull count) | +2–5 pts |

### 4 — Download

Runs `ollama pull <model>` with streaming output. If Ollama isn't present, provides platform-specific installation instructions and GGUF fallback links (llama.cpp, LM Studio).

---

## Installation

**Requirements:** Python ≥ 3.11

```bash
pip install ai-model-detector
```

**From source:**

```bash
git clone https://github.com/eliekh05/AI-Model-Detector-Auto-Downloader
cd AI-Model-Detector-Auto-Downloader
pip install -e .
ai-model-detector
```

---

## CLI Reference

```
usage: ai-model-detector [-h] [--version] [--import FILE] [--category CATEGORY]
                          [--top N] [--json] [--installed] [--pull MODEL]
                          [--no-hf] [--verbose]

options:
  --import FILE        Import a macOS .spx system profile
  --category CATEGORY  Filter: chat | code | vision | math | reasoning | embedding
  --top N              Number of recommendations to show (default: 5)
  --json               Output full results as JSON
  --installed          List already-installed Ollama models
  --pull MODEL         Pull a specific model (e.g. llama3.2:3b)
  --no-hf              Skip Hugging Face supplemental data
  --verbose / -v       Enable debug logging
```

---

## macOS `.spx` Import

Export your system profile from the macOS System Information app:

1. Open **System Information** (`Cmd+Space` → "System Information")
2. **File → Save…** → choose format **System Information (.spx)**
3. Run:
   ```bash
   ai-model-detector --import ~/Desktop/MyMac.spx
   ```

The `.spx` file is a ZIP archive containing plist XML files. The tool reads CPU, RAM, and GPU data directly from it.

---

## Project Structure

```
src/ai_model_detector/
├── __init__.py       # package metadata
├── cli.py            # CLI entry point and interactive flow
├── scanner.py        # deep hardware profiler (live scan + .spx import)
├── registry.py       # live model registry fetcher (Ollama + HuggingFace)
├── scorer.py         # hardware-aware model scoring and ranking
└── display.py        # Rich terminal UI

tests/
├── test_scanner.py
└── test_scorer.py
```

---

## Why Not Just Use GPU Data?

Tools that only look at GPU VRAM miss critical constraints:

- A model might fit in VRAM but not in RAM when quantization layers spill to CPU
- CPU instruction sets (AVX2 vs AVX-512) drastically affect inference speed for CPU-offloaded layers
- Available disk space at download time is often the binding constraint
- Community bug reports reveal models that perform poorly on specific hardware regardless of specs

This tool checks all of these, not just VRAM.

---

## Contributing

Issues, hardware reports, and PRs are welcome at [github.com/eliekh05/AI-Model-Detector-Auto-Downloader](https://github.com/eliekh05/AI-Model-Detector-Auto-Downloader).

---

## License

MIT — see [LICENSE](LICENSE)
