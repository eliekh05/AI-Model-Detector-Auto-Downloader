# AI Model Detector & Auto Downloader

> Deep hardware scanning · Live model registry · Smart recommendations · Auto install

A precise, transparent tool that scans your entire system — from OS and CPU instruction sets to GPU drivers and available VRAM — then queries the **live** Ollama library, Hugging Face, and community issue trackers to recommend and install the best local AI model for your hardware.

Unlike tools that rely solely on GPU data or maintain a hardcoded model list, this tool fetches its registry fresh every run, so newly released models appear automatically without requiring a software update.

---

## Install

```bash
pip3 install ai-model-detector --break-system-packages
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
ai-model-detector --category math
ai-model-detector --category reasoning

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
```

---

## How It Works

### 1 — System Scan

Reads hardware directly from the OS — no config file needed:

| Source | Data collected |
|--------|---------------|
| `/proc/cpuinfo` · `sysctl` · `wmic` | CPU brand, cores, AVX / AVX2 / AVX-512 / F16C flags |
| `psutil` | RAM total, RAM available |
| `dmidecode` · `system_profiler` · `wmic` | RAM speed |
| `nvidia-smi` | NVIDIA GPU name, VRAM, CUDA version |
| `rocm-smi` · `rocminfo` | AMD GPU name, VRAM, ROCm version |
| `system_profiler SPDisplaysDataType` | Apple Silicon GPU, Metal support |
| `psutil.disk_usage` | Free disk space |
| `ollama --version` | Ollama presence and version |

On macOS, `machdep.cpu.features` and `machdep.cpu.leaf7_features` are both queried so AVX2 is correctly detected on Intel Macs (it only appears in `leaf7_features`).

On macOS, pass `--import file.spx` to read a `system_profiler` export instead of scanning live hardware.

### 2 — Live Registry Fetch

Every run fetches fresh data — no model list is stored in the source code:

- **Ollama library** — all available models with tags, sizes, and pull counts
- **Hugging Face API** — top GGUF models by download count (shown for reference; flagged as manual-download only)
- **Ollama GitHub issues** — open bug reports mapped to model names

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

`UNKNOWN` is never treated as `FITS`. If no verified model fits available memory, automatic installation is disabled and you must explicitly override.

### 4 — Install

Runs `ollama pull <model>` with live streaming output. Only models from the Ollama library are offered for auto-install — HuggingFace-only GGUF models are shown in the list but flagged as manual-download. If Ollama isn't installed, platform-specific install instructions are provided.

---

## CLI Reference

```
usage: ai-model-detector [options]

options:
  --import FILE        Import a macOS .spx system profile
  --category CAT       Filter: chat | code | vision | math | reasoning | embedding
  --top N              Number of recommendations to show (default: 5)
  --json               Output full results as JSON
  --installed          List already-installed Ollama models
  --pull MODEL         Pull a specific model (e.g. llama3.2:3b)
  --no-hf              Skip Hugging Face supplemental data
  --verbose / -v       Enable debug logging
  --version            Show version and exit
```

---

## macOS `.spx` Import

Export your system profile from the macOS System Information app:

1. Open **System Information** (`Cmd+Space` → "System Information")
2. **File → Save…** → choose **System Information (.spx)**
3. Run: `ai-model-detector --import ~/Desktop/MyMac.spx`

---

## Why Not Just Use GPU Data?

Tools that only look at GPU VRAM miss critical constraints:

- A model might fit in VRAM but not in RAM when layers spill to CPU
- CPU instruction sets (AVX2 vs AVX-512) drastically affect CPU-offload speed
- Free disk space at download time is often the real bottleneck
- Community bug reports reveal models that perform poorly on specific hardware regardless of specs

This tool checks all of these, not just VRAM.

---

## Project Structure

```
src/ai_model_detector/
├── scanner.py    — deep hardware profiler (live + .spx import)
├── registry.py   — live model registry fetcher (Ollama + HuggingFace)
├── scorer.py     — compatibility evaluation + explainable recommendations
├── downloader.py — ollama pull with ANSI-stripped streaming output
├── display.py    — Rich terminal UI
└── cli.py        — CLI entry point
```

---

## Contributing

Issues, hardware reports, and PRs welcome at  
[github.com/eliekh05/AI-Model-Detector-Auto-Downloader](https://github.com/eliekh05/AI-Model-Detector-Auto-Downloader)

---

## License

MIT — see [LICENSE](LICENSE)
