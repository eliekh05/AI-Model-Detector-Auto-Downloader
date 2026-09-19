"""
downloader.py — Model download & installation orchestrator.

Attempts to install models via `ollama pull`.
If Ollama isn't installed, provides platform-specific install instructions
and fallback GGUF download guidance.
"""

import os
import sys
import shutil
import platform
import subprocess
import threading
import time
import logging
from enum import Enum
from typing import Optional, Callable

logger = logging.getLogger(__name__)


class DownloadResult(Enum):
    SUCCESS     = "success"
    ALREADY_EXISTS = "already_exists"
    OLLAMA_MISSING = "ollama_missing"
    PULL_FAILED = "pull_failed"
    CANCELLED   = "cancelled"


def _ollama_model_exists(model_tag: str) -> bool:
    """Check if a model is already pulled via `ollama list`."""
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True, text=True, timeout=10
        )
        return model_tag.split(":")[0] in result.stdout
    except Exception:
        return False


def _get_ollama_install_instructions() -> str:
    system = platform.system()
    if system == "Darwin":
        return (
            "Ollama is not installed.\n"
            "Install it with:\n"
            "  brew install ollama\n"
            "Or download from: https://ollama.com/download/mac\n"
            "Then run: ollama serve"
        )
    if system == "Linux":
        return (
            "Ollama is not installed.\n"
            "Install it with:\n"
            "  curl -fsSL https://ollama.com/install.sh | sh\n"
            "Then run: ollama serve"
        )
    if system == "Windows":
        return (
            "Ollama is not installed.\n"
            "Download the installer from: https://ollama.com/download/windows\n"
            "Run the installer, then Ollama will start automatically."
        )
    return (
        "Ollama not found. Visit https://ollama.com/download to install."
    )


def _get_gguf_fallback(model_name: str) -> str:
    """Return guidance for manually downloading a GGUF file."""
    slug = model_name.replace(":", "-")
    return (
        f"\nAlternative: Download GGUF manually from Hugging Face:\n"
        f"  https://huggingface.co/models?search={slug}+gguf\n"
        f"\nThen load with llama.cpp:\n"
        f"  ./llama-cli -m path/to/model.gguf -p 'Your prompt here'\n"
        f"\nOr use LM Studio: https://lmstudio.ai"
    )


def pull_model(
    model_tag: str,
    on_output: Optional[Callable[[str], None]] = None,
    check_existing: bool = True,
) -> tuple[DownloadResult, str]:
    """
    Pull a model using `ollama pull`.

    Args:
        model_tag:       Full model tag, e.g. "llama3.2:3b"
        on_output:       Optional callback for streaming pull output lines.
        check_existing:  Skip pull if model is already present.

    Returns:
        (DownloadResult, message)
    """
    if not shutil.which("ollama"):
        instructions = _get_ollama_install_instructions()
        fallback     = _get_gguf_fallback(model_tag)
        return DownloadResult.OLLAMA_MISSING, instructions + fallback

    if check_existing and _ollama_model_exists(model_tag):
        return DownloadResult.ALREADY_EXISTS, f"Model '{model_tag}' is already installed."

    logger.info("Starting ollama pull for %s", model_tag)

    try:
        process = subprocess.Popen(
            ["ollama", "pull", model_tag],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        output_lines: list[str] = []

        for line in process.stdout:  # type: ignore[union-attr]
            stripped = line.rstrip()
            output_lines.append(stripped)
            if on_output:
                on_output(stripped)

        process.wait()

        if process.returncode == 0:
            return DownloadResult.SUCCESS, f"Model '{model_tag}' downloaded successfully."
        else:
            err = "\n".join(output_lines[-10:])
            return DownloadResult.PULL_FAILED, f"ollama pull failed:\n{err}"

    except KeyboardInterrupt:
        try:
            process.terminate()
        except Exception:
            pass
        return DownloadResult.CANCELLED, "Download cancelled by user."

    except Exception as exc:
        logger.exception("Unexpected error during pull")
        return DownloadResult.PULL_FAILED, f"Unexpected error: {exc}"


def start_ollama_serve() -> Optional[subprocess.Popen]:
    """
    Ensure the Ollama server is running.
    Returns the subprocess if we started it, None if it was already running.
    """
    if not shutil.which("ollama"):
        return None

    # Check if already running
    try:
        import urllib.request
        urllib.request.urlopen("http://localhost:11434", timeout=2)
        return None  # already running
    except Exception:
        pass

    logger.info("Starting ollama serve in background…")
    try:
        proc = subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        time.sleep(2)  # give it a moment to bind
        return proc
    except Exception:
        return None


def list_installed_models() -> list[str]:
    """Return list of already-installed model names via `ollama list`."""
    if not shutil.which("ollama"):
        return []
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True, text=True, timeout=10
        )
        lines = result.stdout.strip().splitlines()
        models = []
        for line in lines[1:]:  # skip header
            parts = line.split()
            if parts:
                models.append(parts[0])
        return models
    except Exception:
        return []
