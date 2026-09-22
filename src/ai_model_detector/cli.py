"""
cli.py — Command-line interface (stdlib only, zero dependencies).

Usage:
    ai-model-detector                      # full scan + recommend + optional install
    ai-model-detector --import file.spx    # import macOS .spx system file
    ai-model-detector --category code      # filter by use-case
    ai-model-detector --top 10             # show more recommendations
    ai-model-detector --json               # output full results as JSON
    ai-model-detector --installed          # list already-installed models
    ai-model-detector --pull llama3.2:3b   # pull a specific model directly
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from . import __version__
from .display import (
    print_banner,
    print_download_result,
    print_recommendations,
    print_system_profile,
    spinner,
)
from .downloader import list_installed_models, pull_model, start_ollama_serve
from .registry import fetch_registry
from .scanner import scan_system
from .scorer import EvaluatedModel, RAMFit, partition_recommendations, rank_models, select_install_candidate

logger = logging.getLogger(__name__)


# ── Simple input helpers (replace rich.prompt) ──────────────────────────


def _confirm(prompt: str, default: bool = False) -> bool:
    """Ask a y/n question. Returns True for yes, False for no."""
    suffix = " [Y/n]" if default else " [y/N]"
    try:
        answer = input(f"{prompt}{suffix} ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return default
    if not answer:
        return default
    return answer in ("y", "yes")


def _prompt_choice(prompt: str, choices: list[str], default: str = "") -> str:
    """Ask user to pick from a list of choices."""
    choice_str = "/".join(choices)
    try:
        answer = input(f"{prompt} ({choice_str}) [{default}]: ").strip()
    except (EOFError, KeyboardInterrupt):
        return default
    if not answer:
        return default
    if answer in choices:
        return answer
    return default


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ai-model-detector",
        description=(
            "Deep hardware scanner that recommends and auto-downloads the best local AI model for your system."
        ),
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument(
        "--import",
        dest="spx_file",
        metavar="FILE",
        help="Import a macOS .spx system profile instead of scanning live hardware.",
    )
    parser.add_argument(
        "--category",
        metavar="CATEGORY",
        help=(
            "Filter by task category: asr, audio, chat, coding, reasoning, "
            "embeddings, vision, translation, multimodal, unknown"
        ),
    )
    parser.add_argument(
        "--top",
        type=int,
        default=5,
        metavar="N",
        help="Number of top recommendations to show (default: 5)",
    )
    parser.add_argument(
        "--json",
        dest="output_json",
        action="store_true",
        help="Output full results as JSON and exit.",
    )
    parser.add_argument(
        "--installed",
        action="store_true",
        help="List already-installed Ollama models and exit.",
    )
    parser.add_argument(
        "--pull",
        metavar="MODEL",
        help="Pull a specific model tag directly (e.g. llama3.2:3b).",
    )
    parser.add_argument(
        "--no-hf",
        action="store_true",
        help="Skip Hugging Face supplemental data.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging.",
    )
    return parser


def _confirm_and_pull(target: str) -> None:
    """Always require confirmation before downloading."""
    start_ollama_serve()
    print(f"\n{_ansi('cyan')}Pulling {target}…{_ansi('reset')}  (this may take a while)\n")
    result, msg = pull_model(
        target,
        on_output=lambda line: print(f"  {_ansi('dim')}{line}{_ansi('reset')}"),
    )
    print_download_result(result, msg)


def _ansi(code: str) -> str:
    """Return ANSI escape code if TTY, empty string otherwise."""
    if not (hasattr(sys.stdout, "isatty") and sys.stdout.isatty()):
        return ""
    codes = {
        "reset": "\033[0m",
        "dim": "\033[2m",
        "bold": "\033[1m",
        "cyan": "\033[36m",
        "green": "\033[32m",
        "yellow": "\033[33m",
        "red": "\033[31m",
    }
    return codes.get(code, "")


def _interactive_install(ranked, top_n: int, available_ram_gb: float = 0.0) -> None:
    """
    Offer install with safe defaults.

    - Prefer a verified FITS/TIGHT model as the y/n default.
    - If nothing is verified to fit, say so and require override.
    - Never download without confirmation.
    """
    if not sys.stdin.isatty():
        print(
            f"\n{_ansi('dim')}Non-interactive session — skipping download prompts. "
            f"Re-run in a terminal to install.{_ansi('reset')}"
        )
        return

    default, risky_top, advisory = select_install_candidate(ranked, available_ram_gb=available_ram_gb)
    recommended, potential, partition_advisory = partition_recommendations(ranked)
    recommended_tags = {sm.model.full_tag for sm in recommended}

    print()
    if not recommended:
        print(f"{_ansi('yellow')}{partition_advisory or advisory}{_ansi('reset')}")
    elif advisory:
        print(f"{_ansi('yellow')}{advisory}{_ansi('reset')}")

    def _shortfall_line(sm: EvaluatedModel) -> str:
        """Format a shortfall/fit line for override prompts."""
        if sm.estimated_total_ram_gb <= 0:
            return ""
        avail = available_ram_gb
        shortfall = sm.estimated_total_ram_gb - avail
        if shortfall > 0.2:
            return (f"  {_ansi('red')}Estimated {shortfall:.1f} GB shortfall "
                    f"({sm.estimated_total_ram_gb:.1f} GB needed, {avail:.1f} GB available). "
                    f"Metadata confidence: {sm.memory_confidence.value}.{_ansi('reset')}")
        if shortfall > -0.2:
            return (f"  {_ansi('yellow')}No practical headroom (~0 GB free). "
                    f"({sm.estimated_total_ram_gb:.1f} GB needed, {avail:.1f} GB available). "
                    f"Metadata confidence: {sm.memory_confidence.value}.{_ansi('reset')}")
        return (f"  {_ansi('dim')}Estimated {abs(shortfall):.1f} GB headroom "
                f"({sm.estimated_total_ram_gb:.1f} GB needed, {avail:.1f} GB available). "
                f"Metadata confidence: {sm.memory_confidence.value}.{_ansi('reset')}")

    if default is not None:
        fit_note = default.ram_fit.value
        if default.ram_fit == RAMFit.RISKY:
            fit_note += ", may need freeing RAM"
        if _confirm(
            f"{_ansi('bold')}Install recommended{_ansi('reset')} "
            f"{_ansi('cyan')}{default.model.full_tag}{_ansi('reset')} "
            f"({fit_note}, verified) via `ollama pull`?"
        ):
            _confirm_and_pull(default.model.full_tag)
            return

        # User declined the safer default — optionally offer override
        if risky_top is not None and risky_top.model.full_tag != default.model.full_tag:
            shortfall_line = _shortfall_line(risky_top)
            if _confirm(
                f"{_ansi('red')}{_ansi('bold')}Override{_ansi('reset')} and install "
                f"{_ansi('cyan')}{risky_top.model.full_tag}{_ansi('reset')} "
                f"({risky_top.ram_fit.value}) anyway?\n"
                f"  {_ansi('dim')}This model is not a verified fit. It may be slow, unstable, "
                f"or fail to load.{_ansi('reset')}\n{shortfall_line}",
                default=False,
            ):
                _confirm_and_pull(risky_top.model.full_tag)
                return
    elif risky_top is not None:
        shortfall_line = _shortfall_line(risky_top)
        if _confirm(
            f"{_ansi('red')}{_ansi('bold')}Override{_ansi('reset')} and install "
            f"{_ansi('cyan')}{risky_top.model.full_tag}{_ansi('reset')} "
            f"({risky_top.ram_fit.value}) anyway?\n"
            f"  {_ansi('dim')}No verified fit found. This model may be slow, unstable, "
            f"or fail to load.{_ansi('reset')}\n{shortfall_line}",
            default=False,
        ):
            _confirm_and_pull(risky_top.model.full_tag)
            return
    else:
        print(
            f"{_ansi('yellow')}All top recommendations require manual download — "
            f"no Ollama-library models found in the live registry right now.{_ansi('reset')}"
        )
        return

    # Manual pick — only from models that aren't OVER/disqualified
    pullable_no_over = [sm for sm in ranked if sm.installable and sm.ram_fit != RAMFit.OVER and not sm.disqualified]
    if not pullable_no_over:
        return

    ordered_pullable = [sm for sm in recommended if sm.installable and sm.ram_fit != RAMFit.OVER and not sm.disqualified] + [
        sm for sm in potential if sm.installable and sm.ram_fit != RAMFit.OVER and not sm.disqualified
    ]
    if not ordered_pullable:
        ordered_pullable = pullable_no_over[:top_n]

    pullable_choices = {str(i + 1): sm for i, sm in enumerate(ordered_pullable[:top_n])}
    print(f"\n{_ansi('dim')}Ollama-installable options (DOES_NOT_FIT models excluded):{_ansi('reset')}")
    for k, sm in pullable_choices.items():
        flag = "verified" if sm.verified else "UNVERIFIED"
        section = "recommended" if sm.model.full_tag in recommended_tags else "potential"
        labels = ", ".join(sm.label_names[:2]) if sm.labels else "—"
        ram_note = ""
        if sm.estimated_total_ram_gb > 0:
            shortfall = sm.estimated_total_ram_gb - available_ram_gb
            if shortfall > 0.2:
                ram_note = f" {_ansi('red')}⚠ ~{shortfall:.1f} GB shortfall{_ansi('reset')}"
            elif shortfall > -0.2:
                ram_note = f" {_ansi('yellow')}⚠ no headroom{_ansi('reset')}"
        print(
            f"  [{k}] {sm.model.full_tag}  ({sm.ram_fit.value}, {flag}, {section}; {labels}){ram_note}"
        )
    print("  [s] Skip / exit\n")

    choice = _prompt_choice(
        "Enter number to install, or 's' to skip",
        choices=[*pullable_choices.keys(), "s"],
        default="s",
    )
    if choice == "s":
        return

    chosen = pullable_choices[choice]
    needs_override = chosen.unverified or chosen.ram_fit in (
        RAMFit.UNKNOWN,
        RAMFit.RISKY,
    )
    if needs_override:
        shortfall_line = _shortfall_line(chosen)
        if not _confirm(
            f"{_ansi('red')}{_ansi('bold')}Confirm override{_ansi('reset')} "
            f"for {chosen.model.full_tag} ({chosen.ram_fit.value})?\n"
            f"  {_ansi('dim')}This model is not a verified fit. It may be slow, unstable, "
            f"or fail to load.{_ansi('reset')}\n"
            f"{shortfall_line}",
            default=False,
        ):
            print(f"{_ansi('dim')}Skipped.{_ansi('reset')}")
            return

    if _confirm(
        f"Install {_ansi('cyan')}{chosen.model.full_tag}{_ansi('reset')} via `ollama pull`?",
        default=False,
    ):
        _confirm_and_pull(chosen.model.full_tag)


def run() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    print_banner()

    # ── --installed ─────────────────────────────────────────────────────────
    if args.installed:
        models = list_installed_models()
        if models:
            print(f"\n{_ansi('bold')}{_ansi('cyan')}Installed Ollama models:{_ansi('reset')}")
            for m in models:
                print(f"  • {m}")
        else:
            print(f"{_ansi('yellow')}No Ollama models found (or Ollama not installed).{_ansi('reset')}")
        return

    # ── --pull ───────────────────────────────────────────────────────────────
    if args.pull:
        print(f"\nPulling {_ansi('cyan')}{args.pull}{_ansi('reset')}…\n")
        start_ollama_serve()
        result, msg = pull_model(
            args.pull,
            on_output=lambda line: print(f"  {line}"),
        )
        print_download_result(result, msg)
        return

    # ── Scan system ───────────────────────────────────────────────────────────
    spx_path = Path(args.spx_file) if args.spx_file else None
    scan_label = f"Importing {spx_path.name}" if spx_path else "Scanning system hardware"

    with spinner(scan_label):
        profile = scan_system(spx_path=spx_path)

    print_system_profile(profile)

    # ── Fetch live registry ───────────────────────────────────────────────────
    with spinner("Fetching live model registry from Ollama & HuggingFace…"):
        registry = fetch_registry(include_hf=not args.no_hf)

    if not registry:
        print(f"{_ansi('yellow')}Could not fetch model registry. Check your internet connection and try again.{_ansi('reset')}")
        sys.exit(1)

    print(f"\n{_ansi('dim')}Registry loaded: {len(registry)} model variants from live sources{_ansi('reset')}")

    # ── Evaluate & recommend ─────────────────────────────────────────────────
    with spinner("Evaluating models against your hardware…"):
        ranked = rank_models(
            registry,
            profile,
            top_n=max(args.top, 10),
            category_filter=args.category,
        )

    if not ranked:
        print(f"{_ansi('yellow')}No models matched your filters and hardware constraints.{_ansi('reset')}")
        sys.exit(1)

    # ── JSON output ───────────────────────────────────────────────────────────
    if args.output_json:
        output = {
            "system": profile.to_dict(),
            "recommendations": [
                {
                    "rank": i + 1,
                    "model": sm.model.__dict__,
                    "ram_fit": sm.ram_fit.value,
                    "fits_ram": sm.fits_ram,
                    "fits_vram": sm.fits_vram,
                    "fits_disk": sm.fits_disk,
                    "verified": sm.verified,
                    "unverified": sm.unverified,
                    "installable": sm.installable,
                    "pullable": sm.installable,
                    "runtime_compatible": sm.runtime_compatible,
                    "acceleration": sm.acceleration.value,
                    "gpu_detected": sm.gpu_detected,
                    "performance_basis": sm.performance_basis.value,
                    "estimated_speed": sm.estimated_speed,
                    "estimated_tps": sm.estimated_tps,
                    "estimated_total_ram_gb": sm.estimated_total_ram_gb,
                    "memory_confidence": sm.memory_confidence.value,
                    "missing_metadata": sm.missing_metadata,
                    "params_b": sm.params_b,
                    "will_use_gpu": sm.will_use_gpu,
                    "labels": sm.label_names,
                    "rank_reason": sm.rank_reason,
                    "category_source": getattr(sm.model, "category_source", "unknown"),
                    "explanation": sm.explanation,
                    "warnings": sm.warnings,
                }
                for i, sm in enumerate(ranked[: args.top])
            ],
        }
        print(json.dumps(output, indent=2, default=str))
        return

    # ── Display recommendations ───────────────────────────────────────────────
    print_recommendations(
        ranked,
        top=args.top,
        available_ram_gb=profile.ram.available_gb,
        verbose=args.verbose,
    )

    # ── Interactive download ──────────────────────────────────────────────────
    _interactive_install(ranked, args.top, available_ram_gb=profile.ram.available_gb)

    print(f"\n{_ansi('dim')}Done. Run {_ansi('cyan')}ollama run <model>{_ansi('reset')}{_ansi('dim')} to use an installed model.{_ansi('reset')}\n")


if __name__ == "__main__":
    run()
