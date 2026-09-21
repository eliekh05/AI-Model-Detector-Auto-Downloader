"""
cli.py — Command-line interface for AI Model Detector & Auto Downloader.

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

from rich.prompt import Confirm, Prompt

from . import __version__
from .display import (
    console,
    print_banner,
    print_download_result,
    print_recommendations,
    print_system_profile,
    spinner,
)
from .downloader import list_installed_models, pull_model, start_ollama_serve
from .registry import fetch_registry
from .scanner import scan_system
from .scorer import rank_models

logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ai-model-detector",
        description=(
            "Deep hardware scanner that recommends and auto-downloads "
            "the best local AI model for your system."
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
        help="Filter recommendations by category: chat, code, vision, math, reasoning, embedding",
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
        help="Skip Hugging Face supplemental model data.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging.",
    )
    return parser


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
            console.print("\n[bold cyan]Installed Ollama models:[/]")
            for m in models:
                console.print(f"  • {m}")
        else:
            console.print("[yellow]No Ollama models found (or Ollama not installed).[/]")
        return

    # ── --pull ───────────────────────────────────────────────────────────────
    if args.pull:
        console.print(f"\nPulling [cyan]{args.pull}[/]…\n")
        start_ollama_serve()
        result, msg = pull_model(
            args.pull,
            on_output=lambda line: console.print(f"  {line}"),
        )
        print_download_result(result, msg)
        return

    # ── Scan system ───────────────────────────────────────────────────────────
    spx_path = Path(args.spx_file) if args.spx_file else None
    scan_label = f"Importing {spx_path.name}" if spx_path else "Scanning system hardware"

    with spinner(scan_label) as prog:
        prog.add_task("", total=None)
        profile = scan_system(spx_path=spx_path)

    print_system_profile(profile)

    # ── Fetch live registry ───────────────────────────────────────────────────
    with spinner("Fetching live model registry from Ollama & HuggingFace…") as prog:
        prog.add_task("", total=None)
        registry = fetch_registry(include_hf=not args.no_hf)

    if not registry:
        console.print(
            "[yellow]Could not fetch model registry. "
            "Check your internet connection and try again.[/]"
        )
        sys.exit(1)

    console.print(
        f"\n[dim]Registry loaded: {len(registry)} model variants from live sources[/]"
    )

    # ── Score & rank ─────────────────────────────────────────────────────────
    with spinner("Scoring models against your hardware…") as prog:
        prog.add_task("", total=None)
        ranked = rank_models(
            registry,
            profile,
            top_n=max(args.top, 10),
            category_filter=args.category,
        )

    if not ranked:
        console.print("[yellow]No models matched your filters and hardware constraints.[/]")
        sys.exit(1)

    # ── JSON output ───────────────────────────────────────────────────────────
    if args.output_json:
        output = {
            "system": profile.to_dict(),
            "recommendations": [
                {
                    "rank": i + 1,
                    "model": sm.model.__dict__,
                    "score": sm.score,
                    "fits_ram": sm.fits_ram,
                    "fits_vram": sm.fits_vram,
                    "fits_disk": sm.fits_disk,
                    "estimated_speed": sm.estimated_speed,
                    "will_use_gpu": sm.will_use_gpu,
                    "explanation": sm.explanation,
                    "warnings": sm.warnings,
                }
                for i, sm in enumerate(ranked[: args.top])
            ],
        }
        print(json.dumps(output, indent=2, default=str))
        return

    # ── Display recommendations ───────────────────────────────────────────────
    print_recommendations(ranked, top=args.top)

    # ── Interactive download ──────────────────────────────────────────────────
    # Always install from Ollama library — HuggingFace-only models cannot be
    # pulled via `ollama pull` and must be downloaded manually.
    pullable = [sm for sm in ranked if sm.model.ollama_pullable]
    top_pullable = pullable[0] if pullable else None
    console.print()

    if top_pullable is None:
        console.print(
            "[yellow]All top recommendations require manual download — "
            "no Ollama-library models found in the live registry right now.[/]"
        )
    elif Confirm.ask(
        f"[bold]Install top recommendation[/] "
        f"[cyan]{top_pullable.model.full_tag}[/] via `ollama pull`?"
    ):
        start_ollama_serve()
        target = top_pullable.model.full_tag
        console.print(f"\n[cyan]Pulling {target}…[/]  (this may take a while)\n")
        result, msg = pull_model(
            target,
            on_output=lambda line: console.print(f"  [dim]{line}[/]"),
        )
        print_download_result(result, msg)

    else:
        # Let the user pick from Ollama-pullable models only
        pullable_choices = {
            str(i + 1): sm.model.full_tag
            for i, sm in enumerate(pullable[: args.top])
        }
        pullable_choices["s"] = "skip"

        console.print("\n[dim]Ollama-installable options:[/]")
        for k, v in pullable_choices.items():
            if k != "s":
                console.print(f"  [{k}] {v}")
        console.print("  [s] Skip / exit\n")

        choice = Prompt.ask(
            "Enter number to install, or 's' to skip",
            choices=list(pullable_choices.keys()),
            default="s",
        )

        if choice != "s":
            target = pullable_choices[choice]
            start_ollama_serve()
            console.print(f"\n[cyan]Pulling {target}…[/]  (this may take a while)\n")
            result, msg = pull_model(
                target,
                on_output=lambda line: console.print(f"  [dim]{line}[/]"),
            )
            print_download_result(result, msg)

    console.print("\n[dim]Done. Run [cyan]ollama run <model>[/] to start chatting.[/]\n")


if __name__ == "__main__":
    run()
