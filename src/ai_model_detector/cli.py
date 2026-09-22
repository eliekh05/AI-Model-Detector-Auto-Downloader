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
from .scorer import RAMFit, partition_recommendations, rank_models, select_install_candidate

logger = logging.getLogger(__name__)


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
        help="Skip Hugging Face supplemental model data.",
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
    console.print(f"\n[cyan]Pulling {target}…[/]  (this may take a while)\n")
    result, msg = pull_model(
        target,
        on_output=lambda line: console.print(f"  [dim]{line}[/]"),
    )
    print_download_result(result, msg)


def _interactive_install(ranked, top_n: int) -> None:
    """
    Offer install with safe defaults.

    - Prefer a verified FITS/TIGHT model as the y/n default.
    - If nothing is verified to fit, say so and require override.
    - Never download without confirmation.
    """
    if not sys.stdin.isatty():
        console.print(
            "\n[dim]Non-interactive session — skipping download prompts. "
            "Re-run in a terminal to install.[/]"
        )
        return

    default, risky_top, advisory = select_install_candidate(ranked)
    recommended, potential, partition_advisory = partition_recommendations(ranked)
    pullable = [sm for sm in ranked if sm.installable]
    recommended_tags = {sm.model.full_tag for sm in recommended}

    # Filter out DOES_NOT_FIT models from any install offer — they will not work
    pullable_no_over = [sm for sm in pullable if sm.ram_fit != RAMFit.OVER and not sm.disqualified]

    console.print()
    if not recommended:
        console.print(f"[yellow]{partition_advisory or advisory}[/]")
    elif advisory:
        console.print(f"[yellow]{advisory}[/]")

    if default is not None:
        fit_note = default.ram_fit.value
        if default.ram_fit == RAMFit.RISKY:
            fit_note += ", may need freeing RAM"
        if Confirm.ask(
            f"[bold]Install recommended[/] [cyan]{default.model.full_tag}[/] ({fit_note}, verified) via `ollama pull`?"
        ):
            _confirm_and_pull(default.model.full_tag)
            return

        # User declined the safer default — optionally offer override for unverified top
        # Only offer override for RISKY/UNKNOWN, never for OVER/disqualified
        if (
            risky_top is not None
            and risky_top.model.full_tag != default.model.full_tag
            and risky_top.ram_fit != RAMFit.OVER
            and not risky_top.disqualified
        ):
            risk_detail = ""
            if risky_top.estimated_total_ram_gb > 0:
                risk_detail = (
                    f" (est. ~{risky_top.estimated_total_ram_gb:.1f} GB needed, "
                    f"{risky_top.ram_fit.value})"
                )
            if Confirm.ask(
                f"[bold red]Override[/] and install "
                f"[cyan]{risky_top.model.full_tag}[/]{risk_detail} anyway?\n"
                f"  [dim]This model is not a verified fit. It may be slow, unstable, "
                f"or fail to load.[/]",
                default=False,
            ):
                _confirm_and_pull(risky_top.model.full_tag)
                return
    elif risky_top is not None and risky_top.ram_fit != RAMFit.OVER and not risky_top.disqualified:
        risk_detail = ""
        if risky_top.estimated_total_ram_gb > 0:
            risk_detail = (
                f" (est. ~{risky_top.estimated_total_ram_gb:.1f} GB needed, "
                f"{risky_top.ram_fit.value})"
            )
        if Confirm.ask(
            f"[bold red]Override[/] and install "
            f"[cyan]{risky_top.model.full_tag}[/]{risk_detail} anyway?\n"
            f"  [dim]No verified fit found. This model may be slow, unstable, "
            f"or fail to load.[/]",
            default=False,
        ):
            _confirm_and_pull(risky_top.model.full_tag)
            return
    else:
        console.print(
            "[yellow]All top recommendations require manual download — "
            "no Ollama-library models found in the live registry right now.[/]"
        )
        return

    # Manual pick — only from models that aren't OVER/disqualified
    if not pullable_no_over:
        return

    # Manual pick from pullable list (recommended first, then potential), filtered
    ordered_pullable = [sm for sm in recommended if sm.installable and sm.ram_fit != RAMFit.OVER and not sm.disqualified] + [
        sm for sm in potential if sm.installable and sm.ram_fit != RAMFit.OVER and not sm.disqualified
    ]
    if not ordered_pullable:
        ordered_pullable = pullable_no_over[:top_n]

    pullable_choices = {str(i + 1): sm for i, sm in enumerate(ordered_pullable[:top_n])}
    console.print("\n[dim]Ollama-installable options (DOES_NOT_FIT models excluded):[/]")
    for k, sm in pullable_choices.items():
        flag = "verified" if sm.verified else "UNVERIFIED"
        section = "recommended" if sm.model.full_tag in recommended_tags else "potential"
        labels = ", ".join(sm.label_names[:2]) if sm.labels else "—"
        console.print(
            f"  [{k}] {sm.model.full_tag}  ({sm.ram_fit.value}, {flag}, {section}; {labels})"
        )
    console.print("  [s] Skip / exit\n")

    choice = Prompt.ask(
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
    if needs_override and not Confirm.ask(
        f"[bold red]Confirm override[/] for {chosen.model.full_tag} ({chosen.ram_fit.value})?\n"
        f"  [dim]This model is not a verified fit. It may be slow, unstable, or fail to load.[/]",
        default=False,
    ):
        console.print("[dim]Skipped.[/]")
        return

    if Confirm.ask(
        f"Install [cyan]{chosen.model.full_tag}[/] via `ollama pull`?",
        default=True,
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
        console.print("[yellow]Could not fetch model registry. Check your internet connection and try again.[/]")
        sys.exit(1)

    console.print(f"\n[dim]Registry loaded: {len(registry)} model variants from live sources[/]")

    # ── Evaluate & recommend ─────────────────────────────────────────────────
    with spinner("Evaluating models against your hardware…") as prog:
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
    _interactive_install(ranked, args.top)

    console.print("\n[dim]Done. Run [cyan]ollama run <model>[/] to use an installed model.[/]\n")


if __name__ == "__main__":
    run()
