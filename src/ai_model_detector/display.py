"""
display.py — Rich terminal UI for scan results and recommendations.
"""

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from .scanner import SystemProfile
from .scorer import (
    AccelerationStatus,
    Confidence,
    EvaluatedModel,
    GPUTier,
    PerformanceBasis,
    RAMFit,
    partition_recommendations,
)

console = Console()

# Warnings that apply to the whole machine — print once, not per model card.
_SYSTEM_WARNING_MARKERS = (
    "integrated gpu does not prove",
    "detecting an integrated gpu",
    "llm acceleration: unverified",
    "do not assume llm acceleration",
)


def print_banner() -> None:
    console.print(
        Panel(
            "[bold cyan]AI Model Detector & Auto Downloader[/]\n"
            "[dim]Deep hardware scanning · Live model registry · Smart recommendations[/]",
            box=box.DOUBLE_EDGE,
            style="bold",
        )
    )


def print_system_profile(profile: SystemProfile) -> None:
    table = Table(title="System Profile", box=box.ROUNDED, show_header=False)
    table.add_column("Field", style="bold cyan", width=28)
    table.add_column("Value", style="white")

    table.add_row("OS", f"{profile.os_name} {profile.os_version[:40]}")
    table.add_row("Architecture", profile.os_arch)
    table.add_row("CPU", profile.cpu.brand[:60])
    table.add_row("CPU Cores", f"{profile.cpu.cores_physical}P / {profile.cpu.cores_logical}L")
    table.add_row(
        "CPU Extensions",
        " ".join(
            f
            for f, ok in [
                ("AVX", profile.cpu.supports_avx),
                ("AVX2", profile.cpu.supports_avx2),
                ("AVX-512", profile.cpu.supports_avx512),
                ("F16C", profile.cpu.supports_f16c),
            ]
            if ok
        )
        or "None detected",
    )
    table.add_row("RAM Total", f"{profile.ram.total_gb:.1f} GB")
    table.add_row("RAM Available", f"{profile.ram.available_gb:.1f} GB")
    if profile.ram.speed_mhz:
        table.add_row("RAM Speed", f"{profile.ram.speed_mhz} MHz")

    if profile.gpus:
        for i, gpu in enumerate(profile.gpus):
            label = f"GPU {i + 1}"
            if gpu.is_integrated or (gpu.vram_gb is None or gpu.vram_gb == 0):
                mem = "integrated — shared system memory"
            elif gpu.vram_gb:
                mem = f"{gpu.vram_gb:.1f} GB dedicated VRAM"
            else:
                mem = "VRAM unknown"
            backends = []
            if gpu.cuda_version:
                backends.append(f"CUDA {gpu.cuda_version}")
            if gpu.metal_support:
                backends.append("Metal")
            if gpu.rocm_version:
                backends.append("ROCm")
            backend_str = f" · APIs: {'/'.join(backends)}" if backends else ""
            table.add_row(label, f"{gpu.name} · {mem}{backend_str}")

        # Separate detection from usable LLM acceleration
        from .scorer import _classify_gpu, acceleration_for_tier

        tier, _ = _classify_gpu(profile.gpus, profile.os_name, profile.os_arch)
        accel = acceleration_for_tier(tier, profile.os_name, profile.os_arch, profile.metal_available)
        table.add_row("GPU detected", "yes")
        table.add_row("LLM acceleration", _accel_profile_str(accel, tier))

        # Show what's verified vs unverified for non-established acceleration
        if accel in (AccelerationStatus.UNVERIFIED, AccelerationStatus.METAL_INTEL):
            table.add_row(
                "Acceleration detail",
                "[dim]Detection ≠ confirmed backend use. "
                "GPU detected, compute API "
                + ("Metal available" if profile.metal_available else "not confirmed")
                + " — actual GPU acceleration during inference remains unverified.[/]",
            )
    else:
        table.add_row("GPU", "[yellow]No GPU detected[/]")
        table.add_row("GPU detected", "no")
        table.add_row("LLM acceleration", "CPU-only")

    if profile.disk:
        table.add_row(
            "Disk Free",
            f"{profile.disk.free_gb:.1f} GB / {profile.disk.total_gb:.1f} GB",
        )

    ollama_status = (
        f"[green]✓ Installed[/] ({profile.ollama_version})"
        if profile.ollama_installed
        else "[yellow]✗ Not found — will guide installation[/]"
    )
    table.add_row("Ollama", ollama_status)
    table.add_row("Scan source", profile.source)

    console.print(table)


def _accel_profile_str(accel: AccelerationStatus, tier: GPUTier) -> str:
    if accel == AccelerationStatus.METAL_APPLE:
        return "[green]Metal (Apple Silicon — established)[/]"
    if accel == AccelerationStatus.METAL_INTEL:
        return (
            "[yellow]possible via Metal backend[/] "
            "[dim](Intel GPU supports Metal — acceleration unverified for this model)[/]"
        )
    if accel == AccelerationStatus.CUDA:
        return "[green]CUDA (discrete NVIDIA)[/]"
    if accel == AccelerationStatus.ROCM:
        return "[green]ROCm (discrete AMD)[/]"
    if accel == AccelerationStatus.UNVERIFIED:
        return "[yellow]unverified / backend-dependent[/] [dim](iGPU detected — do not assume LLM acceleration)[/]"
    if tier == GPUTier.NONE:
        return "CPU-only"
    return "CPU-only"


def _ram_fit_badge(ram_fit: RAMFit) -> str:
    return {
        RAMFit.FITS: "[green]FITS[/]",
        RAMFit.TIGHT: "[yellow]TIGHT[/]",
        RAMFit.RISKY: "[orange3]RISKY[/]",
        RAMFit.OVER: "[red]DOES_NOT_FIT[/]",
        RAMFit.UNKNOWN: "[dim]UNKNOWN[/]",
    }[ram_fit]


def _confidence_badge(conf: Confidence) -> str:
    return {
        Confidence.HIGH: "[green]High[/]",
        Confidence.MEDIUM: "[yellow]Medium[/]",
        Confidence.LOW: "[orange3]Low[/]",
        Confidence.UNKNOWN: "[dim]Unknown[/]",
    }[conf]


def _accel_badge(sm: EvaluatedModel) -> str:
    if sm.acceleration == AccelerationStatus.METAL_APPLE:
        return "[green]Metal (Apple Silicon)[/]"
    if sm.acceleration == AccelerationStatus.METAL_INTEL:
        return "[yellow]Metal (Intel — possible, unverified)[/]"
    if sm.acceleration == AccelerationStatus.CUDA:
        return "[green]CUDA[/]" if sm.will_use_gpu else "[yellow]CUDA (partial/offload)[/]"
    if sm.acceleration == AccelerationStatus.ROCM:
        return "[green]ROCm[/]" if sm.will_use_gpu else "[yellow]ROCm (partial/offload)[/]"
    if sm.acceleration == AccelerationStatus.UNVERIFIED:
        return "[yellow]GPU detected — LLM accel unverified[/]"
    return "[dim]CPU-only[/]"


def _perf_basis_str(basis: PerformanceBasis) -> str:
    return {
        PerformanceBasis.MEASURED: "measured",
        PerformanceBasis.ESTIMATED: "estimated",
        PerformanceBasis.INFERRED: "inferred",
        PerformanceBasis.UNKNOWN: "unknown",
    }[basis]


def _is_system_warning(text: str) -> bool:
    lower = text.lower()
    return any(marker in lower for marker in _SYSTEM_WARNING_MARKERS)


def _category_display(sm: EvaluatedModel) -> str:
    cats = ", ".join(sm.model.categories) if sm.model.categories else "unknown"
    src = getattr(sm.model, "category_source", "unknown") or "unknown"
    if cats == "unknown" or cats == "":
        return "[dim]unknown[/]"
    if src == "metadata":
        return f"{cats} [dim](from metadata)[/]"
    if src == "inferred":
        return f"{cats} [dim](inferred)[/]"
    return f"{cats} [dim](uncertain)[/]"


def _fit_title_badge(sm: EvaluatedModel) -> tuple[str, str]:
    """Return (border_style, fit_label) for a model card."""
    if sm.disqualified:
        return "red", "[red]⛔ INCOMPATIBLE[/]"
    if sm.unverified or sm.ram_fit == RAMFit.UNKNOWN:
        return "dim", "[dim]? UNKNOWN[/]"
    if sm.ram_fit == RAMFit.FITS:
        return "green", "[green]✓ FITS[/]"
    if sm.ram_fit == RAMFit.TIGHT:
        return "yellow", "[yellow]⚠ TIGHT[/]"
    if sm.ram_fit == RAMFit.RISKY:
        return "orange3", "[orange3]⚠ RISKY[/]"
    if sm.ram_fit == RAMFit.OVER:
        return "red", "[red]✗ DOES_NOT_FIT[/]"
    return "dim", "[dim]? UNKNOWN[/]"


def _model_card_lines(
    sm: EvaluatedModel,
    *,
    available_ram_gb: float | None,
    verbose: bool,
    suppress_system_warnings: bool,
) -> list[str]:
    m = sm.model
    lines: list[str] = []

    size_str = f"{m.size_gb:.1f} GB" if m.size_gb > 0 else "size unknown"
    quant_str = m.quantization if m.quantization != "unknown" else "quant unknown"
    params_str = f"~{sm.params_b:g}B" if sm.params_b is not None else "unknown"
    lines.append(
        f"[dim]Size:[/] {size_str}  "
        f"[dim]Params:[/] {params_str}  "
        f"[dim]Quant:[/] {quant_str}  "
        f"[dim]Task:[/] {_category_display(sm)}"
    )

    labels_str = ", ".join(sm.label_names) if sm.labels else "—"
    lines.append(f"[dim]Labels:[/] {labels_str}")

    if sm.rank_reason:
        lines.append(f"[dim]Evidence:[/] {sm.rank_reason}")

    lines.append(
        f"[dim]Pullable:[/] {'yes' if sm.installable else 'no'}  "
        f"[dim]Runtime compatible:[/] {'yes' if sm.runtime_compatible else 'no'}  "
        f"[dim]Hardware fit:[/] {_ram_fit_badge(sm.ram_fit)}  "
        f"[dim]Mem confidence:[/] {_confidence_badge(sm.memory_confidence)}"
    )

    if sm.estimated_total_ram_gb > 0:
        avail_str = f"{available_ram_gb:.1f} GB" if available_ram_gb is not None else "see system profile"
        headroom = None
        if available_ram_gb is not None:
            headroom = available_ram_gb - sm.estimated_total_ram_gb
        headroom_str = ""
        if headroom is not None:
            if headroom >= 0:
                headroom_str = f"  [dim]Headroom:[/] ~{headroom:.1f} GB"
            else:
                headroom_str = f"  [dim]Shortfall:[/] ~{abs(headroom):.1f} GB"
        lines.append(
            f"[dim]Est. RAM needed (estimate):[/] ~{sm.estimated_total_ram_gb:.1f} GB  "
            f"[dim]Available:[/] {avail_str}{headroom_str}"
        )
        if verbose:
            lines.append(f"  [dim]{sm.ram_budget_note}[/]")
    elif sm.missing_metadata:
        lines.append(f"[dim]Missing metadata:[/] {', '.join(sm.missing_metadata)}")

    lines.append(
        f"[dim]GPU detected:[/] {'yes' if sm.gpu_detected else 'no'}  "
        f"[dim]LLM accel:[/] {_accel_badge(sm)}"
    )
    if sm.estimated_tps > 0:
        lines.append(
            f"[dim]Performance (estimate):[/] ~{sm.estimated_tps:.1f} tok/s "
            f"({_perf_basis_str(sm.performance_basis)})  "
            f"[dim]Confidence:[/] {_confidence_badge(sm.tps_confidence)}"
        )
    elif sm.performance_basis == PerformanceBasis.UNKNOWN:
        lines.append("[dim]Performance:[/] not estimated")
    else:
        lines.append(f"[dim]Performance:[/] unknown ({_perf_basis_str(sm.performance_basis)})")

    if verbose:
        lines.append(f"[dim]Overall confidence:[/] {_confidence_badge(sm.confidence)}")
        for line in sm.explanation:
            lines.append(f"  [green]✔[/] {line}")
        for w in sm.warnings:
            if suppress_system_warnings and _is_system_warning(w):
                continue
            lines.append(f"  [yellow]⚠[/]  {w}")
        if m.known_issues:
            lines.append(f"  [red]⚠[/]  {len(m.known_issues)} open community bug report(s)")
    else:
        # Compact: at most two model-specific warnings
        shown = 0
        for w in sm.warnings:
            if suppress_system_warnings and _is_system_warning(w):
                continue
            # Skip verbose UNVERIFIED boilerplate already covered by the UNKNOWN badge
            if "UNVERIFIED: size/RAM metadata incomplete" in w:
                continue
            lines.append(f"  [yellow]⚠[/]  {w}")
            shown += 1
            if shown >= 2:
                break

    return lines


def print_recommendations(
    evaluated: list[EvaluatedModel],
    top: int = 5,
    available_ram_gb: float | None = None,
    verbose: bool = False,
) -> None:
    console.print()
    console.rule("[bold cyan]Model Compatibility[/]")

    recommended, potential, advisory = partition_recommendations(evaluated)

    # Hardware-level notes once (avoid repeating iGPU / accel warnings on every card)
    system_notes: list[str] = []
    seen_notes: set[str] = set()
    for sm in evaluated[: max(top * 2, 10)]:
        for w in sm.warnings:
            if _is_system_warning(w) and w not in seen_notes:
                seen_notes.add(w)
                system_notes.append(w)
    if system_notes:
        console.print("\n[bold]Hardware notes[/] [dim](apply to all candidates)[/]")
        for note in system_notes:
            console.print(f"  [yellow]⚠[/]  {note}")

    if advisory:
        console.print(f"\n[bold yellow]{advisory}[/]")

    # Cap each section so the summary stays scannable
    rec_limit = top
    pot_limit = top if not recommended else max(2, top // 2)

    if recommended:
        console.print("\n[bold green]Recommended models[/] [dim](verified FITS / TIGHT)[/]")
        for shown, sm in enumerate(recommended[:rec_limit], start=1):
            border, fit_label = _fit_title_badge(sm)
            verified_badge = "[green]verified[/]"
            install_badge = "[green]● ollama pull[/]" if sm.model.ollama_pullable else "[yellow]● manual download[/]"
            title = (
                f"[bold]#{shown}[/]  [white]{sm.model.full_tag}[/]  "
                f"{fit_label}  {verified_badge}  {install_badge}"
            )
            body = "\n".join(
                _model_card_lines(
                    sm,
                    available_ram_gb=available_ram_gb,
                    verbose=verbose,
                    suppress_system_warnings=True,
                )
            )
            console.print(Panel(body, title=title, border_style=border, padding=(0, 1)))
    else:
        console.print(
            "\n[bold yellow]Recommended models:[/] none — "
            "no candidate is verified to fit available memory."
        )

    show_potential = potential[:pot_limit]
    if show_potential:
        console.print(
            "\n[bold]Potential candidates[/] "
            "[dim](unverified, RISKY, or incomplete metadata — not confirmed fits)[/]"
        )
        for sm in show_potential:
            border, fit_label = _fit_title_badge(sm)
            verified_badge = "[green]verified[/]" if sm.verified else "[dim]UNVERIFIED[/]"
            install_badge = "[green]● ollama pull[/]" if sm.model.ollama_pullable else "[yellow]● manual download[/]"
            # No forced #1 winner numbering for potential candidates
            title = f"[white]{sm.model.full_tag}[/]  {fit_label}  {verified_badge}  {install_badge}"
            body = "\n".join(
                _model_card_lines(
                    sm,
                    available_ram_gb=available_ram_gb,
                    verbose=verbose,
                    suppress_system_warnings=True,
                )
            )
            console.print(Panel(body, title=title, border_style=border, padding=(0, 1)))

    if not recommended and not show_potential:
        console.print("[yellow]No suitable candidates to display for this hardware profile.[/]")

    if not verbose:
        console.print("\n[dim]Tip: re-run with [cyan]-v[/] / [cyan]--verbose[/] for full diagnostics per model.[/]")


def print_download_result(result_code, message: str) -> None:
    from .downloader import DownloadResult

    if result_code == DownloadResult.SUCCESS:
        console.print(f"\n[bold green]✓ {message}[/]")
    elif result_code == DownloadResult.ALREADY_EXISTS:
        console.print(f"\n[bold cyan]ℹ {message}[/]")
    elif result_code == DownloadResult.OLLAMA_MISSING:
        console.print(Panel(message, title="[red]Ollama Not Found[/]", border_style="red"))
    elif result_code == DownloadResult.PULL_FAILED:
        console.print(Panel(message, title="[red]Download Failed[/]", border_style="red"))
    elif result_code == DownloadResult.CANCELLED:
        console.print(f"\n[yellow]{message}[/]")


def spinner(message: str):
    return Progress(
        SpinnerColumn(),
        TextColumn(f"[cyan]{message}[/]"),
        transient=True,
        console=console,
    )
