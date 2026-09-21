"""
display.py — Rich terminal UI for scan results and recommendations.
"""

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from .scanner import SystemProfile
from .scorer import Confidence, GPUTier, RAMFit, ScoredModel

console = Console()


def print_banner() -> None:
    console.print(Panel(
        "[bold cyan]AI Model Detector & Auto Downloader[/]\n"
        "[dim]Deep hardware scanning · Live model registry · Smart recommendations[/]",
        box=box.DOUBLE_EDGE,
        style="bold",
    ))


def print_system_profile(profile: SystemProfile) -> None:
    table = Table(title="System Profile", box=box.ROUNDED, show_header=False)
    table.add_column("Field", style="bold cyan", width=22)
    table.add_column("Value", style="white")

    table.add_row("OS", f"{profile.os_name} {profile.os_version[:40]}")
    table.add_row("Architecture", profile.os_arch)
    table.add_row("CPU", profile.cpu.brand[:60])
    table.add_row("CPU Cores", f"{profile.cpu.cores_physical}P / {profile.cpu.cores_logical}L")
    table.add_row(
        "CPU Extensions",
        " ".join(
            f for f, ok in [
                ("AVX", profile.cpu.supports_avx),
                ("AVX2", profile.cpu.supports_avx2),
                ("AVX-512", profile.cpu.supports_avx512),
                ("F16C", profile.cpu.supports_f16c),
            ] if ok
        ) or "None detected"
    )
    table.add_row("RAM Total", f"{profile.ram.total_gb:.1f} GB")
    table.add_row("RAM Available", f"{profile.ram.available_gb:.1f} GB")
    if profile.ram.speed_mhz:
        table.add_row("RAM Speed", f"{profile.ram.speed_mhz} MHz")

    if profile.gpus:
        for i, gpu in enumerate(profile.gpus):
            label = f"GPU {i+1}"
            vram  = f"{gpu.vram_gb:.1f} GB VRAM" if gpu.vram_gb else "VRAM unknown (shared)"
            accel = []
            if gpu.cuda_version:
                accel.append(f"CUDA {gpu.cuda_version}")
            if gpu.metal_support:
                accel.append("Metal")
            if gpu.rocm_version:
                accel.append("ROCm")
            accel_str = " / ".join(accel) or ""
            table.add_row(label, f"{gpu.name} · {vram}" + (f" · {accel_str}" if accel_str else ""))
    else:
        table.add_row("GPU", "[yellow]No GPU detected — CPU-only inference[/]")

    if profile.disk:
        table.add_row(
            "Disk Free",
            f"{profile.disk.free_gb:.1f} GB / {profile.disk.total_gb:.1f} GB"
        )

    ollama_status = (
        f"[green]✓ Installed[/] ({profile.ollama_version})"
        if profile.ollama_installed
        else "[yellow]✗ Not found — will guide installation[/]"
    )
    table.add_row("Ollama", ollama_status)
    table.add_row("Scan source", profile.source)

    console.print(table)


def _ram_fit_badge(ram_fit: RAMFit) -> str:
    return {
        RAMFit.FIT:     "[green]FIT[/]",
        RAMFit.TIGHT:   "[yellow]TIGHT[/]",
        RAMFit.RISKY:   "[orange3]RISKY[/]",
        RAMFit.OVER:    "[red]DOES NOT FIT[/]",
        RAMFit.UNKNOWN: "[dim]UNKNOWN[/]",
    }[ram_fit]


def _confidence_badge(conf: Confidence) -> str:
    return {
        Confidence.HIGH:    "[green]High[/]",
        Confidence.MEDIUM:  "[yellow]Medium[/]",
        Confidence.LOW:     "[orange3]Low[/]",
        Confidence.UNKNOWN: "[dim]Unknown[/]",
    }[conf]


def _gpu_tier_str(gpu_tier: GPUTier, will_use_gpu: bool) -> str:
    if gpu_tier == GPUTier.APPLE_SILICON:
        return "[green]Apple Silicon (Metal)[/]"
    if gpu_tier in (GPUTier.DISCRETE_CUDA, GPUTier.DISCRETE_ROCM):
        return "[green]Discrete GPU[/]" if will_use_gpu else "[yellow]Discrete GPU (partial)[/]"
    if gpu_tier == GPUTier.INTEGRATED:
        return "[yellow]Integrated GPU (unverified)[/]"
    return "[dim]CPU only[/]"


def print_recommendations(scored: list[ScoredModel], top: int = 5) -> None:
    console.print()
    console.rule("[bold cyan]Model Recommendations[/]")

    for shown, sm in enumerate(scored[:top], start=1):
        m = sm.model

        # ── Colour by RAM fit and disqualification ──────────────────────────
        if sm.disqualified:
            border = "red"
            fit_label = "[red]⛔ INCOMPATIBLE[/]"
        elif sm.ram_fit == RAMFit.FIT:
            border = "green"
            fit_label = "[green]✓ FITS[/]"
        elif sm.ram_fit == RAMFit.TIGHT:
            border = "yellow"
            fit_label = "[yellow]⚠ TIGHT[/]"
        elif sm.ram_fit == RAMFit.RISKY:
            border = "orange3"
            fit_label = "[orange3]⚠ RISKY[/]"
        elif sm.ram_fit == RAMFit.OVER:
            border = "red"
            fit_label = "[red]✗ TOO LARGE[/]"
        else:
            border = "dim"
            fit_label = "[dim]? UNKNOWN SIZE[/]"

        install_badge = (
            "[green]● ollama pull[/]" if m.ollama_pullable
            else "[yellow]● manual download[/]"
        )

        title = (
            f"[bold]#{shown}[/]  [white]{m.full_tag}[/]  "
            f"Score: {sm.score:.0f}/100  "
            f"{fit_label}  {install_badge}"
        )

        lines = []

        # ── Size + quant ────────────────────────────────────────────────────
        size_str = f"{m.size_gb:.1f} GB" if m.size_gb > 0 else "size unknown"
        quant_str = m.quantization if m.quantization != "unknown" else "quant unknown"
        params_str = _approx_params(m.size_gb)
        lines.append(
            f"[dim]Size:[/] {size_str}  "
            f"[dim]Params:[/] {params_str}  "
            f"[dim]Quant:[/] {quant_str}  "
            f"[dim]Categories:[/] {', '.join(m.categories) or '?'}"
        )

        # ── RAM budget ──────────────────────────────────────────────────────
        if sm.estimated_total_ram_gb > 0:
            lines.append(
                f"[dim]RAM needed:[/] ~{sm.estimated_total_ram_gb:.1f} GB  "
                f"[dim]Available:[/] {sm.model.ram_required_gb or '?'} GB  "
                f"[dim]Fit:[/] {_ram_fit_badge(sm.ram_fit)}"
            )
            lines.append(f"  [dim]{sm.ram_budget_note}[/]")

        # ── Speed estimate ──────────────────────────────────────────────────
        if sm.estimated_tps > 0:
            lines.append(
                f"[dim]Est. speed:[/] ~{sm.estimated_tps:.1f} tok/s  "
                f"[dim]Confidence:[/] {_confidence_badge(sm.tps_confidence)}  "
                f"[dim]Accel:[/] {_gpu_tier_str(sm.gpu_tier, sm.will_use_gpu)}"
            )

        # ── Overall confidence ──────────────────────────────────────────────
        lines.append(
            f"[dim]Overall confidence:[/] {_confidence_badge(sm.confidence)}"
        )

        # ── Explanation bullets ─────────────────────────────────────────────
        for line in sm.explanation[:4]:
            lines.append(f"  [green]✔[/] {line}")

        # ── Warnings ────────────────────────────────────────────────────────
        for w in sm.warnings[:3]:
            lines.append(f"  [yellow]⚠[/]  {w}")

        if m.known_issues:
            lines.append(f"  [red]⚠[/]  {len(m.known_issues)} open community bug report(s)")

        console.print(Panel(
            "\n".join(lines),
            title=title,
            border_style=border,
            padding=(0, 1),
        ))


def _approx_params(size_gb: float) -> str:
    """Derive approximate parameter count from file size at Q4 (~0.5 B/param)."""
    if size_gb <= 0:
        return "unknown"
    params_b = size_gb / 0.5
    if params_b < 1.0:
        return f"~{params_b*1000:.0f}M"
    return f"~{params_b:.1f}B"


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
