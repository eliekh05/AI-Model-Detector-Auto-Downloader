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
    GPUTier,
    PerformanceBasis,
    RAMFit,
    ScoredModel,
)

console = Console()


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
        accel = acceleration_for_tier(tier)
        table.add_row("GPU detected", "yes")
        table.add_row("LLM acceleration", _accel_profile_str(accel, tier))
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
    if accel == AccelerationStatus.CUDA:
        return "[green]CUDA (discrete NVIDIA)[/]"
    if accel == AccelerationStatus.ROCM:
        return "[green]ROCm (discrete AMD)[/]"
    if accel == AccelerationStatus.UNVERIFIED:
        return "[yellow]unverified / backend-dependent[/] [dim](iGPU detected — scoring uses CPU-only)[/]"
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


def _accel_badge(sm: ScoredModel) -> str:
    if sm.acceleration == AccelerationStatus.METAL_APPLE:
        return "[green]Metal (Apple Silicon)[/]"
    if sm.acceleration == AccelerationStatus.CUDA:
        return "[green]CUDA[/]" if sm.will_use_gpu else "[yellow]CUDA (partial/offload)[/]"
    if sm.acceleration == AccelerationStatus.ROCM:
        return "[green]ROCm[/]" if sm.will_use_gpu else "[yellow]ROCm (partial/offload)[/]"
    if sm.acceleration == AccelerationStatus.UNVERIFIED:
        return "[yellow]GPU detected — LLM accel unverified (CPU-only scoring)[/]"
    return "[dim]CPU-only[/]"


def _perf_basis_str(basis: PerformanceBasis) -> str:
    return {
        PerformanceBasis.MEASURED: "measured",
        PerformanceBasis.ESTIMATED: "estimated",
        PerformanceBasis.INFERRED: "inferred",
        PerformanceBasis.UNKNOWN: "unknown",
    }[basis]


def print_recommendations(scored: list[ScoredModel], top: int = 5, available_ram_gb: float | None = None) -> None:
    console.print()
    console.rule("[bold cyan]Model Recommendations[/]")

    for shown, sm in enumerate(scored[:top], start=1):
        m = sm.model

        if sm.disqualified:
            border = "red"
            fit_label = "[red]⛔ INCOMPATIBLE[/]"
        elif sm.unverified or sm.ram_fit == RAMFit.UNKNOWN:
            border = "dim"
            fit_label = "[dim]? UNVERIFIED[/]"
        elif sm.ram_fit == RAMFit.FITS:
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
            fit_label = "[red]✗ DOES_NOT_FIT[/]"
        else:
            border = "dim"
            fit_label = "[dim]? UNKNOWN[/]"

        install_badge = "[green]● ollama pull[/]" if m.ollama_pullable else "[yellow]● manual download[/]"
        verified_badge = "[green]verified[/]" if sm.verified else "[dim]UNVERIFIED[/]"

        title = (
            f"[bold]#{shown}[/]  [white]{m.full_tag}[/]  "
            f"Score: {sm.score:.0f}/100  "
            f"{fit_label}  {verified_badge}  {install_badge}"
        )

        lines: list[str] = []

        # Size / params / quant
        size_str = f"{m.size_gb:.1f} GB" if m.size_gb > 0 else "size unknown"
        quant_str = m.quantization if m.quantization != "unknown" else "quant unknown"
        if sm.params_b is not None:
            params_str = f"~{sm.params_b:g}B"
        else:
            params_str = "unknown"
        lines.append(
            f"[dim]Size:[/] {size_str}  "
            f"[dim]Params:[/] {params_str}  "
            f"[dim]Quant:[/] {quant_str}  "
            f"[dim]Categories:[/] {', '.join(m.categories) or '?'}"
        )

        # Why ranked here
        if sm.rank_reason:
            lines.append(f"[dim]Why here:[/] {sm.rank_reason}")

        # Separated concerns
        lines.append(
            f"[dim]Installable:[/] {'yes' if sm.installable else 'no'}  "
            f"[dim]Runtime compatible:[/] {'yes' if sm.runtime_compatible else 'no'}  "
            f"[dim]Fit:[/] {_ram_fit_badge(sm.ram_fit)}  "
            f"[dim]Mem confidence:[/] {_confidence_badge(sm.memory_confidence)}"
        )

        # RAM budget
        avail = available_ram_gb
        if sm.estimated_total_ram_gb > 0:
            avail_str = f"{avail:.1f} GB" if avail is not None else "see system profile"
            lines.append(
                f"[dim]Est. RAM needed:[/] ~{sm.estimated_total_ram_gb:.1f} GB  [dim]Available (system):[/] {avail_str}"
            )
            lines.append(f"  [dim]{sm.ram_budget_note}[/]")
        elif sm.missing_metadata:
            lines.append(f"[dim]Missing metadata:[/] {', '.join(sm.missing_metadata)}")

        # Acceleration + performance
        lines.append(
            f"[dim]GPU detected:[/] {'yes' if sm.gpu_detected else 'no'}  [dim]LLM accel:[/] {_accel_badge(sm)}"
        )
        if sm.estimated_tps > 0:
            lines.append(
                f"[dim]Performance:[/] ~{sm.estimated_tps:.1f} tok/s "
                f"({_perf_basis_str(sm.performance_basis)})  "
                f"[dim]Confidence:[/] {_confidence_badge(sm.tps_confidence)}"
            )
        else:
            lines.append(f"[dim]Performance:[/] unknown ({_perf_basis_str(sm.performance_basis)})")

        lines.append(f"[dim]Overall confidence:[/] {_confidence_badge(sm.confidence)}")

        for line in sm.explanation[:5]:
            lines.append(f"  [green]✔[/] {line}")

        for w in sm.warnings[:4]:
            lines.append(f"  [yellow]⚠[/]  {w}")

        if m.known_issues:
            lines.append(f"  [red]⚠[/]  {len(m.known_issues)} open community bug report(s)")

        console.print(
            Panel(
                "\n".join(lines),
                title=title,
                border_style=border,
                padding=(0, 1),
            )
        )


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
