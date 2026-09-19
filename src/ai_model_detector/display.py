"""
display.py — Rich terminal UI for scan results and recommendations.
"""

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from .scanner import SystemProfile
from .scorer import ScoredModel

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
            vram  = f"{gpu.vram_gb:.1f} GB VRAM" if gpu.vram_gb else "VRAM unknown"
            accel = []
            if gpu.cuda_version:
                accel.append(f"CUDA {gpu.cuda_version}")
            if gpu.metal_support:
                accel.append("Metal")
            if gpu.rocm_version:
                accel.append("ROCm")
            accel_str = " / ".join(accel) or "Vulkan" if gpu.vulkan_support else ""
            table.add_row(label, f"{gpu.name} · {vram} · {accel_str}")
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


def print_recommendations(scored: list[ScoredModel], top: int = 5) -> None:
    console.print()
    console.rule("[bold cyan]Top Model Recommendations[/]")

    for rank, sm in enumerate(scored[:top], start=1):
        m = sm.model
        score_color = "green" if sm.score >= 70 else "yellow" if sm.score >= 45 else "red"

        install_badge = (
            "[green]● ollama pull[/]" if m.ollama_pullable
            else "[yellow]● manual download (HuggingFace)[/]"
        )
        title = (
            f"[bold]#{rank}[/]  [white]{m.full_tag}[/]  "
            f"[{score_color}]Score: {sm.score:.0f}/100[/]  {install_badge}"
        )

        body_lines = []

        if m.size_gb > 0:
            body_lines.append(f"[dim]Size:[/] {m.size_gb:.1f} GB   "
                              f"[dim]Quant:[/] {m.quantization}   "
                              f"[dim]Speed:[/] {sm.estimated_speed}   "
                              f"[dim]GPU:[/] {'Yes' if sm.will_use_gpu else 'CPU-only'}")

        if m.categories:
            body_lines.append(f"[dim]Categories:[/] {', '.join(m.categories)}")

        if m.description and "HuggingFace" not in m.description[:15]:
            body_lines.append(f"[dim]{m.description[:120]}[/]")

        for line in sm.explanation[:3]:
            body_lines.append(f"  [green]✔[/] {line}")

        for w in sm.warnings[:2]:
            body_lines.append(f"  [yellow]⚠[/]  {w}")

        if m.known_issues:
            body_lines.append(
                f"  [red]⚠[/]  {len(m.known_issues)} open community issue(s) reported"
            )

        style = "green" if sm.score >= 70 else "yellow" if sm.score >= 45 else "dim"
        console.print(Panel(
            "\n".join(body_lines),
            title=title,
            border_style=style,
            padding=(0, 1),
        ))


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
    """Context manager returning a Rich spinner progress."""
    return Progress(
        SpinnerColumn(),
        TextColumn(f"[cyan]{message}[/]"),
        transient=True,
        console=console,
    )
