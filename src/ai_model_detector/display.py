"""
display.py — Terminal UI using stdlib only (no rich, no click).

Uses ANSI escape codes for color when the terminal supports it.
Falls back to plain text when output is not a TTY.
"""

import sys
import threading

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

# ── ANSI color support ──────────────────────────────────────────────────

_STDOUT_IS_TTY = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


def _supports_color() -> bool:
    """Detect whether stdout supports ANSI color codes."""
    if not _STDOUT_IS_TTY:
        return False
    # Windows: modern terminals support ANSI
    if sys.platform == "win32":
        return True
    return True


_COLOR_ENABLED = _supports_color()


def _c(code: str, text: str) -> str:
    """Wrap text in ANSI color code if color is enabled."""
    if not _COLOR_ENABLED:
        return text
    return f"\033[{code}m{text}\033[0m"


# Color helpers
def _bold(text: str) -> str:
    return _c("1", text)


def _dim(text: str) -> str:
    return _c("2", text)


def _cyan(text: str) -> str:
    return _c("36", text)


def _green(text: str) -> str:
    return _c("32", text)


def _yellow(text: str) -> str:
    return _c("33", text)


def _red(text: str) -> str:
    return _c("31", text)


def _magenta(text: str) -> str:
    return _c("35", text)


# ── Warnings that apply to the whole machine ────────────────────────────

_SYSTEM_WARNING_MARKERS = (
    "integrated gpu does not prove",
    "detecting an integrated gpu",
    "llm acceleration: not confirmed",
    "do not assume llm acceleration",
    "backend acceleration on this gpu is not established",
)


def _is_system_warning(text: str) -> bool:
    lower = text.lower()
    return any(marker in lower for marker in _SYSTEM_WARNING_MARKERS)


# ── Banner ──────────────────────────────────────────────────────────────


def print_banner() -> None:
    border = "=" * 60
    print()
    print(_cyan(border))
    print(_bold(_cyan("  AI Model Detector & Auto Downloader")))
    print(_dim("  Deep hardware scanning · Live model registry · Smart recommendations"))
    print(_cyan(border))
    print()


# ── System profile ──────────────────────────────────────────────────────


def print_system_profile(profile: SystemProfile) -> None:
    print(_bold(_cyan("System Profile")))
    print("-" * 60)

    rows = [
        ("OS", f"{profile.os_name} {profile.os_version[:40]}"),
        ("Architecture", profile.os_arch),
        ("CPU", profile.cpu.brand[:60]),
        ("CPU Cores", f"{profile.cpu.cores_physical}P / {profile.cpu.cores_logical}L"),
        (
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
        ),
        ("RAM Total", f"{profile.ram.total_gb:.1f} GB"),
        ("RAM Available", f"{profile.ram.available_gb:.1f} GB {_dim('(snapshot at scan time)')}"),
    ]
    if profile.ram.speed_mhz:
        rows.append(("RAM Speed", f"{profile.ram.speed_mhz} MHz"))

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
            rows.append((label, f"{gpu.name} · {mem}{backend_str}"))

        from .scorer import _classify_gpu, acceleration_for_tier

        tier, _ = _classify_gpu(profile.gpus, profile.os_name, profile.os_arch)
        accel = acceleration_for_tier(tier, profile.os_name, profile.os_arch, profile.metal_available)
        rows.append(("GPU detected", "yes"))
        rows.append(("LLM acceleration", _accel_profile_str(accel, tier)))

        if accel == AccelerationStatus.UNVERIFIED:
            rows.append((
                "Acceleration detail",
                _dim("GPU detected but Ollama/llama.cpp backend acceleration "
                     "is not established. Reporting CPU-only performance."),
            ))
    else:
        rows.append(("GPU", _yellow("No GPU detected")))
        rows.append(("GPU detected", "no"))
        rows.append(("LLM acceleration", "CPU-only"))

    if profile.disk:
        rows.append((
            "Disk Free",
            f"{profile.disk.free_gb:.1f} GB / {profile.disk.total_gb:.1f} GB",
        ))

    if profile.ollama_installed:
        ollama_status = _green("✓ Installed") + f" ({profile.ollama_version})"
    else:
        ollama_status = _yellow("✗ Not found — will guide installation")
    rows.append(("Ollama", ollama_status))
    rows.append(("Scan source", profile.source))

    # Print as aligned table
    max_label = max(len(label) for label, _ in rows)
    for label, value in rows:
        print(f"  {_bold(_cyan(label.ljust(max_label)))}  {value}")
    print()


def _accel_profile_str(accel: AccelerationStatus, tier: GPUTier) -> str:
    if accel == AccelerationStatus.METAL_APPLE:
        return _green("Metal (Apple Silicon — established)")
    if accel == AccelerationStatus.CUDA:
        return _green("CUDA (discrete NVIDIA)")
    if accel == AccelerationStatus.ROCM:
        return _green("ROCm (discrete AMD)")
    if accel == AccelerationStatus.UNVERIFIED:
        return (
            _yellow("not confirmed") + " "
            + _dim("(GPU detected but backend acceleration not established — "
                   "do not assume LLM acceleration)")
        )
    if tier == GPUTier.NONE:
        return "CPU-only"
    return "CPU-only"


# ── Model card helpers ──────────────────────────────────────────────────


def _ram_fit_badge(ram_fit: RAMFit) -> str:
    mapping = {
        RAMFit.FITS: (_green, "FITS"),
        RAMFit.TIGHT: (_yellow, "TIGHT"),
        RAMFit.RISKY: (_magenta, "RISKY"),
        RAMFit.OVER: (_red, "DOES_NOT_FIT"),
        RAMFit.UNKNOWN: (_dim, "UNKNOWN"),
    }
    func, label = mapping[ram_fit]
    return func(label)


def _confidence_badge(conf: Confidence) -> str:
    mapping = {
        Confidence.HIGH: (_green, "High"),
        Confidence.MEDIUM: (_yellow, "Medium"),
        Confidence.LOW: (_magenta, "Low"),
        Confidence.UNKNOWN: (_dim, "Unknown"),
    }
    func, label = mapping[conf]
    return func(label)


def _accel_badge(sm: EvaluatedModel) -> str:
    if sm.acceleration == AccelerationStatus.METAL_APPLE:
        return _green("Metal (Apple Silicon)")
    if sm.acceleration == AccelerationStatus.CUDA:
        return _green("CUDA") if sm.will_use_gpu else _yellow("CUDA (partial/offload)")
    if sm.acceleration == AccelerationStatus.ROCM:
        return _green("ROCm") if sm.will_use_gpu else _yellow("ROCm (partial/offload)")
    if sm.acceleration == AccelerationStatus.UNVERIFIED:
        return _yellow("GPU detected — LLM accel unverified")
    return _dim("CPU-only")


def _perf_basis_str(basis: PerformanceBasis) -> str:
    return {
        PerformanceBasis.MEASURED: "measured",
        PerformanceBasis.ESTIMATED: "estimated",
        PerformanceBasis.INFERRED: "inferred",
        PerformanceBasis.UNKNOWN: "unknown",
    }[basis]


def _category_display(sm: EvaluatedModel) -> str:
    cats = ", ".join(sm.model.categories) if sm.model.categories else "unknown"
    src = getattr(sm.model, "category_source", "unknown") or "unknown"
    if cats == "unknown" or cats == "":
        return _dim("unknown")
    if src == "metadata":
        return f"{cats} {_dim('(from metadata)')}"
    if src == "inferred":
        return f"{cats} {_dim('(inferred)')}"
    return f"{cats} {_dim('(uncertain)')}"


def _fit_title_badge(sm: EvaluatedModel) -> str:
    """Return fit label for a model card."""
    if sm.disqualified:
        return _red("⛔ INCOMPATIBLE")
    if sm.unverified or sm.ram_fit == RAMFit.UNKNOWN:
        return _dim("? UNKNOWN")
    if sm.ram_fit == RAMFit.FITS:
        return _green("✓ FITS")
    if sm.ram_fit == RAMFit.TIGHT:
        return _yellow("⚠ TIGHT")
    if sm.ram_fit == RAMFit.RISKY:
        return _magenta("⚠ RISKY")
    if sm.ram_fit == RAMFit.OVER:
        return _red("✗ DOES_NOT_FIT")
    return _dim("? UNKNOWN")


def _fit_border_style(sm: EvaluatedModel) -> str:
    """Return border character for a model card."""
    if sm.disqualified:
        return "red"
    if sm.unverified or sm.ram_fit == RAMFit.UNKNOWN:
        return "dim"
    if sm.ram_fit == RAMFit.FITS:
        return "green"
    if sm.ram_fit == RAMFit.TIGHT:
        return "yellow"
    if sm.ram_fit == RAMFit.RISKY:
        return "magenta"
    if sm.ram_fit == RAMFit.OVER:
        return "red"
    return "dim"


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
        f"{_dim('Size:')} {size_str}  "
        f"{_dim('Params:')} {params_str}  "
        f"{_dim('Quant:')} {quant_str}  "
        f"{_dim('Task:')} {_category_display(sm)}"
    )

    labels_str = ", ".join(sm.label_names) if sm.labels else "—"
    lines.append(f"{_dim('Labels:')} {labels_str}")

    if sm.rank_reason:
        lines.append(f"{_dim('Evidence:')} {sm.rank_reason}")

    lines.append(
        f"{_dim('Pullable:')} {'yes' if sm.installable else 'no'}  "
        f"{_dim('Runtime compatible:')} {'yes' if sm.runtime_compatible else 'no'}  "
        f"{_dim('Hardware fit:')} {_ram_fit_badge(sm.ram_fit)}  "
        f"{_dim('Mem confidence:')} {_confidence_badge(sm.memory_confidence)}"
    )

    if sm.estimated_total_ram_gb > 0:
        avail_str = f"{available_ram_gb:.1f} GB" if available_ram_gb is not None else "see system profile"
        headroom = None
        if available_ram_gb is not None:
            headroom = available_ram_gb - sm.estimated_total_ram_gb
        headroom_str = ""
        if headroom is not None:
            if headroom > 0.2:
                headroom_str = f"  {_dim('Headroom:')} ~{headroom:.1f} GB"
            elif headroom > -0.2:
                headroom_str = f"  {_yellow('No practical headroom')} (~0 GB free)"
            else:
                headroom_str = f"  {_red('Shortfall:')} ~{abs(headroom):.1f} GB"
        lines.append(
            f"{_dim('Est. RAM needed (estimate):')} ~{sm.estimated_total_ram_gb:.1f} GB  "
            f"{_dim('Available:')} {avail_str}{headroom_str}"
        )
        if verbose:
            lines.append(f"  {_dim(sm.ram_budget_note)}")
    elif sm.missing_metadata:
        lines.append(f"{_dim('Missing metadata:')} {', '.join(sm.missing_metadata)}")

    lines.append(
        f"{_dim('GPU detected:')} {'yes' if sm.gpu_detected else 'no'}  "
        f"{_dim('LLM accel:')} {_accel_badge(sm)}"
    )
    if sm.estimated_tps > 0:
        weak_estimate = (
            sm.tps_confidence in (Confidence.LOW, Confidence.UNKNOWN)
            or sm.performance_basis in (PerformanceBasis.INFERRED, PerformanceBasis.UNKNOWN)
        )
        if verbose or not weak_estimate:
            lines.append(
                f"{_dim('Performance (estimate):')} ~{sm.estimated_tps:.1f} tok/s "
                f"({_perf_basis_str(sm.performance_basis)})  "
                f"{_dim('Confidence:')} {_confidence_badge(sm.tps_confidence)}"
            )
        else:
            lines.append(f"{_dim('Performance:')} not estimated (insufficient data)")
    elif sm.performance_basis == PerformanceBasis.UNKNOWN:
        lines.append(f"{_dim('Performance:')} not estimated")
    else:
        lines.append(f"{_dim('Performance:')} unknown ({_perf_basis_str(sm.performance_basis)})")

    if verbose:
        lines.append(f"{_dim('Overall confidence:')} {_confidence_badge(sm.confidence)}")
        for line in sm.explanation:
            lines.append(f"  {_green('✔')} {line}")
        for w in sm.warnings:
            if suppress_system_warnings and _is_system_warning(w):
                continue
            lines.append(f"  {_yellow('⚠')}  {w}")
        if m.known_issues:
            lines.append(f"  {_red('⚠')}  {len(m.known_issues)} open community bug report(s)")
    else:
        shown = 0
        for w in sm.warnings:
            if suppress_system_warnings and _is_system_warning(w):
                continue
            if "UNVERIFIED: size/RAM metadata incomplete" in w:
                continue
            lines.append(f"  {_yellow('⚠')}  {w}")
            shown += 1
            if shown >= 2:
                break

    return lines


# ── Print model card ────────────────────────────────────────────────────


def _print_card(title: str, body_lines: list[str], border_style: str = "dim") -> None:
    """Print a model card with a border."""
    color_func = {
        "red": _red,
        "green": _green,
        "yellow": _yellow,
        "magenta": _magenta,
        "dim": _dim,
    }.get(border_style, _dim)

    border_char = "─"
    width = 60
    print()
    print(color_func(f"┌{border_char * width}┐"))
    # Truncate title to fit
    title_display = title[:width - 2]
    padding = width - len(title_display.replace("\033[", "").replace("[0m", ""))
    print(color_func(f"│ {_bold(title_display)}{' ' * max(0, padding - 1)}│"))
    print(color_func(f"├{border_char * width}┤"))
    for line in body_lines:
        # Strip ANSI for width calculation
        clean = line
        for code in ["\033[0m", "\033[1m", "\033[2m", "\033[31m", "\033[32m",
                      "\033[33m", "\033[35m", "\033[36m"]:
            clean = clean.replace(code, "")
        vis_len = len(clean)
        pad = max(0, width - vis_len - 2)
        print(color_func("│ ") + line + " " * pad + color_func("│"))
    print(color_func(f"└{border_char * width}┘"))


# ── Recommendations ─────────────────────────────────────────────────────


def print_recommendations(
    evaluated: list[EvaluatedModel],
    top: int = 5,
    available_ram_gb: float | None = None,
    verbose: bool = False,
) -> None:
    print()
    print(_bold(_cyan("═" * 60)))
    print(_bold(_cyan("  Model Compatibility")))
    print(_bold(_cyan("═" * 60)))

    recommended, potential, advisory = partition_recommendations(evaluated)

    # Hardware-level notes once
    system_notes: list[str] = []
    seen_notes: set[str] = set()
    for sm in evaluated[: max(top * 2, 10)]:
        for w in sm.warnings:
            if _is_system_warning(w) and w not in seen_notes:
                seen_notes.add(w)
                system_notes.append(w)
    if system_notes:
        print()
        print(_bold("Hardware notes") + _dim(" (apply to all candidates)"))
        for note in system_notes:
            print(f"  {_yellow('⚠')}  {note}")

    if advisory:
        print()
        print(_yellow(_bold(advisory)))

    rec_limit = top
    pot_limit = top if not recommended else max(2, top // 2)

    if recommended:
        print()
        print(_bold(_green("Recommended models")) + _dim(" (verified FITS / TIGHT)"))
        for shown, sm in enumerate(recommended[:rec_limit], start=1):
            fit_label = _fit_title_badge(sm)
            verified_badge = _green("verified")
            install_badge = _green("● ollama pull") if sm.model.ollama_pullable else _yellow("● manual download")
            title = (
                f"{_bold(f'#{shown}')}  {sm.model.full_tag}  "
                f"{fit_label}  {verified_badge}  {install_badge}"
            )
            body = _model_card_lines(
                sm,
                available_ram_gb=available_ram_gb,
                verbose=verbose,
                suppress_system_warnings=True,
            )
            _print_card(title, body, _fit_border_style(sm))
    else:
        print()
        print(_yellow(_bold("Recommended models:")) + " none — "
              "no candidate is verified to fit available memory.")

    show_potential = potential[:pot_limit]
    if show_potential:
        _SPECIALIZED_CATS = {"asr", "audio", "embeddings", "embedding", "translation"}
        general_potential = [
            sm for sm in show_potential
            if not {c.lower() for c in sm.model.categories} & _SPECIALIZED_CATS
        ]
        specialized_potential = [
            sm for sm in show_potential
            if {c.lower() for c in sm.model.categories} & _SPECIALIZED_CATS
        ]

        if general_potential:
            print()
            print(_bold("Potential candidates") +
                  _dim(" (unverified, RISKY, or incomplete metadata — not confirmed fits)"))
            for sm in general_potential:
                verified_badge = _green("verified") if sm.verified else _dim("UNVERIFIED")
                install_badge = _green("● ollama pull") if sm.model.ollama_pullable else _yellow("● manual download")
                title = f"{sm.model.full_tag}  {_fit_title_badge(sm)}  {verified_badge}  {install_badge}"
                body = _model_card_lines(
                    sm,
                    available_ram_gb=available_ram_gb,
                    verbose=verbose,
                    suppress_system_warnings=True,
                )
                _print_card(title, body, _fit_border_style(sm))

        if specialized_potential:
            print()
            print(_bold("Specialized models") +
                  _dim(" (task-specific — not general chat assistants)"))
            for sm in specialized_potential:
                verified_badge = _green("verified") if sm.verified else _dim("UNVERIFIED")
                install_badge = _green("● ollama pull") if sm.model.ollama_pullable else _yellow("● manual download")
                title = f"{sm.model.full_tag}  {_fit_title_badge(sm)}  {verified_badge}  {install_badge}"
                body = _model_card_lines(
                    sm,
                    available_ram_gb=available_ram_gb,
                    verbose=verbose,
                    suppress_system_warnings=True,
                )
                _print_card(title, body, _fit_border_style(sm))

    if not recommended and not show_potential:
        print(_yellow("No suitable candidates to display for this hardware profile."))

    if not verbose:
        print()
        print(_dim(f"Tip: re-run with {_cyan('-v')} / {_cyan('--verbose')} for full diagnostics per model."))


# ── Download result ─────────────────────────────────────────────────────


def print_download_result(result_code, message: str) -> None:
    from .downloader import DownloadResult

    if result_code == DownloadResult.SUCCESS:
        print(f"\n{_green(_bold('✓'))} {message}")
    elif result_code == DownloadResult.ALREADY_EXISTS:
        print(f"\n{_cyan('ℹ')} {message}")
    elif result_code == DownloadResult.OLLAMA_MISSING:
        print(f"\n{_red(_bold('Ollama Not Found'))}")
        print(f"  {message}")
    elif result_code == DownloadResult.PULL_FAILED:
        print(f"\n{_red(_bold('Download Failed'))}")
        print(f"  {message}")
    elif result_code == DownloadResult.CANCELLED:
        print(f"\n{_yellow(message)}")


# ── Spinner (context manager) ───────────────────────────────────────────


class spinner:
    """Simple terminal spinner using stdlib threading."""

    def __init__(self, message: str):
        self.message = message
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self):
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_exc):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1)
        # Clear spinner line
        sys.stdout.write("\r" + " " * (len(self.message) + 4) + "\r")
        sys.stdout.flush()

    def _run(self):
        chars = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
        i = 0
        while not self._stop.is_set():
            char = chars[i % len(chars)]
            sys.stdout.write(f"\r{_cyan(char)} {self.message}")
            sys.stdout.flush()
            i += 1
            self._stop.wait(0.1)
