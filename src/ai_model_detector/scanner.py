"""
scanner.py — Deep system hardware profiler.

Collects OS, CPU, RAM, GPU, disk, and driver information to build
a full hardware profile used for model recommendation scoring.
Supports .spx imports on macOS (system_profiler XML exports).
"""

import os
import sys
import platform
import subprocess
import shutil
import json
import plistlib
import zipfile
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import psutil


@dataclass
class CPUProfile:
    brand: str
    cores_physical: int
    cores_logical: int
    frequency_max_mhz: float
    architecture: str
    supports_avx: bool
    supports_avx2: bool
    supports_avx512: bool
    supports_f16c: bool


@dataclass
class RAMProfile:
    total_gb: float
    available_gb: float
    speed_mhz: Optional[int]  # may not be detectable on all platforms


@dataclass
class GPUDevice:
    name: str
    vram_gb: Optional[float]
    driver_version: Optional[str]
    metal_support: bool        # macOS Metal
    cuda_version: Optional[str]
    rocm_version: Optional[str]
    vulkan_support: bool


@dataclass
class DiskProfile:
    free_gb: float
    total_gb: float
    mount: str


@dataclass
class SystemProfile:
    os_name: str
    os_version: str
    os_arch: str
    cpu: CPUProfile
    ram: RAMProfile
    gpus: list[GPUDevice] = field(default_factory=list)
    disk: Optional[DiskProfile] = None
    ollama_installed: bool = False
    ollama_version: Optional[str] = None
    source: str = "live_scan"   # "live_scan" | "spx_import"

    def to_dict(self) -> dict:
        return asdict(self)


# ── CPU helpers ────────────────────────────────────────────────────────────────

def _detect_cpu_flags() -> dict:
    """Parse /proc/cpuinfo on Linux; use sysctl/cpuinfo on macOS; wmic on Windows."""
    flags = {"avx": False, "avx2": False, "avx512f": False, "f16c": False}
    system = platform.system()

    if system == "Linux":
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("flags"):
                        parts = line.split(":")[1].split()
                        flags["avx"]    = "avx"    in parts
                        flags["avx2"]   = "avx2"   in parts
                        flags["avx512f"]= "avx512f" in parts
                        flags["f16c"]   = "f16c"   in parts
                        break
        except Exception:
            pass

    elif system == "Darwin":
        try:
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.features"],
                capture_output=True, text=True, timeout=5
            )
            feat = result.stdout.upper()
            flags["avx"]    = "AVX1.0" in feat or "AVX " in feat
            flags["avx2"]   = "AVX2" in feat
            flags["avx512f"]= "AVX512F" in feat
            flags["f16c"]   = "F16C" in feat
        except Exception:
            pass

    elif system == "Windows":
        try:
            result = subprocess.run(
                ["wmic", "cpu", "get", "Caption,Name"],
                capture_output=True, text=True, timeout=10
            )
            # Windows doesn't expose flags easily; mark unknown
        except Exception:
            pass

    return flags


def _cpu_brand() -> str:
    """Try to get a clean CPU brand string."""
    system = platform.system()
    if system == "Linux":
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if "model name" in line:
                        return line.split(":")[1].strip()
        except Exception:
            pass
    elif system == "Darwin":
        try:
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, timeout=5
            )
            return result.stdout.strip()
        except Exception:
            pass
    elif system == "Windows":
        try:
            result = subprocess.run(
                ["wmic", "cpu", "get", "Name"],
                capture_output=True, text=True, timeout=10
            )
            lines = [l.strip() for l in result.stdout.splitlines() if l.strip()]
            if len(lines) > 1:
                return lines[1]
        except Exception:
            pass
    return platform.processor() or "Unknown CPU"


def _scan_cpu() -> CPUProfile:
    freq = psutil.cpu_freq()
    flags = _detect_cpu_flags()
    return CPUProfile(
        brand=_cpu_brand(),
        cores_physical=psutil.cpu_count(logical=False) or 1,
        cores_logical=psutil.cpu_count(logical=True) or 1,
        frequency_max_mhz=freq.max if freq else 0.0,
        architecture=platform.machine(),
        supports_avx=flags["avx"],
        supports_avx2=flags["avx2"],
        supports_avx512=flags["avx512f"],
        supports_f16c=flags["f16c"],
    )


# ── RAM helpers ────────────────────────────────────────────────────────────────

def _ram_speed_mhz() -> Optional[int]:
    system = platform.system()
    if system == "Linux":
        try:
            result = subprocess.run(
                ["dmidecode", "--type", "17"],
                capture_output=True, text=True, timeout=5
            )
            for line in result.stdout.splitlines():
                if "Speed:" in line and "MT/s" in line:
                    parts = line.split()
                    for p in parts:
                        if p.isdigit():
                            return int(p)
        except Exception:
            pass
    elif system == "Darwin":
        try:
            result = subprocess.run(
                ["system_profiler", "SPMemoryDataType", "-json"],
                capture_output=True, text=True, timeout=10
            )
            data = json.loads(result.stdout)
            items = data.get("SPMemoryDataType", [])
            if items and "_items" in items[0]:
                speed = items[0]["_items"][0].get("dimm_speed", "")
                num = "".join(c for c in speed if c.isdigit())
                if num:
                    return int(num)
        except Exception:
            pass
    elif system == "Windows":
        try:
            result = subprocess.run(
                ["wmic", "memorychip", "get", "Speed"],
                capture_output=True, text=True, timeout=10
            )
            lines = [l.strip() for l in result.stdout.splitlines() if l.strip()]
            if len(lines) > 1 and lines[1].isdigit():
                return int(lines[1])
        except Exception:
            pass
    return None


def _scan_ram() -> RAMProfile:
    vm = psutil.virtual_memory()
    return RAMProfile(
        total_gb=round(vm.total / (1024**3), 2),
        available_gb=round(vm.available / (1024**3), 2),
        speed_mhz=_ram_speed_mhz(),
    )


# ── GPU helpers ────────────────────────────────────────────────────────────────

def _nvidia_gpus() -> list[GPUDevice]:
    if not shutil.which("nvidia-smi"):
        return []
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True, text=True, timeout=10
        )
        devices = []
        for line in result.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 3:
                continue
            name, vram_mib, driver = parts[0], parts[1], parts[2]
            vram_gb = round(float(vram_mib) / 1024, 2) if vram_mib.replace(".", "").isdigit() else None

            cuda_ver = None
            try:
                cv = subprocess.run(
                    ["nvidia-smi", "--query-gpu=cuda_version", "--format=csv,noheader"],
                    capture_output=True, text=True, timeout=5
                )
                cuda_ver = cv.stdout.strip().splitlines()[0].strip() or None
            except Exception:
                pass

            devices.append(GPUDevice(
                name=name,
                vram_gb=vram_gb,
                driver_version=driver,
                metal_support=False,
                cuda_version=cuda_ver,
                rocm_version=None,
                vulkan_support=True,  # NVIDIA supports Vulkan
            ))
        return devices
    except Exception:
        return []


def _amd_gpus() -> list[GPUDevice]:
    rocm_ver = None
    if shutil.which("rocminfo"):
        try:
            r = subprocess.run(["rocminfo"], capture_output=True, text=True, timeout=10)
            for line in r.stdout.splitlines():
                if "ROCm" in line:
                    rocm_ver = line.strip()
                    break
        except Exception:
            pass

    devices = []
    if shutil.which("rocm-smi"):
        try:
            r = subprocess.run(
                ["rocm-smi", "--showproductname", "--json"],
                capture_output=True, text=True, timeout=10
            )
            data = json.loads(r.stdout)
            for card_id, info in data.items():
                vram = None
                try:
                    vr = subprocess.run(
                        ["rocm-smi", "--showmeminfo", "vram", "--json"],
                        capture_output=True, text=True, timeout=5
                    )
                    vdata = json.loads(vr.stdout)
                    vram_bytes = vdata.get(card_id, {}).get("VRAM Total Memory (B)", 0)
                    vram = round(int(vram_bytes) / (1024**3), 2)
                except Exception:
                    pass
                devices.append(GPUDevice(
                    name=info.get("Card Series", info.get("Card Model", "AMD GPU")),
                    vram_gb=vram,
                    driver_version=None,
                    metal_support=False,
                    cuda_version=None,
                    rocm_version=rocm_ver,
                    vulkan_support=True,
                ))
        except Exception:
            pass
    return devices


def _macos_gpus() -> list[GPUDevice]:
    try:
        result = subprocess.run(
            ["system_profiler", "SPDisplaysDataType", "-json"],
            capture_output=True, text=True, timeout=10
        )
        data = json.loads(result.stdout)
        displays = data.get("SPDisplaysDataType", [])
        devices = []
        for d in displays:
            name = d.get("sppci_model", "Apple GPU")
            vram_str = d.get("sppci_vram", "")
            vram_gb = None
            if "MB" in vram_str:
                try:
                    vram_gb = round(float(vram_str.replace("MB", "").strip()) / 1024, 2)
                except Exception:
                    pass
            elif "GB" in vram_str:
                try:
                    vram_gb = float(vram_str.replace("GB", "").strip())
                except Exception:
                    pass
            devices.append(GPUDevice(
                name=name,
                vram_gb=vram_gb,
                driver_version=None,
                metal_support=True,
                cuda_version=None,
                rocm_version=None,
                vulkan_support=False,
            ))
        return devices
    except Exception:
        return []


def _windows_gpus() -> list[GPUDevice]:
    try:
        result = subprocess.run(
            ["wmic", "path", "win32_VideoController",
             "get", "Name,AdapterRAM,DriverVersion", "/format:csv"],
            capture_output=True, text=True, timeout=10
        )
        devices = []
        lines = [l.strip() for l in result.stdout.splitlines() if l.strip() and "Node" not in l]
        for line in lines[1:]:
            parts = line.split(",")
            if len(parts) < 4:
                continue
            _, vram_bytes, driver, name = parts[0], parts[1], parts[2], parts[3]
            vram_gb = None
            try:
                vram_gb = round(int(vram_bytes) / (1024**3), 2)
            except Exception:
                pass
            devices.append(GPUDevice(
                name=name,
                vram_gb=vram_gb,
                driver_version=driver,
                metal_support=False,
                cuda_version=None,
                rocm_version=None,
                vulkan_support=True,
            ))
        return devices
    except Exception:
        return []


def _scan_gpus() -> list[GPUDevice]:
    system = platform.system()
    gpus = _nvidia_gpus()
    gpus += _amd_gpus()
    if not gpus:
        if system == "Darwin":
            gpus = _macos_gpus()
        elif system == "Windows":
            gpus = _windows_gpus()
    return gpus


# ── Disk ───────────────────────────────────────────────────────────────────────

def _scan_disk() -> DiskProfile:
    home = Path.home()
    usage = psutil.disk_usage(str(home))
    return DiskProfile(
        free_gb=round(usage.free / (1024**3), 2),
        total_gb=round(usage.total / (1024**3), 2),
        mount=str(home),
    )


# ── Ollama detection ───────────────────────────────────────────────────────────

def _detect_ollama() -> tuple[bool, Optional[str]]:
    if not shutil.which("ollama"):
        return False, None
    try:
        r = subprocess.run(["ollama", "--version"], capture_output=True, text=True, timeout=5)
        ver = r.stdout.strip() or r.stderr.strip()
        return True, ver
    except Exception:
        return True, None


# ── .spx import ───────────────────────────────────────────────────────────────

def _parse_spx(spx_path: Path) -> SystemProfile:
    """
    Parse a macOS System Information (.spx) export.
    .spx files are ZIP archives containing plist XML files.
    """
    if not spx_path.exists():
        raise FileNotFoundError(f"SPX file not found: {spx_path}")

    extracted: dict = {}
    with zipfile.ZipFile(str(spx_path), "r") as zf:
        for name in zf.namelist():
            with zf.open(name) as f:
                try:
                    extracted[name] = plistlib.load(f)
                except Exception:
                    pass

    # CPU
    hw_data = extracted.get("Hardware.plist", {})
    hw_items = hw_data.get("SPHardwareDataType", [{}])
    hw = hw_items[0] if hw_items else {}
    cpu_brand = hw.get("cpu_type", "Unknown CPU")
    ram_total_str = hw.get("physical_memory", "0 GB")
    try:
        ram_gb = float(ram_total_str.replace("GB", "").replace("MB", "").strip())
        if "MB" in ram_total_str:
            ram_gb /= 1024
    except Exception:
        ram_gb = 0.0

    cpu = CPUProfile(
        brand=cpu_brand,
        cores_physical=int(hw.get("number_processors", "1")),
        cores_logical=int(hw.get("number_processors", "1")),
        frequency_max_mhz=0.0,
        architecture="arm64" if "Apple" in cpu_brand else "x86_64",
        supports_avx="Apple" not in cpu_brand,
        supports_avx2="Apple" not in cpu_brand,
        supports_avx512=False,
        supports_f16c=False,
    )
    ram = RAMProfile(total_gb=ram_gb, available_gb=ram_gb * 0.5, speed_mhz=None)

    # GPU
    gpu_data = extracted.get("Displays.plist", {})
    gpu_items = gpu_data.get("SPDisplaysDataType", [])
    gpus = []
    for g in gpu_items:
        gpus.append(GPUDevice(
            name=g.get("sppci_model", "Apple GPU"),
            vram_gb=None,
            driver_version=None,
            metal_support=True,
            cuda_version=None,
            rocm_version=None,
            vulkan_support=False,
        ))

    ollama_installed, ollama_ver = _detect_ollama()
    return SystemProfile(
        os_name="Darwin",
        os_version=hw.get("os_version", "macOS"),
        os_arch="arm64" if "Apple" in cpu_brand else "x86_64",
        cpu=cpu,
        ram=ram,
        gpus=gpus,
        disk=_scan_disk(),
        ollama_installed=ollama_installed,
        ollama_version=ollama_ver,
        source="spx_import",
    )


# ── Main scan ─────────────────────────────────────────────────────────────────

def scan_system(spx_path: Optional[Path] = None) -> SystemProfile:
    """
    Run a full system scan and return a SystemProfile.
    If spx_path is provided, import hardware data from a macOS .spx file.
    """
    if spx_path is not None:
        return _parse_spx(spx_path)

    ollama_installed, ollama_ver = _detect_ollama()

    return SystemProfile(
        os_name=platform.system(),
        os_version=platform.version(),
        os_arch=platform.machine(),
        cpu=_scan_cpu(),
        ram=_scan_ram(),
        gpus=_scan_gpus(),
        disk=_scan_disk(),
        ollama_installed=ollama_installed,
        ollama_version=ollama_ver,
        source="live_scan",
    )
