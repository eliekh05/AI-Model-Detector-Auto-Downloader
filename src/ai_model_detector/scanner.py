"""
scanner.py — Deep system hardware profiler (stdlib only, zero dependencies).

Collects OS, CPU, RAM, GPU, disk, and driver information to build
a full hardware profile used for model compatibility evaluation.
Supports .spx imports on macOS (system_profiler XML exports).
"""

import ctypes
import json
import os
import platform
import plistlib
import shutil
import subprocess
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path


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
    speed_mhz: int | None  # may not be detectable on all platforms


@dataclass
class GPUDevice:
    name: str
    vram_gb: float | None
    driver_version: str | None
    metal_support: bool  # macOS Metal
    cuda_version: str | None
    rocm_version: str | None
    vulkan_support: bool
    is_integrated: bool = False  # True for Intel/AMD iGPU sharing system RAM


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
    disk: DiskProfile | None = None
    ollama_installed: bool = False
    ollama_version: str | None = None
    metal_available: bool = False  # Metal API detected on system (macOS)
    vulkan_available: bool = False  # Vulkan driver detected (Linux/Windows)
    source: str = "live_scan"  # "live_scan" | "spx_import"

    def to_dict(self) -> dict:
        return asdict(self)


# ── Low-level system helpers (replace psutil) ────────────────────────────


def _sysctl_int(name: str) -> int | None:
    """Read an integer sysctl value on macOS. Returns None on failure."""
    try:
        r = subprocess.run(
            ["sysctl", "-n", name],
            capture_output=True, text=True, timeout=5, check=False,
        )
        return int(r.stdout.strip())
    except Exception:
        return None


def _sysctl_str(name: str) -> str:
    """Read a string sysctl value on macOS."""
    try:
        r = subprocess.run(
            ["sysctl", "-n", name],
            capture_output=True, text=True, timeout=5, check=False,
        )
        return r.stdout.strip()
    except Exception:
        return ""


def _read_file_int(path: str) -> int | None:
    """Read a single integer from a sysfs/procfs file."""
    try:
        with open(path) as f:
            return int(f.read().strip())
    except Exception:
        return None


def _total_ram_bytes() -> int:
    """Return total physical RAM in bytes using platform APIs."""
    system = platform.system()
    if system == "Darwin":
        val = _sysctl_int("hw.memsize")
        if val:
            return val
    elif system == "Linux":
        val = _read_file_int("/proc/meminfo_total")
        if val is None:
            # Parse MemTotal from /proc/meminfo
            try:
                with open("/proc/meminfo") as f:
                    for line in f:
                        if line.startswith("MemTotal:"):
                            parts = line.split()
                            return int(parts[1]) * 1024  # kB to bytes
            except Exception:
                pass
    elif system == "Windows":
        try:
            kernel32 = ctypes.windll.kernel32
            ctypes.windll.kernel32.GetPhysicallyInstalledMemory = None
            # GlobalMemoryStatusEx
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]
            mem = MEMORYSTATUSEX()
            mem.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if kernel32.GlobalMemoryStatusEx(ctypes.byref(mem)):
                return mem.ullTotalPhys
        except Exception:
            pass
    # Fallback: try os.sysconf
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except Exception:
        pass
    return 0


def _available_ram_bytes() -> int:
    """Return currently available RAM in bytes using platform APIs."""
    system = platform.system()
    if system == "Darwin":
        # pagesize × pages_free + pages_active is approximate;
        # use hw.memsize minus wired+active from vm_stat
        page_size = _sysctl_int("hw.pagesize") or 4096
        try:
            r = subprocess.run(
                ["vm_stat"], capture_output=True, text=True, timeout=5, check=False,
            )
            free = speculative = 0
            for line in r.stdout.splitlines():
                if "Pages free" in line:
                    free = int(line.split(":")[1].strip().rstrip(".")) * page_size
                elif "Pages speculative" in line:
                    speculative = int(line.split(":")[1].strip().rstrip(".")) * page_size
            # Available ≈ free + speculative (macOS keeps active in RAM but counts toward pressure)
            avail = free + speculative
            if avail > 0:
                return avail
        except Exception:
            pass
    elif system == "Linux":
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemAvailable:"):
                        parts = line.split()
                        return int(parts[1]) * 1024  # kB to bytes
        except Exception:
            pass
    elif system == "Windows":
        try:
            kernel32 = ctypes.windll.kernel32

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]
            mem = MEMORYSTATUSEX()
            mem.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if kernel32.GlobalMemoryStatusEx(ctypes.byref(mem)):
                return mem.ullAvailPhys
        except Exception:
            pass
    # Fallback: total / 2 as rough estimate
    return _total_ram_bytes() // 2


def _cpu_count_logical() -> int:
    """Return number of logical CPU cores."""
    count = os.cpu_count()
    if count:
        return count
    system = platform.system()
    if system == "Darwin":
        val = _sysctl_int("hw.logicalcpu")
        if val:
            return val
    return 1


def _cpu_count_physical() -> int:
    """Return number of physical CPU cores."""
    system = platform.system()
    if system == "Darwin":
        val = _sysctl_int("hw.physicalcpu")
        if val:
            return val
    elif system == "Linux":
        try:
            with open("/proc/cpuinfo") as f:
                ids = set()
                for line in f:
                    if line.startswith("physical id"):
                        ids.add(line.split(":")[1].strip())
                # Count unique physical ids × cores per id
                f.seek(0)
                core_ids = set()
                for line in f:
                    if line.startswith("core id"):
                        core_ids.add(line.split(":")[1].strip())
                if ids and core_ids:
                    return len(ids) * len(core_ids)
        except Exception:
            pass
    elif system == "Windows":
        try:
            r = subprocess.run(
                ["wmic", "cpu", "get", "NumberOfCores"],
                capture_output=True, text=True, timeout=10, check=False,
            )
            lines = [l.strip() for l in r.stdout.splitlines() if l.strip() and l.strip() != "NumberOfCores"]
            if lines:
                return int(lines[0])
        except Exception:
            pass
    # Fallback
    return max(1, _cpu_count_logical() // 2)


def _cpu_freq_mhz() -> float:
    """Return max CPU frequency in MHz."""
    system = platform.system()
    if system == "Darwin":
        # hw.cpufrequency is in Hz on some Macs
        val = _sysctl_int("hw.cpufrequency")
        if val and val > 1000:
            return round(val / 1_000_000, 1)
        # Try machdep.cpu.brand_string for "X.XXGHz"
        brand = _sysctl_str("machdep.cpu.brand_string")
        if "GHz" in brand:
            try:
                ghz_str = brand.split("GHz")[0].split()[-1]
                return round(float(ghz_str) * 1000, 1)
            except Exception:
                pass
    elif system == "Linux":
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if "cpu MHz" in line:
                        val = float(line.split(":")[1].strip())
                        return round(val, 1)
        except Exception:
            pass
        # sysfs fallback
        freq = _read_file_int("/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq")
        if freq:
            return round(freq / 1000, 1)
    elif system == "Windows":
        try:
            r = subprocess.run(
                ["wmic", "cpu", "get", "MaxClockSpeed"],
                capture_output=True, text=True, timeout=10, check=False,
            )
            lines = [l.strip() for l in r.stdout.splitlines() if l.strip() and l.strip() != "MaxClockSpeed"]
            if lines:
                return float(lines[0])
        except Exception:
            pass
    return 0.0


# ── CPU helpers ──────────────────────────────────────────────────────────


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
                        flags["avx"] = "avx" in parts
                        flags["avx2"] = "avx2" in parts
                        flags["avx512f"] = "avx512f" in parts
                        flags["f16c"] = "f16c" in parts
                        break
        except Exception:
            pass

    elif system == "Darwin":
        try:
            r1 = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.features"],
                capture_output=True, text=True, timeout=5, check=False,
            )
            feat = r1.stdout.upper()
        except Exception:
            feat = ""

        try:
            r2 = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.leaf7_features"],
                capture_output=True, text=True, timeout=5, check=False,
            )
            leaf7 = r2.stdout.upper()
        except Exception:
            leaf7 = ""

        combined = feat + " " + leaf7
        flags["avx"] = "AVX1.0" in combined or "AVX " in combined
        flags["avx2"] = "AVX2" in combined
        flags["avx512f"] = "AVX512F" in combined
        flags["f16c"] = "F16C" in combined

    elif system == "Windows":
        try:
            subprocess.run(
                ["wmic", "cpu", "get", "Caption,Name"],
                capture_output=True, text=True, timeout=10, check=False,
            )
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
        brand = _sysctl_str("machdep.cpu.brand_string")
        if brand:
            return brand
    elif system == "Windows":
        try:
            result = subprocess.run(
                ["wmic", "cpu", "get", "Name"],
                capture_output=True, text=True, timeout=10, check=False,
            )
            lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
            if len(lines) > 1:
                return lines[1]
        except Exception:
            pass
    return platform.processor() or "Unknown CPU"


def _scan_cpu() -> CPUProfile:
    flags = _detect_cpu_flags()
    return CPUProfile(
        brand=_cpu_brand(),
        cores_physical=_cpu_count_physical(),
        cores_logical=_cpu_count_logical(),
        frequency_max_mhz=_cpu_freq_mhz(),
        architecture=platform.machine(),
        supports_avx=flags["avx"],
        supports_avx2=flags["avx2"],
        supports_avx512=flags["avx512f"],
        supports_f16c=flags["f16c"],
    )


# ── RAM helpers ──────────────────────────────────────────────────────────


def _ram_speed_mhz() -> int | None:
    system = platform.system()
    if system == "Linux":
        try:
            result = subprocess.run(
                ["dmidecode", "--type", "17"],
                capture_output=True, text=True, timeout=5, check=False,
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
                capture_output=True, text=True, timeout=10, check=False,
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
                capture_output=True, text=True, timeout=10, check=False,
            )
            lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
            if len(lines) > 1 and lines[1].isdigit():
                return int(lines[1])
        except Exception:
            pass
    return None


def _scan_ram() -> RAMProfile:
    total = _total_ram_bytes()
    available = _available_ram_bytes()
    return RAMProfile(
        total_gb=round(total / (1024**3), 2),
        available_gb=round(available / (1024**3), 2),
        speed_mhz=_ram_speed_mhz(),
    )


# ── GPU helpers ──────────────────────────────────────────────────────────


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
            capture_output=True, text=True, timeout=10, check=False,
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
                    capture_output=True, text=True, timeout=5, check=False,
                )
                cuda_ver = cv.stdout.strip().splitlines()[0].strip() or None
            except Exception:
                pass

            devices.append(
                GPUDevice(
                    name=name,
                    vram_gb=vram_gb,
                    driver_version=driver,
                    metal_support=False,
                    cuda_version=cuda_ver,
                    rocm_version=None,
                    vulkan_support=True,
                )
            )
        return devices
    except Exception:
        return []


def _amd_gpus() -> list[GPUDevice]:
    rocm_ver = None
    if shutil.which("rocminfo"):
        try:
            r = subprocess.run(["rocminfo"], capture_output=True, text=True, timeout=10, check=False)
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
                capture_output=True, text=True, timeout=10, check=False,
            )
            data = json.loads(r.stdout)
            for card_id, info in data.items():
                vram = None
                try:
                    vr = subprocess.run(
                        ["rocm-smi", "--showmeminfo", "vram", "--json"],
                        capture_output=True, text=True, timeout=5, check=False,
                    )
                    vdata = json.loads(vr.stdout)
                    vram_bytes = vdata.get(card_id, {}).get("VRAM Total Memory (B)", 0)
                    vram = round(int(vram_bytes) / (1024**3), 2)
                except Exception:
                    pass
                devices.append(
                    GPUDevice(
                        name=info.get("Card Series", info.get("Card Model", "AMD GPU")),
                        vram_gb=vram,
                        driver_version=None,
                        metal_support=False,
                        cuda_version=None,
                        rocm_version=rocm_ver,
                        vulkan_support=True,
                    )
                )
        except Exception:
            pass
    return devices


def _macos_gpus() -> list[GPUDevice]:
    try:
        result = subprocess.run(
            ["system_profiler", "SPDisplaysDataType", "-json"],
            capture_output=True, text=True, timeout=10, check=False,
        )
        data = json.loads(result.stdout)
        displays = data.get("SPDisplaysDataType", [])
        devices = []
        integrated_kws = ("iris", "uhd graphics", "hd graphics", "vega", "radeon(tm)")
        for d in displays:
            name = d.get("sppci_model", "Apple GPU")
            name_lower = name.lower()
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
            is_integrated = any(kw in name_lower for kw in integrated_kws)
            devices.append(
                GPUDevice(
                    name=name,
                    vram_gb=vram_gb,
                    driver_version=None,
                    metal_support=True,
                    cuda_version=None,
                    rocm_version=None,
                    vulkan_support=False,
                    is_integrated=is_integrated,
                )
            )
        return devices
    except Exception:
        return []


def _windows_gpus() -> list[GPUDevice]:
    try:
        result = subprocess.run(
            ["wmic", "path", "win32_VideoController", "get", "Name,AdapterRAM,DriverVersion", "/format:csv"],
            capture_output=True, text=True, timeout=10, check=False,
        )
        devices = []
        lines = [line.strip() for line in result.stdout.splitlines() if line.strip() and "Node" not in line]
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
            devices.append(
                GPUDevice(
                    name=name,
                    vram_gb=vram_gb,
                    driver_version=driver,
                    metal_support=False,
                    cuda_version=None,
                    rocm_version=None,
                    vulkan_support=True,
                )
            )
        return devices
    except Exception:
        return []


# ── Compute API detection ───────────────────────────────────────────────


def _detect_metal_support() -> bool:
    """Check if any GPU supports Metal on macOS."""
    if platform.system() != "Darwin":
        return False
    try:
        result = subprocess.run(
            ["system_profiler", "SPDisplaysDataType", "-json"],
            capture_output=True, text=True, timeout=10, check=False,
        )
        data = json.loads(result.stdout)
        for d in data.get("SPDisplaysDataType", []):
            model = d.get("sppci_model", "").lower()
            if any(kw in model for kw in ("apple", "iris", "uhd", "hd graphics", "radeon")):
                return True
    except Exception:
        pass
    return False


def _detect_vulkan_support() -> bool:
    """Check for Vulkan driver availability on Linux/Windows."""
    if platform.system() in ("Linux", "Windows"):
        return shutil.which("vulkaninfo") is not None
    return False


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


# ── Disk ───────────────────────────────────────────────────────────────


def _scan_disk() -> DiskProfile:
    home = Path.home()
    usage = shutil.disk_usage(str(home))
    return DiskProfile(
        free_gb=round(usage.free / (1024**3), 2),
        total_gb=round(usage.total / (1024**3), 2),
        mount=str(home),
    )


# ── Ollama detection ───────────────────────────────────────────────────────


def _detect_ollama() -> tuple[bool, str | None]:
    if not shutil.which("ollama"):
        return False, None
    try:
        r = subprocess.run(["ollama", "--version"], capture_output=True, text=True, timeout=5, check=False)
        ver = r.stdout.strip() or r.stderr.strip()
        return True, ver
    except Exception:
        return True, None


# ── .spx import ───────────────────────────────────────────────────────────


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
    integrated_kws = ("iris", "uhd graphics", "hd graphics", "vega", "radeon(tm)")
    gpus = []
    for g in gpu_items:
        gpu_name = g.get("sppci_model", "Apple GPU")
        is_integrated = any(kw in gpu_name.lower() for kw in integrated_kws)
        gpus.append(
            GPUDevice(
                name=gpu_name,
                vram_gb=None,
                driver_version=None,
                metal_support=True,
                cuda_version=None,
                rocm_version=None,
                vulkan_support=False,
                is_integrated=is_integrated,
            )
        )

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
        metal_available=any(g.metal_support for g in gpus),
        vulkan_available=False,
        source="spx_import",
    )


# ── Main scan ────────────────────────────────────────────────────────────


def scan_system(spx_path: Path | None = None) -> SystemProfile:
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
        metal_available=_detect_metal_support(),
        vulkan_available=_detect_vulkan_support(),
        source="live_scan",
    )
