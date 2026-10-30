import sys
import time
import platform
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import psutil
from src.common import ROOT, write_json

report = {"python":sys.version, "platform":platform.platform(), "torch":torch.__version__,
          "cuda_available":torch.cuda.is_available(), "torch_cuda":torch.version.cuda,
          "logical_cpus":psutil.cpu_count(), "ram_gb":psutil.virtual_memory().total/2**30}
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    report.update(gpu=props.name, vram_gb=props.total_memory/2**30, capability=list(torch.cuda.get_device_capability()))
    a = torch.randn(1024, 1024, device="cuda", requires_grad=True)
    start = time.perf_counter()
    (a @ a).mean().backward()
    torch.cuda.synchronize()
    report["cuda_backward_probe_seconds"] = time.perf_counter()-start
    report["cuda_backward_success"] = bool(torch.isfinite(a.grad).all())
write_json(ROOT / "experiments/hardware.json", report)
print(report)
