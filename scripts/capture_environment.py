import sys
import subprocess
import shutil
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.common import ROOT, digest, write_json, experiment
out=experiment("environment_snapshot")
lock=subprocess.check_output([sys.executable,"-m","pip","freeze"],text=True,encoding="utf-8")
(ROOT / "requirements-lock.txt").write_text(lock,encoding="utf-8")
(out / "requirements-lock.txt").write_text(lock,encoding="utf-8")
files=[]
for folder in ("src","scripts","demo","tests","configs"):
    files += [p for p in (ROOT / folder).rglob("*") if p.is_file() and p.suffix in (".py",".json",".ps1")]
write_json(out / "code_sha256.json",{p.relative_to(ROOT).as_posix():digest(p) for p in files})
for p in files:
    destination=out / "source" / p.relative_to(ROOT)
    destination.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(p,destination)
print(out)
