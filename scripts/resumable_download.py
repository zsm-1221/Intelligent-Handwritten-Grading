"""Verified ranged downloads for unreliable large-file connections, stdlib only."""
import hashlib
import json
import os
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def download(url, target, size, expected, workers=6):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size == size:
        if hashlib.file_digest(target.open("rb"),"sha256").hexdigest() == expected:
            print("Already verified",target.name,flush=True)
            return
    parts = target.parent / (target.name+".parts")
    parts.mkdir(exist_ok=True)
    chunk = 8*1024*1024
    def fetch(start):
        end = min(start+chunk,size)-1
        part = parts / str(start)
        if part.exists() and part.stat().st_size == end-start+1:
            return part.stat().st_size
        for attempt in range(5):
            try:
                request_url = url + ("&" if "?" in url else "?") + f"chunk={start}"
                request = urllib.request.Request(request_url,headers={"Range":f"bytes={start}-{end}"})
                with urllib.request.urlopen(request,timeout=60) as r:
                    if r.status != 206 or r.headers.get("Content-Range") != f"bytes {start}-{end}/{size}":
                        raise RuntimeError(f"Invalid range response: {r.status} {r.headers.get('Content-Range')}")
                    content = r.read()
                if len(content) != end-start+1:
                    raise RuntimeError("Truncated range")
                part.write_bytes(content)
                return len(content)
            except Exception as exc:
                print("Retry",target.name,start,attempt,type(exc).__name__,flush=True)
                if attempt == 4:
                    raise
                time.sleep(min(2**attempt,8))
    total, last, started = 0,0,time.monotonic()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(fetch,start) for start in range(0,size,chunk)]
        for f in as_completed(futures):
            total += f.result()
            if total-last >= 128*1024*1024 or total == size:
                print(f"{target.name} {total/size:.1%} {total/2**20:.0f} MiB {total/2**20/(time.monotonic()-started):.1f} MiB/s",flush=True)
                last=total
    h = hashlib.sha256()
    assembled = target.with_suffix(target.suffix+".verified-download")
    with assembled.open("wb") as out:
        for start in range(0,size,chunk):
            data = (parts / str(start)).read_bytes()
            out.write(data)
            h.update(data)
    if h.hexdigest() != expected:
        raise RuntimeError(f"SHA256 mismatch: {h.hexdigest()} != {expected}")
    os.replace(assembled,target)
    print("VERIFIED",target.name,expected,flush=True)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("kind",choices=["torch","trocr"])
    args = ap.parse_args()
    if args.kind == "torch":
        download("https://download.pytorch.org/whl/cu128/torch-2.8.0%2Bcu128-cp312-cp312-win_amd64.whl",ROOT/"data/installers/torch-2.8.0+cu128-cp312-cp312-win_amd64.whl",3461384651,"0ad925202387f4e7314302a1b4f8860fa824357f9b1466d7992bf276370ebcff")
    else:
        revision="b4648cfa171985a6745f37ddd637e98c0da958ac"
        with urllib.request.urlopen(f"https://huggingface.co/api/models/microsoft/trocr-small-handwritten/revision/{revision}?blobs=true",timeout=30) as r:
            meta=json.load(r)
        sibling=next(x for x in meta["siblings"] if x["rfilename"] == "pytorch_model.bin")
        expected=sibling["lfs"]["sha256"]
        download(f"https://huggingface.co/microsoft/trocr-small-handwritten/resolve/{revision}/pytorch_model.bin", ROOT/"data/pretrained/trocr/pytorch_model.bin",sibling["size"],expected)
