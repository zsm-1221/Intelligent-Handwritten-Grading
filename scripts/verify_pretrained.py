import hashlib
import json
import sys
import urllib.request
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.common import ROOT, digest, write_json

records=[]
for model,folder,revision in [("distilbert/distilbert-base-uncased","distilbert","12040accade4e8a0f71eabdb258fecc2e7e948be"),
                              ("microsoft/trocr-small-handwritten","trocr","b4648cfa171985a6745f37ddd637e98c0da958ac")]:
    with urllib.request.urlopen(f"https://huggingface.co/api/models/{model}/revision/{revision}?blobs=true",timeout=30) as r:
        meta=json.load(r)
    siblings={r["rfilename"]:r for r in meta["siblings"]}
    for file in (ROOT / "data/pretrained" / folder).iterdir():
        if not file.is_file() or file.name not in siblings:
            continue
        info=siblings[file.name]
        if "lfs" in info:
            actual,expected=digest(file),info["lfs"]["sha256"]
            algorithm="sha256"
        else:
            data=file.read_bytes()
            actual=hashlib.sha1(b"blob "+str(len(data)).encode()+b"\0"+data).hexdigest()
            expected=info["blobId"]
            algorithm="git_blob_sha1"
        assert actual==expected,f"Official checksum mismatch: {file}"
        records.append({"model":model,"revision":revision,"file":file.name,"official_algorithm":algorithm,
                        "official_checksum":expected,"sha256":digest(file),"verified":True})
write_json(ROOT / "data/pretrained/verified_manifest.json",records)
print("PRETRAINED_VERIFIED",len(records),flush=True)
