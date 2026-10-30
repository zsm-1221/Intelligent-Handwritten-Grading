import urllib.request
import time
urls = ["https://download.pytorch.org/whl/cu128/torch-2.8.0%2Bcu128-cp312-cp312-win_amd64.whl?probe=20261030", "https://huggingface.co/microsoft/trocr-small-handwritten/resolve/b4648cfa171985a6745f37ddd637e98c0da958ac/pytorch_model.bin?download=true"]
for url in urls:
    try:
        start = time.time()
        with urllib.request.urlopen(urllib.request.Request(url,headers={"Range":"bytes=0-1023"}),timeout=25) as r:
            content = r.read(1024)
            print(r.status,r.headers.get("Content-Length"),r.headers.get("Content-Range"),len(content),repr(content[:8]),time.time()-start,flush=True)
    except Exception as exc:
        print(type(exc).__name__,str(exc),flush=True)
