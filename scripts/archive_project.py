"""Create and verify a complete ZIP64 archive without modifying project inputs."""
import hashlib
import json
import os
import shutil
import time
import zipfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT.parent / (ROOT.name + ".zip")
if DESTINATION.exists():
    DESTINATION = ROOT.parent / (ROOT.name + "_" + datetime.now().strftime("%Y%m%d-%H%M%S") + ".zip")
PARTIAL = DESTINATION.with_suffix(".zip.partial")
RECEIPT = DESTINATION.with_suffix(".zip.manifest.json")
PROGRESS = DESTINATION.with_suffix(".zip.progress.json")
ALREADY_COMPRESSED = {".zip", ".whl", ".png", ".jpg", ".jpeg", ".gz", ".br", ".woff", ".woff2", ".mp4", ".pdf"}

def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")

def main():
    started = time.monotonic()
    files, empty_dirs, groups = [], [], {}
    for directory, directories, names in os.walk(ROOT, followlinks=False):
        directories.sort()
        names.sort()
        current = Path(directory)
        if not directories and not names:
            empty_dirs.append(current)
        for name in directories + names:
            path = current / name
            if path.is_symlink() or path.is_junction():
                raise RuntimeError(f"Archive requires explicit handling of linked path: {path}")
        for name in names:
            path = current / name
            stat = path.stat()
            relative = path.relative_to(ROOT)
            files.append((path, relative, stat.st_size, stat.st_mtime_ns))
            key = relative.parts[0]
            groups[key] = groups.get(key, 0) + stat.st_size
    total = sum(r[2] for r in files)
    if shutil.disk_usage(ROOT.parent).free < total + 1024**3:
        raise RuntimeError("Insufficient free space for a complete verified archive.")
    print(json.dumps({"phase":"inventory", "files":len(files), "source_gib":round(total / 1024**3, 3),
                      "top_level_gib":{k:round(v / 1024**3,3) for k,v in groups.items() if v > 1024**2},
                      "output":str(DESTINATION)}, ensure_ascii=False), flush=True)
    processed, last = 0, 0
    archive_names = set()
    with PARTIAL.open("xb") as raw:
        with zipfile.ZipFile(raw, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as archive:
            for directory in empty_dirs:
                archive.writestr(ROOT.name + "/" + directory.relative_to(ROOT).as_posix().rstrip("/") + "/", b"")
            for i, (path, relative, size, modified) in enumerate(files, 1):
                name = ROOT.name + "/" + relative.as_posix()
                if name in archive_names:
                    raise RuntimeError(f"Duplicate archive path: {name}")
                archive_names.add(name)
                method = zipfile.ZIP_STORED if path.suffix.lower() in ALREADY_COMPRESSED else zipfile.ZIP_DEFLATED
                archive.write(path, name, compress_type=method, compresslevel=1 if method==zipfile.ZIP_DEFLATED else None)
                after = path.stat()
                if (after.st_size, after.st_mtime_ns) != (size, modified):
                    raise RuntimeError(f"File changed during archival: {relative}")
                processed += size
                if time.monotonic()-last >= 15 or i==len(files):
                    progress={"phase":"compressing", "files_done":i, "files_total":len(files),
                              "source_bytes_done":processed, "source_bytes_total":total,
                              "percent":round(processed / max(1,total)*100,1),
                              "archive_bytes_so_far":raw.tell(), "current_file":relative.as_posix()}
                    save_json(PROGRESS, progress)
                    print(json.dumps(progress,ensure_ascii=False),flush=True)
                    last=time.monotonic()
        raw.flush()
        os.fsync(raw.fileno())
    print("VERIFYING every archived file by streamed CRC32",flush=True)
    save_json(PROGRESS,{"phase":"verifying", "files_total":len(files),"source_bytes_total":total})
    import zlib
    verified_bytes, verified_files, last = 0,0,0
    with zipfile.ZipFile(PARTIAL) as archive:
        entries=archive.infolist()
        file_entries=[r for r in entries if not r.is_dir()]
        assert len(file_entries)==len(files)
        assert sum(r.file_size for r in file_entries)==total
        assert set(r.filename for r in file_entries)==archive_names
        for entry in file_entries:
            crc=0
            with archive.open(entry) as reader:
                while chunk:=reader.read(1024*1024):
                    crc=zlib.crc32(chunk,crc)
                    verified_bytes+=len(chunk)
            if crc & 0xffffffff != entry.CRC:
                raise RuntimeError(f"CRC verification failed: {entry.filename}")
            verified_files+=1
            if time.monotonic()-last>=15 or verified_files==len(files):
                progress={"phase":"verifying", "files_done":verified_files,"files_total":len(files),
                          "percent":round(verified_bytes / max(1,total)*100,1)}
                save_json(PROGRESS,progress)
                print(json.dumps(progress),flush=True)
                last=time.monotonic()
    for path,relative,size,modified in files:
        after=path.stat()
        if (after.st_size,after.st_mtime_ns)!=(size,modified):
            raise RuntimeError(f"Source changed before verification finished: {relative}")
    print("HASHING archive SHA256",flush=True)
    save_json(PROGRESS,{"phase":"hashing", "archive_bytes":PARTIAL.stat().st_size})
    with PARTIAL.open("rb") as reader:
        checksum=hashlib.file_digest(reader,"sha256").hexdigest()
    if DESTINATION.exists():
        raise FileExistsError(DESTINATION)
    PARTIAL.rename(DESTINATION)
    result={"status":"verified", "archive":str(DESTINATION), "archive_bytes":DESTINATION.stat().st_size,
            "archive_gib":round(DESTINATION.stat().st_size/1024**3,3), "source_files":len(files),
            "empty_directories":len(empty_dirs), "source_bytes":total, "sha256":checksum,
            "crc32_all_files_passed":True, "excluded_paths":[], "elapsed_seconds":round(time.monotonic()-started,1)}
    save_json(RECEIPT,result)
    save_json(PROGRESS,result)
    print("ARCHIVE_COMPLETED "+json.dumps(result,ensure_ascii=False),flush=True)

if __name__=="__main__":
    main()
