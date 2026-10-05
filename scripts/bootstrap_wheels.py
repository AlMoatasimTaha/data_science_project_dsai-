"""Fallback when pip's HTTP transport stalls: verified wheels from official PyPI.

Run with the project's Python. Installs no files outside this project environment.
Uses pip's bundled packaging solely to resolve wheel tags/requirement markers.
"""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import urllib.request
from pip._vendor.packaging.requirements import Requirement
from pip._vendor.packaging.version import Version, InvalidVersion
from pip._vendor.packaging.tags import sys_tags
from pip._vendor.packaging.utils import parse_wheel_filename, canonicalize_name

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "tools/wheels"
DEST.mkdir(parents=True, exist_ok=True)
TAGS = set(sys_tags())
selected = {}
constraints = {}
queue = [Requirement(x.strip()) for x in (ROOT / "requirements.txt").read_text().splitlines() if x.strip() and not x.startswith("#")]

def get_json(url):
    with urllib.request.urlopen(url, timeout=45) as r:
        return json.load(r)

while queue:
    req = queue.pop(0)
    if req.marker and not req.marker.evaluate():
        continue
    name = canonicalize_name(req.name)
    constraints.setdefault(name, []).append(req.specifier)
    if name in selected:
        if not all(Version(selected[name]["version"]) in s for s in constraints[name]):
            raise RuntimeError(f"Conflicting constraints for {name}; use pip's full resolver")
        continue
    data = get_json(f"https://pypi.org/pypi/{name}/json")
    versions = []
    for value in data["releases"]:
        try:
            v = Version(value)
        except InvalidVersion:
            continue
        if not v.is_prerelease and all(v in s for s in constraints[name]):
            versions.append(v)
    versions.sort(reverse=True)
    chosen = None
    for version in versions:
        files = data["releases"][str(version)]
        compatible = [f for f in files if f["filename"].endswith(".whl") and not f.get("yanked") and parse_wheel_filename(f["filename"])[3] & TAGS]
        if compatible:
            metadata = get_json(f"https://pypi.org/pypi/{name}/{version}/json")
            chosen = {"version": str(version), "file": compatible[0], "metadata": metadata["info"]}
            break
    if chosen is None:
        raise RuntimeError(f"No compatible wheel for {req}")
    selected[name] = chosen
    print(f"Resolved {name}=={chosen['version']}", flush=True)
    for dep in chosen["metadata"].get("requires_dist") or []:
        d = Requirement(dep)
        if d.marker is None or d.marker.evaluate({"extra": ""}) or any(d.marker.evaluate({"extra": extra}) for extra in req.extras):
            d.marker = None
            queue.append(d)

def fetch(item):
    name, record = item
    f = record["file"]
    p = DEST / f["filename"]
    if not p.exists():
        urllib.request.urlretrieve(f["url"], p)
    if hashlib.sha256(p.read_bytes()).hexdigest() != f["digests"]["sha256"]:
        raise RuntimeError(f"Checksum failure: {name}")
    print(f"Verified {p.name}", flush=True)
    return str(p)

with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
    paths = list(pool.map(fetch, selected.items()))
(ROOT / "requirements.lock.txt").write_text("\n".join(f"{n}=={r['version']}" for n, r in sorted(selected.items())) + "\n")
subprocess.run([sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", "--no-compile", *paths], check=True)
subprocess.run([sys.executable, "-m", "pip", "check"], check=True)
