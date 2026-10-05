"""Install the project editably without pip HTTP requests for build dependencies."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import urllib.request

root = Path(__file__).resolve().parents[1]
paths = []
pins = []
for name in ["setuptools", "wheel"]:
    with urllib.request.urlopen(f"https://pypi.org/pypi/{name}/json", timeout=30) as response:
        data = json.load(response)
    item = next(x for x in data["urls"] if x["filename"].endswith("py3-none-any.whl"))
    path = root / "tools/wheels" / item["filename"]
    urllib.request.urlretrieve(item["url"], path)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == item["digests"]["sha256"]
    paths.append(str(path))
    pins.append(f"{name}=={data['info']['version']}")
lock = root / "requirements.lock.txt"
existing = [x for x in lock.read_text().splitlines() if not x.startswith(("setuptools==", "wheel=="))]
lock.write_text("\n".join(sorted(existing + pins)) + "\n")
subprocess.run([sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", "--no-compile", *paths], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-e", str(root), "--no-deps", "--no-build-isolation"], check=True)
