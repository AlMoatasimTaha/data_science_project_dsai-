"""Containment, dataset provenance, model metadata and Git checks."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
def git(*args):
    p = subprocess.run(["git", "-c", f"safe.directory={ROOT.as_posix()}", *args], cwd=ROOT, capture_output=True, text=True)
    return {"exit_code": p.returncode, "output": (p.stdout+p.stderr).strip()}

def main():
    assert ROOT.name == "data_science_project_dsai", f"Wrong root: {ROOT}"
    root_check = git("rev-parse", "--show-toplevel")
    assert root_check["exit_code"] == 0 and Path(root_check["output"]).resolve() == ROOT
    nested_git, escaped = [], []
    paths = []
    for path in ROOT.rglob("*"):
        rel = path.relative_to(ROOT)
        if rel.parts[0] in [".git", ".venv", "tools", ".pytest_cache"] or "__pycache__" in rel.parts:
            continue
        if path.name == ".git":
            nested_git.append(rel.as_posix())
        if not path.resolve().is_relative_to(ROOT):
            escaped.append(rel.as_posix())
        if path.is_file():
            paths.append(rel.as_posix())
    assert not nested_git and not escaped
    manifest = json.loads((ROOT / "data/manifest.json").read_text())
    hashes = {f: hashlib.sha256((ROOT / "data/raw" / f).read_bytes()).hexdigest() == entry["sha256"] for f, entry in manifest["files"].items()}
    assert all(hashes.values())
    registry = json.loads((ROOT / "models/registry.json").read_text())
    assert len(registry) == 5 and all(set(modes) == {"random", "temporal"} for modes in registry.values())
    assert all((ROOT / f"models/{name}/{split}.joblib").exists() for name, modes in registry.items() for split in modes)
    audit = {"project_root_name": ROOT.name, "root_is_git_toplevel": True, "nested_git": nested_git, "escaped_project_paths": escaped, "source_hashes_verified": hashes,
             "tasks": list(registry), "git_status": git("status", "--short"), "git_remote": git("remote", "-v"), "project_files": len(paths), "note": "Project files enumerated within root; runtime executables supplied externally. Virtual environment and wheel cache are local and ignored."}
    (ROOT / "reports/quality_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    (ROOT / "reports/directory_tree.txt").write_text("data_science_project_dsai/\n" + "\n".join("  " + x for x in sorted(paths)) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in audit.items() if k not in ["git_status"]}, indent=2))

if __name__ == "__main__":
    main()
