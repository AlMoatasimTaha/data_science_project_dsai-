"""Exercise all prediction routes against the running localhost API."""
import json
from pathlib import Path
import urllib.request

root = Path(__file__).resolve().parents[1]
posting = {"job_title": "Data Scientist", "company_type": "Startup", "industry": "Tech", "country": "USA", "company_size": "Medium", "posted_year": 2026, "experience_level": "Mid", "remote_type": "Hybrid"}
result = {}
for task in ["salary", "experience", "remote", "skills", "salary_width"]:
    request = urllib.request.Request(f"http://127.0.0.1:8000/predict/{task}", data=json.dumps(posting).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=60) as response:
        assert response.status == 200
        result[task] = json.load(response)
with urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=30) as response:
    result["health"] = json.load(response)
assert result["health"]["ready"]
(root / "reports/api_smoke.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
print("Five live prediction endpoints passed; API readiness confirmed.")
