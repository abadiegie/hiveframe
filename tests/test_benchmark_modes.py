import subprocess
import sys
import json
from pathlib import Path

def test_pandas_isolated(tmp_path):
    sample = tmp_path / "sample.csv"
    sample.write_text("name,city,score\nAlice,Jakarta,85\nBob,Bandung,92\nCharlie,Surabaya,78\n")
    cmd = [sys.executable, "examples/benchmark_modes.py", "--mode", "pandas", "--runs", "1", "--isolate", "--output-json", "--path", str(sample)]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    # Should emit JSON result
    out = proc.stdout.strip().splitlines()
    # Find JSON line
    for line in out:
        if line.startswith("{"):
            result = json.loads(line)
            break
    else:
        raise AssertionError("No JSON output found")
    assert "load" in result and "rows" in result and "memory_avg_mb_load" in result
    assert result["rows"] == 3
    assert result["load"] >= 0
    assert result["memory_avg_mb_load"] > 0
