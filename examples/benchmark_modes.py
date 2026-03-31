# Copyright 2026 Abadi Gilang
# SPDX-License-Identifier: Apache-2.0

"""
Benchmark script for HiveFrame: compare pandas, single-node, and cluster modes.

Usage examples:
  python examples/benchmark_modes.py --mode pandas --runs 1 --isolate --output-json --path /tmp/sample.csv
  python examples/benchmark_modes.py --mode single --runs 1 --isolate --output-json --path /tmp/sample.csv
  python examples/benchmark_modes.py --mode cluster --runs 1 --isolate --nats-url nats://127.0.0.1:4222 --start-ephemeral-nodes --ephemeral-count 1
  python examples/benchmark_modes.py --mode pandas,single,cluster --runs 3 --isolate

For more accurate memory measurement install psutil via `pip install .[benchmark]`.
"""

from __future__ import annotations

import argparse
import asyncio
import gc
import json
import logging
import os
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import hiveframe as hf

try:
    import psutil

    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False

try:
    import tracemalloc

    _HAS_TRACEMALLOC = True
except ImportError:
    _HAS_TRACEMALLOC = False

import resource


class MemorySampler:
    def __init__(self, interval_ms=100):
        self.interval = interval_ms / 1000.0
        self.samples = []
        self._running = False
        self._thread = None
        self._warned = False

    def _sample(self):
        while self._running:
            self.samples.append(self.get_rss_mb())
            time.sleep(self.interval)

    def start(self):
        self.samples.clear()
        self._running = True
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join()
        if self.samples:
            avg = sum(self.samples) / len(self.samples)
            peak = max(self.samples)
        else:
            avg = peak = 0.0
        return avg, peak, len(self.samples)

    def get_rss_mb(self):
        if _HAS_PSUTIL:
            proc = psutil.Process()
            rss = proc.memory_info().rss
            for child in proc.children(recursive=True):
                try:
                    rss += child.memory_info().rss
                except Exception:
                    pass
            return rss / (1024 * 1024)
        else:
            if not self._warned:
                print("WARNING: psutil not installed; using resource/tracemalloc fallback. Install psutil for more accurate memory stats.", file=sys.stderr)
                self._warned = True
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            if sys.platform == "darwin":
                rss = rss / (1024 * 1024)
            else:
                rss = rss / 1024
            return rss


def _get_rss_mb() -> float:
    """Return process resident set size in megabytes, including child processes.

    Prefer psutil for accurate per-process RSS and children; fallback to resource.
    """
    try:
        import psutil

        proc = psutil.Process()
        rss = proc.memory_info().rss
        # include children recursively if any
        try:
            for child in proc.children(recursive=True):
                try:
                    rss += child.memory_info().rss
                except Exception:
                    pass
        except Exception:
            pass
        return float(rss) / (1024.0 * 1024.0)
    except Exception:
        try:
            import resource

            usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            if usage > 10 ** 6:
                return float(usage) / (1024.0 * 1024.0)
            return float(usage) / 1024.0
        except Exception:
            return 0.0


def detect_text_column(df: pd.DataFrame) -> str:
    candidates = [c for c in df.columns if c.lower() in ("name", "title")]
    if candidates:
        return candidates[0]
    obj_cols = [c for c, dt in df.dtypes.items() if dt == object]
    return obj_cols[0] if obj_cols else df.columns[0]


def _read_table_flexible(path: Path) -> pd.DataFrame:
    """Read a CSV or Excel file into a pandas DataFrame with fallbacks.

    Tries pandas.read_excel/read_csv first. If openpyxl raises a parsing ValueError
    (e.g. when boolean cells contain 'true'/'false' strings), fall back to using
    openpyxl.load_workbook(..., data_only=True) and construct a DataFrame from
    worksheet values to be more tolerant.
    """
    suffix = path.suffix.lower()
    try:
        if suffix in (".xls", ".xlsx"):
            return pd.read_excel(path)
        else:
            return pd.read_csv(path)
    except Exception as exc:
        # Try openpyxl fallback for Excel files when pandas/openpyxl parsing fails.
        if suffix in (".xls", ".xlsx"):
            try:
                import openpyxl
                import zipfile
                import tempfile
                from io import BytesIO

                try:
                    wb = openpyxl.load_workbook(filename=path, read_only=False, data_only=True)
                except ValueError:
                    # If openpyxl can't read worksheets due to malformed XML, fall back to
                    # a tolerant XML-based parser that reads worksheet XML and sharedStrings
                    # directly from the .xlsx zipfile and constructs a DataFrame. This avoids
                    # strict openpyxl parsing that may fail on invalid boolean text nodes.
                    import xml.etree.ElementTree as ET

                    def _col_letters_to_index(letters: str) -> int:
                        # Convert Excel column letters (A, B, ..., Z, AA, AB, ...) to 0-based index
                        idx = 0
                        for ch in letters:
                            idx = idx * 26 + (ord(ch.upper()) - ord('A') + 1)
                        return idx - 1

                    def _parse_shared_strings(zf: zipfile.ZipFile) -> list[str]:
                        try:
                            data = zf.read('xl/sharedStrings.xml')
                        except KeyError:
                            return []
                        root = ET.fromstring(data)
                        ss: list[str] = []
                        # sharedStrings.xml contains <si> elements; extract text under <t>
                        for si in root.findall('.//{*}si'):
                            texts: list[str] = []
                            for t in si.findall('.//{*}t'):
                                if t.text:
                                    texts.append(t.text)
                            ss.append(''.join(texts))
                        return ss

                    def _parse_worksheet(zf: zipfile.ZipFile, sheet_path: str, shared: list[str]) -> list[list[str]]:
                        data = zf.read(sheet_path)
                        root = ET.fromstring(data)
                        # Find sheetData/row elements
                        rows_dict: dict[int, dict[int, str]] = {}
                        for row in root.findall('.//{*}row'):
                            r_idx = int(row.attrib.get('r', '0'))
                            cells: dict[int, str] = {}
                            for c in row.findall('.//{*}c'):
                                ref = c.attrib.get('r')  # like 'A1'
                                if not ref:
                                    continue
                                # split letters and numbers
                                letters = ''.join([ch for ch in ref if ch.isalpha()])
                                col_idx = _col_letters_to_index(letters) if letters else 0
                                cell_type = c.attrib.get('t')
                                val = ''
                                v = c.find('{*}v')
                                if v is not None and v.text is not None:
                                    if cell_type == 's':
                                        # shared string index
                                        try:
                                            val = shared[int(v.text)]
                                        except Exception:
                                            val = v.text
                                    else:
                                        val = v.text
                                else:
                                    # inlineStr
                                    is_el = c.find('{*}is')
                                    if is_el is not None:
                                        t = is_el.find('.//{*}t')
                                        if t is not None and t.text is not None:
                                            val = t.text
                                cells[col_idx] = val
                            if cells:
                                rows_dict[r_idx] = cells
                        if not rows_dict:
                            return []
                        max_row = max(rows_dict.keys())
                        max_col = 0
                        for rmap in rows_dict.values():
                            if rmap:
                                max_col = max(max_col, max(rmap.keys()))
                        table: list[list[str]] = []
                        for r in range(1, max_row + 1):
                            rowvals = []
                            cols = rows_dict.get(r, {})
                            for c in range(0, max_col + 1):
                                rowvals.append(cols.get(c, None))
                            table.append(rowvals)
                        return table

                    try:
                        with zipfile.ZipFile(path, 'r') as zf:
                            # pick the first worksheet under xl/worksheets/
                            ws_files = [n for n in zf.namelist() if n.startswith('xl/worksheets/') and n.endswith('.xml')]
                            if not ws_files:
                                raise ValueError('No worksheets found inside xlsx')
                            shared = _parse_shared_strings(zf)
                            table = _parse_worksheet(zf, ws_files[0], shared)
                            if not table:
                                return pd.DataFrame()
                            # determine header
                            header = table[0]
                            if all((h is None or isinstance(h, str)) for h in header):
                                df = pd.DataFrame(table[1:], columns=[str(h) if h is not None else '' for h in header])
                            else:
                                df = pd.DataFrame(table)
                            return df
                    except Exception:
                        # Give up and propagate original exception
                        raise
            except Exception:
                # Re-raise original exception to keep behavior observable
                raise exc
        # Non-excel files: re-raise
        raise


async def bench_pandas(path: Path, runs: int) -> dict:
    timings = {"load": [], "transform": [], "persist": []}
    memory_load = []
    memory_transform = []
    memory_persist = []
    sample_head = None
    total_rows = 0

    for i in range(runs):
        t0 = time.perf_counter()
        pd_df = _read_table_flexible(path)
        if sample_head is None:
            try:
                sample_head = pd_df.head(5).to_dict(orient="list")
            except Exception:
                sample_head = {}
        total_rows = len(pd_df.index)
        t1 = time.perf_counter()
        # sample memory after load
        gc.collect()
        memory_load.append(_get_rss_mb())

        col = detect_text_column(pd_df)
        # transform: uppercase that column on a copy
        t_start = time.perf_counter()
        df2 = pd_df.copy()
        df2[col] = [str(v).upper() if pd.notna(v) else v for v in df2[col].tolist()]
        t2 = time.perf_counter()
        # sample memory after transform
        gc.collect()
        memory_transform.append(_get_rss_mb())

        # persist: write to csv (fast) — optional and synchronous
        out = Path("/tmp/bench_pandas_out.csv")
        t_start_p = time.perf_counter()
        df2.to_csv(out, index=False)
        t3 = time.perf_counter()
        # sample memory after persist
        gc.collect()
        memory_persist.append(_get_rss_mb())

        timings["load"].append(t1 - t0)
        timings["transform"].append(t2 - t1)
        timings["persist"].append(t3 - t_start_p)

    res = {k: statistics.mean(v) if v else 0.0 for k, v in timings.items()}
    res.update({
        # "head": sample_head,
        "rows": total_rows,
        "memory_avg_mb_load": statistics.mean(memory_load) if memory_load else 0.0,
        "memory_peak_mb_load": max(memory_load) if memory_load else 0.0,
        "memory_avg_mb_transform": statistics.mean(memory_transform) if memory_transform else 0.0,
        "memory_peak_mb_transform": max(memory_transform) if memory_transform else 0.0,
        "memory_avg_mb_persist": statistics.mean(memory_persist) if memory_persist else 0.0,
        "memory_peak_mb_persist": max(memory_persist) if memory_persist else 0.0,
    })
    return res


async def bench_hiveframe_single(path: Path, runs: int) -> dict:
    timings = {"load": [], "transform": [], "persist": []}
    memory_load = []
    memory_transform = []
    memory_persist = []
    sample_head = None
    total_rows = 0

    for i in range(runs):
        t0 = time.perf_counter()
        pd_df = _read_table_flexible(path)
        if sample_head is None:
            try:
                sample_head = pd_df.head(5).to_dict(orient="list")
            except Exception:
                sample_head = {}
        total_rows = len(pd_df.index)
        data = pd_df.to_dict(orient="list")
        # create DFrame (this seeds initial data via transactional path)
        dframe = hf.DFrame(data)
        t1 = time.perf_counter()
        # sample memory after load
        gc.collect()
        memory_load.append(_get_rss_mb())

        col = detect_text_column(pd_df)
        t_start = time.perf_counter()
        dframe[col] = [str(v).upper() if pd.notna(v) else v for v in pd_df[col].tolist()]
        t2 = time.perf_counter()
        # sample memory after transform
        gc.collect()
        memory_transform.append(_get_rss_mb())

        t_start_p = time.perf_counter()
        _ = dframe.to_persistent("bench_single")
        t3 = time.perf_counter()
        # sample memory after persist
        gc.collect()
        memory_persist.append(_get_rss_mb())

        timings["load"].append(t1 - t0)
        timings["transform"].append(t2 - t1)
        timings["persist"].append(t3 - t_start_p)

    res = {k: statistics.mean(v) if v else 0.0 for k, v in timings.items()}
    res.update({
        # "head": sample_head,
        "rows": total_rows,
        "memory_avg_mb_load": statistics.mean(memory_load) if memory_load else 0.0,
        "memory_peak_mb_load": max(memory_load) if memory_load else 0.0,
        "memory_avg_mb_transform": statistics.mean(memory_transform) if memory_transform else 0.0,
        "memory_peak_mb_transform": max(memory_transform) if memory_transform else 0.0,
        "memory_avg_mb_persist": statistics.mean(memory_persist) if memory_persist else 0.0,
        "memory_peak_mb_persist": max(memory_persist) if memory_persist else 0.0,
    })
    return res


async def bench_hiveframe_cluster(path: Path, runs: int, node_id: str, port: int, nats_url: str, registry_backend: str, transport_backend: str) -> dict:
    timings = {"load": [], "transform": [], "persist": []}
    memory_load = []
    memory_transform = []
    memory_persist = []
    runtime = None
    sample_head = None
    total_rows = 0

    try:
        # Try to start a runtime configured for cluster. If NATS is unreachable or
        # nats-py not installed, ClusterRuntime will fall back to in-memory registry.
        from core.cluster_runtime import ClusterRuntime, RuntimeConfig

        config = RuntimeConfig(
            node_id=node_id,
            role="write",
            host="127.0.0.1",
            port=port,
            nats_url=nats_url,
            registry_backend=registry_backend,
            transport_backend=transport_backend,
            enable_cluster=True,
        )

        runtime = ClusterRuntime(config)
        await runtime.start()

        for i in range(runs):
            t0 = time.perf_counter()
            pd_df = _read_table_flexible(path)
            if sample_head is None:
                try:
                    sample_head = pd_df.head(5).to_dict(orient="list")
                except Exception:
                    sample_head = {}
            total_rows = len(pd_df.index)
            data = pd_df.to_dict(orient="list")
            dframe = hf.DFrame.from_runtime(runtime, data)
            t1 = time.perf_counter()
            # sample memory after load
            gc.collect()
            memory_load.append(_get_rss_mb())

            col = detect_text_column(pd_df)
            t_start = time.perf_counter()
            dframe[col] = [str(v).upper() if pd.notna(v) else v for v in pd_df[col].tolist()]
            t2 = time.perf_counter()
            # sample memory after transform
            gc.collect()
            memory_transform.append(_get_rss_mb())

            t_start_p = time.perf_counter()
            _ = dframe.to_persistent("bench_cluster")
            t3 = time.perf_counter()
            # sample memory after persist
            gc.collect()
            memory_persist.append(_get_rss_mb())

            # try a global read (fan-out) — in single-node cluster this is local snapshot
            t_read_start = time.perf_counter()
            _ = await dframe.read_fresh_global_async()
            t_read_end = time.perf_counter()

            timings["load"].append(t1 - t0)
            timings["transform"].append(t2 - t1)
            timings["persist"].append(t3 - t_start_p + (t_read_end - t_read_start))

    finally:
        if runtime is not None:
            try:
                await runtime.heartbeat.stop()
            except Exception:
                pass

    res = {k: statistics.mean(v) if v else 0.0 for k, v in timings.items()}
    res.update({
        # "head": sample_head,
        "rows": total_rows,
        "memory_avg_mb_load": statistics.mean(memory_load) if memory_load else 0.0,
        "memory_peak_mb_load": max(memory_load) if memory_load else 0.0,
        "memory_avg_mb_transform": statistics.mean(memory_transform) if memory_transform else 0.0,
        "memory_peak_mb_transform": max(memory_transform) if memory_transform else 0.0,
        "memory_avg_mb_persist": statistics.mean(memory_persist) if memory_persist else 0.0,
        "memory_peak_mb_persist": max(memory_persist) if memory_persist else 0.0,
    })
    return res


async def main_async(args: argparse.Namespace) -> None:
    path = Path(args.path)
    if not path.exists():
        # create a small sample Excel if missing
        sample = pd.DataFrame({
            "name": ["Alice", "Bob", "Charlie"],
            "city": ["jakarta", "bandung", "surabaya"],
            "score": [85, 92, 78],
        })
        if path.suffix.lower() in (".xls", ".xlsx"):
            sample.to_excel(path, index=False)
        else:
            sample.to_csv(path, index=False)
        print(f"Wrote sample to {path}")

    runs = args.runs

    modes = args.mode.split(",") if args.mode else ["pandas", "single", "cluster"]

    results: dict[str, dict] = {}

    if "pandas" in modes:
        print("Running pandas benchmark...")
        results["pandas"] = await bench_pandas(path, runs)

    if "single" in modes:
        print("Running hiveframe single-node benchmark...")
        results["single"] = await bench_hiveframe_single(path, runs)

    if "cluster" in modes:
        print("Running hiveframe cluster benchmark (attempting NATS)...")
        results["cluster"] = await bench_hiveframe_cluster(
            path, runs, args.node_id, args.port, args.nats_url, args.registry_backend, args.transport_backend
        )

    # Print summarized results
    print("\nBenchmark results (averages in seconds):")
    print("mode\tload\ttransform\tpersist(including remote read)")
    for mode, vals in results.items():
        print(f"{mode}\t{vals['load']:.4f}\t{vals['transform']:.4f}\t{vals['persist']:.4f}")
        # print rows and head sample
        print(f"  rows: {vals.get('rows', 0)}")
        head = vals.get('head')
        if head:
            try:
                print("  head:")
                # convert to DataFrame for pretty printing
                print(pd.DataFrame(head).head())
            except Exception:
                print(f"  head: {head}")
        print(f"  memory_avg_mb (load/transform/persist): {vals.get('memory_avg_mb_load'):.2f} / {vals.get('memory_avg_mb_transform'):.2f} / {vals.get('memory_avg_mb_persist'):.2f}")
        print(f"  memory_peak_mb (load/transform/persist): {vals.get('memory_peak_mb_load'):.2f} / {vals.get('memory_peak_mb_transform'):.2f} / {vals.get('memory_peak_mb_persist'):.2f}")

    if args.output_json:
        # Output JSON for test/automation: only print JSON, no extra text
        if len(results) == 1:
            print(json.dumps(list(results.values())[0]))
        else:
            for mode, vals in results.items():
                print(json.dumps({"mode": mode, **vals}))
        return


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="HiveFrame Benchmark Modes")
    p.add_argument("--mode", type=str, default="pandas,single,cluster", help="Comma-separated list of modes to run: pandas,single,cluster")
    p.add_argument("--runs", type=int, default=3, help="Number of runs per mode")
    p.add_argument("--path", type=str, required=True, help="Path to input file (csv or xlsx)")
    p.add_argument("--isolate", action="store_true", default=True, help="Run each mode in a separate process (default: True)")
    p.add_argument("--no-isolate", dest="isolate", action="store_false", help="Run all modes in a single process (legacy)")
    p.add_argument("--output-json", action="store_true", help="Emit JSON result per mode (for subprocess runs)")
    p.add_argument("--start-ephemeral-nodes", action="store_true", help="Start ephemeral cluster nodes for cluster mode")
    p.add_argument("--ephemeral-count", type=int, default=1, help="Number of ephemeral nodes to start for cluster mode")
    p.add_argument("--sample-head-rows", type=int, default=5, help="Number of rows to sample for head output")
    p.add_argument("--no-persist", action="store_true", help="Skip persist stage")
    p.add_argument("--cluster-node", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--run-single-mode", type=str, help=argparse.SUPPRESS)
    p.add_argument("--nats-url", type=str, default="nats://127.0.0.1:4222", help="NATS URL for cluster mode")
    p.add_argument("--node-id", type=str, default="writer-1", help="Node ID for cluster mode")
    p.add_argument("--port", type=int, default=19002, help="Port for cluster node")
    p.add_argument("--registry-backend", type=str, default="nats", help="Registry backend (nats/memory)")
    p.add_argument("--transport-backend", type=str, default="quic", help="Transport backend (quic/tcp)")
    p.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        print("Cancelled")
        sys.exit(1)


if __name__ == "__main__":
    main()
