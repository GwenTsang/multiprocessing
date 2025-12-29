#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import fitz  # PyMuPDF
from concurrent.futures import ProcessPoolExecutor, as_completed


OUT_DIR_BASE: Path = Path("texts")     # output base folder (relative to CWD)


MODE: str = "both"  # "single", "process", or "both"
WORKERS: int = 0    # 0 => os.cpu_count()
SKIP_EXISTING: bool = False  # True => skip PDFs whose output .txt exists and is non-empty
LIMIT: int = 0      # 0 => all PDFs, else first N (deterministic)
ERRORS_SAMPLE: int = 10  # how many (file, error) pairs to include in JSON

# =============================
# Utilities
# =============================


def list_pdfs(root: Path) -> List[Path]:
    """Recursively list PDFs under root (deterministic order)."""
    if not root.exists():
        return []
    pdfs = [p for p in root.rglob("*.pdf") if p.is_file()]
    pdfs.sort()
    return pdfs


def target_txt_path(pdf_path: Path, raw_root: Path, out_root: Path) -> Path:
    """Keep relative structure when possible; otherwise fall back to filename."""
    try:
        rel = pdf_path.relative_to(raw_root)
        out_path = (out_root / rel).with_suffix(".txt")
    except ValueError:
        out_path = (out_root / pdf_path.name).with_suffix(".txt")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return out_path


def write_text(out_path: Path, text: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8", errors="replace")


def extract_text_pymupdf(pdf_path: Path) -> str:
    chunks: List[str] = []
    with fitz.open(pdf_path) as doc:
        for page in doc:
            chunks.append(page.get_text("text"))
    return "\n".join(chunks).strip()


# =============================
# Worker (top-level for Windows ProcessPool pickling)
# =============================


@dataclass(frozen=True)
class FileResult:
    ok: bool
    skipped: bool
    seconds: float
    error: Optional[str]




def process_one_pdf(pdf_path_str: str, raw_root_str: str, out_root_str: str, skip_existing: bool) -> Tuple[str, str, FileResult]:
    t0 = time.perf_counter()
    pdf_path = Path(pdf_path_str)
    raw_root = Path(raw_root_str)
    out_root = Path(out_root_str)
    out_path = target_txt_path(pdf_path, raw_root, out_root)


    try:
        if skip_existing and out_path.exists() and out_path.stat().st_size > 0:
            return (str(pdf_path), str(out_path), FileResult(ok=True, skipped=True, seconds=time.perf_counter() - t0, error=None))


        text = extract_text_pymupdf(pdf_path)
        write_text(out_path, text)
        return (str(pdf_path), str(out_path), FileResult(ok=True, skipped=False, seconds=time.perf_counter() - t0, error=None))


    except Exception as e:
        err = "".join(traceback.format_exception_only(type(e), e)).strip()
        return (str(pdf_path), str(out_path), FileResult(ok=False, skipped=False, seconds=time.perf_counter() - t0, error=err))




# =============================
# Aggregation + Runs
# =============================


def summarize_file_results(results: List[Tuple[str, str, FileResult]], total_seconds: float, errors_sample_n: int) -> Dict[str, Any]:
    oks = [r for (_, _, r) in results if r.ok]
    errs = [(pdf, r.error) for (pdf, _, r) in results if not r.ok]
    skipped = [r for (_, _, r) in results if r.ok and r.skipped]
    processed = [r for (_, _, r) in results if r.ok and not r.skipped]


    def _mean(xs: List[float]) -> float:
        return float(statistics.fmean(xs)) if xs else 0.0


    def _median(xs: List[float]) -> float:
        return float(statistics.median(xs)) if xs else 0.0


    times_ok = [r.seconds for r in oks]
    times_processed = [r.seconds for r in processed]


    out: Dict[str, Any] = {
        "n_total": len(results),
        "n_ok": len(oks),
        "n_err": len(errs),
        "n_skipped": len(skipped),
        "n_processed": len(processed),
        "total_time_sec": float(total_seconds),
        "mean_time_sec_ok": _mean(times_ok),
        "median_time_sec_ok": _median(times_ok),
        "mean_time_sec_processed": _mean(times_processed),
        "median_time_sec_processed": _median(times_processed),
        "errors_sample": errs[: max(0, int(errors_sample_n))],
    }
    out["pdfs_per_sec_ok"] = float(len(oks) / total_seconds) if total_seconds > 0 else 0.0
    out["pdfs_per_sec_processed"] = float(len(processed) / total_seconds) if total_seconds > 0 else 0.0
    return out




def run_single(pdfs: List[Path], raw_root: Path, out_root: Path, skip_existing: bool) -> Tuple[List[Tuple[str, str, FileResult]], float]:
    results: List[Tuple[str, str, FileResult]] = []
    t0 = time.perf_counter()
    for p in pdfs:
        results.append(process_one_pdf(str(p), str(raw_root), str(out_root), skip_existing))
    return results, time.perf_counter() - t0




def run_process_pool(pdfs: List[Path], raw_root: Path, out_root: Path, skip_existing: bool, max_workers: int) -> Tuple[List[Tuple[str, str, FileResult]], float]:
    results: List[Tuple[str, str, FileResult]] = []
    t0 = time.perf_counter()


    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(process_one_pdf, str(p), str(raw_root), str(out_root), skip_existing) for p in pdfs]
        for fut in as_completed(futs):
            results.append(fut.result())


    results.sort(key=lambda x: x[0])  # deterministic
    return results, time.perf_counter() - t0

# Main

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("input_path", type=Path, help="Folder containing PDFs")
    ap.add_argument("--cpu-name", required=True)
    args = ap.parse_args(argv)


    cpu_cores = os.cpu_count() or 0
    max_workers = WORKERS if WORKERS and WORKERS > 0 else (cpu_cores or 1)


    raw_root = args.input_path.resolve()
    out_base = OUT_DIR_BASE.resolve()
    results_path = (raw_root / "results.json").resolve()


    pdfs = list_pdfs(raw_root)
    if LIMIT and LIMIT > 0:
        pdfs = pdfs[:LIMIT]


    # Separate output dirs so MODE="both" doesn't contaminate runs with SKIP_EXISTING
    out_single = out_base / "single"
    out_proc = out_base / "process"


    payload: Dict[str, Any] = {
        "cpu_name": args.cpu_name,
        "cup_name": args.cpu_name,
        "cpu_cores_os_cpu_count": cpu_cores,
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python_version": platform.python_version(),
        },
        "paths": {
            "raw_dir": str(raw_root),
            "out_dir_base": str(out_base),
            "results_json": str(results_path),
        },
        "config": {
            "mode": MODE,
            "skip_existing": bool(SKIP_EXISTING),
            "workers": int(max_workers),
            "limit": int(LIMIT),
            "errors_sample": int(ERRORS_SAMPLE),
        },
        "pdf_count": len(pdfs),
        "results": {},
    }


    try:
        if MODE in ("single", "both"):
            single_results, single_total = run_single(pdfs, raw_root, out_single, SKIP_EXISTING)
            payload["results"]["single"] = summarize_file_results(single_results, single_total, ERRORS_SAMPLE)


        if MODE in ("process", "both"):
            proc_results, proc_total = run_process_pool(pdfs, raw_root, out_proc, SKIP_EXISTING, max_workers=max_workers)
            payload["results"]["process"] = summarize_file_results(proc_results, proc_total, ERRORS_SAMPLE)
            payload["results"]["process"]["max_workers"] = int(max_workers)


        if "single" in payload["results"] and "process" in payload["results"]:
            t1 = float(payload["results"]["single"].get("total_time_sec") or 0.0)
            t2 = float(payload["results"]["process"].get("total_time_sec")
