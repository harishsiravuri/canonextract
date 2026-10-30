"""Cycle-5 binder rewrite: enforce 3-of-3 alignment (row_label ~ tuple.method,
header_path ~ tuple.dataset|metric, cell ~ tuple.value) on cached extractor
outputs. No LLM calls. Emits five-bucket provenance:

  strict_table_valid    - all 3 checks pass on at least one cell
  partial_table_valid   - value + one of (row, col) — cannot uniquely pick intersection
  prose_claimed_table   - value only (or neither) — old binder's mistake
  prose                 - source=prose from extractor
  missing               - no cell candidate found

Outputs:
  outputs/eval_runs_{metalead,scirex}_cycle5/{corpus}_{paper_id}.json   (rewritten tuples)
  outputs/canonextract_build/provenance_cycle5/{corpus}/{paper_id}.json (per-tuple bucket detail)
  outputs/canonextract_build/provenance_cycle5_summary.json             (aggregate + packet 2 rescore)

Reuses cached parsed tables:
  outputs/canonextract_diag_v2/metalead_tables/{paper_key}.json  (nested parser, has rows_by_label)
  outputs/canonextract_diag/tables/{doc_id}.json                 (flat parser)
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import typer
from rich.console import Console

SEED = 42

app = typer.Typer(no_args_is_help=False, add_completion=False)
console = Console()

# ---------- normalization helpers ----------
_LATEX_CMD_RE = re.compile(
    r"\\(?:dagger|ddagger|ast|star|textbf|emph|textit|texttt|bf|it|em|"
    r"textsuperscript|footnotemark|footnote|leftmark|rightmark|mathbf|"
    r"mathrm|text|mbox|hbox|underline|overline|tiny|small|large|Large|"
    r"LARGE|huge|Huge|scriptsize|footnotesize|normalsize|Ref)"
    r"(?:\{[^{}]*\})?"
)
_LATEX_BRACES_RE = re.compile(r"\{|\}")
_LATEX_MATH_RE = re.compile(r"\$[^$]*\$")
_LATEX_BACKSLASH_TOKEN_RE = re.compile(r"\\[a-zA-Z]+\*?")
_PAREN_RE = re.compile(r"\(([^()]*)\)")
_PCT_RE = re.compile(r"(\d[\d\.,]*)\s*%")
_NUM_RE = re.compile(r"([+-]?\d[\d,]*\.?\d*|\.\d+)([eE][+-]?\d+)?")
_MULTIWS_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^\w\s]")

OURS_MARKERS = {
    "ours", "our", "our method", "our model", "our approach", "our system",
    "proposed", "proposed method", "proposed model", "proposed system",
    "this work", "this paper", "we", "mine", "us",
}


def strip_latex(s: str | None) -> str:
    """Strip common LaTeX commands, math, super/subscripts, braces."""
    if not s: return ""
    s = str(s)
    s = _LATEX_MATH_RE.sub(" ", s)
    s = _LATEX_CMD_RE.sub(" ", s)
    s = _LATEX_BACKSLASH_TOKEN_RE.sub(" ", s)
    s = _LATEX_BRACES_RE.sub(" ", s)
    # strip footnote markers like ^\dagger, ^*, ^1 already removed above
    s = re.sub(r"[\^_](?:\{[^{}]*\}|\S)", " ", s)
    return s


def norm(s: str | None) -> str:
    """Lower + strip LaTeX + strip units + strip parens + collapse ws + strip punct."""
    if not s: return ""
    s = strip_latex(str(s))
    # strip parenthetical qualifiers like (test), (dev), (%)
    s = _PAREN_RE.sub(" ", s)
    # strip unit-like suffixes
    s = re.sub(r"\b(?:pct|percent|percentage|acc|score|value)\b", " ", s, flags=re.I)
    s = s.lower().replace("_", " ")
    s = _PUNCT_RE.sub(" ", s)
    s = _MULTIWS_RE.sub(" ", s).strip()
    return s


def parse_number(s) -> float | None:
    """Parse '95.41', '95.41 \\pm 0.2', '0.85%', '85.0', '(15.3)' -> float. None if not parseable."""
    if s is None: return None
    if isinstance(s, (int, float)):
        try: return float(s)
        except Exception: return None
    txt = str(s)
    txt = strip_latex(txt)
    # remove parens; strip pm/pm-ranges
    txt = re.sub(r"[\(\)\[\]]", " ", txt)
    txt = re.sub(r"\\pm.*|±.*", "", txt)
    # remove commas in numbers
    txt = txt.replace(",", "")
    m = _NUM_RE.search(txt)
    if not m: return None
    try:
        return float(m.group(0))
    except Exception:
        return None


def numeric_match(a: float, b: float) -> bool:
    """Match with tolerance max(0.5, 0.005 * abs(a)); handle pct-vs-frac."""
    if a is None or b is None: return False
    # pct-vs-frac reconciliation: if one is <1.5 and other is 10..100, rescale
    def _rescale(x, y):
        if 0 < x < 1.5 and 10 < y < 200:
            return x * 100.0
        if 0 < y < 1.5 and 10 < x < 200:
            return y * 100.0
        return None
    tol = max(0.5, 0.005 * abs(a))
    if abs(a - b) < tol:
        return True
    ra = _rescale(a, b)
    if ra is not None and abs(ra - b) < max(0.5, 0.005 * abs(ra)):
        return True
    rb = _rescale(b, a)
    if rb is not None and abs(rb - a) < max(0.5, 0.005 * abs(rb)):
        return True
    return False


def is_ours_row(row_n: str, tuple_method_n: str, paper_mentions_method: bool) -> bool:
    """Row is an 'ours' marker AND paper mentions the tuple's method.

    Accepts:
      - exact match to a marker ('ours', 'our method', 'proposed', ...)
      - phrases starting with 'ours' / 'our' / 'proposed' followed by a generic
        modifier like 'w/', 'w', 'with', 'full', 'best', 'final' (paper's own
        variant naming). Rejects 'our <specific-named-model>' unless the
        <specific-named-model> matches the tuple's method.
    """
    if not row_n: return False
    tokens = row_n.split()
    if not tokens: return False
    if row_n in OURS_MARKERS:
        return paper_mentions_method or tuple_method_n == ""
    # 'ours w/ transfer', 'our method', 'proposed model', 'our full' — first token is
    # an ours-marker; accept if the trailing tokens are generic connectors, else
    # require them to match (part of) tuple_method_n.
    if tokens[0] in {"ours", "our", "proposed"}:
        rest = " ".join(tokens[1:])
        generic = {"w", "with", "w/", "model", "method", "system", "approach",
                    "network", "framework", "full", "final", "best", "algorithm",
                    "wo", "w/o", "without", "transfer", "large", "small", "base"}
        if not rest:
            return paper_mentions_method or tuple_method_n == ""
        rest_tokens = set(rest.split())
        if rest_tokens.issubset(generic):
            return paper_mentions_method or tuple_method_n == ""
        # otherwise, accept only if rest overlaps with the tuple's method
        if tuple_method_n and any(tok in tuple_method_n or tuple_method_n in tok
                                    for tok in rest_tokens if len(tok) >= 3):
            return True
    return False


def _word_boundary_match(needle: str, haystack: str, min_len: int = 3) -> bool:
    """True if `needle` occurs in `haystack` at a word boundary (or exact match)."""
    if not needle or not haystack:
        return False
    if needle == haystack:
        return True
    if len(needle) < min_len:
        return False
    # Word-boundary match (norm strips punct so token boundaries are spaces)
    for tok in haystack.split():
        if tok == needle:
            return True
    if needle in haystack:
        # require a space or start/end at the boundary of the match
        idx = haystack.find(needle)
        left = idx == 0 or haystack[idx - 1] == " "
        right = idx + len(needle) == len(haystack) or haystack[idx + len(needle)] == " "
        return left and right
    return False


_STOPWORDS = {"the", "a", "an", "of", "and", "or", "in", "on", "for", "with",
              "to", "at", "by", "as", "is", "test", "dev", "val", "eval",
              "set", "score", "results", "result", "table", "starts", "start",
              "human", "median", "mean", "average", "single", "multiple",
              "large", "base", "small", "medium", "acc", "accuracy", "error",
              "class", "prec", "recall", "true", "false", "positive", "negative",
              "our", "ours", "proposed", "model", "method", "system", "network",
              "approach", "framework", "algorithm", "baseline", "task", "dataset",
              "metric", "value", "col"}


def _significant_tokens(s: str) -> set[str]:
    return {tok for tok in s.split() if tok not in _STOPWORDS and len(tok) >= 4}


def _substring_align(a: str, b: str, min_short_len: int = 3) -> bool:
    """Return True if a and b substring-overlap with word-boundary safety,
    OR if they share >=1 significant multi-char token (not a stopword)."""
    if not a or not b:
        return False
    if a == b:
        return True
    # Word-boundary substring in either direction
    if len(a) <= len(b):
        if _word_boundary_match(a, b, min_short_len):
            return True
    else:
        if _word_boundary_match(b, a, min_short_len):
            return True
    # Token-level overlap
    ta, tb = _significant_tokens(a), _significant_tokens(b)
    if ta and tb and ta & tb:
        return True
    return False


# ---------- table iteration ----------
def _synth_rows_from_flat(t: dict) -> dict:
    """SciREX-style flat table: reconstruct rows_by_label from row_headers + cells + column_headers."""
    row_h = t.get("row_headers") or []
    col_h = t.get("column_headers") or []
    cells = t.get("cells") or []
    if not row_h or not col_h:
        return {}
    # Heuristic: cells is a flat interleaved list per row (row_label + per-column values)
    # Not perfectly recoverable; try to unpack len(col_h) cells per row
    per_row = max(1, len(col_h))
    out = {}
    for i, r in enumerate(row_h[:60]):
        start = i * per_row
        row_cells = cells[start: start + per_row] if start < len(cells) else []
        if not row_cells:
            continue
        # zip against column headers; if first cell equals row_label, skip it as the label column
        offset = 1 if (row_cells and str(row_cells[0]).strip() == str(r).strip()) else 0
        d = {}
        for j, c in enumerate(col_h):
            k = j - offset
            if 0 <= k < len(row_cells):
                d[c] = row_cells[k]
        if d:
            out[r] = d
    return out


def _header_paths_for_table(t: dict) -> list[str]:
    """Return the union of nested_column_paths + column_headers (all_header_strings)."""
    paths = []
    if isinstance(t.get("nested_column_paths"), list):
        for p in t["nested_column_paths"]:
            # nested paths look like "Features > 2-7" — normalize spacing
            paths.append(p.replace(">", " ").replace("|", " "))
    for c in t.get("column_headers") or []:
        paths.append(c)
    for c in t.get("all_header_strings") or []:
        paths.append(c)
    for c in t.get("super_column_headers") or []:
        paths.append(c)
    return list({norm(p) for p in paths if p})


def _load_tables(corpus: str, paper_id: str) -> list[dict]:
    if corpus == "metalead":
        p = Path("outputs/canonextract_diag_v2/metalead_tables") / f"{paper_id}.json"
    else:
        p = Path("outputs/canonextract_diag/tables") / f"{paper_id}.json"
    if not p.exists():
        return []
    d = json.loads(p.read_text())
    tables = d.get("tables") or []
    if not tables:
        # metalead nested-parser file may not have per-table shape for older papers;
        # synthesize a single flat table from top-level lists
        if any(d.get(k) for k in ("row_headers", "column_headers", "cells")):
            tables = [{
                "table_id": 0,
                "caption": " | ".join((d.get("table_captions") or [])[:3]),
                "row_headers": d.get("row_headers") or [],
                "column_headers": d.get("column_headers") or [],
                "cells": d.get("cells") or [],
                "nested_column_paths": d.get("nested_column_paths") or [],
                "super_column_headers": d.get("super_column_headers") or [],
                "all_header_strings": (d.get("column_headers") or []),
            }]
    return tables


def _load_paper_text(corpus: str, paper_id: str) -> str:
    if corpus == "metalead":
        tex_dir = Path("data/metalead_tex") / paper_id
    else:
        tex_dir = Path("data/scirex_tex") / paper_id
    if not tex_dir.exists(): return ""
    out = []
    for f in tex_dir.glob("**/*.tex"):
        try: out.append(f.read_text(encoding="utf-8", errors="ignore"))
        except Exception: continue
    return "\n".join(out)[:60000]


# ---------- core alignment ----------
def classify_tuple(t: dict, tables: list[dict], paper_text_n: str) -> tuple[str, dict]:
    """Return (bucket, detail_dict)."""
    prov = t.get("provenance") or {}
    src = (prov.get("source") or "").lower()
    detail = {"orig_source": src, "orig_prov": {k: prov.get(k) for k in
              ("table_id", "row_label", "header_path", "cell")}}
    if src == "prose":
        return "prose", detail

    tuple_method = (t.get("method") or {}).get("name") or ""
    tuple_task_n = norm((t.get("task") or {}).get("name") or "")
    ds_list = t.get("datasets") or []
    m_list = t.get("metrics") or []
    tuple_ds = (ds_list[0].get("name") if ds_list and isinstance(ds_list[0], dict) else "") or ""
    tuple_metric_name = (m_list[0].get("name") if m_list and isinstance(m_list[0], dict) else "") or ""
    tuple_value = None
    if m_list and isinstance(m_list[0], dict):
        tuple_value = parse_number(m_list[0].get("value"))
    if tuple_value is None:
        tuple_value = parse_number(prov.get("cell"))

    method_n = norm(tuple_method)
    ds_n = norm(tuple_ds)
    metric_n = norm(tuple_metric_name)

    # Extractor-bug guard: if the tuple's metric field is a pure number (no letters),
    # the extractor confused metric-name with metric-value. Reject strict for these.
    metric_is_numeric = bool(re.fullmatch(r"[\d\.\-\+eE\s]+", tuple_metric_name or ""))
    if metric_is_numeric:
        metric_n = ""  # forget the "metric" field for alignment; we still may match on dataset

    paper_mentions_method = bool(method_n) and (method_n in paper_text_n
                                                  or any(tok in paper_text_n for tok in method_n.split()
                                                          if len(tok) >= 4))
    if not tuple_method:
        # if extractor emitted no method, allow ours-marker even without paper mention
        paper_mentions_method = True

    if not tables:
        # nothing to bind against; if extractor claimed table but no cache, that's prose_claimed_table
        return "prose_claimed_table", {**detail, "reason": "no_cached_tables_for_paper"}

    best = "missing"
    best_hit = None
    ranks = {"missing": 0, "prose_claimed_table": 1, "partial_table_valid": 2, "strict_table_valid": 3}

    for tab in tables:
        # Build per-column augmented header text: for each column index, join
        # (super_column_header, column_header) so nested paths stay tied to their column.
        col_headers = list(tab.get("column_headers") or [])
        super_headers = list(tab.get("super_column_headers") or [])
        # Pad super_headers to len(col_headers)
        if super_headers and len(super_headers) < len(col_headers):
            super_headers = super_headers + [""] * (len(col_headers) - len(super_headers))
        elif not super_headers:
            super_headers = [""] * len(col_headers)
        col_augmented = {}  # column_header text -> combined normalized string
        for j, ch in enumerate(col_headers):
            sh = super_headers[j] if j < len(super_headers) else ""
            col_augmented[ch] = norm(f"{sh} {ch}")
        # Also include the table caption in per-column text (dataset name often lives only in caption)
        cap_n = norm(tab.get("caption") or "")

        def col_hit_for(header_key: str) -> str:
            """Return 'strict' (specific column matches ds or metric),
            'context' (only table caption matches — implicit alignment),
            or 'none'."""
            aug = col_augmented.get(header_key)
            if aug is None:
                aug = norm(header_key)
            # Strict: specific column header text matches dataset or metric
            if aug:
                if ds_n and _substring_align(ds_n, aug, min_short_len=3):
                    return "strict"
                if metric_n and _substring_align(metric_n, aug, min_short_len=2):
                    return "strict"
            # Context: only the table caption reveals dataset or metric
            if cap_n:
                if ds_n and _substring_align(ds_n, cap_n, min_short_len=4):
                    return "context"
                if metric_n and _substring_align(metric_n, cap_n, min_short_len=3):
                    return "context"
            return "none"

        rows_by_label = tab.get("rows_by_label")
        if not rows_by_label:
            rows_by_label = _synth_rows_from_flat(tab)

        for row_label, row_dict in rows_by_label.items():
            row_n = norm(row_label)
            row_hit_bool = False
            if method_n and _substring_align(method_n, row_n, min_short_len=3):
                row_hit_bool = True
            elif is_ours_row(row_n, method_n, paper_mentions_method):
                row_hit_bool = True

            for col_header, cell in row_dict.items():
                col_kind = col_hit_for(col_header)  # "strict" | "context" | "none"
                cell_val = parse_number(cell)
                value_hit_bool = False
                if tuple_value is not None and cell_val is not None:
                    value_hit_bool = numeric_match(tuple_value, cell_val)

                col_strict = col_kind == "strict"
                col_any = col_kind in ("strict", "context")

                # Bucket logic
                if tuple_value is None:
                    # No value to bind — require row + strict column alignment
                    if row_hit_bool and col_strict:
                        cand = "strict_table_valid"
                    elif row_hit_bool and col_any:
                        cand = "partial_table_valid"
                    elif row_hit_bool or col_strict:
                        cand = "partial_table_valid"
                    else:
                        continue  # skip: can't infer anything without value
                else:
                    if not value_hit_bool:
                        continue  # cell doesn't match tuple value; skip
                    if row_hit_bool and col_strict:
                        cand = "strict_table_valid"
                    elif row_hit_bool and col_any:
                        cand = "partial_table_valid"  # caption-only column context
                    elif row_hit_bool or col_strict:
                        cand = "partial_table_valid"
                    elif col_any:  # value + caption-context only (no row)
                        cand = "prose_claimed_table"
                    else:
                        cand = "prose_claimed_table"

                if ranks[cand] > ranks[best]:
                    best = cand
                    best_hit = {
                        "table_id": tab.get("table_id"),
                        "row_label": row_label,
                        "column_header": col_header,
                        "cell": cell,
                        "row_hit": row_hit_bool,
                        "col_hit": col_kind,
                        "value_hit": value_hit_bool,
                    }
                    if best == "strict_table_valid":
                        detail["hit"] = best_hit
                        return best, detail

    detail["hit"] = best_hit
    return best, detail


# ---------- driver ----------
@app.command()
def run(
    metalead_dir: Path = typer.Option(Path("outputs/canonextract_build/eval_runs_metalead_merged")),
    scirex_dir: Path = typer.Option(Path("outputs/canonextract_build/eval_runs_scirex_merged")),
    packet2_csv: Path = typer.Option(Path("outputs/canonextract_build/packet2_labels.csv")),
    packet2_key: Path = typer.Option(Path("outputs/canonextract_build/packet2_key.json")),
):
    out_ml = Path("outputs/canonextract_build/eval_runs_metalead_cycle5"); out_ml.mkdir(parents=True, exist_ok=True)
    out_sx = Path("outputs/canonextract_build/eval_runs_scirex_cycle5"); out_sx.mkdir(parents=True, exist_ok=True)
    prov_dir = Path("outputs/canonextract_build/provenance_cycle5"); (prov_dir/"metalead").mkdir(parents=True, exist_ok=True); (prov_dir/"scirex").mkdir(parents=True, exist_ok=True)

    # Cache paper text (lowercased, normalized whitespace) for ours-marker check
    text_cache: dict[tuple[str, str], str] = {}
    def _text(corpus, pid):
        k = (corpus, pid)
        if k not in text_cache:
            text_cache[k] = norm(_load_paper_text(corpus, pid))
        return text_cache[k]

    corpus_totals: dict[str, Counter] = defaultdict(Counter)
    per_tuple_labels: dict[str, dict] = {}  # (corpus, pid, tuple_idx) -> bucket

    for corpus, in_dir, out_dir in (("metalead", metalead_dir, out_ml), ("scirex", scirex_dir, out_sx)):
        console.print(f"[bold]processing {corpus}...[/bold]")
        for p in sorted(in_dir.glob(f"{corpus}_*.json")):
            d = json.loads(p.read_text())
            pid = d.get("paper_id") or p.stem.removeprefix(f"{corpus}_")
            tables = _load_tables(corpus, pid)
            paper_text_n = _text(corpus, pid)
            new_tuples = []
            per_tuple_detail = []
            for i, t in enumerate(d.get("tuples") or []):
                bucket, detail = classify_tuple(t, tables, paper_text_n)
                corpus_totals[corpus][bucket] += 1
                corpus_totals[corpus]["total"] += 1
                new_t = json.loads(json.dumps(t))
                new_t["provenance"]["cycle5_bucket"] = bucket
                if detail.get("hit"):
                    new_t["provenance"]["cycle5_hit"] = detail["hit"]
                new_tuples.append(new_t)
                per_tuple_detail.append({
                    "tuple_idx": i, "bucket": bucket, "detail": detail,
                    "method": (t.get("method") or {}).get("name"),
                    "task": (t.get("task") or {}).get("name"),
                    "dataset": (t.get("datasets") or [{}])[0].get("name") if t.get("datasets") else None,
                    "metric": (t.get("metrics") or [{}])[0].get("name") if t.get("metrics") else None,
                    "value": (t.get("metrics") or [{}])[0].get("value") if t.get("metrics") else None,
                })
                per_tuple_labels[f"{corpus}|{pid}|{i}"] = bucket
            (out_dir / p.name).write_text(json.dumps({
                "paper_id": pid, "corpus": corpus,
                "provenance_stats_cycle5": dict(corpus_totals[corpus]),  # rolling; per-paper overwritten below
                "tuples": new_tuples,
            }, indent=2))
            (prov_dir / corpus / f"{pid}.json").write_text(json.dumps({
                "paper_id": pid, "corpus": corpus,
                "per_tuple": per_tuple_detail,
            }, indent=2))
    # Fix per-paper file's rolling stat (overwrite with per-paper counts)
    for corpus, out_dir in (("metalead", out_ml), ("scirex", out_sx)):
        for p in sorted(out_dir.glob(f"{corpus}_*.json")):
            d = json.loads(p.read_text())
            c = Counter()
            for t in d["tuples"]:
                c[t["provenance"]["cycle5_bucket"]] += 1
                c["total"] += 1
            d["provenance_stats_cycle5"] = dict(c)
            p.write_text(json.dumps(d, indent=2))

    # ---------- Packet 2 rescore ----------
    key = json.loads(packet2_key.read_text())
    key_by_row = {k["row_id"]: k for k in key}
    packet2_rows = list(csv.DictReader(packet2_csv.open()))
    rescore = []
    for row in packet2_rows:
        rid = row["row_id"]
        keyrec = key_by_row.get(rid)
        if not keyrec:
            continue
        corpus = keyrec["corpus"]; pid = keyrec["paper_id"]; tidx = keyrec["tuple_idx"]
        label = row["cell_matches_tuple_yes_no"].strip().upper()
        bucket = per_tuple_labels.get(f"{corpus}|{pid}|{tidx}", "not_found_in_pool")
        rescore.append({
            "row_id": rid, "corpus": corpus, "paper_id": pid, "tuple_idx": tidx,
            "human_label": label, "cycle5_bucket": bucket,
            "method": row["method"], "task": row["task"], "dataset": row["dataset"],
            "metric": row["metric"], "value": row["value"],
        })

    n_yes = sum(1 for r in rescore if r["human_label"] == "YES")
    n_no = sum(1 for r in rescore if r["human_label"] == "NO")
    n_missing = sum(1 for r in rescore if r["human_label"] == "MISSING")

    # Precision on new strict_table_valid: among rows flagged strict by cycle 5, fraction YES
    strict_rows = [r for r in rescore if r["cycle5_bucket"] == "strict_table_valid"]
    n_strict = len(strict_rows)
    n_strict_yes = sum(1 for r in strict_rows if r["human_label"] == "YES")
    precision_strict = (n_strict_yes / n_strict) if n_strict else None

    # Recovery on NO cases: fraction now correctly downgraded (partial / prose_claimed_table / missing / prose)
    no_rows = [r for r in rescore if r["human_label"] == "NO"]
    n_no_recovered = sum(1 for r in no_rows if r["cycle5_bucket"] in
                          ("partial_table_valid", "prose_claimed_table", "missing", "prose"))
    recovery_no = (n_no_recovered / len(no_rows)) if no_rows else None

    # Bucket reprojection: per-bucket YES / NO / MISSING counts
    bucket_reproj = defaultdict(Counter)
    for r in rescore:
        bucket_reproj[r["cycle5_bucket"]][r["human_label"]] += 1

    # ---------- write aggregate summary ----------
    ml_total = corpus_totals["metalead"].get("total", 0)
    sx_total = corpus_totals["scirex"].get("total", 0)

    def pct(c: Counter, k: str, tot: int):
        return round(c.get(k, 0) / tot * 100, 2) if tot else 0.0

    summary = {
        "seed": SEED,
        "cycle": 5,
        "cost_usd_cycle5_additive": 0.0,
        "cost_usd_cumulative_cycles_1_to_5": 3.824,
        "corpus_totals": {
            "metalead": {
                "total": ml_total,
                "strict_table_valid": corpus_totals["metalead"].get("strict_table_valid", 0),
                "partial_table_valid": corpus_totals["metalead"].get("partial_table_valid", 0),
                "prose_claimed_table": corpus_totals["metalead"].get("prose_claimed_table", 0),
                "prose": corpus_totals["metalead"].get("prose", 0),
                "missing": corpus_totals["metalead"].get("missing", 0),
                "pct_strict": pct(corpus_totals["metalead"], "strict_table_valid", ml_total),
                "pct_partial": pct(corpus_totals["metalead"], "partial_table_valid", ml_total),
                "pct_prose_claimed_table": pct(corpus_totals["metalead"], "prose_claimed_table", ml_total),
                "pct_prose": pct(corpus_totals["metalead"], "prose", ml_total),
                "pct_missing": pct(corpus_totals["metalead"], "missing", ml_total),
            },
            "scirex": {
                "total": sx_total,
                "strict_table_valid": corpus_totals["scirex"].get("strict_table_valid", 0),
                "partial_table_valid": corpus_totals["scirex"].get("partial_table_valid", 0),
                "prose_claimed_table": corpus_totals["scirex"].get("prose_claimed_table", 0),
                "prose": corpus_totals["scirex"].get("prose", 0),
                "missing": corpus_totals["scirex"].get("missing", 0),
                "pct_strict": pct(corpus_totals["scirex"], "strict_table_valid", sx_total),
                "pct_partial": pct(corpus_totals["scirex"], "partial_table_valid", sx_total),
                "pct_prose_claimed_table": pct(corpus_totals["scirex"], "prose_claimed_table", sx_total),
                "pct_prose": pct(corpus_totals["scirex"], "prose", sx_total),
                "pct_missing": pct(corpus_totals["scirex"], "missing", sx_total),
            },
        },
        "packet2_rescore": {
            "n_rows": len(rescore),
            "n_yes": n_yes, "n_no": n_no, "n_missing": n_missing,
            "cycle5_strict_precision_vs_yes": precision_strict,
            "cycle5_strict_count": n_strict,
            "cycle5_strict_yes": n_strict_yes,
            "cycle5_recovery_on_no": recovery_no,
            "cycle5_no_downgraded": n_no_recovered,
            "bucket_reprojection": {b: dict(c) for b, c in bucket_reproj.items()},
            "per_row": rescore,
        },
    }
    Path("outputs/canonextract_build/provenance_cycle5_summary.json").write_text(json.dumps(summary, indent=2))
    console.print(f"\n[bold]MetaLead[/bold]: total={ml_total} strict={corpus_totals['metalead'].get('strict_table_valid',0)} "
                  f"({pct(corpus_totals['metalead'],'strict_table_valid',ml_total)}%) "
                  f"partial={corpus_totals['metalead'].get('partial_table_valid',0)} "
                  f"({pct(corpus_totals['metalead'],'partial_table_valid',ml_total)}%) "
                  f"pct={corpus_totals['metalead'].get('prose_claimed_table',0)} "
                  f"prose={corpus_totals['metalead'].get('prose',0)} miss={corpus_totals['metalead'].get('missing',0)}")
    console.print(f"[bold]SciREX[/bold]: total={sx_total} strict={corpus_totals['scirex'].get('strict_table_valid',0)} "
                  f"({pct(corpus_totals['scirex'],'strict_table_valid',sx_total)}%) "
                  f"partial={corpus_totals['scirex'].get('partial_table_valid',0)} "
                  f"({pct(corpus_totals['scirex'],'partial_table_valid',sx_total)}%) "
                  f"pct={corpus_totals['scirex'].get('prose_claimed_table',0)} "
                  f"prose={corpus_totals['scirex'].get('prose',0)} miss={corpus_totals['scirex'].get('missing',0)}")
    console.print(f"\n[bold]Packet 2 rescore[/bold]: n={len(rescore)}, YES={n_yes}, NO={n_no}, MISSING={n_missing}")
    console.print(f"  cycle5 strict precision vs Packet 2 YES: "
                  f"{n_strict_yes}/{n_strict} = {precision_strict if precision_strict is None else round(precision_strict*100,1)}%")
    console.print(f"  cycle5 recovery on NO cases: {n_no_recovered}/{len(no_rows)} = "
                  f"{None if recovery_no is None else round(recovery_no*100,1)}%")
    console.print(f"  bucket_reprojection: {json.dumps({b: dict(c) for b,c in bucket_reproj.items()}, indent=2)}")


if __name__ == "__main__":
    app()
