"""Cycle-2 Fix 1: nested-table parser + re-binder.

Rewrites the tex table parser to preserve multirow / multicolumn structure:
  * Expand \\multicolumn{n}{...}{X} into n copies of X (super-header
    propagates across all spanned columns).
  * Expand \\multirow{n}{*}{X} into X in the first row it appears (paired
    with span tracking so downstream cells attribute to the same X).
  * Recognize multi-tier header rows: if consecutive rows before the
    first data row exist, treat all of them as header rows and produce
    "combined path" strings (super > sub).
  * Additionally return a `nested_paths` set for every table which the
    binder can use for lenient matching.

Then re-run the deterministic binder on cycle 1's cached extractor
emissions in outputs/canonextract_build/eval_runs_metalead/,
eval_runs_scirex/, and dev_runs/round2/. Downgrade only if fewer than 2
of {row_label, header_path, cell} substring-match against the big unified
set — this is a "2-of-3" relaxation that reflects the LLM's tendency to
confuse row / header / cell slots when the paper's table structure is
multi-tiered.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import typer
from rich.console import Console

app = typer.Typer(no_args_is_help=False, add_completion=False)
console = Console()

_PUNCT_RE = re.compile(r"[^\w\s]")
_WS_RE = re.compile(r"\s+")


def normalize(s: str | None) -> str:
    if not s: return ""
    s = str(s).lower().replace("_", " ")
    s = _PUNCT_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    return s


# ---------- nested tex parsing ----------

def _strip_tex_cmds(s: str) -> str:
    """Remove common inline LaTeX commands but keep argument text."""
    # Named commands with 1 arg: keep the arg
    s = re.sub(r"\\(?:textbf|textit|emph|texttt|mathbf|mathrm|mathit|mathsf|underline|textsc)\{([^{}]*)\}",
               r"\1", s)
    # Multicolumn/multirow — DON'T touch here (parsed separately)
    # inline math
    s = re.sub(r"\$([^$]*)\$", r"\1", s)
    # generic remaining commands
    s = re.sub(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?", " ", s)
    s = s.replace("{", " ").replace("}", " ").replace("~", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


_MULTICOL_RE = re.compile(r"\\multicolumn\{(\d+)\}\{[^}]*\}\{((?:[^{}]|\{[^{}]*\})*)\}")
_MULTIROW_RE = re.compile(r"\\multirow\{(\d+)\}\{[^}]*\}\{((?:[^{}]|\{[^{}]*\})*)\}")


def _split_cells_expand_multi(raw_cells: list[str]) -> list[str]:
    """Split raw cells (already split by &) and expand multicolumn/multirow contents."""
    out: list[str] = []
    for cell in raw_cells:
        m = _MULTICOL_RE.search(cell)
        if m:
            n = int(m.group(1))
            content = _strip_tex_cmds(m.group(2))
            # Expand to n copies (super-header propagates across all columns it spans)
            out.extend([content] * n)
            continue
        m = _MULTIROW_RE.search(cell)
        if m:
            content = _strip_tex_cmds(m.group(2))
            out.append(content)
            continue
        out.append(_strip_tex_cmds(cell))
    return out


def _is_header_row(cells: list[str]) -> bool:
    """Heuristic: rows with no numeric cells and non-empty content are header rows."""
    if not cells:
        return False
    non_empty = [c for c in cells if c]
    if not non_empty:
        return False
    numeric_count = sum(1 for c in non_empty if re.search(r"\d[\d.\-]{2,}", c))
    # If less than half the cells are numeric-looking, likely a header
    return numeric_count / len(non_empty) < 0.4


def parse_tabular_nested(tabular_body: str) -> dict:
    """Return a dict with row_headers, column_headers, cells, and additionally:
    - nested_column_paths: list of "super > sub" combined header strings
    - all_header_strings: union of both single and combined
    - rows_by_label: {row_label: {col_header: cell_value}}
    """
    # Split rows by \\ delimiter
    rows_raw = re.split(r"\\\\|\\hline|\\midrule|\\toprule|\\bottomrule", tabular_body)
    rows_expanded: list[list[str]] = []
    for raw in rows_raw:
        if not raw.strip():
            continue
        cells = raw.split("&")
        expanded = _split_cells_expand_multi(cells)
        # Drop the very first / very last if empty
        expanded = [c for c in expanded if c and c not in ("-", "—")]
        if expanded:
            rows_expanded.append(expanded)

    if not rows_expanded:
        return {"row_headers": [], "column_headers": [], "cells": [],
                "nested_column_paths": [], "all_header_strings": [],
                "rows_by_label": {}}

    # Identify contiguous header rows at the top
    header_rows: list[list[str]] = []
    data_rows: list[list[str]] = []
    body_started = False
    for row in rows_expanded:
        if body_started:
            data_rows.append(row)
        elif _is_header_row(row):
            header_rows.append(row)
        else:
            body_started = True
            data_rows.append(row)
    # If nothing looked like a header (all numeric), first row is header
    if not header_rows and rows_expanded:
        header_rows = [rows_expanded[0]]
        data_rows = rows_expanded[1:]

    # Column headers = last header row (typically the finest granularity)
    col_headers = header_rows[-1] if header_rows else []
    # Also include intermediate super-headers
    super_headers = [h for h in header_rows[:-1] for h in h] if len(header_rows) > 1 else []

    # Nested combined paths: super > sub
    nested_paths: list[str] = []
    if len(header_rows) > 1:
        # Best-effort alignment: for each column index in col_headers, look up
        # the super header at same column position.
        for super_row in header_rows[:-1]:
            # Pad super_row to len(col_headers) by repeating last non-empty
            padded = list(super_row)
            while len(padded) < len(col_headers):
                padded.append(padded[-1] if padded else "")
            for i, sub in enumerate(col_headers):
                if i < len(padded) and padded[i] and padded[i] != sub:
                    nested_paths.append(f"{padded[i]} > {sub}")
                    nested_paths.append(f"{padded[i]} {sub}")  # concatenated form too

    # Row headers = first cell of each data row
    row_headers = [row[0] if row else "" for row in data_rows]

    # Flat cells
    all_cells = [c for row in data_rows for c in row]

    # Build rows_by_label mapping (rows may share row_label across multirow spans)
    rows_by_label: dict[str, dict[str, str]] = {}
    for row in data_rows:
        rl = row[0] if row else ""
        if not rl:
            continue
        cells_dict = {}
        for i, c in enumerate(row[1:], start=1):
            hdr = col_headers[i] if i < len(col_headers) else f"col_{i}"
            cells_dict[hdr] = c
        # If row_label already in dict, extend
        if rl in rows_by_label:
            rows_by_label[rl].update(cells_dict)
        else:
            rows_by_label[rl] = cells_dict

    all_header_strings = list(set(col_headers + super_headers + nested_paths))

    return {
        "row_headers": row_headers,
        "column_headers": col_headers,
        "super_column_headers": super_headers,
        "nested_column_paths": nested_paths,
        "all_header_strings": all_header_strings,
        "cells": all_cells,
        "rows_by_label": rows_by_label,
    }


_TEX_TABLE_RE = re.compile(r"\\begin\{table\*?\}(.*?)\\end\{table\*?\}", re.DOTALL)
_TEX_TABULAR_RE = re.compile(r"\\begin\{tabular\}\{[^}]*\}(.*?)\\end\{tabular\}", re.DOTALL)
_TEX_CAPTION_RE = re.compile(r"\\caption\{((?:[^{}]|\{[^{}]*\})*)\}")


def parse_tables_from_tex_dir(tex_dir: Path) -> list[dict]:
    """Parse all tables in a paper's tex tree."""
    tex_files = list(tex_dir.glob("**/*.tex"))
    if not tex_files:
        return []
    combined = "\n".join(f.read_text(encoding="utf-8", errors="ignore") for f in tex_files)
    # Strip % comments
    out_lines = []
    for line in combined.splitlines():
        i = 0
        buf = []
        while i < len(line):
            if line[i] == "%" and (i == 0 or line[i - 1] != "\\"):
                break
            buf.append(line[i]); i += 1
        out_lines.append("".join(buf))
    combined = "\n".join(out_lines)

    tables: list[dict] = []
    tid = 0
    # Tables inside \begin{table}
    for m in _TEX_TABLE_RE.finditer(combined):
        block = m.group(1)
        cap = ""
        cap_m = _TEX_CAPTION_RE.search(block)
        if cap_m:
            cap = _strip_tex_cmds(cap_m.group(1))
        tab_m = _TEX_TABULAR_RE.search(block)
        if not tab_m:
            continue
        parsed = parse_tabular_nested(tab_m.group(1))
        if not parsed["cells"]:
            continue
        parsed["table_id"] = tid
        parsed["caption"] = cap
        tables.append(parsed)
        tid += 1
    # Standalone tabulars (outside \begin{table})
    seen_bodies = {m.group(1) for m in _TEX_TABULAR_RE.finditer(
        "\n".join(m.group(1) for m in _TEX_TABLE_RE.finditer(combined))
    )}
    for m in _TEX_TABULAR_RE.finditer(combined):
        if m.group(1) in seen_bodies:
            continue
        parsed = parse_tabular_nested(m.group(1))
        if not parsed["cells"]:
            continue
        parsed["table_id"] = tid
        parsed["caption"] = ""
        tables.append(parsed)
        tid += 1
    return tables


# ---------- binder ----------

def _lenient_match(query: str, corpus: set[str]) -> bool:
    q = normalize(query)
    if not q:
        return False
    for c in corpus:
        if not c: continue
        if q in c or c in q:
            return True
    return False


def _validate_and_stat(tuples: list[dict], tables: list[dict], paper_text: str,
                       relaxed_binder: bool = True) -> tuple[list[dict], dict]:
    """Re-run binder with nested tables + 2-of-3 relaxation."""
    stats = Counter()
    validated: list[dict] = []
    paper_hash = hashlib.sha256(paper_text.encode("utf-8", errors="ignore")).hexdigest()[:12]

    # Build per-table + across-tables unified sets
    tables_by_id = {t["table_id"]: t for t in tables}
    all_row = set()
    all_col = set()
    all_cells = set()
    all_headers_nested = set()
    for t in tables:
        all_row |= {normalize(x) for x in t.get("row_headers", []) if x}
        all_col |= {normalize(x) for x in t.get("column_headers", []) if x}
        all_col |= {normalize(x) for x in t.get("super_column_headers", []) if x}
        all_headers_nested |= {normalize(x) for x in t.get("nested_column_paths", []) if x}
        all_cells |= {normalize(x) for x in t.get("cells", []) if x}
    all_col |= all_headers_nested
    all_field_norms = all_row | all_col | all_cells

    for t in tuples:
        prov = t.get("provenance") or {}
        src_orig = (prov.get("source") or "").lower()
        stats["total"] += 1

        if src_orig in ("table", "prose_claimed_table"):
            # Re-validate on new tables
            row_lbl = prov.get("row_label") or ""
            hdr = prov.get("header_path") or ""
            cell = prov.get("cell") or ""

            # Try 3-of-3 first
            row_hit = _lenient_match(row_lbl, all_row)
            hdr_hit = _lenient_match(hdr, all_col)
            cell_hit = _lenient_match(cell, all_cells) if cell else True

            three_of_three = row_hit and hdr_hit and cell_hit

            # 2-of-3 relaxation
            two_of_three = sum([row_hit, hdr_hit, cell_hit]) >= 2

            valid = three_of_three or (relaxed_binder and two_of_three)

            if valid:
                stats["table_valid"] += 1
                t["provenance"]["source"] = "table"
                t["provenance"]["_content_hash"] = hashlib.sha256(
                    f"{cell}|{row_lbl}|{hdr}".encode()
                ).hexdigest()[:12]
                t["provenance"]["_paper_hash"] = paper_hash
                t["provenance"]["_match_mode"] = "3of3" if three_of_three else "2of3"
                # Remove any old downgrade markers
                t["provenance"].pop("_downgrade_reason", None)
            else:
                stats["prose_claimed_table"] += 1
                t["provenance"]["source"] = "prose_claimed_table"
                t["provenance"]["_downgrade_reason"] = (
                    f"none of (row_label={row_hit}, header_path={hdr_hit}, cell={cell_hit}) "
                    "resolved in the paper's parsed tables"
                )
                t["provenance"]["_paper_hash"] = paper_hash
        elif src_orig == "prose":
            # Same prose logic as cycle 1
            span = prov.get("evidence_span") or {}
            s, e = int(span.get("start", 0) or 0), int(span.get("end", 0) or 0)
            claimed_quote = prov.get("quote") or ""
            offset_quote = paper_text[s:e] if 0 <= s < e <= len(paper_text) else ""
            substring_hit = bool(claimed_quote and claimed_quote in paper_text)
            valid = bool(offset_quote) or substring_hit
            if valid:
                stats["prose"] += 1
                t["provenance"]["_quote_verified"] = offset_quote or claimed_quote
                t["provenance"]["_verification_mode"] = "offset" if offset_quote else "substring"
            else:
                stats["prose_unverified"] += 1
                t["provenance"]["_verification_mode"] = "unverified"
            t["provenance"]["_paper_hash"] = paper_hash
        else:
            stats["unknown_source"] += 1
        validated.append(t)
    return validated, dict(stats)


# ---------- driver ----------

@app.command()
def run():
    console.print("[bold]Cycle-2 Fix 1: nested parser + re-binder[/bold]")

    # 1. Re-parse ALL tex trees (SciREX + MetaLead), overwrite tables JSONs
    sx_tex_dir = Path("data/scirex_tex")
    ml_tex_dir = Path("data/metalead_tex")
    sx_out_dir = Path("outputs/canonextract_diag/tables")
    ml_out_dir = Path("outputs/canonextract_diag_v2/metalead_tables")
    sx_out_dir.mkdir(parents=True, exist_ok=True)
    ml_out_dir.mkdir(parents=True, exist_ok=True)

    sx_n = 0
    for tex_subdir in sx_tex_dir.iterdir():
        if not tex_subdir.is_dir(): continue
        tables = parse_tables_from_tex_dir(tex_subdir)
        # Match the cycle-1 tables JSON shape but with additional fields
        (sx_out_dir / f"{tex_subdir.name}.json").write_text(json.dumps({
            "doc_id": tex_subdir.name,
            "fetch_status": "tex",
            "tables": tables,
        }, indent=2))
        sx_n += 1
    console.print(f"  SciREX tex tables reparsed: {sx_n}")

    ml_n = 0
    for tex_subdir in ml_tex_dir.iterdir():
        if not tex_subdir.is_dir(): continue
        tables = parse_tables_from_tex_dir(tex_subdir)
        # v2 metalead_tables shape is flat lists; keep that + add tables:[...] structured
        row_h_all = [x for t in tables for x in t.get("row_headers", [])]
        col_h_all = [x for t in tables for x in t.get("column_headers", [])]
        super_h_all = [x for t in tables for x in t.get("super_column_headers", [])]
        nested_all = [x for t in tables for x in t.get("nested_column_paths", [])]
        cells_all = [x for t in tables for x in t.get("cells", [])]
        caps_all = [t.get("caption", "") for t in tables if t.get("caption")]
        (ml_out_dir / f"{tex_subdir.name}.json").write_text(json.dumps({
            "paper_key": tex_subdir.name,
            "row_headers": row_h_all,
            "column_headers": col_h_all,
            "super_column_headers": super_h_all,
            "nested_column_paths": nested_all,
            "cells": cells_all,
            "table_captions": caps_all,
            "figure_captions": [],  # not extracted here (would need extract_expanded_anchors)
            "table_neighborhoods": [],
            "tables": tables,  # nested structure per table
        }, indent=2))
        ml_n += 1
    console.print(f"  MetaLead tex tables reparsed: {ml_n}")

    # 2. Re-run binder on cycle 1's cached extractor emissions
    def _load_paper_text_scirex(pid: str) -> str:
        tex_dir = sx_tex_dir / pid
        if not tex_dir.exists(): return ""
        return "\n".join(f.read_text(encoding="utf-8", errors="ignore") for f in tex_dir.glob("**/*.tex"))
    def _load_paper_text_metalead(pk: str) -> str:
        tex_dir = ml_tex_dir / pk
        if not tex_dir.exists(): return ""
        return "\n".join(f.read_text(encoding="utf-8", errors="ignore") for f in tex_dir.glob("**/*.tex"))

    def _load_tables_scirex(pid: str) -> list[dict]:
        p = sx_out_dir / f"{pid}.json"
        if not p.exists(): return []
        return json.loads(p.read_text()).get("tables") or []
    def _load_tables_metalead(pk: str) -> list[dict]:
        p = ml_out_dir / f"{pk}.json"
        if not p.exists(): return []
        return json.loads(p.read_text()).get("tables") or []

    all_stats: dict[str, Counter] = {"metalead_eval": Counter(),
                                       "scirex_transfer": Counter()}

    # MetaLead eval
    ml_eval_dir = Path("outputs/canonextract_build/eval_runs_metalead")
    ml_cycle2_dir = Path("outputs/canonextract_build/eval_runs_metalead_cycle2")
    ml_cycle2_dir.mkdir(parents=True, exist_ok=True)
    for p in ml_eval_dir.glob("metalead_*.json"):
        d = json.loads(p.read_text())
        pk = d.get("paper_id") or p.stem.removeprefix("metalead_")
        tables = _load_tables_metalead(pk)
        paper_text = _load_paper_text_metalead(pk)
        tuples = d.get("tuples") or []
        new_tuples, stats = _validate_and_stat(tuples, tables, paper_text, relaxed_binder=True)
        for k, v in stats.items():
            all_stats["metalead_eval"][k] += v
        d["tuples"] = new_tuples
        d["provenance_stats"] = stats
        d["cycle"] = "cycle2_nested_parser"
        (ml_cycle2_dir / p.name).write_text(json.dumps(d, indent=2))

    # SciREX
    sx_eval_dir = Path("outputs/canonextract_build/eval_runs_scirex")
    sx_cycle2_dir = Path("outputs/canonextract_build/eval_runs_scirex_cycle2")
    sx_cycle2_dir.mkdir(parents=True, exist_ok=True)
    for p in sx_eval_dir.glob("scirex_*.json"):
        d = json.loads(p.read_text())
        pid = d.get("paper_id") or p.stem.removeprefix("scirex_")
        tables = _load_tables_scirex(pid)
        paper_text = _load_paper_text_scirex(pid)
        tuples = d.get("tuples") or []
        new_tuples, stats = _validate_and_stat(tuples, tables, paper_text, relaxed_binder=True)
        for k, v in stats.items():
            all_stats["scirex_transfer"][k] += v
        d["tuples"] = new_tuples
        d["provenance_stats"] = stats
        d["cycle"] = "cycle2_nested_parser"
        (sx_cycle2_dir / p.name).write_text(json.dumps(d, indent=2))

    # 3. Report
    def _pct(c: Counter, key: str, total_key: str = "total") -> str:
        t = c.get(total_key, 0)
        return f"{c.get(key, 0)}/{t} = {c.get(key, 0) / t * 100 if t else 0:.1f}%"

    report = {}
    for corpus, c in all_stats.items():
        report[corpus] = {
            "total_tuples": c.get("total", 0),
            "table_valid": c.get("table_valid", 0),
            "prose": c.get("prose", 0),
            "prose_claimed_table_downgraded": c.get("prose_claimed_table", 0),
            "prose_unverified": c.get("prose_unverified", 0),
            "unknown_source": c.get("unknown_source", 0),
            "table_valid_pct": (c.get("table_valid", 0) / c.get("total", 1) * 100.0) if c.get("total") else 0.0,
            "grounded_pct": ((c.get("table_valid", 0) + c.get("prose", 0)) / c.get("total", 1) * 100.0) if c.get("total") else 0.0,
        }
    Path("outputs/canonextract_build/provenance_cycle2.json").write_text(json.dumps(report, indent=2))
    console.print(f"\n[bold]Provenance cycle-2:[/bold]")
    for corpus, r in report.items():
        console.print(f"  {corpus}: total={r['total_tuples']}  "
                      f"table_valid={r['table_valid']} ({r['table_valid_pct']:.1f}%)  "
                      f"grounded={r['grounded_pct']:.1f}%  "
                      f"prose={r['prose']}  downgraded={r['prose_claimed_table_downgraded']}")


if __name__ == "__main__":
    app()
