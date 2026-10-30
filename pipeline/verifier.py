"""Cycle-6 verifier: Llama-3.3-70B pass on cycle-5 strict tuples.

For each tuple flagged strict_table_valid by cycle 5, send the specific
table (windowed if large) + claimed cell provenance + tuple fields to
Llama 3.3 70B at t=0 and ask YES / NO / UNCLEAR. Keep YES as strict,
downgrade NO/UNCLEAR to `verifier_downgrade_partial`.

Also rescores Packet 2 against the cycle-6 buckets and emits an
aggregate summary.

Zero touches to partial, prose_claimed, prose, or missing tuples.
"""

from __future__ import annotations

import asyncio
import csv
import json
import re
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import typer
from rich.console import Console
from rich.progress import (
    BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn,
)

from paper1.config import load_config
from paper1.openrouter import OpenRouterClient

SEED = 42
CONCURRENCY = 4
COST_CAP = 2.0
MAX_TABLE_ROWS_IN_PROMPT = 20  # window size when table has more rows
MODEL_ID = "meta-llama/llama-3.3-70b-instruct"

app = typer.Typer(no_args_is_help=False, add_completion=False)
console = Console()


@dataclass
class VerifyResult:
    corpus: str
    paper_id: str
    tuple_idx: int
    verdict: str  # YES / NO / UNCLEAR / ERROR
    reason: str
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    error: str | None = None


def _load_tables(corpus: str, paper_id: str) -> list[dict]:
    if corpus == "metalead":
        p = Path("outputs/canonextract_diag_v2/metalead_tables") / f"{paper_id}.json"
    else:
        p = Path("outputs/canonextract_diag/tables") / f"{paper_id}.json"
    if not p.exists(): return []
    d = json.loads(p.read_text())
    return d.get("tables") or []


def _windowed_markdown_table(tab: dict, target_row_label: str) -> tuple[str, str, dict]:
    """Return (markdown_table, caption, tab_meta). Windows to +/-3 rows around
    the claimed row_label if the table has > MAX_TABLE_ROWS_IN_PROMPT rows."""
    caption = (tab.get("caption") or "").strip()
    col_headers = list(tab.get("column_headers") or [])
    super_headers = list(tab.get("super_column_headers") or [])
    rows_by_label = tab.get("rows_by_label")
    if not rows_by_label:
        # synthesize from flat cells
        row_h = tab.get("row_headers") or []
        cells = tab.get("cells") or []
        per_row = max(1, len(col_headers))
        rows_by_label = {}
        for i, r in enumerate(row_h[:60]):
            row_cells = cells[i * per_row: (i + 1) * per_row]
            if not row_cells: continue
            offset = 1 if row_cells and str(row_cells[0]).strip() == str(r).strip() else 0
            d = {}
            for j, c in enumerate(col_headers):
                k = j - offset
                if 0 <= k < len(row_cells):
                    d[c] = row_cells[k]
            if d:
                rows_by_label[r] = d

    labels = list(rows_by_label.keys())
    if len(labels) > MAX_TABLE_ROWS_IN_PROMPT and target_row_label in labels:
        idx = labels.index(target_row_label)
        lo = max(0, idx - 3)
        hi = min(len(labels), idx + 4)
        windowed = labels[lo:hi]
    else:
        windowed = labels[:MAX_TABLE_ROWS_IN_PROMPT]

    # Build markdown
    all_cols = col_headers or (list(next(iter(rows_by_label.values()), {}).keys()) if rows_by_label else [])
    hdr_row = "| row_label | " + " | ".join(c[:40] for c in all_cols) + " |"
    sep_row = "| --- | " + " | ".join("---" for _ in all_cols) + " |"
    body_rows = []
    for lbl in windowed:
        row = rows_by_label.get(lbl, {})
        vals = [str(row.get(c, ""))[:40] for c in all_cols]
        body_rows.append(f"| {str(lbl)[:60]} | " + " | ".join(vals) + " |")

    md = "\n".join([hdr_row, sep_row] + body_rows)
    if super_headers and any(super_headers):
        md = "super_column_headers: " + ", ".join(str(s)[:30] for s in super_headers) + "\n" + md
    return md, caption, {"total_rows": len(labels), "shown_rows": len(windowed)}


VERIFIER_SYSTEM = (
    "You are a verifier. You are given (a) a compact representation of one "
    "table from a scientific paper, and (b) a claim that a specific cell in "
    "that table is the source of a specific extracted tuple. Judge whether "
    "the cell is really the source."
)


def _build_user_prompt(t: dict, table_md: str, caption: str, table_id, prov: dict) -> str:
    method = (t.get("method") or {}).get("name") or ""
    task = (t.get("task") or {}).get("name") or ""
    ds_list = t.get("datasets") or []
    m_list = t.get("metrics") or []
    ds_name = (ds_list[0].get("name") if ds_list and isinstance(ds_list[0], dict) else "") or ""
    m_name = (m_list[0].get("name") if m_list and isinstance(m_list[0], dict) else "") or ""
    m_val = (m_list[0].get("value") if m_list and isinstance(m_list[0], dict) else "") or ""
    row_lbl = prov.get("row_label") or ""
    header_path = prov.get("header_path") or ""
    cell = prov.get("cell") or ""
    return (
        f'TABLE (id: {table_id}, caption: "{caption[:200]}"):\n'
        f"{table_md}\n\n"
        f"CLAIMED TUPLE:\n"
        f"- Method: {method}\n"
        f"- Task: {task}\n"
        f"- Dataset: {ds_name}\n"
        f"- Metric: {m_name}\n"
        f"- Value: {m_val}\n\n"
        f"CLAIMED PROVENANCE:\n"
        f'- Row label: "{row_lbl}"\n'
        f'- Header path: "{header_path}"\n'
        f'- Cell content: "{cell}"\n\n'
        f"QUESTION: Is the claimed cell really the source of the tuple's "
        f"(Method, Dataset, Metric, Value) combination? Consider:\n"
        f'- Does the row label describe the tuple\'s Method? (Handle "ours", "our method", '
        f'"proposed", and paper-canonical name variations.)\n'
        f"- Does the header path describe the tuple's Dataset AND/OR Metric?\n"
        f"- Does the cell content match the tuple's Value (allow rounding, +/-, unit differences)?\n\n"
        f"Answer with exactly one of: YES, NO, UNCLEAR.\n"
        f"Then one short sentence explaining the reason."
    )


_VERDICT_RE = re.compile(r"\b(YES|NO|UNCLEAR)\b", re.IGNORECASE)


def _parse_verdict(text: str) -> tuple[str, str]:
    m = _VERDICT_RE.search(text or "")
    verdict = m.group(1).upper() if m else "UNCLEAR"
    # reason = rest of first two sentences
    reason = (text or "").strip().split("\n", 1)
    reason = reason[0][:400] if reason else ""
    return verdict, reason


async def _verify_one(client: OpenRouterClient, cfg, corpus: str, paper_id: str,
                       tuple_idx: int, t: dict, sem, progress, task_id, spend_state
                       ) -> VerifyResult:
    if spend_state["stopped"]:
        progress.advance(task_id)
        return VerifyResult(corpus, paper_id, tuple_idx, "ERROR", "cost cap tripped")
    async with sem:
        if spend_state["stopped"]:
            progress.advance(task_id)
            return VerifyResult(corpus, paper_id, tuple_idx, "ERROR", "cost cap tripped")
        prov = t.get("provenance") or {}
        table_id = prov.get("table_id")
        tables = _load_tables(corpus, paper_id)
        # Locate the specific table
        the_tab = None
        for tab in tables:
            if tab.get("table_id") == table_id:
                the_tab = tab; break
        if the_tab is None and tables:
            the_tab = tables[0]  # fallback
        if the_tab is None:
            progress.advance(task_id)
            return VerifyResult(corpus, paper_id, tuple_idx, "ERROR",
                                 "no cached table for provenance table_id")
        table_md, cap, meta = _windowed_markdown_table(the_tab, prov.get("row_label") or "")
        user = _build_user_prompt(t, table_md, cap, table_id, prov)
        try:
            r = await client.complete(model_id=MODEL_ID, system=VERIFIER_SYSTEM,
                                       user=user, temperature=0.0, max_tokens=200)
            cost = ((r.tokens_in / 1e6) * cfg.critic.price_in_per_m
                    + (r.tokens_out / 1e6) * cfg.critic.price_out_per_m)
            spend_state["total_cost"] += cost
            spend_state["done"] += 1
            if spend_state["done"] % 20 == 0:
                console.print(f"  [{spend_state['done']}/{spend_state['total']}] "
                              f"cost=${spend_state['total_cost']:.4f}  "
                              f"projected=${spend_state['total_cost'] * spend_state['total'] / spend_state['done']:.4f}")
            if spend_state["total_cost"] > COST_CAP:
                console.print(f"[red]COST CAP HIT: ${spend_state['total_cost']:.4f} > ${COST_CAP}[/red]")
                spend_state["stopped"] = True
            verdict, reason = _parse_verdict(r.text)
            return VerifyResult(corpus, paper_id, tuple_idx, verdict, reason,
                                 tokens_in=r.tokens_in, tokens_out=r.tokens_out, cost_usd=cost)
        except Exception as e:
            return VerifyResult(corpus, paper_id, tuple_idx, "ERROR", "",
                                 error=f"{type(e).__name__}: {str(e)[:200]}")
        finally:
            progress.advance(task_id)


@app.command()
def run(
    metalead_dir: Path = typer.Option(Path("outputs/canonextract_build/eval_runs_metalead_cycle5")),
    scirex_dir: Path = typer.Option(Path("outputs/canonextract_build/eval_runs_scirex_cycle5")),
    packet2_csv: Path = typer.Option(Path("outputs/canonextract_build/packet2_labels.csv")),
    packet2_key: Path = typer.Option(Path("outputs/canonextract_build/packet2_key.json")),
    concurrency: int = typer.Option(CONCURRENCY, "--concurrency"),
):
    cfg = load_config(config_path=Path("config/models.yaml"))
    client = OpenRouterClient(
        api_key=cfg.api_key, base_url=cfg.base_url,
        timeout_s=cfg.defaults.request_timeout_s,
        max_retries=cfg.defaults.max_retries,
        retry_backoff_s=cfg.defaults.retry_backoff_s,
        referer=cfg.referer, title=cfg.title,
    )

    # Collect strict tuples to verify
    to_verify: list[tuple[str, str, int, dict]] = []
    for corpus, in_dir in (("metalead", metalead_dir), ("scirex", scirex_dir)):
        for p in sorted(in_dir.glob(f"{corpus}_*.json")):
            d = json.loads(p.read_text())
            pid = d.get("paper_id") or p.stem.removeprefix(f"{corpus}_")
            for i, t in enumerate(d.get("tuples") or []):
                if (t.get("provenance") or {}).get("cycle5_bucket") == "strict_table_valid":
                    to_verify.append((corpus, pid, i, t))
    console.print(f"[bold]cycle-6 verifier: {len(to_verify)} strict tuples to verify[/bold]")

    asyncio.run(_main(client, cfg, to_verify, concurrency, packet2_csv, packet2_key,
                      metalead_dir, scirex_dir))


async def _main(client, cfg, to_verify, concurrency, packet2_csv, packet2_key,
                 metalead_dir, scirex_dir):
    sem = asyncio.Semaphore(concurrency)
    spend_state = {"total_cost": 0.0, "done": 0, "total": len(to_verify), "stopped": False}
    t0 = time.time()
    with Progress(SpinnerColumn(), TextColumn("{task.description}"), BarColumn(),
                  MofNCompleteColumn(), TimeElapsedColumn(), console=console) as progress:
        task_id = progress.add_task("verifier", total=len(to_verify))
        results = await asyncio.gather(*(
            _verify_one(client, cfg, c, pid, i, t, sem, progress, task_id, spend_state)
            for c, pid, i, t in to_verify
        ))
    await client.aclose()

    n_yes = sum(1 for r in results if r.verdict == "YES")
    n_no = sum(1 for r in results if r.verdict == "NO")
    n_unclear = sum(1 for r in results if r.verdict == "UNCLEAR")
    n_err = sum(1 for r in results if r.verdict == "ERROR")
    total_cost = sum(r.cost_usd for r in results)
    total_ti = sum(r.tokens_in for r in results)
    total_to = sum(r.tokens_out for r in results)
    wall = time.time() - t0
    console.print(f"[bold]verifier done: YES={n_yes} NO={n_no} UNCLEAR={n_unclear} ERR={n_err}[/bold]")
    console.print(f"  cost=${total_cost:.4f}  wall={wall:.0f}s  tokens_in={total_ti}  tokens_out={total_to}")

    # Persist per-paper verifier cache
    verifier_dir = Path("outputs/canonextract_build/verifier_cycle6")
    (verifier_dir/"metalead").mkdir(parents=True, exist_ok=True)
    (verifier_dir/"scirex").mkdir(parents=True, exist_ok=True)
    grouped: dict[tuple[str, str], list[VerifyResult]] = defaultdict(list)
    for r in results: grouped[(r.corpus, r.paper_id)].append(r)
    for (corpus, pid), rs in grouped.items():
        (verifier_dir/corpus/f"{pid}.json").write_text(json.dumps({
            "paper_id": pid, "corpus": corpus,
            "n_verifier_calls": len(rs),
            "verdicts": [{"tuple_idx": r.tuple_idx, "verdict": r.verdict, "reason": r.reason,
                          "tokens_in": r.tokens_in, "tokens_out": r.tokens_out, "cost_usd": r.cost_usd,
                          "error": r.error} for r in rs],
        }, indent=2))

    # Write eval_runs_{corpus}_cycle6/ (copy of cycle 5 + apply verifier downgrade)
    lookup = {(r.corpus, r.paper_id, r.tuple_idx): r for r in results}
    cycle6_totals = {"metalead": Counter(), "scirex": Counter()}
    for corpus, in_dir in (("metalead", metalead_dir), ("scirex", scirex_dir)):
        out_dir = Path(f"outputs/canonextract_build/eval_runs_{corpus}_cycle6")
        out_dir.mkdir(parents=True, exist_ok=True)
        for p in sorted(in_dir.glob(f"{corpus}_*.json")):
            d = json.loads(p.read_text())
            pid = d.get("paper_id") or p.stem.removeprefix(f"{corpus}_")
            for i, t in enumerate(d.get("tuples") or []):
                prov = t.get("provenance") or {}
                c5 = prov.get("cycle5_bucket")
                if c5 == "strict_table_valid":
                    vr = lookup.get((corpus, pid, i))
                    if vr is None:
                        prov["cycle6_bucket"] = "strict_table_valid"
                        prov["cycle6_verifier"] = "NOT_RUN"
                    elif vr.verdict == "YES":
                        prov["cycle6_bucket"] = "strict_table_valid"
                        prov["cycle6_verifier"] = "YES"
                        prov["cycle6_verifier_reason"] = vr.reason
                    elif vr.verdict == "ERROR":
                        # error → keep as strict but flag
                        prov["cycle6_bucket"] = "strict_table_valid"
                        prov["cycle6_verifier"] = f"ERROR:{vr.error or ''}"
                    else:
                        prov["cycle6_bucket"] = "verifier_downgrade_partial"
                        prov["cycle6_verifier"] = vr.verdict
                        prov["cycle6_verifier_reason"] = vr.reason
                else:
                    prov["cycle6_bucket"] = c5  # inherit unchanged
                cycle6_totals[corpus][prov["cycle6_bucket"]] += 1
                cycle6_totals[corpus]["total"] += 1
            d["provenance_stats_cycle6"] = dict(cycle6_totals[corpus])
            (out_dir / p.name).write_text(json.dumps(d, indent=2))
    # Fix per-paper stats (they were rolling)
    for corpus in ("metalead", "scirex"):
        out_dir = Path(f"outputs/canonextract_build/eval_runs_{corpus}_cycle6")
        for p in sorted(out_dir.glob(f"{corpus}_*.json")):
            d = json.loads(p.read_text())
            c = Counter()
            for t in d["tuples"]:
                b = t["provenance"]["cycle6_bucket"]
                c[b] += 1; c["total"] += 1
            d["provenance_stats_cycle6"] = dict(c)
            p.write_text(json.dumps(d, indent=2))

    # ---------- Packet 2 rescore ----------
    key = json.loads(packet2_key.read_text())
    key_by_row = {k["row_id"]: k for k in key}
    packet2_rows = list(csv.DictReader(packet2_csv.open()))
    # Build lookup: (corpus, pid, tidx) -> cycle6_bucket
    c6_bucket_by_key: dict[str, str] = {}
    for corpus in ("metalead", "scirex"):
        out_dir = Path(f"outputs/canonextract_build/eval_runs_{corpus}_cycle6")
        for p in sorted(out_dir.glob(f"{corpus}_*.json")):
            d = json.loads(p.read_text())
            pid = d.get("paper_id") or p.stem.removeprefix(f"{corpus}_")
            for i, t in enumerate(d["tuples"]):
                c6_bucket_by_key[f"{corpus}|{pid}|{i}"] = t["provenance"]["cycle6_bucket"]

    rescore = []
    for row in packet2_rows:
        rid = row["row_id"]
        keyrec = key_by_row.get(rid)
        if not keyrec: continue
        corpus, pid, tidx = keyrec["corpus"], keyrec["paper_id"], keyrec["tuple_idx"]
        rescore.append({
            "row_id": rid, "corpus": corpus, "paper_id": pid, "tuple_idx": tidx,
            "human_label": row["cell_matches_tuple_yes_no"].strip().upper(),
            "cycle6_bucket": c6_bucket_by_key.get(f"{corpus}|{pid}|{tidx}", "not_found"),
        })

    n_yes_h = sum(1 for r in rescore if r["human_label"] == "YES")
    strict_rows = [r for r in rescore if r["cycle6_bucket"] == "strict_table_valid"]
    n_strict = len(strict_rows)
    n_strict_yes = sum(1 for r in strict_rows if r["human_label"] == "YES")
    precision = (n_strict_yes / n_strict) if n_strict else None
    # Recall on Harish YES
    yes_rows = [r for r in rescore if r["human_label"] == "YES"]
    n_yes_strict = sum(1 for r in yes_rows if r["cycle6_bucket"] == "strict_table_valid")
    recall = (n_yes_strict / n_yes_h) if n_yes_h else None

    # Per-corpus precision on cycle-6 strict
    per_corpus_prec = {}
    per_corpus_recall = {}
    for c in ("metalead", "scirex"):
        srows = [r for r in rescore if r["corpus"] == c and r["cycle6_bucket"] == "strict_table_valid"]
        yrows_c = [r for r in rescore if r["corpus"] == c and r["human_label"] == "YES"]
        per_corpus_prec[c] = ((sum(1 for r in srows if r["human_label"] == "YES") / len(srows))
                              if srows else None)
        per_corpus_recall[c] = ((sum(1 for r in yrows_c if r["cycle6_bucket"] == "strict_table_valid") / len(yrows_c))
                                 if yrows_c else None)

    # Per-corpus cycle 6 aggregate rates
    per_corpus_agg = {}
    for corpus in ("metalead", "scirex"):
        c = cycle6_totals[corpus]
        tot = c.get("total", 0)
        per_corpus_agg[corpus] = {
            "total": tot,
            "strict_table_valid": c.get("strict_table_valid", 0),
            "verifier_downgrade_partial": c.get("verifier_downgrade_partial", 0),
            "partial_table_valid": c.get("partial_table_valid", 0),
            "prose_claimed_table": c.get("prose_claimed_table", 0),
            "prose": c.get("prose", 0),
            "missing": c.get("missing", 0),
            "pct_strict": round(c.get("strict_table_valid", 0) / tot * 100, 2) if tot else 0.0,
            "pct_verifier_downgrade_partial": round(c.get("verifier_downgrade_partial", 0) / tot * 100, 2) if tot else 0.0,
            "pct_partial": round(c.get("partial_table_valid", 0) / tot * 100, 2) if tot else 0.0,
            "pct_prose_claimed_table": round(c.get("prose_claimed_table", 0) / tot * 100, 2) if tot else 0.0,
            "pct_prose": round(c.get("prose", 0) / tot * 100, 2) if tot else 0.0,
            "pct_missing": round(c.get("missing", 0) / tot * 100, 2) if tot else 0.0,
        }

    summary = {
        "seed": SEED,
        "cycle": 6,
        "verifier_model_id": MODEL_ID,
        "n_verifier_calls": len(results),
        "verifier_verdicts": {"YES": n_yes, "NO": n_no, "UNCLEAR": n_unclear, "ERROR": n_err},
        "verifier_cost_usd": total_cost,
        "verifier_tokens_in": total_ti,
        "verifier_tokens_out": total_to,
        "verifier_wall_seconds": wall,
        "cumulative_cost_cycles_1_to_6": 3.824 + total_cost,
        "corpus_totals_cycle6": per_corpus_agg,
        "packet2_rescore": {
            "n_rows": len(rescore),
            "cycle6_strict_precision_overall": precision,
            "cycle6_strict_recall_overall": recall,
            "cycle6_strict_precision_metalead": per_corpus_prec["metalead"],
            "cycle6_strict_precision_scirex": per_corpus_prec["scirex"],
            "cycle6_strict_recall_metalead": per_corpus_recall["metalead"],
            "cycle6_strict_recall_scirex": per_corpus_recall["scirex"],
            "per_row": rescore,
        },
    }
    Path("outputs/canonextract_build/verifier_cycle6_summary.json").write_text(json.dumps(summary, indent=2))
    console.print(f"[green]wrote verifier_cycle6_summary.json[/green]")
    for corpus, agg in per_corpus_agg.items():
        console.print(f"  {corpus}: strict={agg['pct_strict']}%  downgrade={agg['pct_verifier_downgrade_partial']}%  "
                      f"partial(binder)={agg['pct_partial']}%  pct={agg['pct_prose_claimed_table']}%  "
                      f"prose={agg['pct_prose']}%  miss={agg['pct_missing']}%")
    console.print(f"  Packet 2: precision overall={precision if precision is None else round(precision*100,1)}%  "
                  f"recall overall={recall if recall is None else round(recall*100,1)}%")
    console.print(f"  Packet 2 per-corpus precision: ML={per_corpus_prec['metalead']}  SX={per_corpus_prec['scirex']}")


if __name__ == "__main__":
    app()
