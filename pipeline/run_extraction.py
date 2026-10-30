"""CanonExtract orchestration — Components 3 + 4 + 5 (light).

For each paper:
  1. Load vocab (local + external) + tables + paper text.
  2. Format the constrained-extractor prompt (config/prompts/canonextract_v1.md
     or whichever version is passed).
  3. Call OpenRouterClient at 3 temperatures [0.0, 0.3, 0.7].
  4. Parse each sample as JSON with lenient recovery.
  5. Tuple-level majority vote on (task, dataset, metric) canonical triples.
  6. Provenance binder: validate table_id/row_label/header_path against the
     paper's parsed tables. Downgrade unresolved table claims to
     "prose_claimed_table".
  7. Emit outputs/canonextract_build/{run_dir}/{paper_id}.json

Note on Critic/Consolidator: paper 1's Critic/Consolidator strictly expect
ExtractorDraft shape without provenance. In this first cycle we run the
extractor + vote + binder only; Critic-on-canonextract is scoped as tech
debt in the SUMMARY.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.progress import (
    BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn,
)

from paper1.config import load_config
from paper1.openrouter import OpenRouterClient, parse_json_response

SEED = 42
COST_CAP_USD = 10.0
LOG_EVERY = 10
TEMPS = [0.0, 0.3, 0.7]
MAX_TABLES = 20
EXT_TOP_N = 500
PAPER_TEXT_CHAR_CAP = 45000  # ~11k tokens; leaves headroom for vocab + tables

app = typer.Typer(no_args_is_help=False, add_completion=False)
console = Console()


_WS_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^\w\s]")


def normalize(s: str | None) -> str:
    if not s: return ""
    s = str(s).lower().replace("_", " ")
    s = _PUNCT_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    return s


@dataclass
class PaperResult:
    paper_id: str
    corpus: str
    status: str
    n_tuples_emitted: int = 0
    n_table_anchored: int = 0
    n_table_valid: int = 0
    n_prose_anchored: int = 0
    n_prose_claimed_table: int = 0
    cost_usd: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    wall_seconds: float = 0.0
    error: str | None = None


def _load_scirex_paper(doc_id: str) -> tuple[str, list[dict]]:
    """Return (paper_text, tables) for a SciREX paper."""
    text = ""
    tex_dir = Path("data/scirex_tex") / doc_id
    if tex_dir.exists():
        combined = []
        for f in tex_dir.glob("**/*.tex"):
            try:
                combined.append(f.read_text(encoding="utf-8", errors="ignore"))
            except Exception:
                continue
        text = "\n".join(combined)
    # Tables from cached v1
    tp = Path("outputs/canonextract_diag/tables") / f"{doc_id}.json"
    tables = []
    if tp.exists():
        d = json.loads(tp.read_text())
        tables = d.get("tables") or []
    return text, tables


def _load_metalead_paper(paper_key: str) -> tuple[str, list[dict]]:
    text = ""
    tex_dir = Path("data/metalead_tex") / paper_key
    if tex_dir.exists():
        combined = []
        for f in tex_dir.glob("**/*.tex"):
            try:
                combined.append(f.read_text(encoding="utf-8", errors="ignore"))
            except Exception:
                continue
        text = "\n".join(combined)
    # For MetaLead, tables were saved as flat lists in v2 diagnostic. Convert to per-table shape.
    p = Path("outputs/canonextract_diag_v2/metalead_tables") / f"{paper_key}.json"
    tables: list[dict] = []
    if p.exists():
        d = json.loads(p.read_text())
        # Flat lists — synthesize one "virtual table" per unique caption if any, else a single table.
        # Since we only have flat lists from v2, wrap as a single table with all row/col/cells.
        row_h = d.get("row_headers", [])
        col_h = d.get("column_headers", [])
        cells = d.get("cells", [])
        caps = d.get("table_captions", [])
        # Convert to a paper-1-style structure: assume simple grid where col_h and row_h zip
        # (Not perfect since v2 didn't preserve per-table grouping, but sufficient for anchor validation)
        if row_h or col_h or cells:
            tables.append({
                "table_id": 0,
                "caption": " | ".join(caps[:3]) if caps else "",
                "row_headers": row_h[:200],
                "column_headers": col_h[:200],
                "cells": cells[:400],
            })
    return text, tables


def _load_vocab(paper_id: str, corpus: str) -> dict:
    p = Path("outputs/canonextract_build/vocab") / f"{corpus}_{paper_id}.json"
    if not p.exists():
        return {"local": {"all_phrases": []}, "external": {"tasks": [], "datasets": [], "metrics": []}}
    return json.loads(p.read_text())


def _format_prompt_bundle(paper_id: str, paper_text: str, tables: list[dict], vocab: dict) -> dict:
    ext = vocab.get("external", {})
    local_phrases = (vocab.get("local") or {}).get("all_phrases") or []

    # Cap tables to MAX_TABLES; compact JSON per table
    compact_tables = []
    for t in tables[:MAX_TABLES]:
        # Row-header × cell-line grid: derive rows heuristically
        rows_compact = []
        col_h = t.get("column_headers") or []
        row_h = t.get("row_headers") or []
        cells = t.get("cells") or []
        # Best-effort row reconstruction: zip row_h and cells if aligned
        if row_h and col_h:
            per_row = max(1, len(col_h))
            for i, r in enumerate(row_h[:60]):
                r_cells = cells[i * per_row: (i + 1) * per_row] if cells else []
                rows_compact.append({
                    "row_label": r,
                    "cells": {col_h[j] if j < len(col_h) else f"col_{j}": (r_cells[j] if j < len(r_cells) else "")
                              for j in range(min(per_row, len(r_cells)) if r_cells else 0)},
                })
        compact_tables.append({
            "table_id": t.get("table_id", 0),
            "caption": (t.get("caption") or "")[:400],
            "headers": col_h[:40],
            "rows": rows_compact,
        })

    # Cap text
    text = paper_text[:PAPER_TEXT_CHAR_CAP]

    return {
        "LOCAL_TASKS": ", ".join(sorted(local_phrases)[:200]) or "(none — use external only)",
        "LOCAL_DATASETS": ", ".join(sorted(local_phrases)[:200]) or "(none)",
        "LOCAL_METRICS": ", ".join(sorted(local_phrases)[:200]) or "(none)",
        "EXTERNAL_TASKS": ", ".join(ext.get("tasks", [])[:EXT_TOP_N]),
        "EXTERNAL_DATASETS": ", ".join(ext.get("datasets", [])[:EXT_TOP_N]),
        "EXTERNAL_METRICS": ", ".join(ext.get("metrics", [])[:EXT_TOP_N]),
        "TABLES_JSON": json.dumps(compact_tables, indent=1)[:15000],
        "paper_id": paper_id,
        "paper_text": text,
    }


def _split_prompt(template: str) -> tuple[str, str]:
    if "USER:" not in template:
        raise ValueError("prompt template missing 'USER:'")
    system_part, user_part = template.split("USER:", 1)
    if "SYSTEM:" in system_part:
        system_part = system_part.split("SYSTEM:", 1)[1]
    return system_part.strip(), user_part.strip()


def _parse_lenient(text: str) -> dict | None:
    try:
        return parse_json_response(text)
    except Exception:
        return None


def _vote_tuples(samples: list[list[dict]]) -> list[dict]:
    """Deterministic tuple-level majority vote across 3 samples.

    Bin each contribution by (norm task, norm dataset, norm metric); at most
    one entry per sample per key. self_consistency = distinct-sample-support / n_samples.
    """
    bins: dict[tuple[str, str, str], dict[int, dict]] = defaultdict(dict)
    for sample_i, contribs in enumerate(samples):
        seen_this_sample: set[tuple[str, str, str]] = set()
        for c in contribs or []:
            task = normalize((c.get("task") or {}).get("name", ""))
            ds_list = c.get("datasets") or []
            m_list = c.get("metrics") or []
            for ds in ds_list:
                dsn = normalize(ds.get("name", "") if isinstance(ds, dict) else str(ds))
                for m in m_list:
                    mn = normalize(m.get("name", "") if isinstance(m, dict) else str(m))
                    if not (task or dsn or mn): continue
                    key = (task, dsn, mn)
                    if key in seen_this_sample:
                        continue  # only one binding per (t,d,m) per sample
                    seen_this_sample.add(key)
                    entry = {**c, "matched_ds": ds, "matched_metric": m}
                    bins[key][sample_i] = entry

    out: list[dict] = []
    n_samples = max(1, len(samples))
    for key, per_sample in bins.items():
        support = len(per_sample)  # distinct samples that emitted this key
        first = next(iter(per_sample.values()))
        merged = dict(first)
        merged["self_consistency"] = min(1.0, support / n_samples)
        merged["datasets"] = [first.get("matched_ds")] if first.get("matched_ds") else []
        merged["metrics"] = [first.get("matched_metric")] if first.get("matched_metric") else []
        merged.pop("matched_ds", None)
        merged.pop("matched_metric", None)
        out.append(merged)
    return out


def _validate_provenance(tuples: list[dict], tables: list[dict], paper_text: str) -> tuple[list[dict], dict]:
    """Validate + enrich provenance. Downgrade table claims that don't resolve."""
    tables_by_id = {t.get("table_id", i): t for i, t in enumerate(tables)}
    stats = Counter()
    validated = []
    paper_hash = hashlib.sha256(paper_text.encode("utf-8", errors="ignore")).hexdigest()[:12]
    for t in tuples:
        prov = t.get("provenance") or {}
        src = (prov.get("source") or "").lower()
        stats["total"] += 1
        if src == "table":
            table_id = prov.get("table_id")
            row_lbl = normalize(prov.get("row_label") or "")
            hdr = normalize(prov.get("header_path") or "")
            cell = normalize(prov.get("cell") or "")
            table = tables_by_id.get(table_id)
            valid = False
            if table is not None:
                row_norms = {normalize(r) for r in table.get("row_headers") or [] if r}
                col_norms = {normalize(c) for c in table.get("column_headers") or [] if c}
                cell_norms = {normalize(c) for c in table.get("cells") or [] if c}
                row_hit = any(row_lbl and (row_lbl in r or r in row_lbl) for r in row_norms)
                hdr_hit = any(hdr and (hdr in c or c in hdr) for c in col_norms)
                cell_hit = any(cell and (cell in c or c in cell) for c in cell_norms) if cell else True
                if row_hit and hdr_hit and cell_hit:
                    valid = True
            if valid:
                stats["table_valid"] += 1
                t["provenance"]["_content_hash"] = hashlib.sha256(
                    f"{prov.get('cell','')}|{prov.get('row_label','')}|{prov.get('header_path','')}".encode()
                ).hexdigest()[:12]
                t["provenance"]["_paper_hash"] = paper_hash
            else:
                stats["prose_claimed_table"] += 1
                t["provenance"]["source"] = "prose_claimed_table"
                t["provenance"]["_downgrade_reason"] = "table_id or row/header/cell not resolvable"
                t["provenance"]["_paper_hash"] = paper_hash
        elif src == "prose":
            # Accept prose provenance if EITHER the claimed offsets resolve OR
            # the quote is a genuine substring of paper_text (offsets are hard
            # for LLMs; substring match is what actually matters).
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
                t["provenance"]["_quote_verified"] = None
                t["provenance"]["_verification_mode"] = "unverified"
            t["provenance"]["_paper_hash"] = paper_hash
        else:
            stats["unknown_source"] += 1
            t["provenance"] = {"source": "unknown", "_paper_hash": paper_hash}
        validated.append(t)
    return validated, dict(stats)


async def _run_one(client: OpenRouterClient, prompt_template: str, cfg,
                   paper_id: str, corpus: str, out_dir: Path,
                   sem: asyncio.Semaphore, progress: Progress, task_id,
                   spend_state: dict) -> PaperResult:
    if spend_state["stopped"]:
        progress.advance(task_id)
        return PaperResult(paper_id=paper_id, corpus=corpus, status="skipped",
                           error="cost cap tripped")
    async with sem:
        if spend_state["stopped"]:
            progress.advance(task_id)
            return PaperResult(paper_id=paper_id, corpus=corpus, status="skipped",
                               error="cost cap tripped")

        t0 = time.time()
        try:
            if corpus == "scirex":
                paper_text, tables = _load_scirex_paper(paper_id)
            else:
                paper_text, tables = _load_metalead_paper(paper_id)
            vocab = _load_vocab(paper_id, corpus)
            bundle = _format_prompt_bundle(paper_id, paper_text, tables, vocab)

            system, user_tmpl = _split_prompt(prompt_template)
            user = user_tmpl.format(paper_id=bundle["paper_id"],
                                    TABLES_JSON=bundle["TABLES_JSON"],
                                    paper_text=bundle["paper_text"])
            # System prompt has vocab placeholders
            system = system.format(**{k: bundle[k] for k in ("LOCAL_TASKS", "LOCAL_DATASETS", "LOCAL_METRICS",
                                                              "EXTERNAL_TASKS", "EXTERNAL_DATASETS", "EXTERNAL_METRICS")})

            # 3 temp samples
            calls = [client.complete(
                model_id=cfg.extractor.model_id,
                system=system, user=user,
                temperature=t, max_tokens=cfg.extractor.max_tokens,
                top_p=cfg.extractor.top_p,
            ) for t in TEMPS]
            results = await asyncio.gather(*calls)
            tokens_in = sum(r.tokens_in for r in results)
            tokens_out = sum(r.tokens_out for r in results)
            cost_usd = sum(
                (r.tokens_in / 1e6) * cfg.extractor.price_in_per_m
                + (r.tokens_out / 1e6) * cfg.extractor.price_out_per_m
                for r in results
            )
            spend_state["total_cost"] += cost_usd
            spend_state["done"] += 1

            # Parse each sample
            samples = []
            for r in results:
                d = _parse_lenient(r.text)
                if d is None:
                    samples.append([])
                    continue
                contribs = d.get("contributions") if isinstance(d, dict) else None
                samples.append(contribs or [])

            # Tuple-level vote
            voted = _vote_tuples(samples)
            # Validate provenance
            validated, prov_stats = _validate_provenance(voted, tables, paper_text)

            # Emit
            out_path = out_dir / f"{corpus}_{paper_id}.json"
            out_path.write_text(json.dumps({
                "paper_id": paper_id, "corpus": corpus,
                "n_tables": len(tables),
                "tuples": validated,
                "provenance_stats": prov_stats,
                "meta": {
                    "tokens_in": tokens_in, "tokens_out": tokens_out,
                    "cost_usd": cost_usd, "wall_seconds": time.time() - t0,
                    "extractor_model": cfg.extractor.model_id,
                    "prompt_hash": hashlib.sha256(prompt_template.encode()).hexdigest()[:12],
                    "voting_samples": len(samples),
                },
            }, indent=2))

            if spend_state["done"] % LOG_EVERY == 0:
                projected = (spend_state["total_cost"] / spend_state["done"]) * spend_state["total_papers"]
                console.print(f"    [checkpoint] papers_done={spend_state['done']}  "
                              f"cost_so_far=${spend_state['total_cost']:.4f}  "
                              f"projected_total=${projected:.4f}")
                # Add other-phase budget already spent
                budget_used = spend_state["total_cost"] + spend_state.get("prior_spend", 0.0)
                projected_total_with_prior = projected + spend_state.get("prior_spend", 0.0)
                if projected_total_with_prior > COST_CAP_USD:
                    console.print(f"[red]COST CAP TRIPPED: projected total ${projected_total_with_prior:.4f} > ${COST_CAP_USD}[/red]")
                    spend_state["stopped"] = True

            return PaperResult(
                paper_id=paper_id, corpus=corpus, status="ok",
                n_tuples_emitted=len(validated),
                n_table_anchored=prov_stats.get("table_valid", 0) + prov_stats.get("prose_claimed_table", 0),
                n_table_valid=prov_stats.get("table_valid", 0),
                n_prose_anchored=prov_stats.get("prose", 0),
                n_prose_claimed_table=prov_stats.get("prose_claimed_table", 0),
                cost_usd=cost_usd, tokens_in=tokens_in, tokens_out=tokens_out,
                wall_seconds=time.time() - t0,
            )
        except Exception as e:  # noqa: BLE001
            return PaperResult(paper_id=paper_id, corpus=corpus, status="error",
                               error=f"{type(e).__name__}: {str(e)[:200]}")
        finally:
            progress.advance(task_id)


@app.command()
def run(
    prompt_path: Path = typer.Option(Path("config/prompts/canonextract_v1.md"), "--prompt"),
    manifest_path: Path = typer.Option(..., "--manifest"),
    corpus: str = typer.Option(..., "--corpus", help="metalead | scirex"),
    out_dir: Path = typer.Option(..., "--out"),
    config_path: Path = typer.Option(Path("config/models.yaml"), "--config"),
    concurrency: int = typer.Option(3, "--concurrency"),
    prior_spend: float = typer.Option(0.0, "--prior-spend",
                                       help="Already-spent budget from earlier phases (for cap)"),
):
    out_dir.mkdir(parents=True, exist_ok=True)
    prompt_template = prompt_path.read_text()
    manifest = json.loads(manifest_path.read_text())

    # Build unique paper list from the manifest
    if isinstance(manifest, list):
        # dev / eval manifest = list of tuple dicts with 'paper_key' or 'doc_id'
        paper_ids = sorted(set(m.get("paper_key") or m.get("doc_id") or "" for m in manifest))
    else:
        raise ValueError("Manifest must be a list of tuples")
    paper_ids = [p for p in paper_ids if p]
    console.print(f"[bold]{corpus} run[/bold]: {len(paper_ids)} unique papers "
                  f"(from {len(manifest)} manifest entries). Prior spend: ${prior_spend:.4f}")

    cfg = load_config(config_path=config_path)
    or_client = OpenRouterClient(
        api_key=cfg.api_key, base_url=cfg.base_url,
        timeout_s=cfg.defaults.request_timeout_s,
        max_retries=cfg.defaults.max_retries,
        retry_backoff_s=cfg.defaults.retry_backoff_s,
        referer=cfg.referer, title=cfg.title,
    )

    asyncio.run(_main(paper_ids, corpus, prompt_template, cfg, or_client, out_dir,
                      concurrency, prior_spend))


async def _main(paper_ids, corpus, prompt_template, cfg, client, out_dir, concurrency, prior_spend):
    sem = asyncio.Semaphore(concurrency)
    spend_state = {"total_cost": 0.0, "done": 0, "total_papers": len(paper_ids),
                   "stopped": False, "prior_spend": prior_spend}
    t0 = time.time()
    with Progress(SpinnerColumn(), TextColumn("{task.description}"), BarColumn(),
                  MofNCompleteColumn(), TimeElapsedColumn(), console=console) as progress:
        task_id = progress.add_task("canonextract", total=len(paper_ids))
        results = await asyncio.gather(*(
            _run_one(client, prompt_template, cfg, pid, corpus, out_dir,
                     sem, progress, task_id, spend_state)
            for pid in paper_ids
        ))
    ok = sum(1 for r in results if r.status == "ok")
    err = sum(1 for r in results if r.status == "error")
    skip = sum(1 for r in results if r.status == "skipped")
    total_cost = sum(r.cost_usd for r in results if r.status == "ok")
    total_ti = sum(r.tokens_in for r in results if r.status == "ok")
    total_to = sum(r.tokens_out for r in results if r.status == "ok")
    total_tuples = sum(r.n_tuples_emitted for r in results if r.status == "ok")
    total_table_valid = sum(r.n_table_valid for r in results if r.status == "ok")
    total_prose = sum(r.n_prose_anchored for r in results if r.status == "ok")
    total_prose_claimed_table = sum(r.n_prose_claimed_table for r in results if r.status == "ok")
    elapsed = time.time() - t0
    console.print(f"[bold]done: ok={ok} err={err} skip={skip} cost=${total_cost:.4f} "
                  f"wall={elapsed:.0f}s tokens_in={total_ti} out={total_to}[/bold]")
    console.print(f"tuples emitted: {total_tuples}  "
                  f"table_valid: {total_table_valid}  "
                  f"prose: {total_prose}  "
                  f"prose_claimed_table: {total_prose_claimed_table}")
    (out_dir / "extraction_summary.json").write_text(json.dumps({
        "papers_total": len(paper_ids), "ok": ok, "err": err, "skip": skip,
        "total_cost_usd": total_cost,
        "tokens_in": total_ti, "tokens_out": total_to,
        "tuples_emitted": total_tuples,
        "provenance_table_valid": total_table_valid,
        "provenance_prose": total_prose,
        "provenance_prose_claimed_table": total_prose_claimed_table,
        "wall_seconds": elapsed,
        "cost_cap_tripped": spend_state["stopped"],
        "per_paper": [asdict(r) for r in results],
    }, indent=2))


if __name__ == "__main__":
    app()
