"""Cost-matched self-consistency baseline on MetaLead.

Paper 1's Extractor prompt only (no Critic, no Consolidator, no table
injection). Sampled 5 times per paper at temperatures {0.0, 0.3, 0.5,
0.7, 0.7}. Voted per contribution field using paper 1's self-consistency
voting rule (fractional agreement >= 3/5, tie-break toward t=0.0 draft).

Cap: $0.50. If projected total exceeds, stop and report partial results.
Log spend every 10 papers. One-shot: no reruns, no prompt tuning, no
sample-count adjustments after seeing results.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import typer
from rich.console import Console
from rich.progress import (
    BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn,
)

from paper1.agents import ExtractorAgent
from paper1.config import load_config, load_prompt
from paper1.openrouter import OpenRouterClient

SEED = 42
CONCURRENCY = 3
COST_CAP = 0.75  # cycle-9 resume: user approved additional spend to finish 7 unfinished papers
TEMPERATURES = (0.0, 0.3, 0.5, 0.7, 0.7)
PAPER_TEXT_CHAR_CAP = 45000  # matches paper 1 baseline

app = typer.Typer(no_args_is_help=False, add_completion=False)
console = Console()


@dataclass
class SamplePaper:
    paper_id: str
    status: str
    n_samples_ok: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    wall_seconds: float = 0.0
    error: str | None = None


def _load_paper_text(paper_id: str) -> str:
    tex_dir = Path("data/metalead_tex") / paper_id
    if not tex_dir.exists(): return ""
    parts = []
    for f in tex_dir.glob("**/*.tex"):
        try: parts.append(f.read_text(encoding="utf-8", errors="ignore"))
        except Exception: continue
    return "\n".join(parts)[:PAPER_TEXT_CHAR_CAP]


def _norm(s: str | None) -> str:
    if not s: return ""
    s = str(s).lower().replace("_", " ")
    s = re.sub(r"\\[a-zA-Z]+\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\[a-zA-Z]+", " ", s)
    s = re.sub(r"[\$\^_\{\}]", " ", s)
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _vote(samples: list[dict]) -> list[dict]:
    """Paper-1 self-consistency: per contribution key, keep fields with fractional agreement
    >= 3/5. Contribution key = (task_norm, first_dataset_norm, first_metric_norm).
    Tie-break: take field values from t=0.0 sample (samples[0]).
    """
    if not samples: return []
    n = len(samples)
    threshold = 3  # per spec: >= 3/5

    # Group emitted contributions across samples by their (task, first-dataset, first-metric) key
    per_key: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for i, sample in enumerate(samples):
        contribs = sample.get("contributions") or []
        for c in contribs:
            task = _norm((c.get("task") or {}).get("name"))
            ds_list = c.get("datasets") or []
            m_list = c.get("metrics") or []
            for ds in ds_list:
                dsn = _norm(ds.get("name") if isinstance(ds, dict) else str(ds))
                for m in m_list:
                    mn = _norm(m.get("name") if isinstance(m, dict) else str(m))
                    if not (task or dsn or mn): continue
                    key = (task, dsn, mn)
                    # Store per-sample entry so we can vote fields and remember origin
                    per_key[key].append({"sample_i": i, "contribution": c,
                                         "matched_ds": ds, "matched_metric": m})

    voted: list[dict] = []
    for key, entries in per_key.items():
        # Support = number of distinct samples that emitted this key
        supports = set(e["sample_i"] for e in entries)
        if len(supports) < threshold:
            continue
        # Prefer the t=0.0 entry if present, else the earliest
        best = min(entries, key=lambda e: e["sample_i"])
        c = dict(best["contribution"])
        c["datasets"] = [best["matched_ds"]] if best["matched_ds"] else []
        c["metrics"] = [best["matched_metric"]] if best["matched_metric"] else []
        c["_sc_support"] = len(supports) / n
        voted.append(c)
    return voted


async def _extract_paper(agent, cfg, paper_id: str, paper_text: str,
                          sem, spend_state, progress, task_id) -> SamplePaper:
    async with sem:
        if spend_state["stopped"]:
            progress.advance(task_id)
            return SamplePaper(paper_id, "skipped", error="cost cap tripped")
        t0 = time.time()
        samples_raw: list[dict] = []
        tokens_in = 0; tokens_out = 0; cost = 0.0
        n_ok = 0; last_err = None
        for i, temp in enumerate(TEMPERATURES):
            if spend_state["stopped"]:
                break
            try:
                draft, r = await agent.run_one(
                    paper_id=paper_id, paper_text=paper_text,
                    retrieval_bundle="",
                    temperature=temp,
                )
                samples_raw.append(draft.model_dump())
                tokens_in += r.tokens_in
                tokens_out += r.tokens_out
                this_cost = ((r.tokens_in / 1e6) * cfg.extractor.price_in_per_m
                             + (r.tokens_out / 1e6) * cfg.extractor.price_out_per_m)
                cost += this_cost
                spend_state["total_cost"] += this_cost
                n_ok += 1
            except Exception as e:
                last_err = f"{type(e).__name__}: {str(e)[:150]}"
                samples_raw.append({"contributions": []})
            # After each sample, project total and stop early if cap will be exceeded
            if spend_state["total_cost"] > COST_CAP:
                console.print(f"[red]cost cap ${COST_CAP:.2f} tripped at ${spend_state['total_cost']:.4f}[/red]")
                spend_state["stopped"] = True

        # Vote across the samples we DID collect
        voted = _vote(samples_raw)

        out = {
            "paper_id": paper_id,
            "contributions": voted,
            "_meta": {
                "extractor_model": cfg.extractor.model_id,
                "voting_samples": len(samples_raw),
                "n_samples_ok": n_ok,
                "temperatures": list(TEMPERATURES),
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
                "cost_usd": cost,
                "wall_seconds": time.time() - t0,
                "seed": SEED,
                "sample_error_if_any": last_err,
            },
            "_raw_samples": samples_raw,
        }
        Path("outputs/canonextract_build/metalead_sc_baseline").mkdir(parents=True, exist_ok=True)
        Path(f"outputs/canonextract_build/metalead_sc_baseline/metalead_{paper_id}.json").write_text(
            json.dumps(out, indent=2))
        spend_state["done"] += 1
        if spend_state["done"] % 10 == 0 or spend_state["stopped"]:
            console.print(f"  [{spend_state['done']}/{spend_state['total']}] "
                          f"cost=${spend_state['total_cost']:.4f}  "
                          f"projected=${spend_state['total_cost'] * spend_state['total'] / spend_state['done']:.4f}")
        progress.advance(task_id)
        return SamplePaper(paper_id, "ok" if n_ok else "error",
                            n_samples_ok=n_ok, tokens_in=tokens_in, tokens_out=tokens_out,
                            cost_usd=cost, wall_seconds=time.time() - t0, error=last_err)


@app.command()
def run(concurrency: int = typer.Option(CONCURRENCY, "--concurrency")):
    cfg = load_config(config_path=Path("config/models.yaml"))
    client = OpenRouterClient(
        api_key=cfg.api_key, base_url=cfg.base_url,
        timeout_s=cfg.defaults.request_timeout_s,
        max_retries=cfg.defaults.max_retries,
        retry_backoff_s=cfg.defaults.retry_backoff_s,
        referer=cfg.referer, title=cfg.title,
    )
    agent = ExtractorAgent(client, cfg.extractor, load_prompt("extractor", cfg.prompts_dir))

    # 27 MetaLead evaluation papers from cycle 5 eval set
    ce_dir = Path("outputs/canonextract_build/eval_runs_metalead_cycle5")
    paper_ids: list[str] = []
    for f in sorted(ce_dir.glob("metalead_*.json")):
        pid = f.stem.removeprefix("metalead_")
        paper_ids.append(pid)

    # Resume: skip papers that already have all 5 samples cached
    cache_dir = Path("outputs/canonextract_build/metalead_sc_baseline")
    if cache_dir.exists():
        already = set()
        for f in cache_dir.glob("metalead_*.json"):
            try:
                d = json.load(f.open())
                if d.get("_meta", {}).get("n_samples_ok", 0) == len(TEMPERATURES):
                    already.add(d["paper_id"])
            except Exception:
                pass
        skipped = [p for p in paper_ids if p in already]
        paper_ids = [p for p in paper_ids if p not in already]
        console.print(f"[bold]resume mode: skipping {len(skipped)} fully-cached papers, "
                      f"processing {len(paper_ids)} remaining[/bold]")

    console.print(f"[bold]cost-matched SC: {len(paper_ids)} MetaLead papers x {len(TEMPERATURES)} samples[/bold]")

    asyncio.run(_main(agent, cfg, paper_ids, concurrency))


async def _main(agent, cfg, paper_ids, concurrency):
    sem = asyncio.Semaphore(concurrency)
    spend_state = {"total_cost": 0.0, "done": 0, "total": len(paper_ids), "stopped": False}
    t0 = time.time()
    # Pre-load paper texts (avoid contention)
    texts = {pid: _load_paper_text(pid) for pid in paper_ids}
    with Progress(SpinnerColumn(), TextColumn("{task.description}"), BarColumn(),
                  MofNCompleteColumn(), TimeElapsedColumn(), console=console) as progress:
        task_id = progress.add_task("sc_baseline", total=len(paper_ids))
        results = await asyncio.gather(*(
            _extract_paper(agent, cfg, pid, texts[pid], sem, spend_state, progress, task_id)
            for pid in paper_ids
        ))
    await agent._client.aclose()

    n_ok = sum(1 for r in results if r.status == "ok")
    n_err = sum(1 for r in results if r.status == "error")
    n_skip = sum(1 for r in results if r.status == "skipped")
    total_cost = sum(r.cost_usd for r in results)
    total_ti = sum(r.tokens_in for r in results)
    total_to = sum(r.tokens_out for r in results)
    wall = time.time() - t0
    console.print(f"[bold]SC baseline done: ok={n_ok} err={n_err} skip={n_skip}[/bold]")
    console.print(f"  cost=${total_cost:.4f}  wall={wall:.0f}s  tokens_in={total_ti}  tokens_out={total_to}")

    summary = {
        "seed": SEED,
        "run": "metalead_cost_matched_sc",
        "extractor_model": cfg.extractor.model_id,
        "temperatures": list(TEMPERATURES),
        "papers_total": len(paper_ids),
        "n_ok": n_ok, "n_err": n_err, "n_skipped": n_skip,
        "cost_usd_total": total_cost,
        "cost_cap": COST_CAP,
        "cap_tripped": spend_state["stopped"],
        "tokens_in": total_ti, "tokens_out": total_to,
        "wall_seconds": wall,
        "per_paper": [{"paper_id": r.paper_id, "status": r.status,
                       "n_samples_ok": r.n_samples_ok,
                       "cost_usd": r.cost_usd, "tokens_in": r.tokens_in,
                       "tokens_out": r.tokens_out, "wall_seconds": r.wall_seconds,
                       "error": r.error} for r in results],
    }
    Path("outputs/canonextract_build/metalead_sc_baseline/extraction_summary.json").write_text(
        json.dumps(summary, indent=2))


if __name__ == "__main__":
    app()
