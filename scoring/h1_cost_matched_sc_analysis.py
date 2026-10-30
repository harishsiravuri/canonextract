"""
H1 whole-record correctness reanalysis on MetaLead (deterministic; $0).
Regenerates outputs/canonextract_build/h1_reanalysis_full.json.

Definitions frozen BEFORE inspecting results (per proposal H1 endpoint):
- Whole record = (Task, Dataset, Metric) triple. Method and Value excluded
  because MetaLead gold does not include them.
- Matching = lenient substring after normalize + LaTeX strip. Identical to
  cycle-1's per-field F1; no new rule.
- Per-paper: Precision (fraction of emitted triples matching gold),
  Recall (fraction of gold triples matched), F1.
- CI: paired bootstrap 95%, 1000 resamples, seed=42, resample unit = paper.
- H1 rule: delta >= +0.05 AND CI excludes 0.

Run from repo root with `data` symlinked to phd_v2/paper1/data.
"""
import json, re, glob
from pathlib import Path
from collections import defaultdict
import numpy as np

SEED = 42
N_BOOT = 1000

def norm(s):
    if not s: return ''
    s = str(s).lower().replace('_',' ')
    s = re.sub(r'\\[a-zA-Z]+\{([^}]*)\}', r'\1', s)
    s = re.sub(r'\\[a-zA-Z]+', ' ', s)
    s = re.sub(r'[\$\^_\{\}]', ' ', s)
    s = re.sub(r'[^a-z0-9 ]', ' ', s)
    return re.sub(r'\s+', ' ', s).strip()

def lenient(pred, gold_set):
    p = norm(pred)
    if not p: return False
    for g in gold_set:
        gg = norm(g)
        if not gg: continue
        if p == gg: return True
        if len(p)>=3 and len(gg)>=3 and (p in gg or gg in p): return True
    return False

def triple_ok(t, gold):
    task, d, m = t
    return lenient(task, gold['tasks']) and lenient(d, gold['datasets']) and lenient(m, gold['metrics'])

def emitted_triples(tuples):
    out = []
    for t in tuples:
        task = (t.get('task') or {}).get('name') or ''
        ds = [(x.get('name') if isinstance(x,dict) else str(x)) for x in t.get('datasets') or []]
        met = [(x.get('name') if isinstance(x,dict) else str(x)) for x in t.get('metrics') or []]
        for d in ds:
            for m in met: out.append((task,d,m))
    return out

def per_paper_metrics(triples, gold_for_paper):
    g = gold_for_paper
    tp = sum(1 for t in triples if triple_ok(t, g))
    matched_gold = sum(1 for gt in g['triples']
                       if any(lenient(pt[0],[gt[0]]) and lenient(pt[1],[gt[1]]) and lenient(pt[2],[gt[2]])
                              for pt in triples))
    prec = tp/len(triples) if triples else 0.0
    rec = matched_gold/len(g['triples']) if g['triples'] else 0.0
    f1 = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0.0
    return prec, rec, f1, tp, len(triples), matched_gold, len(g['triples'])

def load_gold(path='data/metalead/annotations/metalead_annotations.json'):
    raw = json.load(open(path))
    gold = defaultdict(lambda: {'tasks':set(),'datasets':set(),'metrics':set(),'triples':set()})
    for key, entry in raw.items():
        pk = key.replace('.pdf','')
        for tdm in entry.get('TDMs') or []:
            t = norm(tdm.get('Task') or ''); d = norm(tdm.get('Test-Dataset') or ''); m = norm(tdm.get('Metric') or '')
            if t: gold[pk]['tasks'].add(t)
            if d: gold[pk]['datasets'].add(d)
            if m: gold[pk]['metrics'].add(m)
            if t and d and m: gold[pk]['triples'].add((t,d,m))
    return gold

def load_predictions(path_glob, key_pref='tuples'):
    out = {}
    for fp in sorted(glob.glob(path_glob)):
        pth = Path(fp)
        pid = pth.stem.split('_',1)[1] if '_' in pth.stem else pth.stem
        d = json.loads(pth.read_text())
        out[pid] = d.get(key_pref) or d.get('tuples') or d.get('contributions') or []
    return out

def paired_boot(a_vals, b_vals, n_boot=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    a = np.asarray(a_vals); b = np.asarray(b_vals)
    n = len(a)
    if n == 0: return float('nan'), float('nan')
    boot = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot[i] = a[idx].mean() - b[idx].mean()
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))

def evaluate(preds_A, preds_B, gold, name_A='A', name_B='B'):
    paired = sorted(set(preds_A) & set(preds_B))
    rows = []
    for pid in paired:
        aT = emitted_triples(preds_A[pid]); bT = emitted_triples(preds_B[pid])
        g = gold.get(pid, {'tasks':set(),'datasets':set(),'metrics':set(),'triples':set()})
        a_p,a_r,a_f,a_tp,a_ne,a_gc,a_gn = per_paper_metrics(aT, g)
        b_p,b_r,b_f,b_tp,b_ne,b_gc,b_gn = per_paper_metrics(bT, g)
        rows.append({'pid':pid,
                     f'{name_A}_prec':a_p, f'{name_A}_rec':a_r, f'{name_A}_f1':a_f,
                     f'{name_A}_emit':a_ne, f'{name_A}_tp':a_tp,
                     f'{name_B}_prec':b_p, f'{name_B}_rec':b_r, f'{name_B}_f1':b_f,
                     f'{name_B}_emit':b_ne, f'{name_B}_tp':b_tp,
                     'n_gold':a_gn})
    out = {'n_paired': len(rows), 'per_paper': rows, 'name_A': name_A, 'name_B': name_B}
    for m in ['prec','rec','f1']:
        a = [r[f'{name_A}_{m}'] for r in rows]; b = [r[f'{name_B}_{m}'] for r in rows]
        if not a: continue
        am, bm = float(np.mean(a)), float(np.mean(b))
        delta = am - bm
        lo, hi = paired_boot(a, b)
        out[m] = {f'{name_A}_mean':am, f'{name_B}_mean':bm, 'delta':float(delta),
                  'ci_lo':lo,'ci_hi':hi,'ci_excludes_zero':bool(lo>0 or hi<0),
                  'h1_5pp_and_ci_gt_0': bool(delta>=0.05 and lo>0)}
    return out

if __name__ == '__main__':
    gold = load_gold()
    ce = load_predictions('canonextract_build/outputs/eval_runs_metalead_cycle5/metalead_*.json', 'tuples')
    bl = load_predictions('canonextract_build/outputs/baseline_metalead/metalead_*.json', 'contributions')
    print(f'CE={len(ce)}  BL={len(bl)}  gold={len(gold)}')
    res = evaluate(ce, bl, gold, 'CE', 'BL')
    print(f'n_paired={res["n_paired"]}')
    for m in ['prec','rec','f1']:
        r = res[m]
        print(f'{m:>5}: CE={r["CE_mean"]:.4f}  BL={r["BL_mean"]:.4f}  delta={r["delta"]:+.4f}  CI=[{r["ci_lo"]:+.4f},{r["ci_hi"]:+.4f}]  H1={r["h1_5pp_and_ci_gt_0"]}')
    Path('canonextract_build/outputs').mkdir(parents=True, exist_ok=True)
    with open('canonextract_build/outputs/h1_reanalysis_full.json','w') as f:
        json.dump(res, f, indent=2, default=str)
    print('\nWrote canonextract_build/outputs/h1_reanalysis_full.json')
