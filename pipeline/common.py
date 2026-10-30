"""Shared code for canonextract v2 diagnostic.

Expanded anchor extraction (captions + figure captions + table-ref
neighborhoods) and the same normalize + variant matching from v1.
"""

from __future__ import annotations

import re
from pathlib import Path

# ---------- normalization (matches v1) ----------
_PUNCT_RE = re.compile(r"[^\w\s]")
_WS_RE = re.compile(r"\s+")


def normalize(s: str | None) -> str:
    if not s:
        return ""
    s = str(s).lower().replace("_", " ")
    s = _PUNCT_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    return s


METRIC_SYNONYMS = {
    "perplexity": ["ppl"],
    "accuracy": ["acc"],
    "top 1 accuracy": ["top-1 accuracy", "top1 accuracy", "top-1 acc", "top1 acc", "top-1", "top1"],
    "top 5 accuracy": ["top-5 accuracy", "top5 accuracy", "top-5 acc", "top5 acc", "top-5", "top5"],
    "map": ["mean average precision"],
    "mean average precision": ["map", "mean ap"],
    "error rate": ["err", "err rate", "err.", "error"],
    "wer": ["word error rate"],
    "bleu": ["bleu score", "bleu-4"],
    "rouge": ["rouge-l", "rouge score"],
    "f1": ["f-1", "f measure", "f-measure", "f1 score"],
    "iou": ["intersection over union"],
    "miou": ["mean iou", "mean intersection over union"],
}


def metric_variants(metric_norm: str) -> set[str]:
    variants = {metric_norm}
    if metric_norm in METRIC_SYNONYMS:
        variants.update(normalize(v) for v in METRIC_SYNONYMS[metric_norm])
    for canonical, syns in METRIC_SYNONYMS.items():
        norm_canonical = normalize(canonical)
        norm_syns = [normalize(s) for s in syns]
        if metric_norm == norm_canonical or metric_norm in norm_syns:
            variants.add(norm_canonical)
            variants.update(norm_syns)
    return {v for v in variants if v}


def name_matches_field(name_norm: str, corpus_norm: set[str]) -> bool:
    if not name_norm:
        return False
    for c in corpus_norm:
        if not c:
            continue
        if name_norm in c or c in name_norm:
            return True
    return False


def metric_matches_field(metric_name: str, corpus_norm: set[str]) -> bool:
    if not metric_name:
        return False
    variants = metric_variants(normalize(metric_name))
    for v in variants:
        if name_matches_field(v, corpus_norm):
            return True
    return False


# ---------- expanded-anchor extraction from tex ----------
def _strip_tex_cmds(s: str) -> str:
    """Remove common inline LaTeX commands but keep argument text."""
    s = re.sub(r"\\(?:textbf|textit|emph|texttt|mathbf|mathrm|mathit|mathsf|underline)\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"\\multicolumn\{[^}]*\}\{[^}]*\}\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"\\multirow\{[^}]*\}\{[^}]*\}\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"\$([^$]*)\$", r"\1", s)
    s = re.sub(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?", " ", s)
    s = s.replace("{", " ").replace("}", " ").replace("~", " ").replace("&nbsp;", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _strip_tex_comments(tex: str) -> str:
    """Strip % comments from tex, respecting escaped \\%."""
    out_lines = []
    for line in tex.splitlines():
        i = 0
        buf = []
        while i < len(line):
            if line[i] == "%" and (i == 0 or line[i - 1] != "\\"):
                break
            buf.append(line[i])
            i += 1
        out_lines.append("".join(buf))
    return "\n".join(out_lines)


def concat_tex_files(tex_dir: Path) -> str:
    """Concatenate all .tex files under a directory + strip comments."""
    tex_files = list(tex_dir.glob("**/*.tex"))
    if not tex_files:
        return ""
    combined = "\n".join(f.read_text(encoding="utf-8", errors="ignore") for f in tex_files)
    return _strip_tex_comments(combined)


def _brace_matched_arg(text: str, start_after_cmd: int) -> tuple[str, int]:
    """After the LaTeX command name has been consumed, skip optional [foo] and
    return the {balanced-brace} argument's contents plus the index just after
    the closing brace. Returns ('', start_after_cmd) if no argument found.
    """
    i = start_after_cmd
    n = len(text)
    # Skip optional [short caption]
    while i < n and text[i].isspace():
        i += 1
    if i < n and text[i] == "[":
        depth = 1
        i += 1
        while i < n and depth > 0:
            if text[i] == "[":
                depth += 1
            elif text[i] == "]":
                depth -= 1
            i += 1
    # Skip whitespace
    while i < n and text[i].isspace():
        i += 1
    if i >= n or text[i] != "{":
        return "", start_after_cmd
    # Brace match
    depth = 1
    i += 1
    start = i
    while i < n and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start:i], i + 1
        i += 1
    return "", start_after_cmd


def _find_env_blocks(tex: str, env_name: str) -> list[str]:
    """Return the bodies of all \\begin{env}...\\end{env} blocks (env or env*)."""
    pat = re.compile(rf"\\begin\{{{env_name}\*?\}}(.*?)\\end\{{{env_name}\*?\}}", re.DOTALL)
    return [m.group(1) for m in pat.finditer(tex)]


def extract_captions_in_env(tex: str, env_name: str) -> list[str]:
    """Extract every \\caption{...} argument from every {env} block."""
    out: list[str] = []
    blocks = _find_env_blocks(tex, env_name)
    for block in blocks:
        # Find \caption occurrences within this block (there can be multiple)
        i = 0
        n = len(block)
        while i < n:
            j = block.find(r"\caption", i)
            if j == -1:
                break
            after = j + len(r"\caption")
            arg, next_i = _brace_matched_arg(block, after)
            if arg:
                out.append(_strip_tex_cmds(arg))
            i = max(next_i, j + 1)
    return out


# Precompile patterns for finding table/figure references in normalized text
_REF_TAB_PAT = re.compile(r"\\(?:c?ref|Cref|autoref|eqref)\{(?:tab|table)[:_-][^}]+\}", re.IGNORECASE)
_TABLE_N_PAT = re.compile(r"\bTable\s+\d+", re.IGNORECASE)
_FIG_N_PAT = re.compile(r"\bFig(?:ure)?\.?\s+\d+", re.IGNORECASE)
_TAB_SHORT_PAT = re.compile(r"\bTab\.?\s+\d+", re.IGNORECASE)


def extract_reference_neighborhoods(tex: str, window: int = 20) -> list[str]:
    """For every table/figure reference in the (comment-stripped) tex,
    normalize a wider text neighborhood (window words on each side) and return
    those neighborhood strings.
    """
    # First, normalize the whole tex text once
    plain = _strip_tex_cmds(tex)
    words = plain.split()
    plain_lc = plain.lower()

    # Find candidate anchor positions in plain_lc
    anchors: list[tuple[int, int]] = []  # (start, end) char positions
    for pat in (_TABLE_N_PAT, _FIG_N_PAT, _TAB_SHORT_PAT):
        for m in pat.finditer(plain):
            anchors.append((m.start(), m.end()))
    # \ref{...} anchors don't survive strip; run on the raw tex
    for m in _REF_TAB_PAT.finditer(tex):
        # Map back is hard; instead, replace-tokens and look for the substring
        anchors.append((m.start(), m.end()))
    if not anchors:
        return []

    # Convert plain to a word-indexed structure for windowing
    # Build a char->word_index map
    word_starts: list[int] = []
    pos = 0
    for w in words:
        while pos < len(plain) and plain[pos].isspace():
            pos += 1
        word_starts.append(pos)
        pos += len(w)

    def _word_idx(char_pos: int) -> int:
        # binary search word_starts
        lo, hi = 0, len(word_starts)
        while lo < hi:
            mid = (lo + hi) // 2
            if word_starts[mid] <= char_pos:
                lo = mid + 1
            else:
                hi = mid
        return max(0, lo - 1)

    # For anchors originating in the RAW tex (\ref{...} matches), we can't map
    # char positions to plain-text word indices. As a proxy, extract the label
    # (e.g. "tab:res_summ") and search for it in the strip results — the label
    # itself often paraphrases the dataset/method. Instead we just include the
    # label token in a neighborhood-like set:
    ref_label_neighborhoods = []
    for m in _REF_TAB_PAT.finditer(tex):
        label = m.group(0)
        # Extract the {tab:...} label content
        lm = re.search(r"\{([^}]+)\}", label)
        if lm:
            lbl = lm.group(1)
            # Split on : _ - / to get semantic tokens
            tokens = re.split(r"[:_\-/]", lbl)
            ref_label_neighborhoods.append(" ".join(t for t in tokens if t))

    out: list[str] = []
    for start, end in anchors:
        widx_lo = _word_idx(start)
        widx_hi = _word_idx(end - 1) + 1
        lo = max(0, widx_lo - window)
        hi = min(len(words), widx_hi + window)
        neighborhood = " ".join(words[lo:hi])
        out.append(neighborhood)
    out.extend(ref_label_neighborhoods)
    return out


def extract_expanded_anchors(tex_dir: Path) -> dict:
    """Return dict with keys: table_captions, figure_captions, table_neighborhoods
    (each a list of raw strings).
    """
    tex = concat_tex_files(tex_dir)
    if not tex:
        return {"table_captions": [], "figure_captions": [], "table_neighborhoods": []}
    return {
        "table_captions": extract_captions_in_env(tex, "table"),
        "figure_captions": extract_captions_in_env(tex, "figure"),
        "table_neighborhoods": extract_reference_neighborhoods(tex, window=20),
    }


# ---------- analysis ----------
def compute_flags(
    v1_row: list[str],
    v1_col: list[str],
    v1_cells: list[str],
    v2_captions: list[str],
    v2_fig_captions: list[str],
    v2_neighborhoods: list[str],
    gold_task: str, gold_material: str, gold_metric: str, gold_method: str,
) -> dict:
    """Return the expanded booleans + per-source contribution flags."""
    v1_row_n = {normalize(x) for x in v1_row if x}
    v1_col_n = {normalize(x) for x in v1_col if x}
    v1_cells_n = {normalize(x) for x in v1_cells if x}
    v1_all = v1_row_n | v1_col_n | v1_cells_n

    cap_n = {normalize(x) for x in v2_captions if x}
    fig_cap_n = {normalize(x) for x in v2_fig_captions if x}
    nbr_n = {normalize(x) for x in v2_neighborhoods if x}

    expanded_all = v1_all | cap_n | fig_cap_n | nbr_n

    metric = gold_metric or ""
    material = gold_material or ""
    task = gold_task or ""
    method = gold_method or ""

    # v1 booleans (for delta)
    v1_metric_col = metric_matches_field(metric, v1_col_n)
    v1_metric_any = metric_matches_field(metric, v1_all)
    v1_dataset_row = name_matches_field(normalize(material), v1_row_n)
    v1_dataset_any = name_matches_field(normalize(material), v1_all)
    v1_both = v1_metric_any and v1_dataset_any

    # v2 (expanded) booleans
    v2_metric_col = v1_metric_col  # column header set doesn't expand
    v2_metric_any = metric_matches_field(metric, expanded_all)
    v2_dataset_row = v1_dataset_row  # row label set doesn't expand
    v2_dataset_any = name_matches_field(normalize(material), expanded_all)
    v2_both = v2_metric_any and v2_dataset_any

    # Per-source contribution — did adding captions / fig_captions / neighborhoods
    # turn a v1 miss into a v2 hit?
    def _source_that_added(kind: str) -> list[str]:
        """Return which new source(s) surfaced a v1-miss."""
        added = []
        target_norm = normalize(material) if kind == "dataset" else None
        variants = metric_variants(normalize(metric)) if kind == "metric" else None
        for name, corpus in (("captions", cap_n), ("figure_captions", fig_cap_n),
                             ("neighborhoods", nbr_n)):
            hit = False
            if kind == "dataset":
                hit = name_matches_field(target_norm, corpus)
            else:
                hit = any(name_matches_field(v, corpus) for v in variants)
            if hit:
                added.append(name)
        return added

    dataset_added_by = _source_that_added("dataset") if (v2_dataset_any and not v1_dataset_any) else []
    metric_added_by = _source_that_added("metric") if (v2_metric_any and not v1_metric_any) else []

    task_in_expanded = name_matches_field(normalize(task), expanded_all)
    method_in_expanded = name_matches_field(normalize(method), expanded_all)

    return {
        "Task": task, "Material": material, "Metric": metric, "Method": method,
        # v1 (for delta reporting)
        "v1_metric_in_column_header": v1_metric_col,
        "v1_metric_in_any_table_field": v1_metric_any,
        "v1_dataset_in_row_label": v1_dataset_row,
        "v1_dataset_in_any_table_field": v1_dataset_any,
        "v1_both_in_tables": v1_both,
        # v2 (expanded)
        "v2_metric_in_column_header": v2_metric_col,
        "v2_metric_in_expanded": v2_metric_any,
        "v2_dataset_in_row_label": v2_dataset_row,
        "v2_dataset_in_expanded": v2_dataset_any,
        "v2_both_in_expanded": v2_both,
        # per-source contribution
        "dataset_gained_by": dataset_added_by,
        "metric_gained_by": metric_added_by,
        # aux fields
        "task_in_expanded": task_in_expanded,
        "method_in_expanded": method_in_expanded,
    }
