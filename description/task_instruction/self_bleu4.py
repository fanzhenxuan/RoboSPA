"""Compute Self-BLEU-4 for the 56 tasks defined in RoboTwin/envs.

For each task family (e.g. Click_Bell_Clockwise_Order, which has variants
_1.._5 .py in envs/), we read the 5 corresponding JSON instruction files in
description/task_instruction/, pool their `seen` + `unseen` instructions across
the 5 variants, and compute Self-BLEU-4 over three corpora per task:
  - seen   (50 * 5 = 250 sentences)
  - unseen (10 * 5 =  50 sentences)
  - ALL    (60 * 5 = 300 sentences)

Engine: prefers nltk (matches the single-file version), and falls back to a
built-in pure-Python BLEU-4 if nltk is not installed in the current env.
"""

import json
import math
import os
import re
import sys
from collections import Counter

ENVS_DIR = "/home/fzx/RoboTwin/envs"
INSTR_DIR = "/home/fzx/RoboTwin/description/task_instruction"
OUT_CSV = "/home/fzx/RoboTwin/description/task_instruction/self_bleu4_results.csv"

# ---------------------------------------------------------------------------
# BLEU engine selection
# ---------------------------------------------------------------------------
try:
    from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu

    def bleu4(hyp, refs):
        return corpus_bleu(
            [refs], [hyp],
            weights=(0.25, 0.25, 0.25, 0.25),
            smoothing_function=SmoothingFunction().method1,
        )

    ENGINE = "nltk"
except ModuleNotFoundError:
    def _ngram_counts(tokens, n):
        return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))

    def _mod_precision(hyp, refs, n):
        hc = _ngram_counts(hyp, n)
        if not hc:
            return 0, 0
        maxref = {}
        for r in refs:
            for ng, c in _ngram_counts(r, n).items():
                if c > maxref.get(ng, 0):
                    maxref[ng] = c
        overlap = sum(min(c, maxref.get(ng, 0)) for ng, c in hc.items())
        return overlap, sum(hc.values())

    def bleu4(hyp, refs):
        w = (0.25, 0.25, 0.25, 0.25)
        precs = []
        for n in range(1, 5):
            o, t = _mod_precision(hyp, refs, n)
            if t == 0:
                precs.append(0.0)
            elif o == 0:
                precs.append(sys.float_info.epsilon)
            else:
                precs.append(o / t)
        if min(precs) > 0:
            geo = math.exp(sum(wi * math.log(p) for wi, p in zip(w, precs)))
        else:
            geo = 0.0
        ref = min(refs, key=lambda r: (abs(len(r) - len(hyp)), len(r)))
        if len(hyp) > len(ref):
            bp = 1.0
        elif len(hyp) == 0:
            bp = 0.0
        else:
            bp = math.exp(1 - len(ref) / len(hyp))
        return bp * geo

    ENGINE = "pure-python (nltk not installed)"


def self_bleu4(sentences):
    """Mean BLEU-4 of each sentence against all the others (Self-BLEU-4)."""
    toks = [s.lower().split() for s in sentences]
    if len(toks) < 2:
        return float("nan")
    scores = []
    for i, hyp in enumerate(toks):
        refs = [r for j, r in enumerate(toks) if j != i]
        scores.append(bleu4(hyp, refs))
    return sum(scores) / len(scores)


def discover_tasks():
    """Return the 56 task families that have _1.._5 .py files in envs/."""
    files = set(os.listdir(ENVS_DIR))
    fams = {m.group(1) for f in files if (m := re.match(r'^(.*)_[1-5]\.py$', f))}
    return sorted(f for f in fams if all(f"{{}}_{i}.py".format(f) in files for i in range(1, 6)))


def load_task_instructions(family):
    """Pool seen/unseen across the 5 variant JSONs of a task family."""
    seen, unseen, missing = [], [], []
    for i in range(1, 6):
        path = os.path.join(INSTR_DIR, f"{family}_{i}.json")
        if not os.path.exists(path):
            missing.append(os.path.basename(path))
            continue
        data = json.load(open(path))
        seen += data.get("seen", [])
        unseen += data.get("unseen", [])
    return seen, unseen, missing


def main():
    tasks = discover_tasks()
    print(f"Engine: {ENGINE}")
    print(f"Found {len(tasks)} tasks in {ENVS_DIR}\n")

    rows = []
    header = f"{'Task':<42}{'seen':>9}{'unseen':>9}{'ALL':>9}  notes"
    print(header)
    print("-" * len(header))

    sums = {"seen": 0.0, "unseen": 0.0, "ALL": 0.0}
    counted = {"seen": 0, "unseen": 0, "ALL": 0}

    for fam in tasks:
        seen, unseen, missing = load_task_instructions(fam)
        notes = ""
        if missing:
            notes = f"missing: {','.join(missing)}"
        if len(seen) < 2 and len(unseen) < 2:
            notes = (notes + " | not enough sentences").strip(" |")

        s_seen = self_bleu4(seen) if len(seen) >= 2 else float("nan")
        s_unseen = self_bleu4(unseen) if len(unseen) >= 2 else float("nan")
        s_all = self_bleu4(seen + unseen) if len(seen) + len(unseen) >= 2 else float("nan")

        for key, val in (("seen", s_seen), ("unseen", s_unseen), ("ALL", s_all)):
            if val == val:  # not NaN
                sums[key] += val
                counted[key] += 1

        rows.append((fam, len(seen), len(unseen), s_seen, s_unseen, s_all, notes))
        print(f"{fam:<42}{s_seen:>9.4f}{s_unseen:>9.4f}{s_all:>9.4f}  {notes}")

    print("-" * len(header))
    print(f"{'MEAN over tasks':<42}"
          f"{sums['seen']/counted['seen']:>9.4f}"
          f"{sums['unseen']/counted['unseen']:>9.4f}"
          f"{sums['ALL']/counted['ALL']:>9.4f}")

    # save CSV
    import csv
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task", "n_seen", "n_unseen", "self_bleu4_seen",
                    "self_bleu4_unseen", "self_bleu4_all", "notes"])
        for fam, ns, nu, ss, su, sa, notes in rows:
            w.writerow([fam, ns, nu,
                        f"{ss:.4f}" if ss == ss else "",
                        f"{su:.4f}" if su == su else "",
                        f"{sa:.4f}" if sa == sa else "",
                        notes])
    print(f"\nSaved per-task results -> {OUT_CSV}")


if __name__ == "__main__":
    main()
