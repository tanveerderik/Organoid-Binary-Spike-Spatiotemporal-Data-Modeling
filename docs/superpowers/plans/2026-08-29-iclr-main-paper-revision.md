# ICLR 2027 main-paper revision — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bring the scientific body (§1–§7) within the 9-page ICLR limit, fix the Figure 4 layout defect, remove the reviewer-dialogue prose cadence, and add the missing adjacent spike-generation citations — without changing a single number, claim, or negative result.

**Architecture:** Every change is either (a) prose compression, (b) relocation of detail into an appendix section that already exists, (c) a figure-generator fix, or (d) a `.bib` addition. Numbers are never typed by hand: they enter the manuscript through `\input`-ed generated tables and through macros, and `tools/check_paper_numbers.py` is the gate that proves nothing was invented. A new `tools/check_page_budget.py` makes "is the body ≤ 9 pages" a command rather than a judgement call, and it is the acceptance test for Tasks 4–8.

**Tech Stack:** LaTeX (Tectonic at `/home/derik/.local/bin/tectonic`), `pdftotext` for page measurement, Python 3 (`/home/derik/anaconda3/envs/pytorch/bin/python`), matplotlib for the figure generators.

**Spec:** `/home/derik/.claude/uploads/61456d8c-5e42-4e36-9c7a-c042019c2bbd/8e33a57d-iclr_main_paper_revision_suggestions.md`

## Global Constraints

- **Never hand-edit a number.** Every figure in the manuscript comes from a generated table (`paper/tables/*.tex`, written by `tools/make_paper_tables.py`) or a macro (`\NAssays`, `\OracleOurs`, `\CeilingFactor`, `\VoxelRate`, `\ClipFrames`, `\SpikesPerClip`, `\ClipVoxels`, `\ChanMin`, `\ChanMax`, `\NTestClips`, `\OracleFlat`, `\OracleOursSite`, `\OracleFlatSite`). If a number must change, regenerate the artifact.
- **`tools/check_paper_numbers.py` must pass at the end of every task.** Current baseline: `183 numbers checked, 0 unaccounted for (78 structural constants exempt)`.
- **Known gate weakness:** the checker verifies each printed number traces to *some* JSON, not that it matches the *current* value of the field it claims. It passed while a superseded caption was in place. Treat it as a floor, not a proof.
- **Do not change** the interpretation of site AP versus voxel AP; do not present the static site map, DG or GLM as parameter-matched peers; do not imply unseen-preparation generalisation; do not soften the within-token timing weakness; do not turn a failed appendix experiment into a successful ablation; do not claim priority over neural spike generation in general — the defensible novelty is the array-wide HD-MEA binary-volume / shared discrete motif / conditional generative setting.
- **Figures may not be shrunk** to solve the page limit. The motif thumbnails in F2 are already at the legibility limit.
- **The scientific body is §1–§7.** `paper/main.tex:80` records that the Ethics, Reproducibility and Use-of-AI statements sit outside the limit. Task 1 confirms this against the official CFP before any compression is done on its strength.
- Build command used throughout: `cd paper && tectonic -X compile main.tex --outdir <dir>`. Working dir for everything else is the repo root, `/media/derik/Seagate Desktop Drive/organoid_data/MAGVIT_project`. `PYTHONPATH=/media/derik/Seagate\ Desktop\ Drive/organoid_data`.

## Starting state (measured 2026-08-29)

| section | source lines |
|---|---|
| `01_introduction.tex` | 67 |
| `02_related.tex` | 56 |
| `03_data.tex` | 53 |
| `04_method.tex` | 118 |
| `05_experiments.tex` | 222 |
| `06_limitations.tex` | 46 |
| `07_conclusion.tex` | 16 |
| `99_appendix.tex` | 802 |

Compiled: 26 pages. §1–§7 ends on **page 10**; the overflow is **4 typeset lines** — the Conclusion's closing sentence. References start p11. Abstract is ~267 words.

## File structure

- `tools/check_page_budget.py` — **new.** Compiles the paper and asserts the body ends by page 9. The acceptance test for every compression task.
- `tools/tests/test_check_page_budget.py` — **new.** Unit test for the page-parsing logic, so the gate itself is trustworthy.
- `tools/paper_figs/f4_generation.py` — modify. Fix the legend/annotation collision.
- `paper/main.tex` — modify. Abstract compression; `hyperref` link styling.
- `paper/refs.bib` — modify. Add LDNS, Spike-GAN, SpikeProphecy.
- `paper/sections/01_introduction.tex`, `02_related.tex`, `05_experiments.tex`, `06_limitations.tex`, `07_conclusion.tex` — modify. Compression and de-rhetoricisation.
- `paper/sections/99_appendix.tex` — modify. Receives relocated detail.

---

## Task 1: A page-budget gate that can be run, not eyeballed

Every later task claims "this saved N lines." Without a command that answers "is the body within 9 pages," those claims are unverifiable and the last four lines will be argued about repeatedly. Build the gate first.

**Files:**
- Create: `tools/check_page_budget.py`
- Create: `tools/tests/test_check_page_budget.py`

**Interfaces:**
- Produces: `body_end_page(pdf_text: str) -> int` — given the `pdftotext` output of the manuscript, returns the 1-based page on which the scientific body ends (the page before the one where References begins). `main(argv) -> int` returns 0 when the body ends on or before page 9, 1 otherwise.

- [ ] **Step 1: Confirm the page-counting rule against the official CFP**

`paper/main.tex:80` carries the comment `% --- Statements. None of these count toward the 9-page limit. ---`. Confirm this against the ICLR 2027 call for papers before relying on it. If the CFP says otherwise, stop and report: the whole compression target changes from 4 lines to roughly 40, and the plan below needs re-scoping.

Record the finding as a comment in `tools/check_page_budget.py` (Step 3 includes the line).

- [ ] **Step 2: Write the failing test**

```python
# tools/tests/test_check_page_budget.py
"""The page-budget gate has to be trustworthy before anything is compressed on
its say-so, so its page-finding logic is tested against synthetic documents
rather than only against the real manuscript."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from check_page_budget import body_end_page

FF = "\f"


def _doc(pages):
    return FF.join(pages)


def test_body_ends_on_page_before_references():
    doc = _doc(["intro", "method", "results", "R EFERENCES\nsmith et al"])
    assert body_end_page(doc) == 3


def test_statements_between_body_and_references_do_not_count():
    doc = _doc(["intro", "conclusion", "E THICS STATEMENT\ntext",
                "R EFERENCES\nsmith et al"])
    assert body_end_page(doc) == 2


def test_letterspaced_references_heading_is_found():
    doc = _doc(["a", "b", "R E F E R E N C E S"])
    assert body_end_page(doc) == 2


def test_missing_references_raises():
    try:
        body_end_page(_doc(["a", "b"]))
    except ValueError:
        return
    raise AssertionError("expected ValueError when References is absent")
```

- [ ] **Step 3: Run it to verify it fails**

```bash
cd "/media/derik/Seagate Desktop Drive/organoid_data/MAGVIT_project"
/home/derik/anaconda3/envs/pytorch/bin/python -m pytest tools/tests/test_check_page_budget.py -v
```

Expected: FAIL, `ModuleNotFoundError: No module named 'check_page_budget'`.

- [ ] **Step 4: Write the gate**

```python
#!/usr/bin/env python3
"""Assert the scientific body fits the ICLR page limit.

The body is Sections 1-7. The Ethics, Reproducibility and Use-of-AI statements
sit between the Conclusion and the References and do NOT count toward the limit
(paper/main.tex records this; confirmed against the ICLR 2027 CFP on
2026-08-29). So the body ends on the page before whichever page the References
heading first appears on, minus any pages occupied solely by statements.

Counting by hand from a rendered PDF is what produced the earlier, wrong claim
that the paper was already at nine pages. This is the only measurement that
should be quoted.

    python tools/check_page_budget.py            # compile and check
    python tools/check_page_budget.py --pdf X    # check an existing PDF
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
LIMIT = 9

# pdftotext letterspaces small-caps headings, so "REFERENCES" can arrive as
# "R EFERENCES" or "R E F E R E N C E S". Match the letters with optional gaps.
_REFS = re.compile(r"R\s*E\s*F\s*E\s*R\s*E\s*N\s*C\s*E\s*S", re.I)
_STMT = re.compile(
    r"(E\s*THICS\s+S\s*TATEMENT|R\s*EPRODUCIBILITY\s+S\s*TATEMENT|"
    r"U\s*SE\s+OF\s+AI\s+S\s*TATEMENT)", re.I)


def body_end_page(pdf_text: str) -> int:
    """1-based page on which Sections 1-7 end."""
    pages = pdf_text.split("\f")
    refs_page = None
    for i, page in enumerate(pages, start=1):
        if _REFS.search(" ".join(page.split())):
            refs_page = i
            break
    if refs_page is None:
        raise ValueError("no References heading found in the PDF text")

    # Walk back over pages that carry only statements, not body prose.
    page = refs_page - 1
    while page >= 1:
        text = " ".join(pages[page - 1].split())
        if not _STMT.search(text):
            break
        # A page that opens with a statement heading holds no body text.
        head = text[:200]
        if _STMT.search(head):
            page -= 1
            continue
        break
    return page


def _compile(outdir: Path) -> Path:
    tectonic = shutil.which("tectonic") or "/home/derik/.local/bin/tectonic"
    subprocess.run([tectonic, "-X", "compile", "main.tex", "--outdir", str(outdir)],
                   cwd=PAPER, check=True, capture_output=True)
    return outdir / "main.pdf"


def _text(pdf: Path) -> str:
    out = pdf.with_suffix(".txt")
    subprocess.run(["pdftotext", str(pdf), str(out)], check=True, capture_output=True)
    return out.read_text(encoding="utf-8", errors="replace")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", default=None, help="check this PDF instead of compiling")
    args = ap.parse_args(argv)

    with tempfile.TemporaryDirectory() as tmp:
        pdf = Path(args.pdf) if args.pdf else _compile(Path(tmp))
        end = body_end_page(_text(pdf))

    over = end - LIMIT
    print(f"scientific body (Sections 1-7) ends on page {end}; limit is {LIMIT}")
    if over > 0:
        print(f"OVER by {over} page(s)")
        return 1
    print("within the limit")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 5: Run the unit test to verify it passes**

```bash
/home/derik/anaconda3/envs/pytorch/bin/python -m pytest tools/tests/test_check_page_budget.py -v
```

Expected: 4 passed.

- [ ] **Step 6: Run the gate against the real manuscript**

```bash
cd "/media/derik/Seagate Desktop Drive/organoid_data/MAGVIT_project"
PYTHONPATH="/media/derik/Seagate Desktop Drive/organoid_data" \
  /home/derik/anaconda3/envs/pytorch/bin/python tools/check_page_budget.py
```

Expected: `ends on page 10 ... OVER by 1 page(s)`, exit 1. That non-zero exit is the baseline every later task works against.

- [ ] **Step 7: Commit**

```bash
git add tools/check_page_budget.py tools/tests/test_check_page_budget.py
git commit -m "Make the page budget a command rather than a judgement call"
```

---

## Task 2: Fix the Figure 4 legend collision

The review calls this "the clearest current visual defect." It is independent of every compression task, so it can land first and be verified by eye once.

**Files:**
- Modify: `tools/paper_figs/f4_generation.py:87-93`

The cause: `fig.legend(..., loc="upper center")` places the legend just above the axes, and `fig.text(0.5, 1.13, ...)` places the direction annotation at figure-fraction 1.13 — the two occupy overlapping bands above the panels.

- [ ] **Step 1: Reproduce the defect**

```bash
cd "/media/derik/Seagate Desktop Drive/organoid_data/MAGVIT_project"
PYTHONPATH="/media/derik/Seagate Desktop Drive/organoid_data" \
  /home/derik/anaconda3/envs/pytorch/bin/python -c "
from tools.paper_figs import f4_generation as f
f.main()
print('rendered')"
```

Open `reports/paper_figures/f4_generation.pdf` and confirm the annotation text and the legend overlap.

- [ ] **Step 2: Give the legend and the annotation separate bands**

Replace the legend/annotation block with an explicit two-band layout: reserve headroom with `subplots_adjust(top=...)`, then place the annotation above the legend in figure coordinates that cannot collide.

```python
    fig.subplots_adjust(wspace=0.42, top=0.80)
    # The annotation and the legend previously both floated above the axes and
    # overlapped. Give each its own band in figure coordinates, and reserve the
    # headroom for them with `top` above rather than letting them spill past 1.0.
    fig.legend(handles=handles, loc="upper center", ncol=2,
               bbox_to_anchor=(0.5, 0.90), frameon=False)
    fig.text(0.5, 0.975,
             "rightward is better in every panel; the connector is "
             "each arm's random-context null",
             ha="center", va="top", fontsize=7)
```

Keep the existing `handles` construction and the existing annotation wording; only the placement changes.

- [ ] **Step 3: Re-render and inspect**

```bash
PYTHONPATH="/media/derik/Seagate Desktop Drive/organoid_data" \
  /home/derik/anaconda3/envs/pytorch/bin/python tools/make_paper_figures.py 2>&1 | tail -3
```

Open `reports/paper_figures/f4_generation.pdf`. Confirm: no overlap; the annotation sits above the legend; no panel is clipped; the four family panels are unchanged in size.

- [ ] **Step 4: Confirm nothing else moved**

```bash
PYTHONPATH="/media/derik/Seagate Desktop Drive/organoid_data" \
  /home/derik/anaconda3/envs/pytorch/bin/python tools/check_paper_numbers.py
```

Expected: `183 numbers checked, 0 unaccounted for`.

- [ ] **Step 5: Commit**

```bash
git add tools/paper_figs/f4_generation.py paper/figures/f4_generation.pdf reports/paper_figures/f4_generation.pdf
git commit -m "Stop the F4 legend and direction annotation overlapping"
```

---

## Task 3: Neutralise the reviewer-dialogue cadence

The review is explicit that this should be surgical, not a rewrite: "A broad rewrite risks making the prose more generic." Every replacement below is given verbatim by the spec, so this task is mechanical and carries no interpretive risk.

**Files:**
- Modify: `paper/sections/05_experiments.tex`, `paper/sections/06_limitations.tex`, `paper/sections/99_appendix.tex`

- [ ] **Step 1: Write the failing check**

```python
# tools/tests/test_no_rhetorical_phrasing.py
"""The reviewer flagged a rebuttal-like cadence. These are the specific strings
called out; this test keeps them from coming back in a later edit."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SECTIONS = sorted((ROOT / "paper" / "sections").glob("*.tex"))

BANNED = [
    "The interpretation is one sentence",
    "What ranks above us",
    "What is ours is",
    "This is the largest gap in the paper",
    "should read the site-map row as the honest state of the art",
    "Rarefaction is not optional",
    "Beating uniform is worth nothing",
    "deliberately not hiding",
    "Measure at the decoder",
]


def test_no_banned_phrases():
    hits = []
    for f in SECTIONS:
        text = f.read_text()
        for phrase in BANNED:
            if phrase in text:
                hits.append(f"{f.name}: {phrase!r}")
    assert not hits, "rhetorical phrasing still present:\n  " + "\n  ".join(hits)
```

- [ ] **Step 2: Run it to verify it fails**

```bash
/home/derik/anaconda3/envs/pytorch/bin/python -m pytest tools/tests/test_no_rhetorical_phrasing.py -v
```

Expected: FAIL, listing the phrases still present.

- [ ] **Step 3: Apply the replacements**

Each is meaning-preserving. Apply verbatim.

In `05_experiments.tex`:

| replace | with |
|---|---|
| `The interpretation is one sentence: the model improves`<br>`\emph{which electrodes participate}, while electrode-by-frame timing remains`<br>`hard.` | `The improvement is therefore concentrated at the site level, while`<br>`electrode-by-frame timing remains difficult.` |
| `\paragraph{What ranks above us.}`<br>`Across the two metric families and four settings, $23$ of the $32$ paired`<br>`comparisons go to another arm` | `Across the two metric families and four settings, other methods outperform`<br>`ours in $23$ of the $32$ paired comparisons` |
| `What is ours is per-clip discrimination: within a recording our correlation` | `Our model instead shows stronger per-clip discrimination: within a recording`<br>`our correlation` |
| `Rarefaction is not optional: raw overlap tracks recording length and inverts`<br>`the ordering.` | `Rarefaction is necessary because raw overlap is strongly confounded by`<br>`recording length, which inverts the ordering.` |

In `06_limitations.tex`:

| replace | with |
|---|---|
| `This is the largest gap in the paper. The static map is very strong for` | `The static site map therefore remains the strongest spatial reference. It is`<br>`very strong for` |
| `A reader who values per-clip accuracy over`<br>`scalability should read the site-map row as the honest state of the art.` | `For within-recording spatial prediction it remains the strongest reference`<br>`despite its recording-specific storage cost.` |

In `99_appendix.tex`:

| replace | with |
|---|---|
| `Two things this table is deliberately not hiding.` | `The table also shows two important features of checkpoint variability.` |
| `Measure at the decoder.` | `The effect therefore needs to be evaluated at the decoder output rather than`<br>`through a summary statistic alone.` |
| `Beating uniform is worth nothing` | `Uniform is an intentionally weak reference; the relevant comparison is the`<br>`strongest per-recording $\times$ position lookup` |

Note the second row of the `05_experiments.tex` table removes a `\paragraph` mini-heading. The paragraph that followed it becomes a continuation of the preceding text — check the surrounding blank lines so the paragraph break is where you intend.

If a `BANNED` string does not appear verbatim (wording drifted in an earlier session), find the sentence it refers to and apply the same transformation; then update the test's `BANNED` entry to the string that was actually there.

- [ ] **Step 4: Run the check to verify it passes**

```bash
/home/derik/anaconda3/envs/pytorch/bin/python -m pytest tools/tests/test_no_rhetorical_phrasing.py -v
```

Expected: 1 passed.

- [ ] **Step 5: Compile and confirm nothing broke**

```bash
cd paper && tectonic -X compile main.tex --outdir /tmp/pagechk 2>&1 | grep -ci "^error"
cd .. && PYTHONPATH="/media/derik/Seagate Desktop Drive/organoid_data" \
  /home/derik/anaconda3/envs/pytorch/bin/python tools/check_paper_numbers.py
```

Expected: `0` errors; `183 numbers checked, 0 unaccounted for`.

- [ ] **Step 6: Commit**

```bash
git add paper/sections tools/tests/test_no_rhetorical_phrasing.py
git commit -m "Replace the rebuttal cadence with ordinary scientific prose"
```

---

## Task 4: Shorten the Abstract by 30–50 words

Currently ~267 words. The spec names exactly what to keep and what may go.

**Files:**
- Modify: `paper/main.tex` (the `abstract` environment)

- [ ] **Step 1: Record the baseline**

```bash
cd "/media/derik/Seagate Desktop Drive/organoid_data/MAGVIT_project"
/home/derik/anaconda3/envs/pytorch/bin/python -c "
import re
s=open('paper/main.tex').read()
m=re.search(r'\\\\begin{abstract}(.*?)\\\\end{abstract}', s, re.S)
t=re.sub(r'\\\\[a-zA-Z]+\*?(\[[^]]*\])?({[^}]*})?',' ',m.group(1))
print(len(t.split()),'words')"
```

- [ ] **Step 2: Cut the two implementation specifics the spec nominates**

Keep, at full strength: the sparsity statistic and variable-routing motivation; the three-level motif alphabet as one short clause; the `\CeilingFactor` oracle-code comparison; the $1.4$–$2.6\times$ site-level improvement over the matched peer; the within-token timing negative result.

Remove from the Abstract only: the exact `(6,15,14)` patch shape and the exact deduplicated vocabulary size. Both are in §4 and Figure 1, one page later. Do not remove any other number. Do not reword the claims.

- [ ] **Step 3: Verify the reduction and that no claim left with it**

Re-run the Step 1 counter. Expected: 217–237 words. Then read the new Abstract against the keep-list above and confirm all five items are still present.

- [ ] **Step 4: Compile, check numbers, check the budget**

```bash
PYTHONPATH="/media/derik/Seagate Desktop Drive/organoid_data" \
  /home/derik/anaconda3/envs/pytorch/bin/python tools/check_paper_numbers.py
PYTHONPATH="/media/derik/Seagate Desktop Drive/organoid_data" \
  /home/derik/anaconda3/envs/pytorch/bin/python tools/check_page_budget.py
```

The number count will drop by however many numbers left the Abstract; `0 unaccounted for` is the part that must hold.

- [ ] **Step 5: Commit**

```bash
git add paper/main.tex
git commit -m "Trim the abstract to its claims"
```

---

## Task 5: Compress Introduction and Contributions

Target from the spec: 0.25–0.4 page. The named redundancies are the sparsity of the data, the per-recording routing, the shared alphabet, the absence of a learned per-recording table, and why that matters for scalability — each stated in the Introduction, again in the Contributions, and again in Method or Limitations.

**Files:**
- Modify: `paper/sections/01_introduction.tex` (67 lines)

- [ ] **Step 1: Locate each duplicated argument**

```bash
cd "/media/derik/Seagate Desktop Drive/organoid_data/MAGVIT_project"
grep -n "per-recording\|sparse\|sparsity\|routed\|shared" paper/sections/01_introduction.tex
grep -n "per-recording\|sparse\|sparsity\|routed\|shared" paper/sections/04_method.tex paper/sections/06_limitations.tex | head
```

For each of the five arguments, decide which single location states it at full strength. The Introduction should keep the *motivation* framing; Method and Limitations keep the *mechanism* and the *concession*.

- [ ] **Step 2: Shorten the three Contributions bullets**

Per the spec: "Keep the contribution and headline evidence; move interpretation to Results." Each bullet currently states the contribution, its headline number, and an interpretation of why it matters. Delete only the third element from each — the interpretation is already made in §5.

Retain in bullet 1 the `\CeilingFactor` comparison; in bullet 2 the $9\%$ entropy figure and `\NAssays`; in bullet 3 the naming of the peer, the convolutional references and the lookup references. Do not remove a citation.

- [ ] **Step 3: Verify the saving**

```bash
wc -l paper/sections/01_introduction.tex   # expect ~50-56, from 67
PYTHONPATH="/media/derik/Seagate Desktop Drive/organoid_data" \
  /home/derik/anaconda3/envs/pytorch/bin/python tools/check_page_budget.py
```

- [ ] **Step 4: Confirm no claim was lost**

Re-read §1 against the spec's §2 keep-list ("Main-paper content that should remain"). Every item that lives in the Introduction must still be there: the extreme-sparsity and variable-routing motivation, and the statement that the seeded recording identifier does not support unseen-preparation generalisation if the Introduction is where it appears.

- [ ] **Step 5: Commit**

```bash
git add paper/sections/01_introduction.tex
git commit -m "State each motivating argument once, at full strength"
```

---

## Task 6: Compact Related Work, and add the adjacent spike-generation citations

`paper/refs.bib` currently has `lfads` and `ndt`; it has no LDNS, Spike-GAN, SpikeProphecy or Neural Latents Benchmark entry. The spec requires these be added **without growing the section** — by replacing explanatory prose, not appending.

**Files:**
- Modify: `paper/refs.bib`, `paper/sections/02_related.tex` (56 lines)

- [ ] **Step 1: Write the failing check**

```python
# tools/tests/test_related_work_citations.py
"""The revision requires the adjacent spike-generation literature to be cited in
Related Work, and requires the section not to grow while doing it."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RELATED = ROOT / "paper" / "sections" / "02_related.tex"
BIB = ROOT / "paper" / "refs.bib"

REQUIRED_KEYS = ["kapoor2024ldns", "molano2018spikegan"]
MAX_LINES = 56   # the section may not grow


def test_bib_has_the_adjacent_generative_work():
    bib = BIB.read_text()
    missing = [k for k in REQUIRED_KEYS if f"{{{k}," not in bib]
    assert not missing, f"missing bib entries: {missing}"


def test_related_work_cites_them():
    text = RELATED.read_text()
    missing = [k for k in REQUIRED_KEYS if k not in text]
    assert not missing, f"not cited in Related Work: {missing}"


def test_related_work_did_not_grow():
    n = len(RELATED.read_text().splitlines())
    assert n <= MAX_LINES, f"Related Work grew to {n} lines (cap {MAX_LINES})"
```

- [ ] **Step 2: Run it to verify it fails**

```bash
/home/derik/anaconda3/envs/pytorch/bin/python -m pytest tools/tests/test_related_work_citations.py -v
```

Expected: the first two tests FAIL; the third passes.

- [ ] **Step 3: Add the bib entries**

Look up the true bibliographic details before writing these — do not copy a guessed year, venue or author list. LDNS is the latent-diffusion spiking-data paper; Spike-GAN is Molano-Mazón et al.'s spike-train GAN. Add SpikeProphecy and the Neural Latents Benchmark only if Step 4 finds prose to trade for them.

```bibtex
@inproceedings{kapoor2024ldns,
  title     = {...},
  author    = {...},
  booktitle = {...},
  year      = {...},
}

@inproceedings{molano2018spikegan,
  title     = {...},
  author    = {...},
  booktitle = {...},
  year      = {...},
}
```

- [ ] **Step 4: Restructure the section into the spec's four blocks**

The target order is: statistical neural forward models (DG, GLM); discrete/video generative models (VQ-VAE, residual quantisation, MaskGIT/MAGVIT); neural latent and spike-generative models (LFADS/NDT, then LDNS and Spike-GAN, marked as adjacent rather than identical); HD-MEA organoid electrophysiology as domain context, not as generative competitors.

Pay for the new sentences by cutting the prose that explains *why* each adjacent literature is not the same problem — the spec says that explanation is what is consuming the space. Two or three sentences of new citation is the budget; do not write a survey.

- [ ] **Step 5: Apply the baseline-wording fix from spec §11**

The spec asks that the baseline taxonomy be defined once and then used with neutral labels, rather than re-argued. Where Related Work currently says DG and GLM are "unusable as peers," use:

> DG and GLM are included as recording-specific statistical references rather than parameter-matched peers because their fitted spatial parameters scale with the number of recordings.

The detailed withholding experiment stays in the appendix, where Task 0's earlier relocation already put it.

- [ ] **Step 6: Run the checks**

```bash
/home/derik/anaconda3/envs/pytorch/bin/python -m pytest tools/tests/test_related_work_citations.py -v
PYTHONPATH="/media/derik/Seagate Desktop Drive/organoid_data" \
  /home/derik/anaconda3/envs/pytorch/bin/python tools/check_paper_numbers.py
```

Expected: 3 passed; numbers still `0 unaccounted for`.

- [ ] **Step 7: Commit**

```bash
git add paper/refs.bib paper/sections/02_related.tex tools/tests/test_related_work_citations.py
git commit -m "Position the adjacent spike-generation work without growing related work"
```

---

## Task 7: Remove repeated interpretation in Results

Target from the spec: 0.25–0.4 page across §5.1–§5.7. **"Do not remove results. Remove repeated interpretation."** The five distinctions that are re-argued in more than one subsection: direct supervision versus reusable representation; memorisation lookup versus shared learned capacity; site accuracy versus voxel timing; discrimination versus calibration; free generation versus reconstruction.

**Files:**
- Modify: `paper/sections/05_experiments.tex` (222 lines)

- [ ] **Step 1: Find every restatement**

```bash
cd "/media/derik/Seagate Desktop Drive/organoid_data/MAGVIT_project"
grep -n "direct supervision\|directly-supervised\|reusable\|memoris\|lookup\|site level\|voxel\|calibrat\|discriminat\|reconstruct" \
  paper/sections/05_experiments.tex
```

For each of the five distinctions, keep the first full statement and reduce every later one to the claim plus a `Section~\ref{...}` cross-reference.

- [ ] **Step 2: Cut, one distinction at a time, compiling between each**

Work through them individually rather than in one sweep. After each, run:

```bash
PYTHONPATH="/media/derik/Seagate Desktop Drive/organoid_data" \
  /home/derik/anaconda3/envs/pytorch/bin/python tools/check_page_budget.py
```

Stop as soon as it reports "within the limit" — there is no benefit to over-cutting, and each further cut costs prose the reviewer asked to keep.

- [ ] **Step 3: Confirm every result survived**

Diff the numbers before and after:

```bash
git diff paper/sections/05_experiments.tex | grep '^-' | grep -oE '[0-9]+\.[0-9]{3,4}' | sort -u > /tmp/removed.txt
git diff paper/sections/05_experiments.tex | grep '^+' | grep -oE '[0-9]+\.[0-9]{3,4}' | sort -u > /tmp/added.txt
comm -23 /tmp/removed.txt /tmp/added.txt
```

Every number this prints has left §5. For each one, confirm it is either (a) present in a generated table the section `\input`s, or (b) present in the appendix section the sentence now cross-references. If it is in neither, put it back — that is a removed result, not a removed interpretation.

- [ ] **Step 4: Run both gates**

```bash
PYTHONPATH="/media/derik/Seagate Desktop Drive/organoid_data" \
  /home/derik/anaconda3/envs/pytorch/bin/python tools/check_paper_numbers.py
PYTHONPATH="/media/derik/Seagate Desktop Drive/organoid_data" \
  /home/derik/anaconda3/envs/pytorch/bin/python tools/check_page_budget.py
```

Expected: `0 unaccounted for`, and `within the limit`.

- [ ] **Step 5: Commit**

```bash
git add paper/sections/05_experiments.tex
git commit -m "Argue each distinction once and cross-reference the rest"
```

---

## Task 8: Bring the generation evidence into the main paper — only if the budget allows

**Conditional.** Attempt only if Task 7 ended with slack. The spec is explicit: "Do not simply add a table on top of the current 9-page body." Generation is currently discussed in §5.4 while its visualisation sits in the appendix, which the spec calls a narrative-coherence problem.

**Files:**
- Modify: `paper/sections/05_experiments.tex`, `paper/sections/99_appendix.tex`, possibly `tools/paper_figs/f3_task_axis.py` and `tools/paper_figs/f4_generation.py`

- [ ] **Step 1: Measure the available slack**

```bash
PYTHONPATH="/media/derik/Seagate Desktop Drive/organoid_data" \
  /home/derik/anaconda3/envs/pytorch/bin/python tools/check_page_budget.py
```

If the body ends on page 9 with less than roughly a third of a page free, stop here and record in the commit message that generation stays in the appendix. Do not create a tenth page to improve narrative flow.

- [ ] **Step 2: Prefer the combined figure over a new table**

The spec's stated preference: combine F3 (completion, `1x2` panels) and F4 (generation, `1x4` panels) into one multi-panel figure, which "would bring the generation result into the main paper while potentially using less total vertical space than a separate figure plus several paragraphs."

Both generators already use `fig.subplots`, so a combined figure is a new module that imports the panel-drawing logic from each rather than duplicating it. Keep axis labels and symbols consistent across the two halves — the spec requires this explicitly.

- [ ] **Step 3: Pay for it**

Whatever the combined figure occupies must be recovered from the prose it replaces — the §5.4 paragraphs the figure now carries visually. Re-run the budget gate; if it goes over, revert this task entirely rather than shrinking figures.

- [ ] **Step 4: Verify**

```bash
PYTHONPATH="/media/derik/Seagate Desktop Drive/organoid_data" \
  /home/derik/anaconda3/envs/pytorch/bin/python tools/make_paper_figures.py 2>&1 | tail -3
PYTHONPATH="/media/derik/Seagate Desktop Drive/organoid_data" \
  /home/derik/anaconda3/envs/pytorch/bin/python tools/check_paper_numbers.py
PYTHONPATH="/media/derik/Seagate Desktop Drive/organoid_data" \
  /home/derik/anaconda3/envs/pytorch/bin/python tools/check_page_budget.py
```

Inspect the combined figure at 100% zoom. If any panel is less legible than it was standalone, revert — the spec forbids solving layout by making figures unreadable.

- [ ] **Step 5: Commit or revert**

```bash
git add -A paper tools/paper_figs
git commit -m "Combine the completion and generation panels into one main-paper figure"
```

---

## Task 9: Reduce the boxed-hyperlink clutter

The spec: "The colored hyperlink/reference boxes are visually distracting in the rendered PDF. If the template permits, use unobtrusive links rather than boxed links."

**Files:**
- Modify: `paper/main.tex:5`

- [ ] **Step 1: Confirm the ICLR template permits it**

The style file is loaded before `hyperref` (`main.tex:2` then `:5`). Check the ICLR 2027 author instructions for any requirement about link colouring before changing it; some templates mandate the default.

- [ ] **Step 2: Switch to unboxed links**

```latex
\usepackage[colorlinks=true, linkcolor=black, citecolor=black,
            urlcolor=blue, breaklinks=true]{hyperref}
```

`colorlinks=true` removes the boxes; black internal links keep the printed page looking unmarked while URLs stay visibly clickable.

- [ ] **Step 3: Compile and confirm the page count did not move**

```bash
PYTHONPATH="/media/derik/Seagate Desktop Drive/organoid_data" \
  /home/derik/anaconda3/envs/pytorch/bin/python tools/check_page_budget.py
```

`colorlinks` changes box metrics slightly; confirm the body still ends by page 9.

- [ ] **Step 4: Commit**

```bash
git add paper/main.tex
git commit -m "Use unboxed hyperlinks"
```

---

## Task 10: Final verification and release rebuild

**Files:**
- Modify: `paper/main_preview.pdf`, release export

- [ ] **Step 1: Run every gate**

```bash
cd "/media/derik/Seagate Desktop Drive/organoid_data/MAGVIT_project"
export PYTHONPATH="/media/derik/Seagate Desktop Drive/organoid_data"
PY=/home/derik/anaconda3/envs/pytorch/bin/python
$PY -m pytest tools/tests/ -v
$PY tools/check_paper_numbers.py
$PY tools/check_page_budget.py
```

Expected: all tests pass; `0 unaccounted for`; `within the limit`.

- [ ] **Step 2: Walk the spec's own priority checklist**

Confirm each item in the spec's §17 "Must fix before submission":
page-count treatment confirmed (Task 1); body ≤ 9 pages (Task 7); Figure 4 overlap fixed (Task 2); adjacent spike-generation work cited without expanding Related Work (Task 6); conspicuous rhetorical phrases removed (Task 3); key negative results still in the main paper.

For the last one, confirm by reading, not by grep — the static-site-map result, the within-token timing limitation, and the no-unseen-preparation limitation must all still be in §1–§7, per spec §13.

- [ ] **Step 3: Refresh the preview PDF and the source zip**

```bash
cd paper && tectonic -X compile main.tex --outdir /tmp/finalpdf && cp /tmp/finalpdf/main.pdf main_preview.pdf && cd ..
```

- [ ] **Step 4: Rebuild the anonymised release**

`tools/make_release.py` has not been run since the full-coverage data rebuild. It self-verifies, and its name rules are case-insensitive so the LICENSE does not leak an identity.

```bash
$PY tools/make_release.py 2>&1 | tail -20
```

Confirm the scrubber reports success and that no author identity appears in the export.

- [ ] **Step 5: Commit**

```bash
git add -A paper reports tools
git commit -m "Fit the scientific body to nine pages and rebuild the release"
```

---

## Out of scope, and why

- **The `4B` sample set and `reports/generation_regimes_4b`.** Dropped from `compare_table.DEFAULT_SETS` on 2026-08-29 because it had become bit-identical to the ship arm. Unrelated to this revision.
- **`preproc_stats.json`'s voxel rate from 48 validation clips.** A real open question, flagged separately, but the spec does not raise it and changing it would change a number in §3.
- **Tightening `tools/check_paper_numbers.py`.** It validates existence rather than currency, which is a genuine gap — but hardening it mid-revision would churn the gate every task depends on. Do it after submission.

## Self-review

**Spec coverage.** §1 page budget → Tasks 1, 5, 7. §2 content to retain → verified in Tasks 5, 7, 10. §3.1 Introduction → Task 5. §3.2 Related Work → Task 6. §3.3 Results repetition → Task 7. §4 table or combined figure → Task 8. §5 figure recommendations → Task 2 (F4), Task 8 (combining); F1/F2/F3 keep, no action needed. §6–§7 rhetorical cadence → Task 3. §9 Abstract → Task 4. §10 citations → Task 6. §11 baseline wording → Task 6 Step 5. §12 compression order → the Task 4–8 ordering follows it. §13 what must stay → Task 10 Step 2. §14 formatting → Task 2, Task 9. §16 editing constraints → Global Constraints. §17 checklist → Task 10 Step 2.

Not covered by a task, deliberately: §14 item 3 (do not shrink F2) and §5's optional "slightly enlarge F2 thumbnails" — the first is a prohibition already in Global Constraints, the second is optional and would cost page budget this plan does not have.

**Placeholders.** The bib entries in Task 6 Step 3 are intentionally left with `...` fields: filling them from memory would fabricate citations. The step says to look them up. That is the one place where writing content would be worse than not writing it.

**Type consistency.** `body_end_page(pdf_text: str) -> int` is defined in Task 1 Step 4 and used by the test in Step 2 and by `main()`; `check_page_budget.py` is invoked identically in Tasks 4, 5, 7, 8, 9, 10. `LIMIT = 9` is defined once.
