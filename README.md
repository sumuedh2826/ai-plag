# nw-ai-code-detector

Heavy worker plus a small Streamlit UI for AI-generated DSA code detection. Decision support only:
the system reports a calibrated score and nearest reference solutions, never a verdict, and never
auto-fails a candidate.

## Naming: folder vs project vs package

These three names differ on purpose. Nothing here should be renamed.

| Layer | Name | Why |
| --- | --- | --- |
| On-disk folder | `ai-plag` | Local checkout directory only; carries no meaning downstream. |
| Poetry project | `nw-ai-code-detector` | Worker repo name used for the deployable artifact. |
| Python package | `nw_ai_code_detector` | Importable module name (`src/` layout). |
| Future Django app | `nkb_ai_detector` | In-process app inside the assessment backend wheel. |

## Environment

- Python 3.12 (`python = ">=3.12,<3.15"`; the upper bound comes from `voyageai`)
- Poetry for dependency management
- `constraints.txt` holds the assessments-backend pins (`Django`, `boto3`, `botocore`, `requests`)
  for the future in-process app; the worker itself pins only `boto3` to the backend version.

```bash
poetry install
```

## Layout

```
src/nw_ai_code_detector/    # importable worker package
data/                       # datasets (gitignored: real user_ids + full programs)
outputs/                    # generated artifacts (gitignored)
constraints.txt             # backend-aligned pins for the future in-process app
```

`data/` and `outputs/` deliberately sit outside the package and are never committed.

## Reference index (production bank)

`config.REFERENCE_INDEX_DIR` points at `outputs/reference_index_d10/`, which is **not in this
repo**. It is a build artifact (~92 MB), produced by:

```bash
poetry run python -m nw_ai_code_detector.build_reference_index_d10 --stage embed
poetry run python -m nw_ai_code_detector.build_reference_index_d10 --stage build
```

It is kept out of git to keep the repo light, and is intended to ship via S3 and be fetched at
deploy time.

**The bank contains no student data.** Every text in it is an output of this repo's own
generators; nothing is read from, derived from, or attributable to a student submission.
Per `(question_id, language)` cluster it holds only: the embedding vectors, the persona and
model that generated each reference, that reference's source text, and its content hash.
No submissions, no user ids, no free-text fields.

One honest nuance: 34 of the 9,423 reference texts are byte-identical to some student
submission. These are one-line canonical answers (`return sum(arr)`, `len(str(n))`,
`s.swapcase()`) where any correct solution converges on the same characters — each still
carries the persona and model that produced it. No student text was ingested; the collision is
in the answer, not the provenance.

The bank is **self-contained**: it bundles its own vectors, exact-match hashes
(`reference_hashes.json`) and reference texts (`reference_texts.json`), so serving needs no
`data/` directory. Rebuilding it, however, does require the generated reference banks under
`data/`, which are gitignored.

## Stripper validation

```bash
poetry run python -m nw_ai_code_detector.validate_stripper
```

Prints raw versus stripped code for three Python and three C++ human submissions plus a synthetic
AI-shaped fixture per sample, and asserts that every stripped output parses under Tree-sitter.

## Integration alignment

The detector mirrors the existing plagiarism plugin, so integration points stay predictable.

| Plagiarism (existing) | AI detector (planned) |
| --- | --- |
| app `nkb_plagiarism` | app `nkb_ai_detector` |
| wheel host `nkb_assessment_backend` | same wheel |
| gate `is_code_plagiarism_enabled` | gate `is_ai_code_detection_enabled` (off by default) |
| SQS `PLAGIARISM_CHECK_RESPONSE` | SQS `AI_CODE_DETECTION_RESPONSE` |
| worker | `nw-ai-code-detector` (this repo) |

All heavy ML stays in this worker repo.

## Scoring eligibility

The worker runs **strip → eligibility → (eligible only) Voyage embed → exact `(qid, lang)` cluster
score**. Eligibility is not a silent drop, a 0.5 fill, or a score cap.

We **only score submissions that are 100% correct and parseable**. We do **not** attempt to score
partial, wrong, or error solutions. That is a firm product decision, not a data gap.

Integrity checks run first. Failure is `{status: missing_or_invalid, score: null, decision: null}`:

| Check | Rule |
| --- | --- |
| Language | Supported languages only: `CPP` and `PYTHON`. |
| Stripped body | `stripped_code` exists after the stripper runs. |
| Parse | Stripped body parses cleanly under Tree-sitter (`root_node.has_error` → skip). |
| Tokens | Tokenization of the stripped body succeeds. |
| Cluster | An exact mixed-v1 cluster exists for `(question_id, language)`. |

Then the significant-token floor (`significant_code_token_count`). Below the floor the worker
abstains **before** calling Voyage:

`{status: "insufficient_evidence", reason: "insufficient_tokens", score: null, decision: null}`

Thresholds are provisional, to be validated against a labeled test set: **CPP ≥ 80**, **PYTHON ≥ 60**.

Entropy is not part of eligibility. It belongs to later calibration/confidence on the score.

### Skip reasons (kept as-is)

| Reason | What happens | Rationale (real examples) |
| --- | --- | --- |
| Function-name mismatch | Skip | Author renamed the boilerplate function (e.g. `findluminary` vs `findCelebrity` on `0179f157`). The program can be correct but is not routable to the expected target. **Kept as-is by decision.** |
| Compiles in a compiler, not in Tree-sitter | Skip | Tree-sitter `has_error` is the parseability check. A C++ submission can compile with `#define` macros and still fail the grammar (e.g. `39ee164e`). **Compiles ≠ parses.** |
| Truncated / empty / broken raw | Skip | Unparseable or empty source cannot be stripped or embedded (e.g. held-out `79bf23ea` truncated, `9b5fad86` empty, `a48eb2fe` incomplete `if`). |

### Calibration (later, not an eligibility exclude)

Short solutions that exact-match an AI reference (for example a ~25-token one-liner) are a known
low-confidence case. They stay in the scored set. Confidence handling belongs in the **calibration**
layer, not in this integrity gate.

