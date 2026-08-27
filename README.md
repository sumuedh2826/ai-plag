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
