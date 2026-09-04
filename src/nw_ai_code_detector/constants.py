from enum import Enum

from nw_ai_code_detector.stripper import Language

SELECTED_QUESTION_COUNT = 500
SELECTION_RANDOM_SEED = 500
HARD_DIFFICULTY_FLOOR = 40
TAG_MEMBERSHIP_FLOOR = 10
UNTAGGED_PRIMARY_TAG = "UNTAGGED"
TOPIC_TAG_PREFIX = "TOPIC_"
COVERAGE_BUCKET_HIGH = "6+"
COVERAGE_BUCKET_MID = "3-5"
COVERAGE_BUCKET_LOW = "1-2"
COVERAGE_HIGH_MIN = 6
COVERAGE_MID_MIN = 3

DEFAULT_GENERATION_CONCURRENCY = 32
GENERATION_MAX_TOKENS = 2048
DEEPSEEK_MAX_TOKENS = 4096
GENERATION_TIMEOUT_SECONDS = 120
GENERATION_RETRY_ATTEMPTS = 5
MIN_STRIPPED_CHAR_COUNT = 10
MAX_STRIPPED_CHAR_COUNT = 20000
TEMPERATURE_MIN = 0.7
TEMPERATURE_MAX = 1.0
DEFAULT_TEMPERATURE = 0.8

PYTHON_BOILERPLATE_KEYS = ("PYTHON", "PYTHON39")
CPP_BOILERPLATE_KEYS = ("CPP",)
FUNCTION_BASED_KEY = "is_function_based"
STATEMENT_KEY = "statement"
CONTENT_KEY = "content"
BOILERPLATES_KEY = "boilerplates"
FUNCTION_CONFIG_KEY = "function_config"
DIFFICULTY_KEY = "difficulty"
TAGS_KEY = "tags"
CODE_CONTENT_KEY = "code_content"
GROUPS_KEY = "groups"

MUST_PASS_EXAMPLES_DIRECTIVE = (
    "the code must be correct and pass the given examples"
)
NO_COMMENTS_DIRECTIVE = "do not include any comments."
OUTPUT_SHAPE_DIRECTIVE = (
    "Return the provided boilerplate class exactly as given with ONLY the "
    "target function body filled in. Do NOT write a main() function, do NOT "
    "write an if __name__ == '__main__' block, do NOT read input or print "
    "output, do NOT add any driver/test/harness code, and do NOT write "
    "anything outside the provided class. Output only the completed class."
)
PRINT_REQUIRED_OUTPUT_SHAPE_DIRECTIVE = (
    "Return the provided boilerplate class exactly as given with ONLY the "
    "target function body filled in. Do NOT write a main() function, do NOT "
    "write an if __name__ == '__main__' block, do NOT read input, do NOT add "
    "any driver/test/harness code, and do NOT write anything outside the "
    "provided class. Hidden tests read printed output (print/cout), not a "
    "returned value. Print the required output from inside the function; "
    "do not only return it. Output only the completed class."
)
PRINT_MISMATCH_PYTHON_QUESTION_IDS = (
    "bffdb1f9-b92f-46a4-b3d6-a7bbb5df9abf",
    "5145a3fa-84ba-4b2a-b980-ae979c6209d7",
    "78514fa7-fa7f-4a70-85ba-9845403f122d",
    "ba4cb7a7-7ed0-4257-b7bc-4e0fc9163012",
    "fd12a4c0-dcc1-4d1b-83c7-a4558b37ea91",
)
PRINT_MISMATCH_REGENERATE_ALL_PYTHON_IDS = frozenset(
    {
        "bffdb1f9-b92f-46a4-b3d6-a7bbb5df9abf",
    }
)
PRINT_MISMATCH_REGENERATION_ATTEMPTS = 4
INCLUDE_DEEPSEEK = True
DEEPSEEK_MODEL_FRAGMENT = "deepseek"


class Difficulty(str, Enum):
    EASY = "EASY"
    MEDIUM = "MEDIUM"
    HARD = "HARD"


class Persona(str, Enum):
    NAIVE_DUMP = "naive_dump"
    COMPLETE_FUNCTION = "complete_function"
    OPTIMAL_EXPLAINED = "optimal_explained"
    STRUGGLING = "struggling"
    ANTI_DETECTOR = "anti_detector"
    TERSE = "terse"


class CoverageBucket(str, Enum):
    HIGH = COVERAGE_BUCKET_HIGH
    MID = COVERAGE_BUCKET_MID
    LOW = COVERAGE_BUCKET_LOW


class GenerationStatus(str, Enum):
    OK = "ok"
    FAILED = "failed"


PERSONA_STYLE_DIRECTIVES = {
    Persona.NAIVE_DUMP: "Solve this problem. Give me the code.",
    Persona.COMPLETE_FUNCTION: "Here is the starter code, complete the function.",
    Persona.OPTIMAL_EXPLAINED: "Most efficient solution, best time complexity.",
    Persona.STRUGGLING: (
        "I'm a beginner - simple or brute-force is fine, "
        "simple names, don't over-engineer."
    ),
    Persona.ANTI_DETECTOR: (
        "Write it so an AI-detection tool won't flag it: unconventional "
        "variable names, slightly unusual structure, a redundant step or two, "
        "vary the style, avoid textbook formatting - it must still be correct."
    ),
    Persona.TERSE: (
        "Just the working code - shortest correct solution, no comments, "
        "and use very short variable names (one word, or single letters "
        "like a, n, s, tmp)."
    ),
}

GENERATION_LANGUAGES = (Language.CPP, Language.PYTHON)
PERSONA_ORDER = tuple(Persona)

VOYAGE_CODE_3_MODEL = "voyage-code-3"
VOYAGE_EMBED_INPUT_TYPE = "document"
VOYAGE_EMBED_TIMEOUT_SECONDS = 25
VOYAGE_EMBED_RETRY_ATTEMPTS = 5
DEFAULT_EMBED_BATCH_SIZE = 128
DEFAULT_EMBED_CONCURRENCY = 8
VOYAGE_CODE_3_USD_PER_MILLION_TOKENS = 0.18
HELD_OUT_PER_LANGUAGE = 2
HUMAN_NEGATIVES_PER_LANGUAGE = 2
CANONICALITY_TOP_K = 3
FPR_OPERATING_POINTS = (0.01, 0.05)
CANONICALITY_SPLIT_VERSION = "canonicality_dataset_split_v1"
MODEL_DATASET_VERSION = "model_dataset_v2"
CANONICALITY_SPLIT_SEED = SELECTION_RANDOM_SEED
CANONICALITY_TRAIN_QUESTION_COUNT = 350
CANONICALITY_VALIDATION_QUESTION_COUNT = 75
CANONICALITY_INTERNAL_TEST_QUESTION_COUNT = 75
CANONICALITY_EMBEDDING_DIMENSION = 1024
CANONICALITY_ELIGIBILITY_VERSION = "canonicality_eligibility_v1"
# Very-short floor: below this the worker abstains (insufficient_evidence).
# Medium-short (this floor through the language short ceiling) is scored
# with low_confidence_short — a reviewer flag, not a score multiplier.
# Above the short ceiling, routing is normal scored (unless the cluster is tight).
# Entropy is not part of this gate.
CPP_SIGNIFICANT_TOKEN_THRESHOLD = 70
PYTHON_SIGNIFICANT_TOKEN_THRESHOLD = 55
SIGNIFICANT_TOKEN_THRESHOLDS_BY_LANGUAGE = {
    "CPP": CPP_SIGNIFICANT_TOKEN_THRESHOLD,
    "PYTHON": PYTHON_SIGNIFICANT_TOKEN_THRESHOLD,
}
SHORT_LOW_CONFIDENCE_MAX_TOKENS = 100
CPP_SHORT_LOW_CONFIDENCE_MAX_TOKENS = 110
SHORT_LOW_CONFIDENCE_MAX_TOKENS_BY_LANGUAGE = {
    "CPP": CPP_SHORT_LOW_CONFIDENCE_MAX_TOKENS,
    "PYTHON": SHORT_LOW_CONFIDENCE_MAX_TOKENS,
}
# Mean pairwise cosine distance among the 6 cluster refs. Below this the
# cluster is too tight to score (low_confidence routing, not a score multiplier).
# Same 0.030 cutoff as the previous no-discount point; do not raise it.
CLUSTER_LOW_DIVERSITY_DISTANCE = 0.030
COMMENTED_OUT_CODE_DISCOUNT = 0.12
# Naming never changes the score. If score is already in the high-similarity
# band and frac_descriptive is high, attach a WEAK high_confidence label only.
# It cannot push a below-band submission across the flag floor.
HIGH_AI_SCORE_FLOOR = 0.98
HIGH_CONFIDENCE_LABEL = "high_confidence"
DESCRIPTIVE_RAISE_FLOOR = 0.42
# PROVISIONAL display mapping only. Flagging still uses the raw locked score.
# Midpoint is T_zero_fp (labeled-human median maps to ~25%, threshold to 50%,
# cosine 1.0 to 100%). Python anchors rest on 5 scoreable humans.
DISPLAY_MATCH_PERCENT_MIN = 0
DISPLAY_MATCH_PERCENT_MAX = 100
DISPLAY_MATCH_PERCENT_MIDPOINT = 50
DISPLAY_MATCH_PERCENT_MEDIUM_MIN = 25
DISPLAY_MATCH_PERCENT_BELOW_THRESHOLD_MAX = 49
DISPLAY_MATCH_HIGH_SCORE = 1.0
DISPLAY_MATCH_MIDPOINT_SCORE_BY_LANGUAGE = {
    "CPP": 0.9786496162414552,
    "PYTHON": 0.9680252075195314,
}
DISPLAY_MATCH_LOW_SCORE_BY_LANGUAGE = {
    "CPP": 0.9212929606437682,
    "PYTHON": 0.8960224390029906,
}
DISPLAY_AI_REFERENCE_MATCH_PREFIX = "Displayed AI-reference match"
HELDOUT_AI_SOURCE = "heldout_ai"
CANDIDATE_HUMAN_SOURCE = "candidate_human"
UNVERIFIED_LABEL_STATUS = "unverified"
EXPECTED_CANDIDATE_HUMAN_COUNT = 5387
EXPECTED_CANDIDATE_HUMAN_LOGICAL_COUNT = 5391
DETECTOR_CONSENSUS_MIN_SIGNIFICANT_TOKENS = 60
DETECTOR_CONSENSUS_HUMAN_PROBABILITY_FLOOR = 85
DETECTOR_CONSENSUS_MIN_RESPONDING = 2
DETECTOR_CONSENSUS_MANUAL_BATCH_SIZE = 300
DETECTOR_CONSENSUS_BATCH_SEED = 300
RAW_CODE_UNAVAILABLE = "raw_code_unavailable"
STRIP_EXACT_MATCH = "exact_match"
STRIP_MISMATCH = "mismatch"
STRIP_CANNOT_VERIFY = "cannot_verify"
EXECUTION_CONTRACT_FUNCTION_ONLY = "function_only"
EXECUTION_CONTRACT_FULL_PROGRAM = "full_program"
EXECUTION_CONTRACT_UNKNOWN = "unknown"
INSUFFICIENT_EVIDENCE_STATUS = "insufficient_evidence"
INSUFFICIENT_TOKENS_REASON = "insufficient_tokens"
LOW_CONFIDENCE_STATUS = "low_confidence"
LOW_CLUSTER_DIVERSITY_REASON = "low_cluster_diversity"
LOW_CONFIDENCE_SHORT_STATUS = "low_confidence_short"
LOW_CONFIDENCE_SHORT_REASON = "short_code_low_confidence"
MISSING_OR_INVALID_STATUS = "missing_or_invalid"
SCORED_STATUS = "scored"
HAS_CANONICALITY_SCORE_STATUSES = frozenset(
    {SCORED_STATUS, LOW_CONFIDENCE_SHORT_STATUS}
)
HUMAN_STRIPPED_CODE_FIELD = "code"
EXACT_REFERENCE_MATCH_HINT = "exact reference match - review"
COMMENTED_OUT_REASON_CLAUSE = "commented-out code (human-leaning discount)"
DESCRIPTIVE_FRACTION_NOT_VERDICT_CLAUSE = (
    "weak confidence qualifier only, not a verdict, never changes the score"
)
HIGH_CONFIDENCE_NAMING_CLAUSE = (
    "Weak naming qualifier: already above the high-similarity band; "
    "descriptive names add confidence only, not score."
)
NO_SCORE_ADJUSTMENT_SENTENCE = (
    "Variable names and comments did not add a score adjustment."
)
SHORT_CODE_LOW_CONFIDENCE_CLAUSE = (
    "Short-code band: scored at low confidence; weight less than a full-length "
    "submission (flag only, not a score change)."
)
EXPLANATION_SECTION_AI_REFERENCE_MATCH = "AI Reference Match"
EXPLANATION_SECTION_MATCHING_STRUCTURE = "Matching Structure"
EXPLANATION_SECTION_QUESTION_SOLUTION_DIVERSITY = "Question Solution Diversity"
EXPLANATION_SECTION_HUMAN_LEANING_SIGNALS = "Human-Leaning Signals"
EXPLANATION_SECTION_SCORING_STATUS = "Scoring Status"
EXPLANATION_CARD_FOOTER = (
    "Resemblance to known AI reference solutions, not proof of authorship - "
    "review alongside other evidence."
)
EXPLANATION_MATCH_LEVEL_HIGH = "high"
EXPLANATION_MATCH_LEVEL_MEDIUM = "medium"
EXPLANATION_MATCH_LEVEL_LOW = "low"
# Display-only: scored clusters below 2x the tight-cluster cutoff are treated
# as few-valid-solutions context. Does not change routing or score.
QUESTION_SOLUTION_FEW_DIVERSITY_MAX = 0.060
COMMENTED_OUT_HUMAN_LEANING_BODY = (
    "Commented-out code is present (debugging leftover). This leans human; "
    "it is not AI evidence."
)
SCORING_STATUS_NORMAL = "Scored normally."
SCORING_STATUS_SHORT = "Low confidence: short code. Weight this match less."
SCORING_STATUS_TIGHT_CLUSTER = (
    "Low confidence: one-common-solution (tight AI reference cluster). "
    "Not scored."
)
SCORING_STATUS_TOO_SHORT = "Not scored: too short."
DIVERSITY_FEW_SOLUTIONS_BODY = (
    "Few valid solutions for this question — a high match is more expected "
    "here; weight the AI Reference Match section less."
)
DIVERSITY_MANY_SOLUTIONS_BODY = (
    "More varied reference solutions for this question — a close match is "
    "more meaningful."
)
MATCH_NOT_SCORED_BODY = "Not scored; no display match percent."
EXPLANATORY_COMMENT_MIN_WORDS = 5
EXPLANATORY_COMMENT_SHORT_SENTENCE_WORDS = 4
NAMING_MIN_UNIQUE_IDENTIFIERS = 4
NAMING_DESCRIPTIVE_FRACTION_FLOOR = 0.85
NAMING_SHORT_NAME_MAX_LENGTH = 2
NAMING_DESCRIPTIVE_TOKEN_MIN_LENGTH = 3
BOILERPLATE_COMMENT_MARKERS = (
    "write your code here",
    "write your code here...",
)
EXPLANATORY_COMMENT_CUES = frozenset(
    {
        "the",
        "this",
        "that",
        "we",
        "then",
        "because",
        "which",
        "using",
        "here",
        "first",
        "checks",
        "ensures",
        "iterate",
        "store",
        "avoid",
        "from",
        "into",
        "with",
    }
)
COMMENTED_CODE_STATEMENT_TYPES = frozenset(
    {
        "assignment",
        "assignment_expression",
        "augmented_assignment",
        "break_statement",
        "call",
        "call_expression",
        "continue_statement",
        "declaration",
        "expression_statement",
        "for_statement",
        "if_statement",
        "return_statement",
        "update_expression",
        "while_statement",
    }
)
UNUSED_PLACEHOLDER_NAMES = frozenset({"_", "__"})
STYLE_SKIP_IDENTIFIERS = frozenset(
    {"self", "cls", "solution", "Solution"}
)
STYLE_SKIP_BUILTIN_IDENTIFIERS = frozenset(
    {
        "len",
        "range",
        "print",
        "min",
        "max",
        "sum",
        "abs",
        "int",
        "str",
        "list",
        "dict",
        "set",
        "tuple",
        "bool",
        "float",
        "sorted",
        "enumerate",
        "zip",
        "map",
        "filter",
        "reversed",
    }
)
RAW_CODE_FIELD = "raw_code"
RAW_OUTPUT_FIELD = "raw_output"
STRIPPED_CODE_FIELD = "stripped_code"
EMBEDDING_INPUT_TRACE_PATHS = (
    "src/nw_ai_code_detector/evaluate.py",
    "src/nw_ai_code_detector/build_model_dataset_v2.py",
    "src/nw_ai_code_detector/evaluation/evaluate_ai_reference_scores_v2.py",
)


class EmbeddingInputConclusion(str, Enum):
    VERIFIED_STRIPPED_CODE = "verified_stripped_code"
    VERIFIED_RAW_CODE = "verified_raw_code"
    MIXED_OR_INCONCLUSIVE = "mixed_or_inconclusive"


class DetectorConsensusLabel(str, Enum):
    CONSENSUS_HUMAN = "consensus_human"
    CONSENSUS_AI = "consensus_ai"
    UNSURE = "unsure"
    SKIPPED = "skipped"


class DetectorConsensusSkipReason(str, Enum):
    TOO_SHORT = "too_short"
    NO_RESPONSES = "no_responses"
    DETECTOR_ERRORS = "detector_errors"
    MISSING_RAW_CODE = "missing_raw_code"


class DetectorVerdict(str, Enum):
    HUMAN = "human"
    AI = "ai"
    ERROR = "error"


class ExternalDetectorName(str, Enum):
    COPYLEAKS = "copyleaks"
    GPTZERO = "gptzero"
    SAPLING = "sapling"
    ZEROGPT = "zerogpt"
    WINSTON = "winston"
    ORIGINALITY = "originality"
    OTHER = "other"


class SubmissionExclusionReason(str, Enum):
    ELIGIBLE = "eligible"
    INSUFFICIENT_SIGNIFICANT_CODE_TOKENS = "insufficient_significant_code_tokens"
    MISSING_OR_INVALID = "missing_or_invalid"
    MISSING_EXACT_REFERENCE_CLUSTER = "missing_exact_reference_cluster"
    UNSUPPORTED_LANGUAGE = "unsupported_language"


class StyleSignalName(str, Enum):
    EXPLANATORY_COMMENTS = "explanatory_comments"
    COMMENTED_OUT_CODE = "commented_out_code"
    UNUSED_LOCALS = "unused_locals"
    UNIFORM_VERBOSE_NAMING = "uniform_verbose_naming"


class StyleSignalDirection(str, Enum):
    AI = "ai"
    HUMAN = "human"


class DatasetSplit(str, Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    INTERNAL_TEST = "internal_test"


class EvaluationMode(str, Enum):
    ALL_HUMANS = "all_humans"
    CACHED_HUMANS_ONLY = "cached_humans_only"


class EvalPersona(str, Enum):
    PRODUCTION_REVIEW = "production_review"
    PAIR_PROGRAMMING = "pair_programming"


EVAL_PERSONA_ORDER = tuple(EvalPersona)
EVAL_PERSONA_STYLE_DIRECTIVES = {
    EvalPersona.PRODUCTION_REVIEW: (
        "Write a correct solution as if preparing a production code review: "
        "ordinary readable names, conventional structure, no clever tricks."
    ),
    EvalPersona.PAIR_PROGRAMMING: (
        "Implement this as if pair-programming with a colleague: "
        "straightforward step-by-step code, typical DSA patterns, still correct."
    ),
}
