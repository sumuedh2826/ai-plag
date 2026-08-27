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
HUMAN_REFERENCE_BANK_VERSION = "human_reference_bank_v1"
CANONICALITY_SPLIT_SEED = SELECTION_RANDOM_SEED
CANONICALITY_TRAIN_QUESTION_COUNT = 350
CANONICALITY_VALIDATION_QUESTION_COUNT = 75
CANONICALITY_INTERNAL_TEST_QUESTION_COUNT = 75
CANONICALITY_EMBEDDING_DIMENSION = 1024


class DatasetSplit(str, Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    INTERNAL_TEST = "internal_test"


class HumanRole(str, Enum):
    REFERENCE_BANK = "human_reference_bank"
    LABELED = "labeled_human"


class HumanBankStatus(str, Enum):
    AVAILABLE = "available"
    SPARSE = "sparse_human_bank"
    UNAVAILABLE = "human_bank_unavailable"


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
