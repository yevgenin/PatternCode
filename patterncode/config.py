import os
from pathlib import Path
from time import strftime
from typing import TypeVar

import coolname
from numba import njit

import patterncode


def get_environ_flag(flag, default=True):
    return bool(int(os.environ.get(flag, default)))


def generate_name(date=True, time=True, num_words=2):
    words = coolname.generate(num_words)
    words_str = '-'.join(words)

    if date and time:
        return strftime('%Y%m%dT%H%M%SZ') + '-' + words_str
    elif date:
        return strftime('%Y%m%d') + '-' + words_str
    elif time:
        return strftime('%H%M%SZ') + '-' + words_str
    else:
        return words_str


T = TypeVar('T')
DATA_DIR = Path().home() / 'data'
OUT_DIR = Path().home() / 'Dropbox/out'
CACHE_DIR = DATA_DIR / 'out/cache'
ENV_PREFIX = 'PC_'
ENV_RUN_NAME = ENV_PREFIX + 'RUN_NAME'
ENV_LOAD_RUN_NAME = ENV_PREFIX + 'LOAD_RUN_NAME'
ENV_QUICK_RUN = ENV_PREFIX + 'QUICK_RUN'
ENV_LOAD = ENV_PREFIX + 'LOAD'
PLOT_COMPUTATIONS_FLAG = False
ENV_SAVE = ENV_PREFIX + 'SAVE'
ENV_TEX = ENV_PREFIX + 'TEX'
RUN_NAME = os.environ.get(ENV_RUN_NAME, default=generate_name())
LOAD_RUN_NAME = os.environ.get(ENV_LOAD_RUN_NAME)
QUICK_RUN_FLAG = get_environ_flag(ENV_QUICK_RUN, default=False)
SAVE_FLAG = get_environ_flag(ENV_SAVE, default=True)
LOAD_FLAG = get_environ_flag(ENV_LOAD, default=True)

SEED = 1799144579

PROJECT_DATA_DIR = Path(patterncode.__file__).parent.parent / 'data'
DETERMINISTIC_MAKE = True
PATTERN_LEN = 6
BACTERIA_GENOME_TITLE = 'Bacterial genomes'
HUMAN_GENOME_TITLE = 'Human genome'
FIG_SIZE = 4
SHOW_CI = True
ACGT = "ACGT"
HIGHLIGHT_TEXT_COLOR = 'g'
CONFIDENCE_ALPHA = 0.05
MOLECULE_DATA_FILE = 'MoleculesExport_20230421T225112Z-grumpy-coyote.pkl'
RANDOM_GENOME = 'random_genome'
HUMAN_GENOME = 'human_genome'
BACTERIAL_GENOMES = 'bacterial_genomes'
DEFAULT_MOLECULE_LEN = 50000
DEFAULT_ALIGN_LENGTH = DEFAULT_MOLECULE_LEN

LIMIT_NUM_SEQ = 2 if QUICK_RUN_FLAG else None
LIMIT_SEQ_LEN = 2 * 10 ** 6 if QUICK_RUN_FLAG else None
RANDOM_GENOME_LEN = LIMIT_SEQ_LEN if QUICK_RUN_FLAG else int(1e8)
LIMIT_PATTERNS = 1024 if QUICK_RUN_FLAG else None
NUM_SIMULATE_PATTERNS = 16 if QUICK_RUN_FLAG else 64
NUM_ANNOTATE_PATTERNS = 0
NUM_SIMULATE_LENS = 16 if QUICK_RUN_FLAG else 64
MAX_INT32 = 2 ** 32 - 1
PARALLEL_FLAG = 1
MAX_WORKERS = None
VERBOSE = 1
NUM_BIN_SIZES = 128
numba_kw = dict(nogil=True, fastmath=False, boundscheck=True, error_model='numpy', cache=False)
numba_parallel = njit(**numba_kw, parallel=True)

DEFAULT_NUM_TRIALS = 64 if QUICK_RUN_FLAG else 512
DEFAULT_COUNT_CAP = 2
DEFAULT_BIN_SIZE = 1000
DEFAULT_MOLECULE_LEN_PER_GENOME = {HUMAN_GENOME: DEFAULT_MOLECULE_LEN, BACTERIAL_GENOMES: DEFAULT_MOLECULE_LEN,
                                   RANDOM_GENOME: DEFAULT_MOLECULE_LEN}
DATASET_BACTERIA = 'bacteria'
DATASET_HUMAN = 'human'
MOLECULE_LENGTH = 'DNA fragment length (bp)'
CHANNEL_MODEL_ATTR = {"bin_size": 'Bin Size (bp)', "align_length": 'Alignment Length (bp)'}
ERROR_PROBABILITY = "Error probability"
EXPERIMENTAL = 'Experimental'
THEORY = "Theory"
SIMULATION = "Simulation"
PLOT_LEN_LIM = 500e3
DENSITY = 'Genome pattern density $(bp^{-1})$'
PATTERN = 'Pattern'
CTTAAG = "CTTAAG"
DEFAULT_PATTERN = CTTAAG
ORGANISM_NAME = 'organism.organismName'
ORGANISM_TAX_ID = 'organism.taxId'
DOWNLOAD_DIR = DATA_DIR / 'download'
BACTERIA_DATA_DIR = DOWNLOAD_DIR / 'ncbi_dataset/data'
BACTERIA_SUMMARY_FILE = DOWNLOAD_DIR / 'summary.jsonl'

ASSEMBLY_SUMMARY_FILE = Path('ncbi_dataset') / 'data' / 'assembly_data_report.jsonl'
UNPACK_DIR_BACTERIA = DOWNLOAD_DIR / 'bacteria'
UNPACK_DIR_HUMAN = DOWNLOAD_DIR / 'human'

HUMAN_GENOME_FASTA_FILE = UNPACK_DIR_HUMAN / "ncbi_dataset/data/GCF_000001405.40/GCF_000001405.40_GRCh38.p14_genomic.fna"

IUPAC_DNA = {
    'A': ['A'],
    'C': ['C'],
    'G': ['G'],
    'T': ['T'],
    'U': ['U'],
    'R': ['A', 'G'],
    'Y': ['C', 'T'],
    'S': ['G', 'C'],
    'W': ['A', 'T'],
    'K': ['G', 'T'],
    'M': ['A', 'C'],
    'B': ['C', 'G', 'T'],
    'D': ['A', 'G', 'T'],
    'H': ['A', 'C', 'T'],
    'V': ['A', 'C', 'G'],
    'N': ['A', 'C', 'G', 'T'],
    '.': ['.'],
    '-': ['-'],
}
ACCESSION = 'accession'

BACTERIA_NAMES = """
Salmonella enterica
Escherichia coli K12
Staphylococcus aureus
Streptococcus pyogenes
Klebsiella pneumoniae
Mycobacterium tuberculosis
Pseudomonas aeruginosa
Proteus mirabilis
Citrobacter koseri
"""


def get_species_names(names_str):
    names_str = [
        _.strip() for _ in names_str.split('\n')
        if _ and not _.startswith('#')
    ]
    return names_str


GENOME_NAMES = [
    HUMAN_GENOME,
    BACTERIAL_GENOMES,
    RANDOM_GENOME,
]
