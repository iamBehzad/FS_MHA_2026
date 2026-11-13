
# ============================================================
# Reproducibility
# ============================================================

RANDOM_STATE = 42


# ============================================================
# Dataset Configuration
# ============================================================

DATASET_NAMES = [
    'Audit',
    'Breast',
    'Chess',
    'CTG',
    'Faults',
    'German',
    'Ionosphere',
    'Sonar',
    'Statlog',
    'Steel',
    'WDVG1',
    'QSAR',
    'Mfeat_Zernike',
    'MiniBooNE',
    'Parkinsons',
    'Phishing',
    'Sensorless',
    'Wall_Following',
    'Spambase',
    'Waveform',
]

# ============================================================
# Data Splitting
# ============================================================

TEST_SIZE = 0.25


# ============================================================
# Classification Configuration
# ============================================================

CLASSIFIER = "knn"  # Available: knn, rf, svm
AVERAGE_METRIC = "weighted"

DRAW_CONFUSION_MATRIX = False


# ============================================================
# Optimization Configuration
# ============================================================

MIN_MAX_PROBLEM = "min"

# Metric order: [accuracy, precision, recall, f1]
OBJ_WEIGHTS = [1, 0, 0, 0]

EPOCH = 100
POP_SIZE = 50


# ============================================================
# Execution and Display
# ============================================================

PRINT_ALL = False


# ============================================================
# Output Configuration
# ============================================================

BEST_FIT_FOLDER = 'history/best_fit'
CONVERGENCE_FOLDER = 'history/convergence'

ALGORITHMS = [
    'ARBOS',
    'BinaryDE',
    'BinaryGA',
    'BinaryGWO',
    'BinaryPSO',
]

PROPOSED_ALGORITHM = 'ARBOS'

RESULT_FILENAME = 'results.xlsx'
OUTPUT_FOLDER = 'convergence_curves'
