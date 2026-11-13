
# ============================================================
# Standard Libraries
# ============================================================

import os
from pathlib import Path


# ============================================================
# Numerical and Data Processing
# ============================================================

import numpy as np
import pandas as pd

import scipy.io as sio
from scipy.stats import wilcoxon, friedmanchisquare


# ============================================================
# Data Visualization
# ============================================================

import seaborn as sns
import matplotlib.pyplot as plt


# ============================================================
# Scikit-learn: Data Preprocessing
# ============================================================

import sklearn.preprocessing as preprocessing

from sklearn.model_selection import train_test_split


# ============================================================
# Scikit-learn: Classification Algorithms
# ============================================================

from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.svm import SVC as SVM
from sklearn.neural_network import MLPClassifier


# ============================================================
# Scikit-learn: Evaluation Metrics
# ============================================================

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix
)


# ============================================================
# Mealpy
# ============================================================

from mealpy import (
    Problem,
    Optimizer,
    Multitask,
    BinaryVar,
    FloatVar,
    PSO,
    WOA,
    GA,
    DE,
    AOA,
    SCSO,
    GWO
)


# ============================================================
# Project Configuration
# ============================================================

from Config import *
