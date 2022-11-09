import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs
from initialize import initalize

# Initialized data
# returns standardized data
X, y_4_Math, y_8_Math, y_4_Reading, y_8_Reading = initalize()

# split data into train and test sets (per y set)
X_4_Math_train, X_4_Math_test, y_4_Math_train, y_4_Math_test = train_test_split(
    X, y_4_Math, test_size=0.3, random_state=0)

X_8_Math_train, X_8_Math_test, y_8_Math_train, y_8_Math_test = train_test_split(
    X, y_8_Math, test_size=0.3, random_state=0)

X_4_Reading_train, X_4_Reading_test, y_4_Reading_train, y_4_Reading_test = train_test_split(
    X, y_4_Reading, test_size=0.3, random_state=0)

X_8_Reading_train, X_8_Reading_test, y_8_Reading_train, y_8_Reading_test = train_test_split(
    X, y_8_Reading, test_size=0.3, random_state=0)

