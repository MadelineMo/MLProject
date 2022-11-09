import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs

# Initialized data
def initalize():
    df = pd.read_csv('states_all.csv', header=0, encoding='utf-8')
    df = df.drop(['PRIMARY_KEY'])
    # State
    df = df.replace('ALABAMA', 1)
    df = df.replace('ALASKA', 2)
    df = df.replace('ARIZONA', 3)
    df = df.replace('ARKANSAS', 4)
    df = df.replace('CALIFORNIA', 5)
    df = df.replace('COLORADO', 6)
    df = df.replace('CONNECTICUT', 7)
    df = df.replace('DELAWARE', 8)
    df = df.replace('DISTRICT_OF_COLUMBIA', 9)
    df = df.replace('FLORIDA', 10)
    df = df.replace('GEORGIA', 11)
    df = df.replace('HAWAII', 12)
    df = df.replace('IDAHO', 13)
    df = df.replace('ILLINOIS', 14)
    df = df.replace('INDIANA', 15)
    df = df.replace('IOWA', 16)
    df = df.replace('KANSAS', 17)
    df = df.replace('KENTUCKY', 18)
    df = df.replace('LOUISIANA', 19)
    df = df.replace('MAINE', 20)
    df = df.replace('MARYLAND', 21)
    df = df.replace('MASSACHUSETTS', 22)
    df = df.replace('MICHIGAN', 23)
    df = df.replace('MINNESOTA', 24)
    df = df.replace('MISSISSIPPI', 25)
    df = df.replace('MISSOURI', 26)
    df = df.replace('MONTANA', 27)
    df = df.replace('NEBRASKA', 28)
    df = df.replace('NEVADA', 29)
    df = df.replace('NEW_HAMPSHIRE', 30)
    df = df.replace('NEW_JERSEY', 31)
    df = df.replace('NEW_MEXICO', 32)
    df = df.replace('NEW_YORK', 33)
    df = df.replace('NORTH_CAROLINA', 34)
    df = df.replace('NORTH_DAKOTA', 35)
    df = df.replace('OHIO', 36)
    df = df.replace('OKLAHOMA', 37)
    df = df.replace('OREGON', 38)
    df = df.replace('PENNSYLVANIA', 39)
    df = df.replace('RHODE_ISLAND', 40)
    df = df.replace('SOUTH_CAROLINA', 41)
    df = df.replace('TENNESSEE', 42)
    df = df.replace('TEXAS', 43)
    df = df.replace('UTAH', 44)
    df = df.replace('VERMONT', 45)
    df = df.replace('VIRGINIA', 46)
    df = df.replace('WASHINGTON', 47)
    df = df.replace('WEST_VIRGINIA', 48)
    df = df.replace('WISCONSIN', 49)
    df = df.replace('WYOMING', 50)
    df = df.replace('DODEA', 51)
    df = df.replace('NATIONAL', 52)

    # fill all empty boxes with 0
            # !!! what should we do about missing scores data? (y collums)
            # remove rows with empty test score data???
    df = df.fillna(0)
    df = df.astype(int)
    data = df.to_numpy()

    # sort data X and y, y = scores average
    y_4_Math = data[:, 20]
    y_8_Math = data[:, 21]
    y_4_Reading = data[:, 22]
    y_8_Reading = data[:, 23]
    X = data[:, 0:19]

    # Standardize X
    sc = StandardScaler()
    sc.fit(X)
    X_train_std = sc.transform(X)
    return(X_train_std, y_4_Math, y_8_Math, y_4_Reading, y_8_Reading)