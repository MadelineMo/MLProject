import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs

# Initialized data
def initalize():
    df = pd.read_csv('weatherAUS.csv', header=0, encoding='utf-8')

    # remove dates colum
    df = df.drop(['Date'], axis=1)

    # replace names with numbers
    df = df.replace('Adelaide', 1)
    df = df.replace('Albany', 2)
    df = df.replace('Albury', 3)
    df = df.replace('AliceSprings', 4)
    df = df.replace('BadgerysCreek', 5)
    df = df.replace('Ballarat', 6)
    df = df.replace('Bendigo', 7)
    df = df.replace('Brisbane', 8)
    df = df.replace('Cairns', 9)
    df = df.replace('Canberra', 10)
    df = df.replace('Cobar', 11)
    df = df.replace('CoffsHarbour', 12)
    df = df.replace('Dartmoor', 13)
    df = df.replace('Darwin', 14)
    df = df.replace('GoldCoast', 15)
    df = df.replace('Hobart', 16)
    df = df.replace('Katherine', 17)
    df = df.replace('Launceston', 18)
    df = df.replace('MelbourneAirport', 20)
    df = df.replace('Melbourne', 19)
    df = df.replace('Mildura', 21)
    df = df.replace('Moree', 22)
    df = df.replace('MountGambier', 23)
    df = df.replace('MountGinini', 24)
    df = df.replace('Newcastle', 25)
    df = df.replace('Nhil', 26)
    df = df.replace('NorahHead', 27)
    df = df.replace('NorfolkIsland', 28)
    df = df.replace('Nuriootpa', 29)
    df = df.replace('PearceRAAF', 30)
    df = df.replace('Penrith', 31)
    df = df.replace('PerthAirport', 33)
    df = df.replace('Perth', 32)
    df = df.replace('Portland', 34)
    df = df.replace('Richmond', 35)
    df = df.replace('Sale', 36)
    df = df.replace('SalmonGums', 37)
    df = df.replace('SydneyAirport', 38)
    df = df.replace('Sydney', 39)
    df = df.replace('Townsville', 40)
    df = df.replace('Tuggeranong', 41)
    df = df.replace('Uluru', 42)
    df = df.replace('WaggaWagga', 43)
    df = df.replace('Walpole', 44)
    df = df.replace('Watsonia', 45)
    df = df.replace('Williamtown', 46)
    df = df.replace('Witchcliffe', 47)
    df = df.replace('Wollongong', 48)
    df = df.replace('Woomera', 49)

    # replace north, south, east, west
    df = df.replace('E', 1)
    df = df.replace('ENE', 2)
    df = df.replace('ESE', 3)
    df = df.replace('N', 4)
    df = df.replace('NA', 5)
    df = df.replace('NE', 6)
    df = df.replace('NNE', 7)
    df = df.replace('NNW', 8)
    df = df.replace('NW', 9)
    df = df.replace('S', 10)
    df = df.replace('SE', 11)
    df = df.replace('SSE', 12)
    df = df.replace('SSW', 13)
    df = df.replace('SW', 14)
    df = df.replace('W', 15)
    df = df.replace('WNW', 16)
    df = df.replace('WSW', 17)

    # replace yes and no
    df = df.replace('Yes', 1)
    df = df.replace('No', -1)
    df['RainToday'] = df['RainToday'].fillna(-1)
    df['RainTomorrow'] = df['RainTomorrow'].fillna(-1)

    # fill all empty boxes with 0
    df = df.fillna(0)
    df = df.astype(float)
    data = df.to_numpy()

    # sort data
    y = data[:, 21]  # column 21
    X = data[:, 0:21]

    # Standardize
    sc = StandardScaler()
    sc.fit(X)
    X_std = sc.transform(X)

    return(X_std, y)