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
    # df = df.fillna(0)
    #df = df.astype(float)
    #data = df.to_numpy()
    
    
    
    
    #Isabel's edit (dealing with NA Values - if we don't want to fill them al with 0 or NA) - please edit! I'm sure there is a better way to go about this :)
    
    #Before runnig comment out line 92!!
    
    #View the column names
    print(df.columns.values)

    #Finding different data types
    df.info()

    #look at null values
    df.isnull().sum() 

    #percent of null missing 
    percent_missing = df.isnull().sum() * 100 / len(df)
    missing_value_df = pd.DataFrame({'column_name': df.columns,
                                     'percent_missing': percent_missing})
    print(missing_value_df)
    
    #Filling averages for missing some missing variables 
    df["MinTemp"] = df["MinTemp"].fillna(df['MinTemp'].mean())
    df["MaxTemp"] = df["MaxTemp"].fillna(df['MaxTemp'].mean())    
    df["Rainfall"] = df["Rainfall"].fillna(df['Rainfall'].mean())
    df["Humidity9am"] = df["Humidity9am"].fillna(df['Humidity9am'].mean())
    df["Humidity3pm"] = df["Humidity3pm"].fillna(df['Humidity3pm'].mean())
    df["Pressure9am"] = df["Pressure9am"].fillna(df['Pressure9am'].mean())
    df["Pressure3pm"] = df["Pressure3pm"].fillna(df['Pressure3pm'].mean())
    df["Temp9am"] = df["Temp9am"].fillna(df['Temp9am'].mean())
    df["Temp3pm"] = df["Temp3pm"].fillna(df['Temp3pm'].mean())
    df["WindGustSpeed"] = df["WindGustSpeed"].fillna(df['WindGustSpeed'].mean())
    df["WindSpeed9am"] = df["WindSpeed9am"].fillna(df['WindSpeed9am'].mean())
    df["WindSpeed3pm"] = df["WindSpeed3pm"].fillna(df['WindSpeed3pm'].mean())
    df["Sunshine"] = df["Sunshine"].fillna(df['Sunshine'].mean())
    df["Evaporation"] = df["Evaporation"].fillna(df['Evaporation'].mean())
    df["Cloud3pm"] = df["Cloud3pm"].fillna(df['Cloud3pm'].mean())
    df["Cloud9am"] = df["Cloud9am"].fillna(df['Cloud9am'].mean())
    
    #Filling the missing values for continuous variables with mode
    df['RainToday']=df['RainToday'].fillna(df['RainToday'].mode()[0])
    df['RainTomorrow']=df['RainTomorrow'].fillna(df['RainTomorrow'].mode()[0])
    df['WindDir9am'] = df['WindDir9am'].fillna(df['WindDir9am'].mode()[0])
    df['WindGustDir'] = df['WindGustDir'].fillna(df['WindGustDir'].mode()[0])
    df['WindDir3pm'] = df['WindDir3pm'].fillna(df['WindDir3pm'].mode()[0])
    
    
    #Making sure there are no null values left in the end
    print(df.isnull().sum() )
    
    #Notes:
        #Should we drop variables with little correlation to RainTomorrow?
        #Should we drop variables with a large % of missing data?
        #For variables with a small % of missing data, should we fill with NA/0 or their average?


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

