from pandas import read_csv, DataFrame, Series
from numpy import array, concatenate, random
from math import *

def CalculateHeuristic(prices):
    if (len(prices) < 30):
        return len(prices)/100
    return (max(prices)-prices[0])

def ImportData(filename):
    df = read_csv("Data/" + filename + ".csv")
    df = df[df.Volume != 0]

    print(df.head(5))
    print(len(df.values.tolist()))
    print(len(df.values.tolist()[0]))

    df['date'] = df['Local time'].str.extract('(\d\d.\d\d\.\d\d\d\d)', expand=True)
    df['time'] = df['Local time'].str.extract('(\d\d:\d\d\:\d\d)', expand=True)

    numTimes = df.groupby('date').count()
    shortDays = numTimes[numTimes.time != 390].index.values
    print(shortDays)

    # Remove short days
    df = df[~df['date'].isin(shortDays)]

    openData = df['Open'].values.tolist()
    volumeData = df['Volume'].values.tolist()

    # Heuristic
    heuristicData = [0] * len(openData)
    for index, price in enumerate(openData):
        # how many elements to go forward 390 - (index % 390)
        # calculate heuristic for index-->forward (start with max price?)
        numForward = 390-(index%390)
        heuristicData[index] = CalculateHeuristic(openData[index:index+numForward])

    for i in range(int(len(openData)/390)):
        minIndex = i * 390
        maxIndex = (i * 390) + 390

        # min and max in first 30 mins (with varience)
        minOpen = min(openData[minIndex:minIndex + 30]) * 0.995
        maxOpen = max(openData[minIndex:minIndex + 30]) * 1.005

        minVolume = min(volumeData[minIndex:minIndex + 30]) * 0.95
        maxVolume = max(volumeData[minIndex:minIndex + 30]) * 1.15

        minHeuristic = min(heuristicData[minIndex:minIndex + 30]) * 0.995
        maxHeuristic = max(heuristicData[minIndex:minIndex + 30]) * 1.005

        for j in range(minIndex, maxIndex):
            if openData[j] > maxOpen:
                openData[j] = 1
            elif openData[j] < minOpen:
                openData[j] = 0
            else:
                openData[j] = (openData[j] - minOpen) / (maxOpen - minOpen)

            if volumeData[j] > maxVolume:
                volumeData[j] = 1
            elif volumeData[j] < minVolume:
                volumeData[j] = 0
            else:
                volumeData[j] = (volumeData[j] - minVolume) / (maxVolume - minVolume)

            if heuristicData[j] > maxHeuristic:
                heuristicData[j] = 1
            elif heuristicData[j] < minHeuristic:
                heuristicData[j] = 0
            else:
                heuristicData[j] = (heuristicData[j] - minHeuristic) / (maxHeuristic - minHeuristic)

    seHe = Series(heuristicData)
    df['Output'] = seHe.values

    df.rename(columns={'Open': 'Price'}, inplace=True)
    seOp = Series(openData)
    df['Open'] = seOp.values

    seOp = Series(volumeData)
    df['Volume'] = seOp.values

    dfnew = df[['Open', 'Volume','Output','Price']].copy()
    print(dfnew.head(5))

    finalData = dfnew.values.tolist()
    blah = array(finalData)
    print(blah.shape)

    meep = blah.reshape((int(blah.shape[0]/390), 390, blah.shape[1]))
    print(meep.shape)
    print(meep[0][0])
    print(meep[0][1])
    return meep

def Concat():
    print("=== Tesla ===================================")
    dataTesla = ImportData('TSLA')

    print("=== NFLX ===================================")
    dataNflx = ImportData('NFLX')

    print("=== JNJ ===================================")
    dataJnj = ImportData('JNJ')

    print("=== CMG ===================================")
    dataCmg = ImportData('CMG')

    print("=== DAL ===================================")
    dataDal = ImportData('DAL')

    print("=== Concat ===================================")
    dataConcat = concatenate((dataTesla, dataNflx, dataJnj, dataCmg, dataDal), axis=0)
    print(dataConcat.shape)

    random.shuffle(dataConcat)
    print(dataConcat.shape)
    return dataConcat
