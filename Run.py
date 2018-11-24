from pandas import read_csv, DataFrame, Series, to_datetime, Timedelta
from numpy import array, column_stack, concatenate, random, array_split
from matplotlib import pyplot
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.layers import LSTM
import time
import os
import os.path

def ImportData(importFolderPath):
    combinedData = DataFrame(columns=['Datetime','Ticker','Volume','Open'])

    files = os.listdir(importFolderPath)
    for file in files:
        (ticker, ext) = os.path.splitext(file)
        if ext != '.csv':
            continue

        fullPath = os.path.join(importFolderPath, file)

        tempData = read_csv(fullPath)

        # Remove times with no volume
        tempData = tempData[tempData.Volume != 0]

        # Remove short days (days with < 380 minutes of data)
        tempData['date'] = tempData['Local time'].str.extract('(\d\d.\d\d\.\d\d\d\d)', expand=True)
        numTimes = tempData.groupby('date').count()
        shortDays = numTimes[numTimes.Open != 390].index.values
        tempData = tempData[~tempData['date'].isin(shortDays)]

        # Add ticker
        tempData['Ticker'] = ticker

        # Convert to datetime
        tempData['Datetime'] = to_datetime(tempData['Local time'].str[:-9], format="%d.%m.%Y %H:%M:%S.%f")
        tempData['Datetime'] += Timedelta(hours=3)
        tempData = tempData.drop(['Local time','Close','High','Low', 'date'], axis=1)
        tempData = tempData.reindex(['Datetime','Ticker','Volume','Open'], axis=1)

        combinedData = combinedData.append(tempData)

    combinedData = combinedData.drop_duplicates()
    combinedData = combinedData.reset_index(drop=True)

    finalData = array(combinedData)
    finalData = finalData.reshape((int(finalData.shape[0]/390), 390, finalData.shape[1]))
    random.shuffle(finalData)
    return finalData

def main():
    data = ImportData('./RawData')
    n = int(0.95*data.shape[0])
    trainData = data[:n,:,:]
    testData = data[n:,:,:]

    for i in range(60,360,30):
        input = [] # numDays x i x 2
        output = [] # numDays

        for day in trainData:
            prices = array([item[-1] for item in day[:i]])
            volumes = array([item[-2] for item in day[:i]])

            minPrice = min(prices)
            maxPrice = max(prices)
            minVolume = min(volumes)
            maxVolume = max(volumes)

            prices = (prices - minPrice) / (maxPrice - minPrice)
            volumes = (volumes - minVolume) / (maxVolume - minVolume)

            dayInput = column_stack((prices,volumes)).tolist()
            input.append(dayInput)

            lastPrice = day[i-1][-1]
            maxPrice = max([item[-1] for item in day[i:]])
            heuristic = (maxPrice/lastPrice - 0.97) / (1.03 - 0.97)
            output.append(heuristic)
        # TODO: change input/output range depending on volume/price, muffle, start at 0.5 and up/down, later in day goes to 0

        lstm = LSTM(12, input_shape=(i, 2))
        model = Sequential()
        model.add(lstm)
        model.add(Dense(1))
        model.compile(loss='mae', optimizer='adam')

        history = model.fit([input], output, epochs=30, batch_size=72)

        model.save("./Models/model_" + str(i) + ".h5")

    modelGain = 1
    dataGain = 1

    for day in testData:
        dataGain = dataGain * day[-1][-1] / day[0][-1]

        bought = False
        price = 0
        buyTime = 0

        for i in range (1,390):
            modelPath = "./Models/model_" + str(i) + ".h5"
            if not os.path.isfile(modelPath):
                continue

            model = load_model(modelPath)

            prices = array([item[-1] for item in day[:i]])
            volumes = array([item[-2] for item in day[:i]])

            minPrice = min(prices)
            maxPrice = max(prices)
            minVolume = min(volumes)
            maxVolume = max(volumes)

            prices = (prices - minPrice) / (maxPrice - minPrice)
            volumes = (volumes - minVolume) / (maxVolume - minVolume)

            dayInput = column_stack((prices,volumes)).tolist()
            pred = model.predict([[dayInput]])[0][0]

            if pred > 0.58 and not bought:
                bought = True
                price = day[i][-1]
                buyTime = i
            elif pred < 0.53 and bought:
                bought = False
                modelGain = modelGain * day[i][-1] / price
                print("Bought at minute " + str(buyTime) + " for "+ str(price) + ". Sold at minute " + str(i) + " for " + str(day[i][-1]) + ".")
        if bought:
            modelGain = modelGain * day[-1][-1] / price
            print("Bought at minute " + str(buyTime) + " for "+ str(price) + ". Sold at minute 390 for " + str(day[i][-1]) + ".")

    print(dataGain)
    print(modelGain)

if __name__ == "__main__":
    main()
