from matplotlib import pyplot
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from numpy import random
from Parse import Concat

#data = random.rand(1000, 390, 3) # day, timesteps, inputs + output
# (data, prices) = ImportData('TSLA')
data = Concat() # real stock data. Range of inputs/outputs [0,1]
print(data.shape)

lenData = len(data)
timesteps = len(data[0]) # minutes in day
inputDim = len(data[0][0]) - 2 # price and volume, minus output and price

train_X = data[:int(lenData*3/4),:,:inputDim] # day, timesteps, inputs
train_y = data[:int(lenData*3/4),:,inputDim:inputDim+1] # results for each day+timestep

# Separate validate and test data: While model doesn't see validation set, architect tunes parameters based on validation set
validate_X = data[int(lenData*3/4):int(lenData*7/8),:,:inputDim]
validate_y = data[int(lenData*3/4):int(lenData*7/8),:,inputDim:inputDim+1]
test_X = data[int(lenData*7/8):,:,:inputDim]
test_y = data[int(lenData*7/8):,:,inputDim:inputDim+1]
test_prices = data[int(lenData*7/8):,:,inputDim+1:]

# design network
lstm = LSTM(12, input_shape=(train_X.shape[1], train_X.shape[2]), return_sequences=True) # Play around with number of units
model = Sequential()
model.add(lstm)
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam') # or mse (larger have bigger impacts on error --> go for big gains or consistent small gains)

# fit network
history = model.fit(train_X, train_y, epochs=30, batch_size=72, validation_data=(validate_X, validate_y), verbose=2, shuffle=False)

# plot error with training and validation sets (both should go down, and validation should be above/near train)
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
#pyplot.show()

# Test_X accuracy. Find "would have bought" (> ?) and "would have sold" (< ?) criteria. Compare to price. Plot performance

# Start actually performing on realtime stock data (fill future with zeroes)
yhat = model.predict(test_X)
gain = 1.0
totalMinutes = 0
baseline = 1.0

print(len(yhat))

for day in range(len(yhat)):
    bought = False
    priceBought = 0
    t1 = 0

    baseline *= test_prices[day][389][0] / test_prices[day][0][0]

    # start 30 minutes into day
    for minute in range(30, len(yhat[day])):

        if not bought and yhat[day][minute][0] > 0.475:
            bought = True
            priceBought = test_prices[day][minute][0]
            t1 = minute
        if bought and yhat[day][minute][0] < 0.425:
            bought = False
            print("Bought at " + str(priceBought) + ". Sold at " + str(test_prices[day][minute][0]) + ". Held for " + str(minute - t1) + " minutes.")
            gain *= test_prices[day][minute][0] / priceBought
            totalMinutes += minute-t1
    if bought:
        print("Bought at " + str(priceBought) + ". Sold at " + str(test_prices[day][389][0]) + ". Held for " + str(389 - t1) + " minutes. EOD.")
        gain *= test_prices[day][389][0] / priceBought
        totalMinutes += 389-t1

print("Overall gain is " + str(gain) + "%" + " in " + str(totalMinutes) + " minutes.")
print("Baseline gain is " + str(baseline) + "%" + " in " + str(len(yhat)*350) + " minutes.")

# test_X[0][9][0] = 0
# test_X[0][9][1] = 0
# yhat = model.predict(test_X)

# test_X[0][5][0] = 0
# test_X[0][5][1] = 0
# yhat = model.predict(test_X)
