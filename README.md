<!-- ABOUT THE PROJECT -->

## Google Stock Price Prediction Using LSTM

This project aims to predict Google's stock prices using historical data and Long Short-Term Memory (LSTM) neural networks. LSTM is a type of recurrent neural network (RNN) that is particularly effective in learning patterns and trends in time-series data, such as stock prices. The model predicts future stock prices based on past performance and can be a helpful tool for stock market analysis.

### Installation 

```js
pip install pandas numpy matplotlib seaborn scikit-learn imblearn keras tensorflow
```

### Data Preprocessing
Load the dataset, detect and remove outliers, and normalize the stock prices using MinMaxScaler.
```js
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from scipy import stats

# Load the data
training_data = pd.read_csv('GOOG.csv')

# Outlier detection using Z-scores
z_scores = np.abs(stats.zscore(training_data.iloc[:, 1:2]))
outliers = np.where(z_scores > 3)
print("Outliers detected at indices:", outliers)

# Remove outliers
training_data_no_outliers = training_data[(z_scores < 3).all(axis=1)]

# Preprocess the data (use stock prices)
training_data = training_data_no_outliers.iloc[:, 1:2]

# Normalize the data
mm = MinMaxScaler(feature_range=(0, 1))
training_data_scaled = mm.fit_transform(training_data)
```

## Building and Training the Model
```js
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

# Split the data into training and testing sets
train_size = int(len(training_data_scaled) * 0.7)
train_data = training_data_scaled[:train_size]
test_data = training_data_scaled[train_size:]

# Prepare training data
x_train = train_data[:-1]
y_train = train_data[1:]

# Reshape data for LSTM input
x_train = np.reshape(x_train, (x_train.shape[0], 1, 1))

# Build the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.1))
model.add(LSTM(units=50))
model.add(Dense(units=1))

# Compile and fit the model
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, epochs=100, batch_size=32)
```

Results
The model is evaluated by comparing the predicted stock prices with the real stock prices. Visualizations are provided to show how well the predictions match the actual stock prices over time.

###  Predictions and Evaluation

```js
# Predicting stock prices
input_value = test_data
input_value = mm.transform(input_value)
input_value = np.reshape(input_value, (input_value.shape[0], 1, 1))

predictions = model.predict(input_value)
predictions = mm.inverse_transform(predictions)

# Plot real vs predicted stock prices
plt.figure(figsize=(15, 8))
plt.plot(test_data, color='red', label='Real Stock Price')
plt.plot(predictions, color='blue', label='Predicted Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()

# Calculate RMSE and accuracy
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(test_data, predictions)
rmse = np.sqrt(mse)
accuracy = 100 - (rmse / np.max(test_data) * 100)
print(f'Accuracy: {accuracy:.2f}%')
```
![Screenshot 2024-11-11 051108](https://github.com/user-attachments/assets/53193d0a-a9b3-423a-bc6b-ebde07641490)

