<!-- ABOUT THE PROJECT -->
##Google Stock Price Prediction Using LSTM

This project aims to predict Google's stock prices using historical data and Long Short-Term Memory (LSTM) neural networks. LSTM is a type of recurrent neural network (RNN) that is particularly effective in learning patterns and trends in time-series data, such as stock prices. The model predicts future stock prices based on past performance and can be a helpful tool for stock market analysis.

Data Preprocessing
The dataset consists of historical stock prices, and the following preprocessing steps are applied:

Outlier Detection:

Outliers are identified using boxplots and Z-scores. Data points with Z-scores greater than 3 are considered outliers and removed.
Scaling:

The stock prices are scaled between 0 and 1 using MinMaxScaler to help the LSTM model learn more effectively.
Data Splitting:

The dataset is split into training (70%) and testing (30%) datasets.
Model Overview
This project uses Long Short-Term Memory (LSTM), a type of recurrent neural network (RNN) designed to handle time-series data. LSTM is well-suited for stock price prediction because it can capture long-term dependencies in data.

The model consists of several LSTM layers with Dropout to prevent overfitting.
The output layer consists of a single neuron to predict the stock price.
LSTM Model Architecture:
Input Layer: Accepts the input features (scaled stock prices).
LSTM Layers: Four LSTM layers to capture time dependencies.
Dropout Layer: Prevents overfitting by randomly setting a fraction of inputs to 0 during training.
Dense Layer: Final layer that outputs the predicted stock price.
Evaluation
The model’s performance is evaluated using the mean squared error (MSE) and root mean squared error (RMSE), which are standard metrics for regression tasks. Lower values of MSE and RMSE indicate better model performance.

Model Accuracy:
The accuracy is calculated based on how closely the model’s predictions align with the actual stock prices in the test dataset.

Results
The model is evaluated by comparing the predicted stock prices with the real stock prices. Visualizations are provided to show how well the predictions match the actual stock prices over time.

