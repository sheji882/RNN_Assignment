Weather Temperature Prediction using SimpleRNN
Objective

The goal of this project is to forecast the next day’s temperature using past weather data with a Recurrent Neural Network (RNN) implemented in TensorFlow/Keras.

 Dataset

Source: Daily Weather Dataset – Kaggle

Features used:

Date

Temperature (C) (Target)

Humidity

Wind Speed (km/h)

Pressure (optional)

⚙️ Steps Performed
1. Data Loading & Exploration

Loaded dataset using Pandas.

Displayed first 10 rows.

Plotted temperature trends over time using Matplotlib/Seaborn.

Checked for missing values.

2. Data Preprocessing

Handled missing values (imputation/removal).

Renamed columns for easier access.

Converted Date column to datetime.

Normalized features (Temperature, Humidity, Wind Speed) using MinMaxScaler.

3. Sequence Preparation

Used past 14 days as input sequence (X).

Target (y) is next day’s temperature.

Split dataset into train (70%), validation (15%), and test (15%) sets.

4. Model Development

Built a stacked SimpleRNN model:

SimpleRNN(50, return_sequences=True)

SimpleRNN(50, return_sequences=False)

Dense(1) (output layer for regression)

Compiled with:

Optimizer: adam

Loss: mse

Metric: mae

5. Model Training

Trained model for 50 epochs, batch size = 32.

Used EarlyStopping to avoid overfitting.

Plotted training vs validation loss curves.

6. Model Evaluation

Predictions made on test set.

Calculated metrics:

RMSE (Root Mean Squared Error)

MAE (Mean Absolute Error)

R² Score

Plotted actual vs predicted temperatures.

7. Forecasting Future Temperatures

Used the last test sequence to forecast next 7 days’ temperatures iteratively.

Visualized forecast vs recent historical temperatures.
