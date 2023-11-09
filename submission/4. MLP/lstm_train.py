import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

col = ['longitude','t82usi','z74si','v03si','vc2si','u96si','u14si','u06si','s63si','s58si',
       'n2iusi','ken','grin','g13si',
       'ap4si','0799hk','n_shopping_malls_0.2_km','n_primary_schools_0.2_km',
       'n_shopping_malls_0.1_km','n_primary_schools_0.1_km','rent_approved_year']
col2 = ['z25si','u11si','s51si','s08si','p15si','o39si','me8usi','maxn','m44usi'
       ,'k71usi','hmnsi','grab','g07si','flex','f34si','e5hsi','d05si','cjlusi','c52si','c07si'
       ,'buousi','bn4si','asln','1792hk']
col3 = ['c09si','a17usi']

# Load the preprocessed data from the CSV file
data = pd.read_csv("train-data-cleaned-v2.csv")

# Define the feature columns (X) and the target column (Y)
X = data.drop(columns=['monthly_rent']).drop(columns=col).drop(columns=col2).drop(columns=col3)
Y = data['monthly_rent']

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Reshape data for LSTM
X_train = X_train.values.reshape(-1, 1, X_train.shape[1])
X_test = X_test.values.reshape(-1, 1, X_test.shape[1])

# Define an LSTM model for regression
model = Sequential()
model.add(LSTM(100, activation='relu', input_shape=(1, X_train.shape[2]), return_sequences=True))
model.add(LSTM(50, activation='relu'))
model.add(Dense(1))  # Output layer with a single neuron for regression

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
history = model.fit(X_train, Y_train, epochs=100, batch_size=32, verbose=1, validation_data=(X_test, Y_test))

# Make predictions on the test data
Y_pred = model.predict(X_test)

# Calculate the Mean Squared Error (MSE) for regression
mse = mean_squared_error(Y_test, Y_pred)
rmse = np.sqrt(mse)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)

# Save the model
model.save("lstm_model.h5")

# Load and predict on test data
test_data = pd.read_csv("test-data-cleaned-v2.csv").drop(columns=col).drop(columns=col2).drop(columns=col3)
test_data = test_data.values.reshape(-1, 1, test_data.shape[1])
predictions = model.predict(test_data)
predictions_df = pd.DataFrame({'Predicted': predictions.flatten()})
predictions_df.to_csv('lstm_predictions.csv', index=False)

# Plot the training loss over epochs
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss and Validation Loss Over Epochs')
plt.legend()
plt.show()