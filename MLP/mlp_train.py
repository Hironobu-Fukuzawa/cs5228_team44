import datetime

import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt  # Add this import for plotting
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
import joblib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor

col = ['longitude','t82usi','z74si','v03si','vc2si','u96si','u14si','u06si','s63si','s58si',
       'n2iusi','ken','grin','g13si',
       'ap4si','0799hk','n_shopping_malls_0.2_km','n_primary_schools_0.2_km',
       'n_shopping_malls_0.1_km','n_primary_schools_0.1_km','rent_approved_year']
col2 = ['z25si','u11si','s51si','s08si','p15si','o39si','me8usi','maxn','m44usi'
       ,'k71usi','hmnsi','grab','g07si','flex','f34si','e5hsi','d05si','cjlusi','c52si','c07si'
       ,'buousi','bn4si','asln','1792hk']
col3 = ['c09si','a17usi']

print(datetime.datetime.now())
# Load the preprocessed data from the CSV file
data = pd.read_csv("train-data-cleaned-v2.csv")

# Define the feature columns (X) and the target column (Y)
X = data.drop(columns=['monthly_rent']).drop(columns=col).drop(columns=col2).drop(columns=col3)
Y = data['monthly_rent']

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Define a function to create the model for KerasRegressor
def create_model(learning_rate=0.001, hidden_layer_sizes=(100, 50, 10), activation='relu'):
    model = Sequential([
        Dense(hidden_layer_sizes[0], activation=activation, input_shape=(X_train.shape[1],)),
        Dense(hidden_layer_sizes[1], activation=activation),
        Dense(hidden_layer_sizes[2], activation=activation),
        Dense(1)  # Output layer with a single neuron for regression
    ])
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model

# Create a KerasRegressor model for GridSearchCV
keras_regressor = KerasRegressor(build_fn=create_model, verbose=0)

# Define hyperparameters to search
param_grid = {
    'learning_rate': [0.01],
    'hidden_layer_sizes': [(100,50,10), (200, 50, 10), (500, 250, 100)],
    'activation': ['relu', 'tanh', 'sigmoid']
}

# Lists to record the training loss for each epoch
training_loss_history = []

# Define a function to record the training loss for each epoch
def record_training_loss(epoch, logs):
    training_loss_history.append(logs['loss'])

# Callback to record training loss at the end of each epoch
loss_callback = tf.keras.callbacks.LambdaCallback(on_epoch_end=record_training_loss)

# Perform grid search to find the best hyperparameters
grid_search = GridSearchCV(estimator=keras_regressor, param_grid=param_grid, scoring='neg_mean_squared_error', cv=3)
grid_search.fit(X_train, Y_train)

# Get the best model and its hyperparameters
best_model = grid_search.best_estimator_
best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)

# Train the best model and record the training loss
history = best_model.fit(X_train, Y_train, epochs=100, batch_size=32, verbose=1, callbacks=[loss_callback])
print(datetime.datetime.now())
# Make predictions on the test data
Y_pred = best_model.predict(X_test)

# Calculate the Mean Squared Error (MSE) for regression
mse = mean_squared_error(Y_test, Y_pred)
rmse = np.sqrt(mse)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)

# Save the best model
joblib.dump(best_model, 'best_mlp_model.pkl')

# Load and predict on test data
test_data = pd.read_csv("test-data-cleaned-v2.csv").drop(columns=col).drop(columns=col2).drop(columns=col3)
predictions = best_model.predict(test_data)
predictions_df = pd.DataFrame({'Predicted': predictions.flatten()})
predictions_df.to_csv('best_mlp_predictions2.csv', index=False)

# Plot the training loss over epochs
plt.plot(range(1, len(training_loss_history) + 1), training_loss_history)
plt.xlabel('Epochs')
plt.ylabel('Training Loss')
plt.title('Training Loss Over Epochs')
plt.show()
