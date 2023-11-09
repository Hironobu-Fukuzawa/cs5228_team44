import pandas as pd
import joblib

# Load the trained Random Forest model
mlp_model = joblib.load('mlp_model.pkl')

data = pd.read_csv("test-data-cleaned-v2.csv")
# Make predictions on the test data
predictions = mlp_model.predict(data)

# Create a DataFrame to store the predictions (you can customize this part as needed)
predictions_df = pd.DataFrame({'Predicted': predictions})
# Save the predictions to a CSV file or perform any other required actions
predictions_df.to_csv('mlp_predictions.csv', index=False)
