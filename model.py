import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from fastapi import FastAPI

# Step 2: Generate synthetic dataset
# Creating a DataFrame with a single feature column "X"
data = pd.DataFrame({
    "X": np.random.rand(100) * 10,  # Generating 100 random values between 0 and 10
})
# Generating target variable "y" using a linear equation with some noise
data["y"] = 2.5 * data["X"] + np.random.randn(100) * 2

# Step 3: Train a simple Linear Regression model
# Splitting the dataset into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(data[["X"]], data["y"], test_size=0.2, random_state=42)

# Initializing the Linear Regression model
model = LinearRegression()

# Training the model using the training dataset
model.fit(X_train, y_train)

# Step 4: Evaluate and save the model
# Making predictions on the test dataset
y_pred = model.predict(X_test)

# Printing the Mean Squared Error (MSE) to evaluate model performance
print(f"MSE: {mean_squared_error(y_test, y_pred)}")

# Saving the trained model to a file using joblib
joblib.dump(model, "model.pkl")

# Step 5: Create an API using FastAPI
# Initializing the FastAPI app
app = FastAPI()

# Loading the trained model from the saved file
loaded_model = joblib.load("model.pkl")

# Creating an endpoint to get predictions
@app.get("/predict")
def predict(x: float):
    """Predict function to return model output for a given input value."""
    prediction = loaded_model.predict(np.array([[x]]))  # Converting input to a 2D array for prediction
    return {"prediction": prediction.tolist()}  # Returning the prediction as a dictionary