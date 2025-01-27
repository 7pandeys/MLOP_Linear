from fastapi import FastAPI
import joblib
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