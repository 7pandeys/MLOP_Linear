# Linear Regression API

This project implements a simple Linear Regression model and deploys it using FastAPI, Docker, and GitHub Actions.

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/linear-regression-api.git
   cd linear-regression-api
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the FastAPI server:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```
4. Use the API endpoint:
   ```bash
   curl "http://127.0.0.1:8000/predict?x=5.0"
   ```

## Docker Instructions

1. Build the Docker image:
   ```bash
   docker build -t linear-regression-api .
   ```
2. Run the container:
   ```bash
   docker run -p 8000:8000 linear-regression-api
   ```

## CI/CD using GitHub Actions

- The workflow in `.github/workflows/mlops.yml` automates building, testing, and deploying the model.