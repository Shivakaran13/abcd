# Heart Disease Prediction

A machine learning project to predict the presence of heart disease using clinical and diagnostic data. The project includes data preprocessing, model training and evaluation, experiment tracking with MLflow, and a simple Flask app for serving predictions.

## About The Project

Heart disease prediction helps identify individuals at risk of cardiovascular conditions so that early interventions can be made. This repository demonstrates an end-to-end workflow: dataset preparation, exploratory data analysis, model training with CatBoost / XGBoost (and other baselines), experiment tracking with MLflow, and a small Flask-based web interface to serve the trained model.

## Dataset

This dataset contains clinical and diagnostic measurements commonly used to predict heart disease. There are 13 features and 1 target column.

Attributes:
- Age
- Sex (Gender)
- Chest pain type (typical angina, atypical angina, non-anginal pain, asymptomatic)
- Resting blood pressure (in mm Hg)
- Serum cholesterol in mg/dl
- Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)
- Resting electrocardiographic results
- Maximum heart rate achieved
- Exercise-induced angina (1 = yes; 0 = no)
- ST depression induced by exercise relative to rest (called "oldpeak")
- Slope of the peak exercise ST segment
- Number of major vessels (0-3) colored by fluoroscopy
- Thalassemia (normal, fixed defect, reversible defect)
- Target (presence of heart disease: 0 = no disease, 1 = disease)

Note: Check the original dataset documentation or README in the data folder for exact encoding of categorical fields.

## Built With

- Python 3.x
- Pandas
- NumPy
- Scikit-learn
- CatBoost
- XGBoost
- MLflow
- DVC (optional, for data versioning)
- Flask (for the demo web app)
- Matplotlib / Seaborn (visualizations)

## Getting Started

These instructions will help you run the project locally for development and testing.

### Prerequisites

- Git
- Python 3.8+ (or your preferred version; ensure compatibility)
- Conda (optional, recommended) or venv
- Docker (optional, if using the Docker image)

### Option 1 — Install and run locally (recommended for development)

1. Clone the repository
```bash
git clone https://github.com/Shivakaran13/abcd.git
cd abcd
```

2. Create and activate a virtual environment (Conda example)
```bash
conda create -p ./env python=3.8 -y
conda activate ./env
```
Or using venv:
```bash
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows
.venv\Scripts\activate
```

3. Install Python dependencies
```bash
pip install -r requirements.txt
```

4. Prepare data / DVC (if using DVC)
- If the repo uses DVC to fetch large data, run (example):
```bash
dvc pull
```

5. Run the Flask app (example)
```bash
python app.py
```
By default the app may run on http://127.0.0.1:5000 — check `app.py` for configuration.

6. Use the API or web UI to send prediction requests (see the `app.py` / `README` within the app folder for usage examples).

### Option 2 — Run with Docker

1. Pull the Docker image (example — replace with actual image name on DockerHub if available)
```bash
docker pull shivakaran13/abcd:latest
```

2. Run the container and map required ports (example)
```bash
docker run -d --name heart-app -p 5000:5000 shivakaran13/abcd:latest
```

3. Access the app at http://localhost:5000

If you produce your own Docker build:
```bash
docker build -t shivakaran13/abcd:local .
docker run -p 5000:5000 shivakaran13/abcd:local
```

## MLflow Tracking Setup

This project uses MLflow to log experiments. You can configure MLflow by setting the `MLFLOW_TRACKING_URI` environment variable to point to your MLflow server (e.g., DagsHub or a self-hosted tracking server).

Example (Linux / macOS):
```bash
export MLFLOW_TRACKING_URI=https://dagshub.com/<username>/<repo>.mlflow
```

Example (Windows PowerShell):
```powershell
$env:MLFLOW_TRACKING_URI = "https://dagshub.com/<username>/<repo>.mlflow"
```

Run training scripts after setting the tracking URI so runs are recorded:
```bash
python run_training.py
```
(Replace `run_training.py` with the actual training script name in the repo.)

## Project Structure (example)
- data/               — raw and processed data
- src/                — code for training, evaluation, and utilities
- models/             — saved model artifacts
- app.py              — Flask application to serve predictions
- requirements.txt
- mlruns/             — local MLflow runs (if used locally)

Adjust to match the actual structure in this repository.

## Usage

- To train a model: run the training script (e.g., `python src/train.py`) and monitor runs in MLflow.
- To evaluate: run evaluation scripts in `src/` (e.g., `python src/evaluate.py`).
- To serve predictions locally: run `python app.py` and POST JSON payloads to the prediction endpoint.

Refer to the code in `src/` for exact script names, CLI arguments, and expected input formats.

## Contributing

Contributions are welcome! To contribute:

1. Fork the project
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Commit your changes: `git commit -m "Add my feature"`
4. Push to the branch: `git push origin feature/my-feature`
5. Open a Pull Request and describe the change

Please follow any existing CONTRIBUTING.md or style guidelines in the repo.

## License

Add a LICENSE file to your repository to indicate the project's license. If you don't have one yet, consider an open-source license such as MIT.

## Acknowledgements

Thanks to the contributors and data providers that made this project possible. This project references classical heart disease datasets and commonly used ML benchmarks — please cite or acknowledge the original dataset sources if reused in publications.
