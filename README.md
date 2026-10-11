# Country Development Classification

A machine learning project that classifies countries using economic, trade, and health indicators. The project uses a trained classification model, data preprocessing, and a Flask API to generate predictions for individual countries or multiple countries at once.

## Project Overview

This project explores how indicators such as income, exports, imports, health expenditure, and inflation can be used in a machine learning classification task.

A Flask API connects the trained model to prediction requests and returns predicted labels, confidence scores, class probabilities, and feature contribution values.

## Features

- Machine learning classification using a saved model.
- Data preprocessing with logarithmic transformations.
- StandardScaler preprocessing.
- Individual country predictions through a Flask API.
- Batch predictions for up to 2,000 rows per request.
- Prediction confidence and class probability outputs.
- Feature contribution values for individual predictions.

## Technology Stack

- Python
- Flask
- pandas
- NumPy
- scikit-learn
- Joblib
- REST API and JSON

## Input Indicators

The application expects nine numeric indicators. The exact feature names and order are determined by the saved scaler.

The code explicitly applies logarithmic transformations to:

- Income
- GDP per capita (`gdpp`)
- Exports
- Imports
- Health expenditure (`health`)
- Inflation, using a signed logarithmic transformation

The original training notebook should be consulted to confirm the meaning, units, and complete list of indicators.

## Project Structure

```text
country-development-classification/
├── app.py
├── classifier.pkl
├── scaler.pkl
├── templates/
│   └── index.html
└── README.md
```

This is an example structure. Your actual files may differ, and the model and scaler filenames must match those referenced in the Python application.

## Getting Started

### Prerequisites

- Python
- pip
- The trained classifier and scaler files

### Install Dependencies

```bash
pip install flask pandas numpy scikit-learn joblib
```

### Run the Application

Ensure the model and scaler files are in the same directory as the Flask application.

Run:

```bash
python app.py
```

The Flask application starts on port `5000`.

The home page requires an `index.html` file inside the `templates` directory.

## API Endpoints

### 1. Individual Prediction

**Endpoint:** `POST /predict`

Accepts a JSON object containing all nine required numeric indicators.

The API returns:

- `label`: predicted class
- `confidence`: highest predicted class probability as a percentage
- `probabilities`: probabilities for the available classes
- `contributions`: feature contribution values calculated by the application

The exact JSON field names depend on the features recorded by the saved scaler.

### 2. Batch Prediction

**Endpoint:** `POST /predict_batch`

Accepts a JSON object containing a `rows` array. Each row must include all nine required numeric indicators.

The API supports between 1 and 2,000 rows per request and returns:

- `labels`: predicted class for each row
- `confidence`: maximum predicted class probability for each row

Invalid or incomplete input produces an error response.

## How It Works

1. The application loads the trained classifier and saved scaler.
2. Incoming numeric data is organized into a DataFrame using the expected feature order.
3. Logarithmic transformations are applied to the specified indicators.
4. The saved scaler transforms the processed features.
5. The classifier generates predictions and class probabilities.
6. The API returns the results as JSON.

## Limitations

- Model accuracy and other evaluation metrics have not been provided.
- The original training dataset and notebook are not included in the files reviewed.
- The predicted labels depend on the original training labels and cannot be interpreted reliably without their definitions.
- Confidence values are model probabilities and may not represent calibrated real-world certainty.
- Feature contribution values are calculated using model coefficients and scaled features; their interpretation depends on the trained model and preprocessing.

## Future Improvements

- Document the dataset, feature definitions, and target classes.
- Add verified evaluation metrics.
- Include automated tests for individual and batch predictions.
- Improve input validation and error handling.
- Add visualizations to help interpret country classifications.
- Configure the application securely before deployment.

## Author

**Tayyaba Maqsood**

BSCS Student | AI/ML Enthusiast | Python & Machine Learning

## Disclaimer

This project is intended for educational and portfolio purposes. Predictions reflect the patterns learned by the trained model and should not be treated as definitive assessments of a country's development.
