# EV Charging Infrastructure Gap Prediction

## Project Overview

This project predicts Electric Vehicle (EV) charging infrastructure demand at the U.S. census tract level and identifies potential infrastructure gaps. A trained machine learning model estimates the expected number of EV charging ports in each tract using demographic, economic, and built-environment features. By comparing predicted charging demand with existing infrastructure, the project highlights underserved areas where additional EV charging stations may be needed.

## How to Run the Project

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/ev-gap-prediction.git
cd ev-gap-prediction
```

### 2. Install Required Dependencies

Install required Python libraries:

```bash
pip install -r requirements.txt
```

If a requirements file is not provided, install manually:

```bash
pip install pandas numpy geopandas shapely scikit-learn lightgbm xgboost shap streamlit
```

### 3. Pre-trained Model

The trained model used for prediction is already saved in the repository:

```
model/xgb_total_ports_reduced.pkl
```

### 4. Run the Prediction Pipeline

Open the notebook:

```
prediction.ipynb
```

Run the pipeline using:

```python
pipeline.run("AZ")
```

Replace `"AZ"` with any U.S. state abbreviation to generate predictions for that state.

Examples:

```python
pipeline.run("CA")
pipeline.run("TX")
pipeline.run("NY")
```

The pipeline will automatically:

* Download required datasets
* Clean and preprocess the data
* Perform spatial integration
* Generate predictions using the trained model
* Calculate EV infrastructure gaps

The output will be saved as:

* **GeoJSON** file for spatial visualization
* **CSV** file containing prediction results

### 5. Run the Dashboard

To visualize results, start the Streamlit dashboard:

```bash
streamlit run dashboard.py
```

The dashboard allows you to explore:

* EV infrastructure gap map
* Underserved census tracts
* Overserved census tracts
* Balanced regions

Use the state selector to view predictions for different states.

## Author

Sanjeeb Adhikari
MS Data Science – Regis University
