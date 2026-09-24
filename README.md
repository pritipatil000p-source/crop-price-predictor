# 🌾 Crop Price Predictor

Predict Indian mandi (wholesale market) commodity prices using historical price data and a Random Forest machine learning model.

---

## 📋 Project Overview

This application predicts **modal prices** of agricultural commodities in Indian mandis. It uses real historical daily wholesale price data from Kaggle to train a **RandomForestRegressor** and serves predictions through a **FastAPI** backend, displayed via a **Streamlit** frontend.

---

## ✨ Features

- 📊 **Market Overview** — Latest modal, min, and max prices for selected commodity/market
- 📈 **Historical Price Chart** — Interactive Plotly chart showing price trends
- 🤖 **ML Price Prediction** — Predict future prices by selecting State → District → Market → Commodity → Date
- 📊 **Model Performance Metrics** — MAE, RMSE, and R² displayed in the dashboard
- 📋 **Historical Data Table** — Browse recent records
- 🗄️ **Prediction History** — All predictions logged to SQLite

---

## 📁 Project Structure

```
crop-price-predictor/
│
├── backend/
│   ├── app/
│   │   ├── main.py              ← FastAPI app entry point
│   │   ├── schemas.py           ← Pydantic request/response models
│   │   ├── database.py          ← SQLite + SQLAlchemy setup
│   │   ├── api/
│   │   │   └── routes.py        ← All API route handlers
│   │   ├── services/
│   │   │   ├── prediction_service.py  ← Load model + predict
│   │   │   └── data_service.py        ← Dataset access helpers
│   │   └── models/
│   │       └── prediction.py    ← ORM model
│   └── requirements.txt
│
├── frontend/
│   ├── app.py                   ← Streamlit dashboard
│   ├── api_client.py            ← HTTP client for FastAPI
│   └── requirements.txt
│
├── data/
│   ├── daily_price.csv          ← ← ← Place Kaggle CSV here
│   └── README.md                ← Dataset download instructions
│
├── ml/
│   ├── preprocess.py            ← Data loading and cleaning
│   ├── train.py                 ← Model training script
│   ├── evaluate.py              ← Model evaluation script
│   ├── saved_model.pkl          ← (Generated after training)
│   ├── metrics.json             ← (Generated after training)
│   └── metadata.json            ← (Generated after training)
│
├── notebooks/
│   └── exploration.ipynb        ← EDA notebook
│
├── README.md
└── .gitignore
```

---

## 📦 Dataset

### Source

This project uses the **Indian Mandi (Agricultural Market) Daily Price Dataset** from Kaggle.

**Recommended dataset:**
> https://www.kaggle.com/datasets/srinivas1/agriculturalcommoditiesdataset

Search Kaggle for: **"Price of Agricultural Commodities Data"**

### Required columns

| Column        | Description                            |
|---------------|----------------------------------------|
| State         | Indian state name                      |
| District      | District name                          |
| Market        | Mandi / market name                    |
| Commodity     | Crop or commodity name                 |
| Variety       | Crop variety                           |
| Grade         | Quality grade (FAQ, etc.)              |
| Min Price     | Minimum wholesale price (INR/quintal)  |
| Max Price     | Maximum wholesale price (INR/quintal)  |
| Modal Price   | **Target variable** — modal price      |
| Date          | Date of price record                   |

---

## ⚙️ Installation (Windows)

### 1. Clone the repository

```cmd
git clone <your-repo-url>
cd crop-price-predictor
```

### 2. Create a virtual environment

```cmd
python -m venv venv
```

### 3. Activate the virtual environment

```cmd
venv\Scripts\activate
```

### 4. Install backend dependencies

```cmd
pip install -r backend/requirements.txt
```

### 5. Install frontend dependencies

```cmd
pip install -r frontend/requirements.txt
```

---

## 📂 Dataset Placement

### Option A — Manual Download

1. Visit: https://www.kaggle.com/datasets/srinivas1/agriculturalcommoditiesdataset
2. Click **Download** → extract the ZIP
3. Rename the CSV to `daily_price.csv`
4. Place it at:

```
data/daily_price.csv
```

### Option B — Kaggle CLI

```cmd
pip install kaggle
```

Set up your API token (download `kaggle.json` from https://www.kaggle.com/account):

```cmd
mkdir %USERPROFILE%\.kaggle
copy kaggle.json %USERPROFILE%\.kaggle\kaggle.json
```

Download:

```cmd
kaggle datasets download -d srinivas1/agriculturalcommoditiesdataset --unzip -p data/
```

Then rename the CSV file:

```cmd
ren data\*.csv daily_price.csv
```

---

## 🤖 Training the Model

From the `crop-price-predictor/` directory, run:

```cmd
python ml/train.py
```

This will:
1. Load and clean `data/daily_price.csv`
2. Print a data inspection report
3. Split data chronologically (80% train / 20% test)
4. Train a `RandomForestRegressor` pipeline
5. Evaluate on the test set (MAE, RMSE, R²)
6. Save the trained model to `ml/saved_model.pkl`
7. Save metrics to `ml/metrics.json`
8. Save dropdown metadata to `ml/metadata.json`

Expected output:
```
Shape: 1,000,000+ rows × 9 columns
...
MAE  : 350.42 INR/quintal
RMSE : 820.15 INR/quintal
R²   : 0.9120
✅ Model saved to: ml/saved_model.pkl
```

### Re-evaluate the model (optional)

```cmd
python ml/evaluate.py
```

---

## 🚀 Starting the Backend (FastAPI)

Open **Terminal 1**. From the `crop-price-predictor/` directory:

```cmd
venv\Scripts\activate
uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at:
- API root: http://localhost:8000/
- Health check: http://localhost:8000/health
- Interactive docs (Swagger UI): http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

---

## 🌐 Starting the Frontend (Streamlit)

Open **Terminal 2**. From the `crop-price-predictor/` directory:

```cmd
venv\Scripts\activate
streamlit run frontend/app.py
```

The dashboard will open automatically at: http://localhost:8501

---

## 🔗 API Endpoints

| Method | Endpoint             | Description                          |
|--------|----------------------|--------------------------------------|
| GET    | `/`                  | API root message                     |
| GET    | `/health`            | Health check                         |
| GET    | `/commodities`       | List all commodities                 |
| GET    | `/states`            | List all states                      |
| GET    | `/districts`         | List districts (filter: `?state=`)   |
| GET    | `/markets`           | List markets (filter: `?state=&district=`) |
| GET    | `/varieties`         | List varieties                       |
| GET    | `/grades`            | List grades                          |
| GET    | `/historical-prices` | Historical prices (filter by commodity, market, dates) |
| POST   | `/predict`           | Predict modal price                  |
| GET    | `/model-metrics`     | MAE, RMSE, R² of the trained model   |
| GET    | `/prediction-history`| Past predictions from the database   |

---

## 📬 Example Prediction Request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "state": "Maharashtra",
    "district": "Pune",
    "market": "Pune",
    "commodity": "Tomato",
    "variety": "Local",
    "grade": "FAQ",
    "prediction_date": "2026-10-15"
  }'
```

**Response:**

```json
{
  "commodity": "Tomato",
  "market": "Pune",
  "prediction_date": "2026-10-15",
  "predicted_price": 2450.50,
  "unit": "INR per quintal",
  "disclaimer": "This is an ML model estimate based on historical data. It is NOT a guaranteed market price."
}
```

---

## 📊 Model Evaluation

The model is evaluated on the **most recent 20%** of the dataset (chronological split — no data leakage).

| Metric | Description                              |
|--------|------------------------------------------|
| MAE    | Mean Absolute Error (INR/quintal)        |
| RMSE   | Root Mean Squared Error (INR/quintal)    |
| R²     | Coefficient of determination (0–1 scale) |

Results vary depending on the size and quality of your downloaded dataset.

---

## ⚠️ Limitations

1. **Predictions are estimates**, not guaranteed market prices.
2. The model cannot predict sudden price spikes due to weather, policy changes, or supply shocks.
3. Accuracy depends on the dataset size and how representative the training data is.
4. For commodities or markets with very few historical records, predictions will be less reliable.
5. The model uses chronological splitting — it does **not** train on future data.
6. Min/max prices for future dates are estimated from recent historical averages.

---

## 🛠️ Troubleshooting

| Problem | Solution |
|---------|----------|
| `FileNotFoundError: data/daily_price.csv` | Download the Kaggle dataset and place it at `data/daily_price.csv` |
| `FileNotFoundError: ml/saved_model.pkl` | Run `python ml/train.py` |
| `Cannot connect to backend API` | Start the FastAPI server: `uvicorn backend.app.main:app --reload` |
| `No states found` | Train the model first (generates `ml/metadata.json`) |
| Column mapping error | Check `ml/preprocess.py` COLUMN_MAP and match to your CSV columns |
| `sparse_output` error | Upgrade scikit-learn: `pip install scikit-learn>=1.2` |

---

## 📜 License

This project is for educational and research purposes. The dataset is subject to Kaggle's terms of use.
