# Dataset Instructions

## Required Dataset

This project uses the **"Price of Agricultural Commodities Data"** from Kaggle.

### Dataset Source

**Kaggle URL:**
https://www.kaggle.com/datasets/kianwee/agricultural-raw-material-prices-1990-2020

Or search on Kaggle for:
> "Price of Agricultural Commodities Data" OR "daily price mandi india"

The recommended dataset is the Indian mandi (wholesale market) daily prices dataset.
A commonly used source:
https://www.kaggle.com/datasets/srinivas1/agriculturalcommoditiesdataset

---

## How to Download

### Option 1: Kaggle Web UI

1. Visit the Kaggle dataset page (see URL above).
2. Click the **Download** button (top-right).
3. Extract the downloaded ZIP file.
4. Place the CSV file in this folder as:

```
data/daily_price.csv
```

### Option 2: Kaggle CLI

Install the Kaggle CLI if you haven't already:

```bash
pip install kaggle
```

Set up your Kaggle API token:
1. Go to https://www.kaggle.com/account
2. Click "Create New API Token"
3. Save the downloaded `kaggle.json` to `C:\Users\<YourName>\.kaggle\kaggle.json`

Then run (example — replace with actual dataset slug):

```bash
kaggle datasets download -d srinivas1/agriculturalcommoditiesdataset --unzip -p data/
```

Rename the downloaded file to `daily_price.csv` if needed.

---

## Expected Columns

The CSV must contain these columns (exact names may vary — the code normalizes them):

| Original Column | Normalized Name |
|----------------|-----------------|
| State          | state           |
| District       | district        |
| Market         | market          |
| Commodity      | commodity       |
| Variety        | variety         |
| Grade          | grade           |
| Min Price      | min_price       |
| Max Price      | max_price       |
| Modal Price    | modal_price     |
| Date/Arrival Date | date         |

---

## File Location

After placing the file, your directory should look like:

```
data/
└── daily_price.csv    ← Place the downloaded CSV here
```

---

## Note

Do NOT rename or move the file. The ML training scripts and backend service both
expect the file at exactly: `data/daily_price.csv`
