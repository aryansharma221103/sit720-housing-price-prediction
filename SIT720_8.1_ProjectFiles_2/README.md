# Sydney Housing Price Prediction and Decision Support System
SIT720 8.1 Distinction Task — Machine Learning Mini Project (Aryan)

## Contents

- `data/raw_collection/*.txt` — raw sold-listing data, hand-transcribed from domain.com.au's sold-listings
  search pages for Mosman, Parramatta and Mount Druitt (one file per browsed page; see report Part 1.2 for
  the full collection method).
- `data/parse_real_listings.py` — parses `data/raw_collection/*.txt` into `data/sydney_housing_sold.csv`,
  the dataset itself (115 real sold properties across the three suburbs).
- `data/sydney_housing_sold.csv` — the manually collected dataset used throughout the notebook and app.
- `notebook/build_notebook.py` — script that programmatically builds the analysis notebook.
- `notebook/SIT720_8.1_Distinction_Task.ipynb` — the executed analysis notebook (Parts 1–5): EDA, feature
  engineering, three regression models with 5-fold CV, error analysis, and the ML vs LLM vs human comparison.
  Re-run with `jupyter nbconvert --to notebook --execute --inplace notebook/SIT720_8.1_Distinction_Task.ipynb`.
- `app/app.py` — the Streamlit decision-support application (Part 6). Run with `streamlit run app/app.py`
  from the project root (or `cd app && streamlit run app.py`). Requires `app/best_model.joblib` and
  `app/model_meta.json`, both produced by running the notebook end-to-end.
- `figures/` — all charts and app screenshots used in the report.
- `report_data/` — exact result tables (CSV/JSON) exported by the notebook's last run, used to build the report.
- `report/build_report.py`, `report/render_pdf.py` — scripts that build the final PDF report from
  `report_data/` and `figures/`.
- `report/build_docx.js` — script that builds the final Word (.docx) report from the same data.
- `report/SIT720_8.1_Report.pdf`, `report/SIT720_8.1_Report.docx` — the final submitted report (Parts 1–6).

## Quick start

```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn joblib streamlit jupyter nbconvert nbformat

# 1. (Re)parse the manually collected raw listings into the dataset
python3 data/parse_real_listings.py

# 2. (Re)build and execute the notebook
python3 notebook/build_notebook.py
jupyter nbconvert --to notebook --execute --inplace notebook/SIT720_8.1_Distinction_Task.ipynb

# 3. Run the deployed app
cd app && streamlit run app.py
```

## Data collection

Unlike an earlier draft of this project, the dataset is **not synthetic**. All 115 rows in
`data/sydney_housing_sold.csv` were manually collected from domain.com.au's sold-listings search pages
(one page opened and read at a time), transcribed by hand into `data/raw_collection/*.txt`, and parsed with
`data/parse_real_listings.py`. Listings marked "Price Withheld" were excluded. See report Part 1.2 for the
full method and Part 1.3 for the data-quality limitations this introduces (particularly price-disclosure
selection bias, which differs sharply by suburb).

## GenAI acknowledgement

See the acknowledgement box at the top of `report/SIT720_8.1_Report.pdf` / `report/SIT720_8.1_Report.docx`
and the first markdown cell of the notebook for the full disclosure of how GenAI (Claude, Anthropic) was
used in this project, including its role in the browser-assisted manual data collection described above.
