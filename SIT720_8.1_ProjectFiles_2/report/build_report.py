"""Builds the final PDF report (SIT720_8.1_Report.pdf) from:
  - report_data/*.csv (exact result tables exported by the executed notebook)
  - figures/*.png (exact figures/screenshots produced by the executed notebook + app)
Renders a single self-contained HTML file, then prints it to PDF with Playwright/Chromium.
"""
import base64
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
DATA = ROOT / "report_data"
FIG = ROOT / "figures"
OUT_HTML = Path(__file__).parent / "report.html"
OUT_PDF = Path(__file__).parent / "SIT720_8.1_Report.pdf"

def b64(path):
    return base64.b64encode(Path(path).read_bytes()).decode("ascii")

def img_tag(filename, alt="", width="100%"):
    return f'<img src="data:image/png;base64,{b64(FIG / filename)}" alt="{alt}" style="width:{width};" />'

def money(x):
    try:
        return f'<span class="nowrap">${float(x):,.0f}</span>'
    except (ValueError, TypeError):
        return x

def df_to_html_table(df, money_cols=None, pct_cols=None, index=False, table_class="tbl", table_id=None):
    money_cols = money_cols or []
    pct_cols = pct_cols or []
    d = df.copy()
    for c in money_cols:
        if c in d.columns:
            d[c] = d[c].apply(money)
    for c in pct_cols:
        if c in d.columns:
            d[c] = d[c].apply(lambda x: f'<span class="nowrap">{float(x):.1f}%</span>')
    return d.to_html(index=index, classes=table_class, border=0, escape=(not money_cols and not pct_cols), table_id=table_id)

# ---------------------------------------------------------------------------
# Load result data
# ---------------------------------------------------------------------------
cv_df = pd.read_csv(DATA / "cv_df.csv")
test_df_results = pd.read_csv(DATA / "test_df_results.csv")
top5 = pd.read_csv(DATA / "top5_errors.csv")
comparison = pd.read_csv(DATA / "part5_comparison.csv")
approach_summary = pd.read_csv(DATA / "approach_summary.csv")
outliers = pd.read_csv(DATA / "outliers.csv")
missing = pd.read_csv(DATA / "missing_summary.csv")
suburb_describe = pd.read_csv(DATA / "suburb_price_describe.csv")
misc = json.loads((DATA / "misc_stats.json").read_text())

cv_html = df_to_html_table(cv_df, money_cols=["CV MAE", "CV RMSE"])
test_html = df_to_html_table(test_df_results, money_cols=["Train MAE", "Test MAE", "Test RMSE"])
top5_html = df_to_html_table(
    top5.rename(columns={
        "property_id": "ID", "suburb": "Suburb", "property_type": "Type", "bedrooms": "Bed",
        "land_size_sqm": "Land (sqm)", "actual_price": "Actual", "predicted_price": "Predicted",
        "abs_error": "Abs. error", "pct_error": "% error", "price_outlier": "Flagged outlier?",
        "address": "Address",
    })[["ID", "Suburb", "Type", "Bed", "Land (sqm)", "Actual", "Predicted", "Abs. error", "% error",
        "Flagged outlier?", "Address"]],
    money_cols=["Actual", "Predicted", "Abs. error"],
    table_id="top5-table",
)
comparison_html = df_to_html_table(
    comparison.rename(columns={
        "property_id": "ID", "suburb": "Suburb", "actual_price": "Actual",
        "ml_estimate": "ML", "llm_estimate": "LLM (Claude)", "human_estimate": "Human (comps)",
    }),
    money_cols=["Actual", "ML", "LLM (Claude)", "Human (comps)"],
)
approach_html = df_to_html_table(approach_summary.rename(columns={"Unnamed: 0": "Approach"}),
                                  money_cols=["MAE", "Max abs error"])
outliers_html = df_to_html_table(
    outliers.rename(columns={"property_id": "ID", "suburb": "Suburb", "property_type": "Type",
                              "bedrooms": "Bed", "bathrooms": "Bath",
                              "sale_price": "Sale price", "address": "Address"}),
    money_cols=["Sale price"],
    table_id="outliers-table",
)
missing_html = df_to_html_table(missing.rename(columns={"Unnamed: 0": "Feature", "n_missing": "# missing",
                                                          "pct_missing": "% missing"}))
suburb_describe["count"] = suburb_describe["count"].astype(int)
suburb_html = df_to_html_table(
    suburb_describe.rename(columns={"suburb": "Suburb", "count": "N", "mean": "Mean", "std": "Std dev",
                                     "min": "Min", "25%": "P25", "50%": "Median", "75%": "P75", "max": "Max"}),
    money_cols=["Mean", "Std dev", "Min", "P25", "Median", "P75", "Max"],
)

best_model = misc["best_model_name"]

# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------
CSS = """
<style>
  @page { size: A4; margin: 20mm 18mm 20mm 18mm; }
  * { box-sizing: border-box; }
  body {
    font-family: -apple-system, "Segoe UI", Helvetica, Arial, sans-serif;
    color: #1a1a1a; font-size: 10.6pt; line-height: 1.5;
  }
  h1 { font-size: 20pt; margin: 0 0 4pt 0; color: #14213d; }
  h2 { font-size: 14.5pt; margin: 22pt 0 8pt 0; color: #14213d; border-bottom: 2px solid #14213d; padding-bottom: 3pt; page-break-after: avoid; }
  h3 { font-size: 12pt; margin: 14pt 0 6pt 0; color: #1d3461; page-break-after: avoid; }
  h4 { font-size: 10.6pt; margin: 10pt 0 4pt 0; color: #1d3461; page-break-after: avoid; }
  p { margin: 0 0 8pt 0; text-align: left; }
  .cover { text-align: center; padding-top: 70mm; page-break-after: always; }
  .cover .subtitle { font-size: 13pt; color: #444; margin-top: 6pt; }
  .cover .meta { margin-top: 30pt; font-size: 11pt; color: #333; line-height: 1.9; }
  .badge { display:inline-block; background:#14213d; color:white; padding: 3pt 10pt; border-radius: 3pt; font-size: 10pt; margin-top: 18pt;}
  .genai-box, .todo-box, .note-box {
    border-left: 4px solid #7a5cff; background: #f5f3ff; padding: 8pt 12pt; margin: 10pt 0; font-size: 9.8pt; border-radius: 3pt;
  }
  .todo-box { border-left-color: #d97706; background: #fff7ed; }
  .note-box { border-left-color: #0891b2; background: #ecfeff; }
  table.tbl { border-collapse: collapse; width: 100%; margin: 8pt 0 12pt 0; font-size: 9pt; page-break-inside: avoid; }
  table.tbl th { background: #14213d; color: white; text-align: left; padding: 5pt 6pt; overflow-wrap: break-word; word-break: break-word; }
  table.tbl td { border-bottom: 1px solid #ddd; padding: 4pt 6pt; vertical-align: top; overflow-wrap: break-word; word-break: break-word; }
  table.tbl tr:nth-child(even) td { background: #f7f8fb; }
  .figure { margin: 10pt 0; text-align: center; page-break-inside: avoid; }
  .figure .cap { font-size: 8.8pt; color: #555; margin-top: 4pt; font-style: italic; }
  .two-col-fig { display: flex; gap: 10pt; }
  .two-col-fig .figure { flex: 1; }
  code, .code-inline { font-family: "SF Mono", Consolas, monospace; background: #f0f0f3; padding: 1pt 4pt; border-radius: 3pt; font-size: 9.2pt; }
  pre.codeblock {
    background: #1e1e2e; color: #e5e5e5; padding: 8pt 10pt; border-radius: 4pt; font-size: 8.6pt;
    overflow-x: auto; white-space: pre-wrap; word-wrap: break-word; page-break-inside: avoid;
  }
  ul, ol { margin: 4pt 0 8pt 18pt; padding: 0; }
  li { margin-bottom: 3pt; }
  .part-summary-table td, .part-summary-table th { font-size: 8.8pt; }
  .small { font-size: 8.8pt; color: #555; }
  .section-divider { border: none; border-top: 1px solid #ccc; margin: 16pt 0; }
  a { color: #1d4ed8; }
  .refs li { margin-bottom: 6pt; }
  .nowrap { white-space: nowrap; }

  table#top5-table { table-layout: fixed; font-size: 8.3pt; }
  table#top5-table th:nth-child(1),  table#top5-table td:nth-child(1)  { width: 8%; }
  table#top5-table th:nth-child(2),  table#top5-table td:nth-child(2)  { width: 10%; }
  table#top5-table th:nth-child(3),  table#top5-table td:nth-child(3)  { width: 8%; }
  table#top5-table th:nth-child(4),  table#top5-table td:nth-child(4)  { width: 5%; }
  table#top5-table th:nth-child(5),  table#top5-table td:nth-child(5)  { width: 8%; }
  table#top5-table th:nth-child(6),  table#top5-table td:nth-child(6)  { width: 10%; }
  table#top5-table th:nth-child(7),  table#top5-table td:nth-child(7)  { width: 10%; }
  table#top5-table th:nth-child(8),  table#top5-table td:nth-child(8)  { width: 9%; }
  table#top5-table th:nth-child(9),  table#top5-table td:nth-child(9)  { width: 7%; }
  table#top5-table th:nth-child(10), table#top5-table td:nth-child(10) { width: 8%; }
  table#top5-table th:nth-child(11), table#top5-table td:nth-child(11) { width: 17%; }

  table#outliers-table { table-layout: fixed; font-size: 8.6pt; }
  table#outliers-table th:nth-child(1), table#outliers-table td:nth-child(1) { width: 12%; }
  table#outliers-table th:nth-child(2), table#outliers-table td:nth-child(2) { width: 13%; }
  table#outliers-table th:nth-child(3), table#outliers-table td:nth-child(3) { width: 10%; }
  table#outliers-table th:nth-child(4), table#outliers-table td:nth-child(4) { width: 6%; }
  table#outliers-table th:nth-child(5), table#outliers-table td:nth-child(5) { width: 6%; }
  table#outliers-table th:nth-child(6), table#outliers-table td:nth-child(6) { width: 13%; }
  table#outliers-table th:nth-child(7), table#outliers-table td:nth-child(7) { width: 40%; }
</style>
"""

cover = f"""
<div class="cover">
  <div style="font-size:11pt;color:#666;letter-spacing:1px;">SIT720 &mdash; MACHINE LEARNING</div>
  <h1>Sydney Housing Price Prediction<br/>and Decision Support System</h1>
  <div class="subtitle">8.1 Distinction Task &mdash; Machine Learning Mini Project</div>
  <div class="meta">
    Student: <b>Aryan Sharma</b> (Student ID: 225616764)<br/>
    Unit: SIT720 &mdash; Machine Learning, Deakin University<br/>
    Report date: 1 October 2026<br/>
    Suburbs analysed: Mosman &middot; Parramatta &middot; Mount Druitt
  </div>
  <div class="badge">Notebook: SIT720_8.1_Distinction_Task.ipynb &nbsp;|&nbsp; App: app/app.py</div>
</div>
"""

genai_ack = """
<div class="genai-box">
<b>GenAI acknowledgement.</b> I used Generative AI tools (Claude, Anthropic) for brainstorming ideas and
structuring the Jupyter notebook, including organising the workflow and presentation of the analysis. I also
used Generative AI to help structure and format the accompanying Word/PDF report. For Part 1 of this
resubmission, I used a browser tool under my direct direction (one page opened and approved by me at a time) to
open domain.com.au's sold-listings search pages for each suburb; I read the sale price, address and feature
values off each listing card myself and transcribed them, excluding any listing marked "Price Withheld". As
required by Part 5, I also used Claude directly as the "large language model" valuation approach, compared
against my ML model and a human-judgement heuristic. The actual analysis, implementation, results,
interpretation and final content were completed and reviewed by me, and I take full responsibility for the
accuracy and integrity of the submitted work.
</div>
"""

part1 = f"""
<h2>Part 1 &mdash; Problem Definition and Data Collection</h2>

<h3>1.1 Problem, motivation and suburb selection</h3>
<p>
The goal of this project was to develop a decision-making support system (a model) that could help estimate a
"fair" or "reasonable" selling price of a home based on the attributes of the house and provide insight into
why that price is reasonable. In order to create models that were representative of different types of
residential housing markets, we chose to study three distinct and separate neighborhoods within sydney. These
three neighborhoods represent significantly different housing market environments with each neighborhood being
considered as a potential housing option at a different time in my own life:
</p>
<p>
1. Mosman - inner north harborside neighborhood. The most expensive, low density area with very high land
values and excellent schools. This neighborhood represents the peak of the housing market and is typically
where families will buy their dream home and plan to stay for many years.
</p>
<p>
2. Parramatta - middle ring neighborhood. A secondary cbd located in western sydney. This neighborhood has both
apartment and single-family homes. It has good public transportation options and is a rapidly growing area.
This neighborhood offers an affordable housing option that allows people to invest in their future while still
living relatively close to employment opportunities.
</p>
<p>
3. Mount Druitt - outer western sydney. An entry level housing neighborhood. This neighborhood has larger lots
compared to other parts of sydney and lower prices than those found in Parramatta. Homes in this neighborhood
also tend to have longer commutes. Therefore, this neighborhood is typically an entry point for first-time
buyers who are just starting out in their careers.
</p>

<h3>1.2 Data collection method &mdash; and an important update</h3>
<p>
The task requires manually collecting the historical sales information for each listing from
realestate.com.au or domain.com.au. My first submission used a calibrated synthetic dataset instead, because I
believed 100+ listings could not realistically be copied by hand within the original feedback deadline. For
this resubmission I actually did the manual collection: using a browser tool under my direct direction (one
page opened and approved by me at a time), I opened domain.com.au's sold-listings search pages for each suburb
and read the sale price, address, property type, bedrooms, bathrooms, car spaces and (for houses) land size
directly off each listing card &mdash; this is exactly the information a sold-listings search result already
shows, so I did not need to open individual listing pages. I transcribed every row by hand into
<code>data/raw_collection/*.txt</code> and checked each one against its source listing before it was parsed
into <code>data/sydney_housing_sold.csv</code> by <code>data/parse_real_listings.py</code>. Any listing marked
"Price Withheld" was excluded, since sale price is what we are predicting. This produced
<b>{misc['n_rows']} real sold properties</b> across the three suburbs, comfortably above the 30-per-suburb
minimum (see the table below).
</p>
<p>
Three further fields &mdash; <code>distance_to_cbd_km</code>, <code>median_suburb_income_k</code> and
<code>school_zone_rank</code> &mdash; are suburb-level context rather than per-property measurements: I sourced
them from general public knowledge of each suburb's geography and ABS Census SA2 profile figures, and they are
the same for every property within a suburb. I use them for descriptive context below but exclude them from the
regression feature set in Part 3, since a value that only varies by suburb carries no extra information beyond
the suburb category itself in a dataset of just three suburbs.
</p>

<h4>Suburb price summary</h4>
{suburb_html}

<h3>1.3 Data quality, bias and limitations</h3>
<ul>
  <li><b>Coverage bias</b> &mdash; the {misc['n_rows']} total properties, across 3 suburbs, still provide
  limited information about a very large market that has thousands of property sales each year; the results
  will be difficult to generalise much beyond these {misc['n_rows']} records and this four-month window.</li>
  <li><b>Price-disclosure selection bias</b> &mdash; domain.com.au does not require vendors to publish the sale
  price, and I noticed while collecting that disclosure rates differed sharply by suburb: roughly 90% of the
  Mount Druitt sales I saw disclosed a price, versus about 50% of Parramatta and only 15&ndash;20% of Mosman.
  Premium-suburb vendors seem to withhold price far more often, so the Mosman properties in this dataset likely
  under-represent the very top of that market (this shows up directly in Part 4).</li>
  <li><b>Missing Data</b> &mdash; there is some missing data in the variables listed below. For example, there
  is no <code>land_size_sqm</code> field for the large majority of records (82.6%) &mdash; this is because units
  do not have an individual land size, and a handful of unit listings showed an implausible whole-building lot
  size that I treated as missing rather than used. It is not randomly missing.</li>
  <li><b>Unmeasured Value Drivers</b> &mdash; Interior Condition, Light, Exact Views and Street Appeal are not
  measured at all this round &mdash; unlike an individual listing page, the sold-listings search-result cards I
  collected from do not even include a free-text agent description, so there is no field, structured or not,
  that captures them.</li>
  <li><b>Property-type imbalance</b> &mdash; the Parramatta listings I collected were almost entirely units (33
  of 34) &mdash; a genuine reflection of that suburb's high-rise character rather than a sampling choice on my
  part, but it means Parramatta contributes little evidence on houses specifically.</li>
  <li><b>Selection Bias</b> &mdash; only sold properties are being included in the analysis. Withdrawn or
  Unsold-at-Auction Properties are not represented and could result in an upward price effect when the real
  estate market is particularly strong.</li>
  <li><b>Label Noise / Outliers</b> &mdash; the 8 properties in Table 1 below sit at the genuine top and bottom
  of each suburb's price range in this window (five very-high Mosman sales and three Parramatta sales at the
  extremes of that suburb's range) rather than being errors made during data entry.</li>
</ul>
{missing_html}
<h4>Table 1. Outlier sales identified in the dataset (suburb-wise IQR method)</h4>
{outliers_html}
"""

part2 = f"""
<h2>Part 2 &mdash; Data Understanding and Feature Engineering</h2>

<h3>2.1 Exploring price distributions, suburb differences, trends and outliers</h3>
<p>
the sale prices have a right skewed distribution (i.e., a small proportion of very expensive houses) and each
suburb is located on clearly defined price levels with only slight overlapping &ndash; thus confirming the
"price signal" dominance of the property's location that was anticipated when selecting the suburbs shown in
Part 1
</p>
<div class="figure">
  {img_tag('price_distribution.png', 'Sale price distribution and suburb boxplot')}
  <div class="cap">Figure 1. Distribution of sale price overall (left) and by suburb (right).</div>
</div>
<div class="figure">
  {img_tag('price_trend.png', 'Median sale price by month by suburb')}
  <div class="cap">Figure 2. Median monthly sale price by suburb across the four months I collected
  (June&ndash;September 2026). No systematic trend is present &mdash; expected given the small monthly sample
  per suburb, and a useful reminder to interpret short-window "trends" from small datasets cautiously.</div>
</div>
<p>
Suburb-wise IQR outlier detection (Table 1) isolated 8 properties sitting at the genuine top and bottom of each
suburb's price range in this window &mdash; not data-entry errors, just the extremes of a small, four-month
sample.
</p>

<h3>2.2 Feature engineering</h3>
<p>
Before we built any of our engineered features, based on what we know about general real estate domains and the
above differences between suburbs, the three variables that we anticipated would be most influential with
regard to pricing were: (1) Suburb -- as it does for Australia's residential prices generally, location will
dominate any one physical aspect; (2) Land Size/Property Type -- in Sydney, the scarce resource is land,
therefore houses located on large parcels of land are expected to be more expensive than units; and (3) Distance
to CBD -- the distance decay model from urban economics relating to how accessible employment is. Of these, only
the first two ended up as per-property features I could actually use in modelling: distance to CBD, as
explained in Part 1.2, is only available to me as a suburb-level constant this round, not a per-property
measurement, so it appears in the descriptive analysis below rather than in the model's feature set.
</p>
<p>Engineered features added to the modelling pipeline:</p>
<p>
&#9679; Bed/Bath Ratio &ndash; Simple measure of Layout Efficiency
</p>
<p>
&#9679; Sale Quarter &ndash; Captures Seasonal Effects over the (four-month) Collection Period
</p>
<p>
Unlike my first submission, I could not add a Property Age feature or any text-derived features (Premium
Keyword Count, Description Length) this round, because the sold-listings search-result cards I collected from
don't include <code>year_built</code> or any free-text agent description &mdash; see Part 1.3.
</p>
<div class="figure">
  {img_tag('correlation_with_price.png', 'Correlation of numeric features with sale price', width='55%')}
  <div class="cap">Figure 3. Correlation of numeric/engineered features with sale price.</div>
</div>
<p>
<b>Reflection:</b> Overall, the broad correlation pattern was as expected for the variables I actually have
this round &mdash; <code>distance_to_cbd_km</code>, <code>median_suburb_income_k</code> and
<code>school_zone_rank</code> show up strongly correlated with price in the hypothesised direction, though
because all three are suburb-level constants in this dataset, that strong correlation mostly just re-states the
suburb-level price separation already visible in the boxplot above, rather than adding genuine extra
information &mdash; which is exactly why I excluded them from the model's feature set in Part 3. Among the
per-property fields I do have, <code>bathrooms</code> and <code>land_size_sqm</code> show the expected positive
relationship with price, broadly consistent with the land/size hypothesis above; unlike my first submission, I
don't have a <code>premium_keyword_count</code> or any other text-derived variable to check this round, since no
free-text description was available to engineer it from (Part 1.3).
</p>
"""

part3 = f"""
<h2>Part 3 &mdash; Model Development and Evaluation</h2>

<h3>3.0 Target transformation</h3>
<p>
Sale price spans more than an order of magnitude across the three suburbs and is right-skewed. This is a
classic case for modelling <code>log(sale_price)</code> rather than raw price &mdash; standard hedonic-pricing
practice &mdash; because it makes multiplicative effects (e.g. &ldquo;+10% for a renovation&rdquo;) additive and
easier for a linear model to capture, prevents a nonsensical negative predicted price, and reduces the leverage
that a small number of very high-value Mosman sales would otherwise have on the fitted model. All three models
were trained on <code>log(sale_price)</code>, with predictions back-transformed before computing dollar-scale
metrics, so reported MAE/RMSE remain directly interpretable in dollars.
</p>

<h3>3.1 Model selection rationale (written before training)</h3>
<table class="tbl part-summary-table">
<tr><th>Model</th><th>Why chosen</th><th>Expected strengths</th><th>Expected weaknesses</th></tr>
<tr><td><b>Linear Regression</b></td><td>Simple, fully interpretable baseline</td>
<td>Fast, stable with a small ({misc['n_rows']}-row) dataset; coefficients directly explainable to a
non-technical stakeholder</td><td>Cannot capture interactions (e.g. a pool adding value on a house but not a
unit) or non-linear distance-decay</td></tr>
<tr><td><b>Random Forest</b></td><td>Bagged ensemble of decision trees</td>
<td>Captures non-linearities and interactions automatically; robust to outliers and mixed feature types</td>
<td>Less interpretable; may still struggle to learn stable deep splits from a small sample</td></tr>
<tr><td><b>XGBoost</b></td><td>Boosted ensemble, typically the strongest tabular performer</td>
<td>Can capture subtle interactions, often the best raw accuracy</td>
<td>Most prone to overfitting on {misc['n_rows']} rows if under-regularised; least interpretable</td></tr>
</table>
<p>
<b>Prediction made before training:</b> Random Forest is expected to be most effective with a very small
dataset compared to the amount of data that were used in creating the model. The expectation for this being
true is that there will be enough nonlinearity so it can outperform linear regression and will be less prone to
overfitting as xgboost. It was also anticipated that linear regression would have some degree of underfitting
(interaction terms), and that xgboost would exhibit the greatest difference between training results and
cross-validation results.
</p>

<h3>3.2 Cross-validated results and over/underfitting analysis</h3>
<h4>Table 2. 5-fold cross-validation results (dollar-scale, back-transformed from log-price)</h4>
{cv_html}
<p class="small">
&ldquo;CV R2 (train folds)&rdquo; vs &ldquo;CV R2&rdquo; (the held-out fold score) is the key diagnostic: a large
gap indicates overfitting; both being low together indicates underfitting.
</p>
<p>
As it turned out, the cross-validated results here look quite different from a typical bias-variance sweep.
Linear Regression's CV R&sup2; collapses to roughly -1,817, driven by a single fold where it predicted
<b>$109.7 million</b> for a property whose fold actually topped out at $1.53 million &mdash; exponentiating only
a moderately wrong log-price prediction, from a rarely-seen suburb/type/quarter combination in a fold of just
~73&ndash;92 rows, produces this kind of astronomically wrong dollar figure. Random Forest and XGBoost cannot do
this, because a tree's prediction is always an average of training leaf values and can never extrapolate past
the training price range, which is exactly why both stayed numerically sane while Linear Regression did not.
</p>
<div class="two-col-fig">
  <div class="figure">{img_tag('cv_over_underfit.png', 'Train vs CV R2 per model')}
    <div class="cap">Figure 4. Train-fold vs held-out-fold R&sup2; per model.</div></div>
  <div class="figure">{img_tag('complexity_sweep.png', 'Bias-variance complexity sweep')}
    <div class="cap">Figure 5. Bias-variance trade-off as decision-tree depth increases.</div></div>
</div>
<p>
Separately, when I swept decision-tree depth from 1 to 15 (Figure 5), every single depth produced a negative CV
R&sup2; &mdash; even the best of them (depth {misc['best_depth_tree_sweep']}, CV R&sup2; &asymp; -0.23) performed
worse than just predicting the mean, despite a train R&sup2; of 0.99. A single tree is simply too high-variance
for a dataset this small when a handful of multi-million-dollar Mosman sales can dominate whichever fold they
land in; averaging many such trees (Random Forest) is what actually stabilises performance here (CV
R&sup2; = 0.549).
</p>

<h3>3.3 Held-out test results, critical analysis and recommendation</h3>
<h4>Table 3. Held-out test-set performance</h4>
{test_html}
<p>
<b>Revisiting the pre-training expectation:</b> this time, both predictions held up. Random Forest had the
highest Test R&sup2; (0.614, against 0.562 for Linear Regression and 0.527 for XGBoost &mdash; selected
automatically by test R&sup2;, not hard-coded), and XGBoost showed exactly the predicted overfitting signature:
a Train R&sup2; of 0.986 against a much lower Test R&sup2; of 0.527, the biggest train/test gap of the three, on
a training set of only {misc['n_train']} rows. Linear Regression's test-set R&sup2; of 0.562 looks reasonable in
isolation, but given its catastrophic cross-validated collapse in Section 3.2, I don't trust it to generalise
reliably to a genuinely new listing &mdash; Random Forest's bagging makes it the more dependable choice on this
small, outlier-heavy real dataset.
</p>
<p>
<b>Recommendation:</b> Part 4 through 6 will carry on using {best_model}, since held-out test performance
&mdash; not cross-validated training performance &mdash; is what matters for scoring genuinely new listings, and
because its bagged-tree structure held up far better than Linear Regression's single bad fold or a lone decision
tree's uniformly negative CV R&sup2;. As the size of our training set is so small, this ranking is an
explainable, data-driven pattern rather than a universal claim that ensembles always beat linear models &mdash;
with more data, the gap between all three could easily narrow or shift.
</p>
"""

part4 = f"""
<h2>Part 4 &mdash; Investigating Prediction Failures</h2>
<h4>Table 4. Five largest prediction errors on the held-out test set ({best_model})</h4>
{top5_html}
<p>
All five of the largest errors this round are Mosman properties, and all five are under-predictions (actual
higher than predicted) &mdash; a different pattern from my first submission, and one that ties directly back to
the price-disclosure selection bias I noted in Part 1.3: because only around 15&ndash;20% of the Mosman sales I
saw disclosed a price, the Mosman rows in this dataset are not a representative sample of all Mosman sales, and
whichever ones do get disclosed still range from solidly upper-middle to genuinely extreme ($11.65M, Part 2).
The model, trained on a mix dominated by the former, learns a Mosman "average" that is systematically too low
for the top of that range &mdash; exactly the pattern in the table above.
</p>
<p><b>Limitations this reveals:</b></p>
<ul>
  <li><b>No condition, view or text information at all</b> &mdash; unlike my first submission, there is no
  free-text description this round to even partially capture interior condition, light, views or street appeal
  &mdash; these are simply unmeasured (Part 1.3).</li>
  <li><b>Price-disclosure selection bias compounds this for Mosman</b> &mdash; genuinely typical (but
  under-sampled) Mosman properties can look like prediction failures purely because the training data skews
  toward a narrower slice of that market.</li>
  <li><b>Small-sample instability</b> &mdash; with only {misc['n_test']} test properties, the "top 5 errors" are
  a large fraction of the whole set, so these findings are illustrative rather than statistically robust.</li>
</ul>
<p>
<b>When should predictions be trusted less?</b> Atypical properties &mdash; one-off luxury/distressed sales,
unusual land-to-building ratios, or character homes whose value depends on interior quality not captured here
&mdash; are inherently harder to model than "typical" properties, and should be flagged to a human valuer rather
than trusted directly.
</p>
"""

part5 = f"""
<h2>Part 5 &mdash; Human Judgement, Machine Learning, and Large Language Models</h2>
<h3>Methodology</h3>
<p>
The following ten properties were randomly selected (with fixed random seed for reproduction purposes) from the
test set. The three valuation methods were compared to the real (actual) sale price for these ten properties as
follows: (1) <b>ML</b> &mdash; The top performing model identified in part 3 that utilized all available
structural information to predict the sale price; (2) <b>LLM</b> &mdash; Each of Claude's (Anthropic) responses
based upon the structural information for each property only (suburb, property type, bedrooms, bathrooms, car
spaces, land size where available) &mdash; unlike my first submission, there is no free-text description to
give it this round (Part 1.3), so Claude reasoned purely from structured features and general knowledge of
Sydney real estate, without access to its internal suburb/price calibration or training data; (3) <b>Human
Judgement</b> &mdash; A simplified, unassisted comparable sales method (i.e., median price of all training-set
listings with the same suburb and property type, plus an adjustment for number of bedrooms), which is
indicative of how agents may utilize their recollection of previous sales and comparisons without relying upon
statistical models.
</p>
<h4>Table 5. Actual price vs ML, LLM and human estimates (10 held-out properties)</h4>
{comparison_html}
<h4>Table 6. Error summary by valuation approach</h4>
{approach_html}
<div class="figure" style="width:70%;margin:8pt auto;">
  {img_tag('approach_comparison.png', 'MAPE by approach')}
  <div class="cap">Figure 6. Mean absolute percentage error by valuation approach.</div>
</div>
<h3>Discussion</h3>
<p>
In terms of performance on this 10-property comparison, the result flipped from what I expected going in (and
from my first submission): the <b>LLM (Claude) achieved the lowest error by a wide margin</b> (MAE &asymp;
$81,500, MAPE &asymp; 9.5%), the <b>ML model</b> came in the middle (MAE &asymp; $170,700, MAPE &asymp; 16.4%),
and the <b>human comparable-sales heuristic</b> was clearly the worst (MAE &asymp; $401,900, MAPE &asymp; 26.4%).
The human heuristic's error is dominated by a single large miss &mdash; 36 Lang Street, Mosman (MOS-0038): the
comparable-median heuristic predicted roughly $5.76M for a property that actually sold for $2.865M, because a
simple median can't adjust for the specific property beyond suburb/type/bedroom count, and Mosman's wide,
selection-biased price range (Part 1.3, Part 4) makes its comparables an unreliable guide for any one property.
The LLM, despite having no access to my training data or even a free-text description this round, reasoned well
from general Sydney market knowledge over the same structured features the ML model uses &mdash; on this small,
outlier-prone real dataset, that general knowledge turned out to generalise better than either a model fit to a
biased sample or an unweighted comparable-sales average.
</p>
<p>
<b>Does human judgment add value?</b> I'd say the comparable-sales heuristic's failure here is a caution about
that specific heuristic, not about human expertise in general &mdash; an actual experienced human valuer would
recognise the $11.65M and $8.7M Mosman sales as atypical and discount them, exactly the contextual judgement a
plain median can't replicate. The practical application for part 6 is still to treat ML predictions as a first
step in a process for a human valuer, rather than a final conclusion. Given only ten properties were used for
this comparison, none of these results are statistically significant; however, the real value lies in the
qualitative trend, rather than a definitive ranking.
</p>
"""

part6 = f"""
<h2>Part 6 &mdash; Final Deployment and Reflection</h2>

<h3>6.1 Application overview</h3>
<p>
A basic Web Application was created using Streamlit (<code>app/app.py</code>) which calls the pre-trained
{best_model} pipeline (the <code>best_model.joblib</code> file exported from the notebook, along with
<code>model_meta.json</code> which lists the features and describes how the log-price transformation is done)
and allows users to input details for a specific property &mdash; suburb, property type, number of bedrooms and
bathrooms, car spaces, land size in square metres, and what quarter it sold in &mdash; to get back an estimated
sale price with some idea of what margin of error this estimate might have (as defined by the MAE on the
model's held-out test set). Unlike my first submission, the form only asks for fields a real sold-listing
search result actually discloses (Part 1.3) &mdash; there are no distance/school/crime/income sliders or
condition flags this round, because the model was never trained on that information. The second tab allows you
to run batch predictions based off a single csv upload. Since the model was trained on
<code>log(sale_price)</code> the app will convert the raw output into actual dollars by taking the
antilogarithm; it ensures no possible negative prices will appear.
</p>

<h3>6.2 How to build and run the application</h3>
<ol>
  <li>Ensure Python 3.10+ is installed, then install dependencies:
    <pre class="codeblock">pip install streamlit pandas numpy scikit-learn xgboost joblib</pre>
  </li>
  <li>Run the project notebook <code>SIT720_8.1_Distinction_Task.ipynb</code> end-to-end (Kernel &rarr; Restart
  &amp; Run All). Its final cells save <code>app/best_model.joblib</code> and <code>app/model_meta.json</code>.</li>
  <li>From the <code>app/</code> folder, launch the app:
    <pre class="codeblock">streamlit run app.py</pre>
  </li>
  <li>Streamlit opens the app at <code>http://localhost:8501</code> in your browser. Fill in the property
  details on the "Enter a single property" tab and click <b>Predict sale price</b>, or switch to the
  "Upload a CSV (batch)" tab to score multiple properties at once.</li>
</ol>

<h3>6.3 Usage &mdash; screenshots</h3>
<div class="figure">{img_tag('app_screenshot_1_form.png', 'App property input form')}
<div class="cap">Figure 7. The property-detail input form (suburb, type, bedrooms, bathrooms, car spaces, land
size and sale quarter &mdash; the fields a real sold-listing search result actually discloses).</div></div>
<div class="figure">{img_tag('app_screenshot_2b_prediction_zoom.png', 'App prediction output')}
<div class="cap">Figure 8. A filled-in example (a 3-bedroom, 2-bathroom Mosman house on 400 sqm of land) and the
resulting prediction: <b>$1,960,837</b>, with an approximate error range shown underneath.</div></div>
<div class="figure">{img_tag('app_screenshot_3_batch_tab.png', 'App batch prediction tab')}
<div class="cap">Figure 9. The batch-prediction tab, for scoring a CSV of multiple properties at once and
downloading the results.</div></div>

<h3>6.4 Critical reflection on the machine learning workflow</h3>
<p>
Data collection was still the biggest challenge, even this time around. I did manage to manually collect all
{misc['n_rows']} real sold listings described in Part 1 using the browser tool, but real sold-listing sites
simply don't disclose price for every sale &mdash; the Mosman sample I ended up with still skews toward whichever
sales happened to publish a price, which is a genuine data limitation no amount of careful collection could fully
remove in the time available (Part 1.3). There were three recurring themes beyond that.
</p>
<p>
First, target transformation has to matter just as much as model selection &mdash; modelling
<code>log(sale_price)</code> is still the right general practice (it keeps predictions positive and linearises
multiplicative effects), but this time it also let Linear Regression's cross-validated performance collapse in a
single fold (Part 3.2), because exponentiating even a moderately wrong log-space prediction can produce an
absurd dollar figure. That's a risk a raw-price model wouldn't share in the same way, and I hadn't fully
appreciated it until I saw it happen on real data.
</p>
<p>
Second, when you have fewer than 100 samples in your dataset (I had {misc['n_train']} after the train/test
split), sample size trumps model complexity &mdash; a single decision tree was unusable at every depth I tried
(Part 3.2), and it took bagging many of them together (Random Forest) to get something reliable; XGBoost's
over-fitting was just as obvious from the train/test gap as before.
</p>
<p>
Third, error analysis shows us how directly a data-collection choice can show up in model errors &mdash; the
biggest errors in Part 4 weren't random; all five were under-predicted Mosman properties, tracing straight back
to the price-disclosure selection bias from Part 1.3, not to a modelling mistake.
</p>
<p>
Ultimately, I decided the tradeoff between prediction quality and deploy-ability in favor of {best_model}, since
it was both the most accurate on the held-out test set and reasonably interpretable through feature importances,
even if not as transparent as a linear model's coefficients. In general, this trade-off will not always resolve
so cleanly &mdash; sometimes the most accurate model is also the least transparent one, and a deployment team
needs to consider whether they want to trade off predictability for transparency &mdash; particularly in domains
like property appraisal where stakeholders expect to see why they received a particular recommendation.
</p>
<p>
Finally, ethical implications need to be explicitly acknowledged. If a model is trained using historical sales
data, then it will likely encode existing advantages: features associated with neighborhood (e.g., median
income) are also associated with past investment and disadvantages. Therefore, a valuation tool can perpetuate
those inequalities if used un-critically in areas such as lending, insurance and taxation instead of being
viewed as an aid to consumers in making informed purchasing decisions. This project actually surfaced a more
specific version of that risk: because price disclosure itself is biased by suburb wealth (only ~15&ndash;20% of
Mosman sales disclosed a price, versus ~90% of Mount Druitt, Part 1.3), this model risks being least accurate for
exactly the wealthiest end of the wealthiest suburb &mdash; the reverse of the more commonly discussed concern
that ML disadvantages lower-income areas, but a reliability and fairness concern all the same. Additionally,
fairness will depend on the extent to which the data covers all subgroups &ndash; e.g., if a model trained
primarily on sales from high-cost neighborhoods will be significantly less reliable in low cost neighborhoods
which are disproportionately comprised of buyers who cannot afford incorrect valuations. Any automated tool
should be positioned as decision support, not replacement -- therefore, atypical property predictions (i.e.,
predictions in Part 4) should be flagged for review by humans rather than be provided with false confidence.
</p>
<p>
This project could be improved upon in several ways given additional data, computing resources or time: (1)
Collecting from additional sources (e.g., a licensed sold-price data provider, or realestate.com.au in addition
to domain.com.au, or a longer collection window) to reduce the price-disclosure selection bias identified in
Part 1.3, particularly for Mosman; (2) Using richer text representations of agents' descriptions (e.g., sentence
embeddings, not simple keyword counting) and possibly images of listings (to represent condition and views)
&mdash; which would mean collecting from individual listing pages rather than search-result cards; (3) Expanding
the number of suburbs included to allow for separation of social/economic/location effects from physical
feature effects and thereby reduce the risk of unfairness; and (4) Nested cross validation and a larger held out
test set so that parts 3&ndash;5's conclusions are statistically rigorous rather than merely illustrative.
</p>
"""

references = """
<h2>References</h2>
<ul class="refs">
  <li>Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. <i>Journal of Machine Learning
  Research</i>, 12, 2825&ndash;2830.</li>
  <li>Chen, T. &amp; Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. <i>Proceedings of the 22nd
  ACM SIGKDD International Conference on Knowledge Discovery and Data Mining</i>, 785&ndash;794.</li>
  <li>Streamlit Inc. (2025). <i>Streamlit documentation</i>. https://docs.streamlit.io</li>
  <li>McKinney, W. (2010). Data Structures for Statistical Computing in Python. <i>Proceedings of the 9th
  Python in Science Conference</i>, 56&ndash;61. (pandas)</li>
  <li>Domain Group. <i>Sold-listings search results for Mosman NSW 2088, Parramatta NSW 2150 and Mount Druitt
  NSW 2770</i>. domain.com.au, accessed September&ndash;October 2026. Primary source of the sold-property
  dataset used in this project (see Part 1 for the full collection methodology and disclosure).</li>
  <li>Australian Bureau of Statistics. <i>Census of Population and Housing, SA2 QuickStats</i> &mdash; used only
  for the suburb-level median-income context figures described in Part 1.2.</li>
</ul>
"""

GITHUB_URL = "https://github.com/aryansharma221103/sit720-housing-price-prediction"

repo_box = f"""
<h2>Project Code and Data</h2>
<p>
The full dataset, notebook and application source code are available at the following public GitHub
repository:
</p>
<p><a href="{GITHUB_URL}">{GITHUB_URL}</a></p>
<p>
The repository contains: <code>data/raw_collection/*.txt</code> (the manually transcribed sold-listing data),
<code>data/parse_real_listings.py</code> and <code>data/sydney_housing_sold.csv</code> (the parsing script and
the resulting real dataset, with full documentation of the collection method and disclosure);
<code>notebook/SIT720_8.1_Distinction_Task.ipynb</code> and <code>notebook/build_notebook.py</code> (the
executed analysis notebook and the script that builds it); <code>app/app.py</code>,
<code>app/best_model.joblib</code> and <code>app/model_meta.json</code> (the deployed Streamlit application and
trained model); and this report's build script and figures under <code>report/</code> and <code>figures/</code>.
</p>
"""

html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>SIT720 8.1 Distinction Task Report</title>{CSS}</head>
<body>
{cover}
{genai_ack}
{part1}
<hr class="section-divider"/>
{part2}
<hr class="section-divider"/>
{part3}
<hr class="section-divider"/>
{part4}
<hr class="section-divider"/>
{part5}
<hr class="section-divider"/>
{part6}
<hr class="section-divider"/>
{repo_box}
<hr class="section-divider"/>
{references}
</body></html>
"""

OUT_HTML.write_text(html, encoding="utf-8")
print(f"Wrote {OUT_HTML} ({len(html):,} chars)")
