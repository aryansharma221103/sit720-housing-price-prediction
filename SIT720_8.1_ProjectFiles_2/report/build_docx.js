const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, Table, TableRow, TableCell,
  WidthType, BorderStyle, ShadingType, AlignmentType, ImageRun, PageBreak,
  ExternalHyperlink, LevelFormat, convertInchesToTwip, VerticalAlign, Header, Footer,
  PageNumber, NumberFormat,
} = require("docx");

const ROOT = path.resolve(__dirname, "..");
const FIG = path.join(ROOT, "figures");
const DATA = JSON.parse(fs.readFileSync(path.join(__dirname, "data.json"), "utf8"));
const FIG_DIMS = JSON.parse(fs.readFileSync(path.join(__dirname, "fig_dims.json"), "utf8"));

const GITHUB_URL = "https://github.com/aryansharma221103/sit720-housing-price-prediction";
const BEST_MODEL = DATA.misc.best_model_name;

// ---------------------------------------------------------------------------
// Palette / constants
// ---------------------------------------------------------------------------
const NAVY = "14213D";
const NAVY_DARK = "0F1A30";
const PURPLE_BG = "F5F3FF";
const PURPLE_BORDER = "7A5CFF";
const AMBER_BG = "FFF7ED";
const AMBER_BORDER = "D97706";
const CYAN_BG = "ECFEFF";
const CYAN_BORDER = "0891B2";
const ROW_ALT = "F7F8FB";
const PAGE_WIDTH_TWIPS = 11906; // A4
const MARGIN = convertInchesToTwip(0.79); // ~20mm
const CONTENT_WIDTH = PAGE_WIDTH_TWIPS - MARGIN * 2;

function money(x) {
  const n = typeof x === "string" ? parseFloat(x) : x;
  if (Number.isNaN(n)) return String(x);
  return "$" + Math.round(n).toLocaleString("en-US");
}
function pct(x) {
  const n = typeof x === "string" ? parseFloat(x) : x;
  return n.toFixed(1) + "%";
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
function h1(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_1,
    spacing: { before: 360, after: 160 },
    border: { bottom: { color: NAVY, space: 4, style: BorderStyle.SINGLE, size: 12 } },
    children: [new TextRun({ text, bold: true, color: NAVY, size: 30 })],
  });
}
function h2(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_2,
    spacing: { before: 260, after: 100 },
    children: [new TextRun({ text, bold: true, color: "1D3461", size: 24 })],
  });
}
function h3(text) {
  return new Paragraph({
    spacing: { before: 160, after: 80 },
    children: [new TextRun({ text, bold: true, color: "1D3461", size: 20 })],
  });
}

// runs: array of {text, bold, italic, code}
function p(runs, opts = {}) {
  const children = (Array.isArray(runs) ? runs : [{ text: runs }]).map((r) => {
    if (r.code) {
      return new TextRun({
        text: r.text, font: "Consolas", size: 18, color: "AD1457",
        shading: { type: ShadingType.CLEAR, fill: "F0F0F3" },
      });
    }
    return new TextRun({ text: r.text, bold: !!r.bold, italics: !!r.italic, size: 20 });
  });
  return new Paragraph({ children, spacing: { after: 160, line: 300 }, ...opts });
}

function bullet(runs) {
  return p(runs, { bullet: { level: 0 }, spacing: { after: 90, line: 280 } });
}

function caption(text) {
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { after: 220, before: 40 },
    children: [new TextRun({ text, italics: true, size: 17, color: "555555" })],
  });
}

function calloutBox(label, text, { bg, border }) {
  return new Table({
    width: { size: CONTENT_WIDTH, type: WidthType.DXA },
    borders: {
      top: { style: BorderStyle.NONE }, bottom: { style: BorderStyle.NONE },
      left: { style: BorderStyle.SINGLE, size: 24, color: border },
      right: { style: BorderStyle.NONE },
      insideHorizontal: { style: BorderStyle.NONE }, insideVertical: { style: BorderStyle.NONE },
    },
    rows: [
      new TableRow({
        children: [
          new TableCell({
            width: { size: CONTENT_WIDTH, type: WidthType.DXA },
            shading: { type: ShadingType.CLEAR, fill: bg },
            margins: { top: 140, bottom: 140, left: 200, right: 200 },
            children: [
              new Paragraph({
                spacing: { after: 60 },
                children: [
                  new TextRun({ text: label, bold: true, size: 19 }),
                ],
              }),
              new Paragraph({
                spacing: { after: 0, line: 280 },
                children: [new TextRun({ text, size: 19 })],
              }),
            ],
          }),
        ],
      }),
    ],
  });
}

function spacerPara(h = 200) {
  return new Paragraph({ spacing: { after: h }, children: [] });
}

function imagePara(filename, widthPx) {
  const filePath = path.join(FIG, filename);
  const data = fs.readFileSync(filePath);
  const dims = FIG_DIMS[filename] || { width: 800, height: 450 };
  const w = widthPx || 560;
  const h = Math.round((dims.height / dims.width) * w);
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 100, after: 40 },
    children: [
      new ImageRun({ data, type: "png", transformation: { width: w, height: h } }),
    ],
  });
}

function codeBlock(lines) {
  return new Table({
    width: { size: CONTENT_WIDTH, type: WidthType.DXA },
    borders: {
      top: { style: BorderStyle.NONE }, bottom: { style: BorderStyle.NONE },
      left: { style: BorderStyle.NONE }, right: { style: BorderStyle.NONE },
      insideHorizontal: { style: BorderStyle.NONE }, insideVertical: { style: BorderStyle.NONE },
    },
    rows: [
      new TableRow({
        children: [
          new TableCell({
            width: { size: CONTENT_WIDTH, type: WidthType.DXA },
            shading: { type: ShadingType.CLEAR, fill: "1E1E2E" },
            margins: { top: 120, bottom: 120, left: 200, right: 200 },
            children: (Array.isArray(lines) ? lines : [lines]).map(
              (l) => new Paragraph({
                spacing: { after: 0 },
                children: [new TextRun({ text: l, font: "Consolas", size: 18, color: "E5E5E5" })],
              })
            ),
          }),
        ],
      }),
    ],
  });
}

function cell(text, { header = false, width, align = AlignmentType.LEFT, shade } = {}) {
  return new TableCell({
    width: width ? { size: width, type: WidthType.DXA } : undefined,
    shading: header
      ? { type: ShadingType.CLEAR, fill: NAVY }
      : shade ? { type: ShadingType.CLEAR, fill: shade } : undefined,
    verticalAlign: VerticalAlign.CENTER,
    margins: { top: 80, bottom: 80, left: 100, right: 100 },
    children: [
      new Paragraph({
        alignment: align,
        children: [
          new TextRun({
            text: String(text),
            bold: header,
            color: header ? "FFFFFF" : "1A1A1A",
            size: 17,
          }),
        ],
      }),
    ],
  });
}

function simpleTable(headers, rows, colWidths) {
  const total = CONTENT_WIDTH;
  const widths = colWidths || headers.map(() => Math.floor(total / headers.length));
  const headerRow = new TableRow({
    tableHeader: true,
    children: headers.map((h, i) => cell(h, { header: true, width: widths[i] })),
  });
  const bodyRows = rows.map((r, ri) =>
    new TableRow({
      children: r.map((v, i) => cell(v, { width: widths[i], shade: ri % 2 === 1 ? ROW_ALT : undefined })),
    })
  );
  return new Table({
    width: { size: total, type: WidthType.DXA },
    columnWidths: widths,
    borders: {
      top: { style: BorderStyle.SINGLE, size: 2, color: "DDDDDD" },
      bottom: { style: BorderStyle.SINGLE, size: 2, color: "DDDDDD" },
      left: { style: BorderStyle.NONE }, right: { style: BorderStyle.NONE },
      insideHorizontal: { style: BorderStyle.SINGLE, size: 2, color: "DDDDDD" },
      insideVertical: { style: BorderStyle.NONE },
    },
    rows: [headerRow, ...bodyRows],
  });
}

function tableCaption(text) {
  return new Paragraph({
    spacing: { before: 160, after: 60 },
    children: [new TextRun({ text, bold: true, size: 19, color: "1D3461" })],
  });
}

const DIVIDER = new Paragraph({
  spacing: { before: 200, after: 200 },
  border: { bottom: { color: "CCCCCC", space: 1, style: BorderStyle.SINGLE, size: 4 } },
  children: [],
});

// ---------------------------------------------------------------------------
// Build document sections
// ---------------------------------------------------------------------------
const children = [];

// Cover page
children.push(
  new Paragraph({ spacing: { before: 2400 }, children: [] }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "SIT720 — MACHINE LEARNING", size: 20, color: "666666", characterSpacing: 20 })],
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 200 },
    children: [new TextRun({ text: "Sydney Housing Price Prediction", bold: true, size: 44, color: NAVY })],
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { after: 200 },
    children: [new TextRun({ text: "and Decision Support System", bold: true, size: 44, color: NAVY })],
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { after: 500 },
    children: [new TextRun({ text: "8.1 Distinction Task — Machine Learning Mini Project", size: 24, color: "444444" })],
  }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 60 },
    children: [new TextRun({ text: "Student: ", size: 22, color: "333333" }), new TextRun({ text: "Aryan Sharma", bold: true, size: 22 }), new TextRun({ text: " (Student ID: 225616764)", size: 22, color: "333333" })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 60 },
    children: [new TextRun({ text: "Unit: SIT720 — Machine Learning, Deakin University", size: 22, color: "333333" })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 60 },
    children: [new TextRun({ text: "Report date: 1 October 2026", size: 22, color: "333333" })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 600 },
    children: [new TextRun({ text: "Suburbs analysed: Mosman · Parramatta · Mount Druitt", size: 22, color: "333333" })] }),
  new Table({
    alignment: AlignmentType.CENTER,
    width: { size: 7200, type: WidthType.DXA },
    borders: {
      top: { style: BorderStyle.NONE }, bottom: { style: BorderStyle.NONE },
      left: { style: BorderStyle.NONE }, right: { style: BorderStyle.NONE },
      insideHorizontal: { style: BorderStyle.NONE }, insideVertical: { style: BorderStyle.NONE },
    },
    rows: [new TableRow({ children: [new TableCell({
      shading: { type: ShadingType.CLEAR, fill: NAVY },
      width: { size: 7200, type: WidthType.DXA },
      margins: { top: 120, bottom: 120, left: 200, right: 200 },
      children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [
        new TextRun({ text: "Notebook: SIT720_8.1_Distinction_Task.ipynb   |   App: app/app.py", color: "FFFFFF", bold: true, size: 19 }),
      ]})],
    })]})],
  }),
  new Paragraph({ children: [new PageBreak()] })
);

// GenAI acknowledgement
children.push(calloutBox(
  "GenAI acknowledgement.",
  "I used Generative AI tools (Claude, Anthropic) for brainstorming ideas and structuring the Jupyter notebook, " +
  "including organising the workflow and presentation of the analysis. I also used Generative AI to help " +
  "structure and format the accompanying Word/PDF report. For Part 1 of this resubmission, I used a browser " +
  "tool under my direct direction (one page opened and approved by me at a time) to open domain.com.au's " +
  "sold-listings search pages for each suburb; I read the sale price, address and feature values off each " +
  "listing card myself and transcribed them, excluding any listing marked “Price Withheld”. As required by " +
  "Part 5, I also used Claude directly as the “large language model” valuation approach, compared against my " +
  "ML model and a human-judgement heuristic. The actual analysis, implementation, results, interpretation and " +
  "final content were completed and reviewed by me, and I take full responsibility for the accuracy and " +
  "integrity of the submitted work.",
  { bg: PURPLE_BG, border: PURPLE_BORDER }
));
children.push(spacerPara(200));

// ============================== PART 1 ==============================
children.push(h1("Part 1 — Problem Definition and Data Collection"));
children.push(h3("1.1 Problem, motivation and suburb selection"));
children.push(p("The goal of this project was to develop a decision-making support system (a model) that could help estimate a “fair” or “reasonable” selling price of a home based on the attributes of the house and provide insight into why that price is reasonable. In order to create models that were representative of different types of residential housing markets, we chose to study three distinct and separate neighborhoods within sydney. These three neighborhoods represent significantly different housing market environments with each neighborhood being considered as a potential housing option at a different time in my own life:"));
children.push(p("1. Mosman - inner north harborside neighborhood. The most expensive, low density area with very high land values and excellent schools. This neighborhood represents the peak of the housing market and is typically where families will buy their dream home and plan to stay for many years."));
children.push(p("2. Parramatta - middle ring neighborhood. A secondary cbd located in western sydney. This neighborhood has both apartment and single-family homes. It has good public transportation options and is a rapidly growing area. This neighborhood offers an affordable housing option that allows people to invest in their future while still living relatively close to employment opportunities."));
children.push(p("3. Mount Druitt - outer western sydney. An entry level housing neighborhood. This neighborhood has larger lots compared to other parts of sydney and lower prices than those found in Parramatta. Homes in this neighborhood also tend to have longer commutes. Therefore, this neighborhood is typically an entry point for first-time buyers who are just starting out in their careers."));

children.push(h3("1.2 Data collection method — and an important update"));
children.push(p([
  { text: "The task requires manually collecting the historical sales information for each listing from realestate.com.au or domain.com.au. My first submission used a calibrated synthetic dataset instead, because I believed 100+ listings could not realistically be copied by hand within the original feedback deadline. For this resubmission I actually did the manual collection: using a browser tool under my direct direction (one page opened and approved by me at a time), I opened domain.com.au's sold-listings search pages for each suburb and read the sale price, address, property type, bedrooms, bathrooms, car spaces and (for houses) land size directly off each listing card — this is exactly the information a sold-listings search result already shows, so I did not need to open individual listing pages. I transcribed every row by hand into " },
  { text: "data/raw_collection/*.txt", code: true },
  { text: " and checked each one against its source listing before it was parsed into " },
  { text: "data/sydney_housing_sold.csv", code: true },
  { text: " by " }, { text: "data/parse_real_listings.py", code: true },
  { text: ". Any listing marked “Price Withheld” was excluded, since sale price is what we are predicting. This produced " },
  { text: "115 real sold properties", bold: true },
  { text: " across the three suburbs, comfortably above the 30-per-suburb minimum (see the table below)." },
]));
children.push(p([
  { text: "Three further fields — " }, { text: "distance_to_cbd_km", code: true }, { text: ", " }, { text: "median_suburb_income_k", code: true }, { text: " and " }, { text: "school_zone_rank", code: true },
  { text: " — are suburb-level context rather than per-property measurements: I sourced them from general public knowledge of each suburb's geography and ABS Census SA2 profile figures, and they are the same for every property within a suburb. I use them for descriptive context below but exclude them from the regression feature set in Part 3, since a value that only varies by suburb carries no extra information beyond the suburb category itself in a dataset of just three suburbs." },
]));

children.push(tableCaption("Suburb price summary"));
children.push(simpleTable(
  ["Suburb", "N", "Mean", "Std dev", "Min", "P25", "Median", "P75", "Max"],
  DATA.suburb.map(r => [r.suburb, Math.round(r.count), money(r.mean), money(r.std), money(r.min), money(r["25%"]), money(r["50%"]), money(r["75%"]), money(r.max)]),
));
children.push(spacerPara(160));

children.push(h3("1.3 Data quality, bias and limitations"));
children.push(bullet([{ text: "Coverage bias", bold: true }, { text: " — the 115 total properties, across 3 suburbs, still provide limited information about a very large market that has thousands of property sales each year; the results will be difficult to generalise much beyond these 115 records and this four-month window." }]));
children.push(bullet([{ text: "Price-disclosure selection bias", bold: true }, { text: " — domain.com.au does not require vendors to publish the sale price, and I noticed while collecting that disclosure rates differed sharply by suburb: roughly 90% of the Mount Druitt sales I saw disclosed a price, versus about 50% of Parramatta and only 15–20% of Mosman. Premium-suburb vendors seem to withhold price far more often, so the Mosman properties in this dataset likely under-represent the very top of that market (this shows up directly in Part 4)." }]));
children.push(bullet([{ text: "Missing Data", bold: true }, { text: " — there is some missing data in the variables listed below. For example, there is no " }, { text: "land_size_sqm", code: true }, { text: " field for the large majority of records (82.6%) — this is because units do not have an individual land size, and a handful of unit listings showed an implausible whole-building lot size that I treated as missing rather than used. It is not randomly missing." }]));
children.push(bullet([{ text: "Unmeasured Value Drivers", bold: true }, { text: " — Interior Condition, Light, Exact Views and Street Appeal are not measured at all this round — unlike an individual listing page, the sold-listings search-result cards I collected from do not even include a free-text agent description, so there is no field, structured or not, that captures them." }]));
children.push(bullet([{ text: "Property-type imbalance", bold: true }, { text: " — the Parramatta listings I collected were almost entirely units (33 of 34) — a genuine reflection of that suburb's high-rise character rather than a sampling choice on my part, but it means Parramatta contributes little evidence on houses specifically." }]));
children.push(bullet([{ text: "Selection Bias", bold: true }, { text: " — only sold properties are being included in the analysis. Withdrawn or Unsold-at-Auction Properties are not represented and could result in an upward price effect when the real estate market is particularly strong." }]));
children.push(bullet([{ text: "Label Noise / Outliers", bold: true }, { text: " — the 8 properties in Table 1 below sit at the genuine top and bottom of each suburb's price range in this window (five very-high Mosman sales and three Parramatta sales at the extremes of that suburb's range) rather than being errors made during data entry." }]));
children.push(spacerPara(120));
children.push(simpleTable(
  ["Feature", "# missing", "% missing"],
  DATA.missing.map(r => [r.Feature, r.n_missing, pct(r.pct_missing)]),
));
children.push(tableCaption("Table 1. Outlier sales identified in the dataset (suburb-wise IQR method)"));
children.push(simpleTable(
  ["ID", "Suburb", "Type", "Bed", "Bath", "Sale price", "Address"],
  DATA.outliers.map(r => [r.property_id, r.suburb, r.property_type, r.bedrooms, r.bathrooms, money(r.sale_price), r.address]),
  [900, 1400, 900, 600, 600, 1400, Math.max(1000, CONTENT_WIDTH - 5800)],
));
children.push(DIVIDER);

// ============================== PART 2 ==============================
children.push(h1("Part 2 — Data Understanding and Feature Engineering"));
children.push(h3("2.1 Exploring price distributions, suburb differences, trends and outliers"));
children.push(p("the sale prices have a right skewed distribution (i.e., a small proportion of very expensive houses) and each suburb is located on clearly defined price levels with only slight overlapping – thus confirming the “price signal” dominance of the property's location that was anticipated when selecting the suburbs shown in Part 1"));
children.push(imagePara("price_distribution.png", 580));
children.push(caption("Figure 1. Distribution of sale price overall (left) and by suburb (right)."));
children.push(imagePara("price_trend.png", 520));
children.push(caption("Figure 2. Median monthly sale price by suburb across the four months I collected (June–September 2026). No systematic trend is present — expected given the small monthly sample per suburb, and a useful reminder to interpret short-window “trends” from small datasets cautiously."));
children.push(p("Suburb-wise IQR outlier detection (Table 1) isolated 8 properties sitting at the genuine top and bottom of each suburb's price range in this window — not data-entry errors, just the extremes of a small, four-month sample."));

children.push(h3("2.2 Feature engineering"));
children.push(p("Before we built any of our engineered features, based on what we know about general real estate domains and the above differences between suburbs, the three variables that we anticipated would be most influential with regard to pricing were: (1) Suburb -- as it does for Australia's residential prices generally, location will dominate any one physical aspect; (2) Land Size/Property Type -- in Sydney, the scarce resource is land, therefore houses located on large parcels of land are expected to be more expensive than units; and (3) Distance to CBD -- the distance decay model from urban economics relating to how accessible employment is. Of these, only the first two ended up as per-property features I could actually use in modelling: distance to CBD, as explained in Part 1.2, is only available to me as a suburb-level constant this round, not a per-property measurement, so it appears in the descriptive analysis below rather than in the model's feature set."));
children.push(p("Engineered features added to the modelling pipeline:"));
children.push(p("● Bed/Bath Ratio – Simple measure of Layout Efficiency"));
children.push(p("● Sale Quarter – Captures Seasonal Effects over the (four-month) Collection Period"));
children.push(p([
  { text: "Unlike my first submission, I could not add a Property Age feature or any text-derived features (Premium Keyword Count, Description Length) this round, because the sold-listings search-result cards I collected from don't include " },
  { text: "year_built", code: true }, { text: " or any free-text agent description — see Part 1.3." },
]));
children.push(imagePara("correlation_with_price.png", 420));
children.push(caption("Figure 3. Correlation of numeric/engineered features with sale price."));
children.push(p([
  { text: "Reflection:", bold: true },
  { text: " Overall, the broad correlation pattern was as expected for the variables I actually have this round — " },
  { text: "distance_to_cbd_km", code: true }, { text: ", " }, { text: "median_suburb_income_k", code: true }, { text: " and " }, { text: "school_zone_rank", code: true },
  { text: " show up strongly correlated with price in the hypothesised direction, though because all three are suburb-level constants in this dataset, that strong correlation mostly just re-states the suburb-level price separation already visible in the boxplot above, rather than adding genuine extra information — which is exactly why I excluded them from the model's feature set in Part 3. Among the per-property fields I do have, " },
  { text: "bathrooms", code: true }, { text: " and " }, { text: "land_size_sqm", code: true },
  { text: " show the expected positive relationship with price, broadly consistent with the land/size hypothesis above; unlike my first submission, I don't have a " },
  { text: "premium_keyword_count", code: true }, { text: " or any other text-derived variable to check this round, since no free-text description was available to engineer it from (Part 1.3)." },
]));
children.push(DIVIDER);

// ============================== PART 3 ==============================
children.push(h1("Part 3 — Model Development and Evaluation"));
children.push(h3("3.0 Target transformation"));
children.push(p([
  { text: "Sale price spans more than an order of magnitude across the three suburbs and is right-skewed. This is a classic case for modelling " },
  { text: "log(sale_price)", code: true },
  { text: " rather than raw price — standard hedonic-pricing practice — because it makes multiplicative effects (e.g. “+10% for a renovation”) additive and easier for a linear model to capture, prevents a nonsensical negative predicted price, and reduces the leverage that a small number of very high-value Mosman sales would otherwise have on the fitted model. All three models were trained on " },
  { text: "log(sale_price)", code: true },
  { text: ", with predictions back-transformed before computing dollar-scale metrics, so reported MAE/RMSE remain directly interpretable in dollars." },
]));

children.push(h3("3.1 Model selection rationale (written before training)"));
children.push(simpleTable(
  ["Model", "Why chosen", "Expected strengths", "Expected weaknesses"],
  [
    ["Linear Regression", "Simple, fully interpretable baseline", `Fast, stable with a small (${DATA.misc.n_rows}-row) dataset; coefficients directly explainable to a non-technical stakeholder`, "Cannot capture interactions (e.g. land size mattering differently by suburb) or non-linear effects"],
    ["Random Forest", "Bagged ensemble of decision trees", "Captures non-linearities and interactions automatically; robust to outliers and mixed feature types", "Less interpretable; may still struggle to learn stable deep splits from a small sample"],
    ["XGBoost", "Boosted ensemble, typically the strongest tabular performer", "Can capture subtle interactions, often the best raw accuracy", `Most prone to overfitting on ${DATA.misc.n_rows} rows if under-regularised; least interpretable`],
  ],
  [1600, 2400, 2900, 2900],
));
children.push(spacerPara(120));
children.push(p("Prediction made before training: Random Forest is expected to be most effective with a very small dataset compared to the amount of data that were used in creating the model. The expectation for this being true is that there will be enough nonlinearity so it can outperform linear regression and will be less prone to overfitting as xgboost. It was also anticipated that linear regression would have some degree of underfitting (interaction terms), and that xgboost would exhibit the greatest difference between training results and cross-validation results."));

children.push(h3("3.2 Cross-validated results and over/underfitting analysis"));
children.push(tableCaption("Table 2. 5-fold cross-validation results (dollar-scale, back-transformed from log-price)"));
children.push(simpleTable(
  ["Model", "CV MAE", "CV RMSE", "CV R2", "CV R2 (train folds)", "R2 std (folds)"],
  DATA.cv_df.map(r => [r.Model, money(r["CV MAE"]), money(r["CV RMSE"]), r["CV R2"].toFixed(3), r["CV R2 (train folds)"].toFixed(3), r["R2 std (folds)"].toFixed(3)]),
));
children.push(p([{ text: "“CV R2 (train folds)” vs “CV R2” (the held-out fold score) is the key diagnostic: a large gap indicates overfitting; both being low together indicates underfitting.", italic: true }], { spacing: { before: 100, after: 140 } }));
children.push(p([
  { text: "As it turned out, the cross-validated results here look quite different from a typical bias-variance sweep. Linear Regression's CV R² collapses to roughly -1,817, driven by a single fold where it predicted " },
  { text: "$109.7 million", bold: true }, { text: " for a property whose fold actually topped out at $1.53 million — exponentiating only a moderately wrong log-price prediction, from a rarely-seen suburb/type/quarter combination in a fold of just ~73–92 rows, produces this kind of astronomically wrong dollar figure. Random Forest and XGBoost cannot do this, because a tree's prediction is always an average of training leaf values and can never extrapolate past the training price range, which is exactly why both stayed numerically sane while Linear Regression did not." },
], { spacing: { before: 100, after: 140 } }));
children.push(imagePara("cv_over_underfit.png", 470));
children.push(caption("Figure 4. Train-fold vs held-out-fold R² per model."));
children.push(imagePara("complexity_sweep.png", 470));
children.push(caption("Figure 5. Bias-variance trade-off as decision-tree depth increases."));
children.push(p(`Separately, when I swept decision-tree depth from 1 to 15 (Figure 5), every single depth produced a negative CV R² — even the best of them (depth ${DATA.misc.best_depth_tree_sweep}, CV R² ≈ -0.23) performed worse than just predicting the mean, despite a train R² of 0.99. A single tree is simply too high-variance for a dataset this small when a handful of multi-million-dollar Mosman sales can dominate whichever fold they land in; averaging many such trees (Random Forest) is what actually stabilises performance here (CV R² = 0.549).`));

children.push(h3("3.3 Held-out test results, critical analysis and recommendation"));
children.push(tableCaption("Table 3. Held-out test-set performance"));
children.push(simpleTable(
  ["Model", "Train MAE", "Train R2", "Test MAE", "Test RMSE", "Test R2"],
  DATA.test_df.map(r => [r.Model, money(r["Train MAE"]), r["Train R2"].toFixed(3), money(r["Test MAE"]), money(r["Test RMSE"]), r["Test R2"].toFixed(3)]),
));
children.push(spacerPara(120));
children.push(p([
  { text: "Revisiting the pre-training expectation:", bold: true },
  { text: " this time, both predictions held up. Random Forest had the highest Test R² (0.614, against 0.562 for Linear Regression and 0.527 for XGBoost — selected automatically by test R², not hard-coded), and XGBoost showed exactly the predicted overfitting signature: a Train R² of 0.986 against a much lower Test R² of 0.527, the biggest train/test gap of the three, on a training set of only " },
  { text: `${DATA.misc.n_train}`, }, { text: " rows. Linear Regression's test-set R² of 0.562 looks reasonable in isolation, but given its catastrophic cross-validated collapse in Section 3.2, I don't trust it to generalise reliably to a genuinely new listing — Random Forest's bagging makes it the more dependable choice on this small, outlier-heavy real dataset." },
]));
children.push(p([
  { text: "Recommendation:", bold: true },
  { text: ` Part 4 through 6 will carry on using ${BEST_MODEL}, since held-out test performance — not cross-validated training performance — is what matters for scoring genuinely new listings, and because its bagged-tree structure held up far better than Linear Regression's single bad fold or a lone decision tree's uniformly negative CV R². As the size of our training set is so small, this ranking is an explainable, data-driven pattern rather than a universal claim that ensembles always beat linear models — with more data, the gap between all three could easily narrow or shift.` },
]));
children.push(DIVIDER);

// ============================== PART 4 ==============================
children.push(h1("Part 4 — Investigating Prediction Failures"));
children.push(tableCaption(`Table 4. Five largest prediction errors on the held-out test set (${BEST_MODEL})`));
children.push(simpleTable(
  ["ID", "Suburb", "Type", "Bed", "Land (sqm)", "Actual", "Predicted", "Abs. error", "% error", "Outlier?", "Address"],
  DATA.top5.map(r => [r.property_id, r.suburb, r.property_type, r.bedrooms, r.land_size_sqm == null ? "NaN" : r.land_size_sqm, money(r.actual_price), money(r.predicted_price), money(r.abs_error), pct(r.pct_error), String(r.price_outlier), r.address]),
  [800, 1100, 850, 500, 900, 1150, 1150, 1100, 750, 800, Math.max(900, CONTENT_WIDTH - 9100)],
));
children.push(spacerPara(120));
children.push(p([
  { text: "All five of the largest errors this round are Mosman properties, and all five are under-predictions (actual higher than predicted) — a different pattern from my first submission, and one that ties directly back to the price-disclosure selection bias I noted in Part 1.3: because only around 15–20% of the Mosman sales I saw disclosed a price, the Mosman rows in this dataset are not a representative sample of all Mosman sales, and whichever ones do get disclosed still range from solidly upper-middle to genuinely extreme ($11.65M, Part 2). The model, trained on a mix dominated by the former, learns a Mosman “average” that is systematically too low for the top of that range — exactly the pattern in the table above." },
]));
children.push(p([{ text: "Limitations this reveals:", bold: true }]));
children.push(bullet([{ text: "No condition, view or text information at all", bold: true }, { text: " — unlike my first submission, there is no free-text description this round to even partially capture interior condition, light, views or street appeal — these are simply unmeasured (Part 1.3)." }]));
children.push(bullet([{ text: "Price-disclosure selection bias compounds this for Mosman", bold: true }, { text: " — genuinely typical (but under-sampled) Mosman properties can look like prediction failures purely because the training data skews toward a narrower slice of that market." }]));
children.push(bullet([{ text: "Small-sample instability", bold: true }, { text: " — with only 23 test properties, the “top 5 errors” are a large fraction of the whole set, so these findings are illustrative rather than statistically robust." }]));
children.push(p([
  { text: "When should predictions be trusted less?", bold: true },
  { text: " Atypical properties — one-off luxury/distressed sales, unusual land-to-building ratios, or character homes whose value depends on interior quality not captured here — are inherently harder to model than “typical” properties, and should be flagged to a human valuer rather than trusted directly." },
]));
children.push(DIVIDER);

// ============================== PART 5 ==============================
children.push(h1("Part 5 — Human Judgement, Machine Learning, and Large Language Models"));
children.push(h3("Methodology"));
children.push(p([
  { text: "The following ten properties were randomly selected (with fixed random seed for reproduction purposes) from the test set. The three valuation methods were compared to the real (actual) sale price for these ten properties as follows: (1) " },
  { text: "ML", bold: true }, { text: " — The top performing model identified in part 3 that utilized all available structural information to predict the sale price; (2) " },
  { text: "LLM", bold: true }, { text: " — Each of Claude's (Anthropic) responses based upon the structural information for each property only (suburb, property type, bedrooms, bathrooms, car spaces, land size where available) — unlike my first submission, there is no free-text description to give it this round (Part 1.3), so Claude reasoned purely from structured features and general knowledge of Sydney real estate, without access to its internal suburb/price calibration or training data; (3) " },
  { text: "Human Judgement", bold: true }, { text: " — A simplified, unassisted comparable sales method (i.e., median price of all training-set listings with the same suburb and property type, plus an adjustment for number of bedrooms), which is indicative of how agents may utilize their recollection of previous sales and comparisons without relying upon statistical models." },
]));
children.push(tableCaption("Table 5. Actual price vs ML, LLM and human estimates (10 held-out properties)"));
children.push(simpleTable(
  ["ID", "Suburb", "Actual", "ML", "LLM (Claude)", "Human (comps)"],
  DATA.comparison.map(r => [r.property_id, r.suburb, money(r.actual_price), money(r.ml_estimate), money(r.llm_estimate), money(r.human_estimate)]),
));
children.push(spacerPara(140));
children.push(tableCaption("Table 6. Error summary by valuation approach"));
children.push(simpleTable(
  ["Approach", "MAE", "MAPE (%)", "Max abs error"],
  DATA.approach.map(r => [r.Approach, money(r.MAE), r["MAPE (%)"].toFixed(1), money(r["Max abs error"])]),
));
children.push(imagePara("approach_comparison.png", 420));
children.push(caption("Figure 6. Mean absolute percentage error by valuation approach."));
children.push(h3("Discussion"));
children.push(p([
  { text: "In terms of performance on this 10-property comparison, the result flipped from what I expected going in (and from my first submission): the " }, { text: "LLM (Claude) achieved the lowest error by a wide margin", bold: true },
  { text: " (MAE ≈ $81,500, MAPE ≈ 9.5%), the " }, { text: "ML model", bold: true }, { text: " came in the middle" },
  { text: " (MAE ≈ $170,700, MAPE ≈ 16.4%), and the " }, { text: "human comparable-sales heuristic", bold: true }, { text: " was clearly the worst" },
  { text: " (MAE ≈ $401,900, MAPE ≈ 26.4%). The human heuristic's error is dominated by a single large miss — 36 Lang Street, Mosman (MOS-0038): the comparable-median heuristic predicted roughly $5.76M for a property that actually sold for $2.865M, because a simple median can't adjust for the specific property beyond suburb/type/bedroom count, and Mosman's wide, selection-biased price range (Part 1.3, Part 4) makes its comparables an unreliable guide for any one property. The LLM, despite having no access to my training data or even a free-text description this round, reasoned well from general Sydney market knowledge over the same structured features the ML model uses — on this small, outlier-prone real dataset, that general knowledge turned out to generalise better than either a model fit to a biased sample or an unweighted comparable-sales average." },
]));
children.push(p([
  { text: "Does human judgment add value?", bold: true },
  { text: " I'd say the comparable-sales heuristic's failure here is a caution about that specific heuristic, not about human expertise in general — an actual experienced human valuer would recognise the $11.65M and $8.7M Mosman sales as atypical and discount them, exactly the contextual judgement a plain median can't replicate. The practical application for part 6 is still to treat ML predictions as a first step in a process for a human valuer, rather than a final conclusion. Given only ten properties were used for this comparison, none of these results are statistically significant; however, the real value lies in the qualitative trend, rather than a definitive ranking." },
]));
children.push(DIVIDER);

// ============================== PART 6 ==============================
children.push(h1("Part 6 — Final Deployment and Reflection"));
children.push(h3("6.1 Application overview"));
children.push(p([
  { text: "A basic Web Application was created using Streamlit (" }, { text: "app/app.py", code: true },
  { text: ") which calls the pre-trained " }, { text: `${BEST_MODEL}`, }, { text: " pipeline (the " },
  { text: "best_model.joblib", code: true }, { text: " file exported from the notebook, along with " }, { text: "model_meta.json", code: true },
  { text: " which lists the features and describes how the log-price transformation is done) and allows users to input details for a specific property — suburb, property type, number of bedrooms and bathrooms, car spaces, land size in square metres, and what quarter it sold in — to get back an estimated sale price with some idea of what margin of error this estimate might have (as defined by the MAE on the model's held-out test set). Unlike my first submission, the form only asks for fields a real sold-listing search result actually discloses (Part 1.3) — there are no distance/school/crime/income sliders or condition flags this round, because the model was never trained on that information. The second tab allows you to run batch predictions based off a single csv upload. Since the model was trained on " },
  { text: "log(sale_price)", code: true },
  { text: " the app will convert the raw output into actual dollars by taking the antilogarithm; it ensures no possible negative prices will appear." },
]));

children.push(h3("6.2 How to build and run the application"));
children.push(p([{ text: "1. Ensure Python 3.10+ is installed, then install dependencies:" }]));
children.push(codeBlock("pip install streamlit pandas numpy scikit-learn xgboost joblib"));
children.push(p([
  { text: "2. Run the project notebook " }, { text: "SIT720_8.1_Distinction_Task.ipynb", code: true },
  { text: " end-to-end (Kernel → Restart & Run All). Its final cells save " },
  { text: "app/best_model.joblib", code: true }, { text: " and " }, { text: "app/model_meta.json", code: true }, { text: "." },
], { spacing: { before: 140, after: 140 } }));
children.push(p([{ text: "3. From the " }, { text: "app/", code: true }, { text: " folder, launch the app:" }]));
children.push(codeBlock("streamlit run app.py"));
children.push(p([
  { text: "4. Streamlit opens the app at " }, { text: "http://localhost:8501", code: true },
  { text: " in your browser. Fill in the property details on the “Enter a single property” tab and click " },
  { text: "Predict sale price", bold: true }, { text: ", or switch to the “Upload a CSV (batch)” tab to score multiple properties at once." },
], { spacing: { before: 140, after: 160 } }));

children.push(h3("6.3 Usage — screenshots"));
children.push(imagePara("app_screenshot_1_form.png", 480));
children.push(caption("Figure 7. The property-detail input form (suburb, type, bedrooms, bathrooms, car spaces, land size and sale quarter — the fields a real sold-listing search result actually discloses)."));
children.push(imagePara("app_screenshot_2b_prediction_zoom.png", 480));
children.push(caption("Figure 8. A filled-in example (a 3-bedroom, 2-bathroom Mosman house on 400 sqm of land) and the resulting prediction: $1,960,837, with an approximate error range shown underneath."));
children.push(imagePara("app_screenshot_3_batch_tab.png", 480));
children.push(caption("Figure 9. The batch-prediction tab, for scoring a CSV of multiple properties at once and downloading the results."));

children.push(h3("6.4 Critical reflection on the machine learning workflow"));
children.push(p(`Data collection was still the biggest challenge, even this time around. I did manage to manually collect all ${DATA.misc.n_rows} real sold listings described in Part 1 using the browser tool, but real sold-listing sites simply don't disclose price for every sale — the Mosman sample I ended up with still skews toward whichever sales happened to publish a price, which is a genuine data limitation no amount of careful collection could fully remove in the time available (Part 1.3). There were three recurring themes beyond that.`));
children.push(p([
  { text: "First, target transformation has to matter just as much as model selection — modelling " }, { text: "log(sale_price)", code: true },
  { text: " is still the right general practice (it keeps predictions positive and linearises multiplicative effects), but this time it also let Linear Regression's cross-validated performance collapse in a single fold (Part 3.2), because exponentiating even a moderately wrong log-space prediction can produce an absurd dollar figure. That's a risk a raw-price model wouldn't share in the same way, and I hadn't fully appreciated it until I saw it happen on real data." },
]));
children.push(p(`Second, when you have fewer than 100 samples in your dataset (I had ${DATA.misc.n_train} after the train/test split), sample size trumps model complexity — a single decision tree was unusable at every depth I tried (Part 3.2), and it took bagging many of them together (Random Forest) to get something reliable; XGBoost's over-fitting was just as obvious from the train/test gap as before.`));
children.push(p("Third, error analysis shows us how directly a data-collection choice can show up in model errors — the biggest errors in Part 4 weren't random; all five were under-predicted Mosman properties, tracing straight back to the price-disclosure selection bias from Part 1.3, not to a modelling mistake."));
children.push(p(`Ultimately, I decided the tradeoff between prediction quality and deploy-ability in favor of ${BEST_MODEL}, since it was both the most accurate on the held-out test set and reasonably interpretable through feature importances, even if not as transparent as a linear model's coefficients. In general, this trade-off will not always resolve so cleanly — sometimes the most accurate model is also the least transparent one, and a deployment team needs to consider whether they want to trade off predictability for transparency — particularly in domains like property appraisal where stakeholders expect to see why they received a particular recommendation.`));
children.push(p([
  { text: "Finally, ethical implications need to be explicitly acknowledged. If a model is trained using historical sales data, then it will likely encode existing advantages: features associated with neighborhood (e.g., median income) are also associated with past investment and disadvantages. Therefore, a valuation tool can perpetuate those inequalities if used un-critically in areas such as lending, insurance and taxation instead of being viewed as an aid to consumers in making informed purchasing decisions. This project actually surfaced a more specific version of that risk: because price disclosure itself is biased by suburb wealth (only ~15–20% of Mosman sales disclosed a price, versus ~90% of Mount Druitt, Part 1.3), this model risks being least accurate for exactly the wealthiest end of the wealthiest suburb — the reverse of the more commonly discussed concern that ML disadvantages lower-income areas, but a reliability and fairness concern all the same. Additionally, fairness will depend on the extent to which the data covers all subgroups – e.g., if a model trained primarily on sales from high-cost neighborhoods will be significantly less reliable in low cost neighborhoods which are disproportionately comprised of buyers who cannot afford incorrect valuations. Any automated tool should be positioned as decision support, not replacement -- therefore, atypical property predictions (i.e., predictions in Part 4) should be flagged for review by humans rather than be provided with false confidence." },
]));
children.push(p([
  { text: "This project could be improved upon in several ways given additional data, computing resources or time: (1) Collecting from additional sources (e.g., a licensed sold-price data provider, or realestate.com.au in addition to domain.com.au, or a longer collection window) to reduce the price-disclosure selection bias identified in Part 1.3, particularly for Mosman; (2) Using richer text representations of agents' descriptions (e.g., sentence embeddings, not simple keyword counting) and possibly images of listings (to represent condition and views) — which would mean collecting from individual listing pages rather than search-result cards; (3) Expanding the number of suburbs included to allow for separation of social/economic/location effects from physical feature effects and thereby reduce the risk of unfairness; and (4) Nested cross validation and a larger held out test set so that parts 3–5's conclusions are statistically rigorous rather than merely illustrative." },
]));
children.push(DIVIDER);

// ============================== PROJECT CODE AND DATA ==============================
children.push(h1("Project Code and Data"));
children.push(p([
  { text: "The full dataset, notebook and application source code are available at the following public GitHub repository:" },
]));
children.push(new Paragraph({
  spacing: { after: 200 },
  children: [
    new ExternalHyperlink({
      link: GITHUB_URL,
      children: [new TextRun({ text: GITHUB_URL, style: "Hyperlink", size: 20 })],
    }),
  ],
}));
children.push(p([
  { text: "The repository contains: " }, { text: "data/raw_collection/*.txt", code: true }, { text: " and " }, { text: "data/parse_real_listings.py", code: true },
  { text: " (the hand-transcribed raw listing data and the script that parses it into " }, { text: "data/sydney_housing_sold.csv", code: true }, { text: "); " },
  { text: "notebook/SIT720_8.1_Distinction_Task.ipynb", code: true }, { text: " and " }, { text: "notebook/build_notebook.py", code: true },
  { text: " (the executed analysis notebook and the script that builds it); " },
  { text: "app/app.py", code: true }, { text: ", " }, { text: "app/best_model.joblib", code: true }, { text: " and " }, { text: "app/model_meta.json", code: true },
  { text: " (the deployed Streamlit application and trained model); and this report's build script and figures under " },
  { text: "report/", code: true }, { text: " and " }, { text: "figures/", code: true }, { text: "." },
]));
children.push(DIVIDER);

// ============================== REFERENCES ==============================
children.push(h1("References"));
children.push(bullet([{ text: "Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research, 12, 2825–2830." }]));
children.push(bullet([{ text: "Chen, T. & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 785–794." }]));
children.push(bullet([{ text: "Streamlit Inc. (2025). Streamlit documentation. https://docs.streamlit.io" }]));
children.push(bullet([{ text: "McKinney, W. (2010). Data Structures for Statistical Computing in Python. Proceedings of the 9th Python in Science Conference, 56–61. (pandas)" }]));
children.push(bullet([{ text: "Domain Group. Sold-listings search results for Mosman, Parramatta and Mount Druitt, domain.com.au, accessed September–October 2026 (manually collected, listing-by-listing, for Part 1 of this project; see Part 1.2 for the full collection method)." }]));
children.push(bullet([{ text: "Australian Bureau of Statistics. Census of Population and Housing, SA2 QuickStats for Mosman, Parramatta and Mount Druitt — used only as general background for the suburb-level context fields described in Part 1.2." }]));

// ---------------------------------------------------------------------------
// Assemble document
// ---------------------------------------------------------------------------
const doc = new Document({
  styles: {
    default: {
      document: { run: { font: "Calibri", size: 20 } },
    },
  },
  numbering: {
    config: [
      {
        reference: "default-bullets",
        levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 360, hanging: 260 } } } }],
      },
    ],
  },
  sections: [
    {
      properties: {
        page: {
          size: { width: PAGE_WIDTH_TWIPS, height: 16838 },
          margin: { top: MARGIN, bottom: MARGIN, left: MARGIN, right: MARGIN },
        },
      },
      footers: {
        default: new Footer({
          children: [new Paragraph({
            alignment: AlignmentType.CENTER,
            children: [
              new TextRun({ children: [PageNumber.CURRENT], size: 16, color: "888888" }),
            ],
          })],
        }),
      },
      children,
    },
  ],
});

Packer.toBuffer(doc).then((buffer) => {
  const outPath = path.join(__dirname, "SIT720_8.1_Report.docx");
  fs.writeFileSync(outPath, buffer);
  console.log("Wrote", outPath, buffer.length, "bytes");
});
