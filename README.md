# TRACE — Tracking Resource Allocation for Climate Equity

- By: Sanjog Singh Kadayat


**Does climate adaptation aid reach the countries that need it most?**

TRACE is an interactive analytics application that scores how far each country's
climate adaptation aid departs from what its climate vulnerability, size, wealth, and
governance would predict. It pairs a reproducible statistical model with a five-page
Streamlit dashboard and an AI-assisted policy-brief generator.

**Live app:** https://trace-climate-aid.streamlit.app/

This repository is the deliverable for a DATA 6999 Business Analytics Applications
capstone at the Fairfield University Dolan School of Business.

---

## The question

Global climate adaptation finance is growing, but growth alone says nothing about
*fairness*. The animating question here is allocative: of the adaptation aid that flows
each year, does more of it reach the countries that are most exposed to climate harm, or
does it track size, wealth, politics, and visibility instead? TRACE answers this by
modeling expected aid for every country-year, then reading the gap between expected and
actual as a **misallocation score**.

## Key findings

- **Aid does track need, but only once country size is controlled.** In a naive model,
  vulnerability looks statistically irrelevant to how aid is distributed (p ≈ 0.5). Adding
  a population control flips the result: vulnerability becomes a positive, significant
  driver (coefficient +3.64, p = 0.007). The honest headline is not "aid ignores need" but
  "aid tracks need once you account for size, while specific countries deviate
  systematically."
- **Allocation is highly concentrated.** The aid Gini is 0.62. The top 15 recipients take
  44% of all adaptation aid; the lowest-funded half of countries share just 8%; the top
  13% of countries hold half of the total.
- **Some countries are systematically short-changed relative to their risk.** The widest
  negative gaps are Malaysia (-2.30), Eritrea (-2.28, thin data), Iran (-1.98),
  Turkmenistan (-1.80), and Thailand (-1.76). The widest positive gaps are Serbia (+1.83),
  Albania (+1.77), Colombia (+1.73), Jordan (+1.68), and Tunisia (+1.62).
- **Two of six world regions are chronically underfunded** relative to their risk:
  Sub-Saharan Africa (mean -0.30) and South Asia (mean -0.22).
- **The gaps are mostly widening.** Four of six regional gaps are projected to drift
  further from need toward 2030 rather than close, with South Asia widening fastest.

About half of all allocation remains unexplained by the model, so a score is a flag for
investigation, not a verdict.

## What the app does (five pages)

| Page | What it shows |
| --- | --- |
| **Home** | World map of misallocation scores, headline KPIs, the most under- and over-funded countries, and the aid-concentration (Lorenz) curve. Includes a per-country detail panel. |
| **Regression** | The model's drivers as plain-language scorecards, the pivotal role of the population control, and how well the model fits within each income tier. |
| **2030 projections** | Where regional gaps and individual countries are trending if current trajectories hold. |
| **Policy briefs** | An AI-generated, seven-section policy brief for any country, built on the app's verified figures. |
| **Methods & data** | Every modeling choice justified with its effect on the result, the exact exclusions, all caveats, and downloadable processed data. |

## Data sources

The analysis merges four sources into a country-year panel covering **138 countries from
2010 to 2023**:

- **OECD Rio Markers** — bilateral and multilateral aid flagged as climate adaptation
  (the outcome variable).
- **ND-GAIN** — each country's climate vulnerability score (the measure of need).
- **World Bank** — GDP per capita, Government Effectiveness (WGI), income classification,
  and population.
- **UN lists** — landlocked and small-island developing state (SIDS) status.

## Methodology in brief

Adaptation aid is modeled with pooled OLS on a log scale:

```
log(1 + aid) ~ vulnerability + log GDP per capita + log population
             + governance + landlocked + SIDS + income tier
             + year fixed effects
```

Standard errors are clustered by country and year fixed effects are included so that
each country is compared to its peers *within the same year* rather than to a growing
global pool. The residual, how much more or less a country receives than predicted, is
its misallocation score; a country's headline score is the mean residual across its
observed years. The model explains about half of allocation (R² = 0.501, n = 1,806
country-years). The **Methods & data** page documents every choice and its effect.

## Tech stack

- **Python** 3.13
- **Streamlit** 1.58 (multipage app via `st.navigation`)
- **Plotly** 6.7 (choropleth, Lorenz, trajectory charts)
- **statsmodels** (OLS with country-clustered standard errors)
- **Anthropic API** (`claude-sonnet-4-6`) for policy-brief generation

## Repository structure

```
climate-adaptation-and-vulnaribility/
├── streamlit_app.py          # Entry point: navigation, theming, page registry
├── requirements.txt          # Runtime dependencies (slim, pinned)
├── .streamlit/
│   ├── config.toml           # Theme and base font size
│   └── secrets.toml          # ANTHROPIC_API_KEY (git-ignored, never committed)
├── app/
│   ├── data.py               # Cached loaders for the processed CSVs
│   ├── model.py              # OLS fit with clustered standard errors
│   ├── charts.py             # Choropleth, Lorenz, trajectory builders
│   ├── geo.py                # Country centroids for on-map labels
│   ├── details.py            # Per-country summary helpers
│   ├── briefs.py             # Anthropic call + verified-figures assembly
│   ├── assets/               # TRACE logo and icon
│   ├── skills/
│   │   └── policy_brief.md    # The brief format and guardrails
│   └── views/
│       ├── overview.py        # Home
│       ├── regression.py      # Regression
│       ├── projections.py     # 2030 projections
│       ├── briefs.py          # Policy briefs
│       └── methods.py         # Methods & data
├── data/
│   └── processed/            # Model-ready CSVs (see below)
└── notebooks/                # 01_clean, 02_model, 03_analysis, 04_projection, eda
```

Processed data files in `data/processed/`:

- `panel.csv` — the full merged country-year panel (includes excluded countries).
- `model_scored.csv` — model sample with per-row residuals (1,806 country-years).
- `country_scored.csv` — one row per scored country (138): mean score, income tier,
  region, funding profile, thin-data flag.
- `country_projection.csv` — per-country trend fit and 2030 projection (134 countries).
- `regional_gap_trend.csv` — regional gap at the start versus end of the period (6 rows).

## Running locally

```bash
# 1. Clone
git clone https://github.com/sanjogkadayat-web/climate-adaptation-and-vulnaribility.git
cd climate-adaptation-and-vulnaribility

# 2. Create and activate a virtual environment (Python 3.13)
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your Anthropic key for the Policy briefs page
mkdir -p .streamlit
echo 'ANTHROPIC_API_KEY = "sk-ant-your-key-here"' > .streamlit/secrets.toml

# 5. Run
streamlit run streamlit_app.py
```

The first four pages work without an API key. Only **Policy briefs** needs one; without
it, that page shows a friendly "no key" state instead of crashing.

## Deployment

The app is deployed on **Streamlit Community Cloud**, pointed at `streamlit_app.py` on the
`main` branch with Python 3.13. The Anthropic key is stored in the Cloud app's **Secrets**
(Settings → Secrets) as:

```toml
ANTHROPIC_API_KEY = "sk-ant-..."
```

Every push to `main` triggers an automatic rebuild.

## How the policy briefs work

The **Policy briefs** page sends `claude-sonnet-4-6` a compact block of the country's
*verified* figures (its score, rank, profile, projection, regional context) drawn straight
from the processed data, together with the format defined in `app/skills/policy_brief.md`.
The guardrails are deliberate:

- Every number in a brief is locked to the app's figures and is never altered.
- The model may add qualitative, widely-known context about a country, but is instructed
  not to invent specifics (no fabricated statistics, dates, dollar amounts, programs, or
  treaties).
- No web access is used.

Each brief has seven sections: executive summary, context, the allocation gap, why the gap
persists, outlook to 2030, recommendations, and risks and caveats. The page exposes both
the exact figures passed to the model and a plain-language explanation of these guardrails.

## Limitations

- **Five countries cannot be scored** because they lack a model input, most often the
  ND-GAIN vulnerability score. Three are immaterial (North Korea, Saint Kitts and Nevis,
  Kosovo), but two are genuinely large recipients the analysis cannot place: Palestine
  (~$2.0B over the period) and South Sudan (~$1.8B).
- **Five scored countries rest on only one or two years** of data (Croatia, Oman, Trinidad
  and Tobago, Barbados, Eritrea) and are flagged throughout and kept out of headline lists.
- **Scores are model-relative and associational.** A negative score means a country gets
  less than the model predicts, not proof of unfair treatment, and the model describes how
  aid is distributed, not why any single decision was made.
- **Aid is donor-reported** through the Rio Markers, an imperfect proxy that can be
  over-claimed.
- **Projections are extrapolations, not forecasts**, and assume current trends hold.

## Credits

Data provided by the **OECD** (Rio Markers), **University of Notre Dame** (ND-GAIN),
the **World Bank** (WDI and WGI), and the **United Nations**. Analysis and application by
Sanjog Kadayat, DATA 6999 capstone, Fairfield University Dolan School of Business.

This project is academic work; please credit the original data providers if you reuse the
processed data.