# ClutchSheets MLB F5 Forecaster

An interactive MLB First 5 Innings forecasting platform for evaluating projected run environments, win probabilities, model calibration, and performance against the betting market.

## Live Application

https://mlb-f5-forecaster-production.up.railway.app/

---

## Overview

The ClutchSheets MLB F5 Forecaster generates daily First 5 Innings projections using pitcher performance, handedness-adjusted team offense, park factors, lineup context, rest variables, and probability modeling.

The platform converts projected runs for each team into First 5 win probabilities and compares those forecasts against available market prices.

Beyond displaying daily projections, the application evaluates whether the model is improving over time through historical results, closing-line analysis, probability calibration, Log Loss, and market comparison metrics.

The project is designed as a transparent forecasting and model-evaluation platform rather than a source of guaranteed betting outcomes.

---

## Features

- Daily MLB First 5 forecasts
- Expected First 5 run projections
- Home and away win probabilities
- Model-generated fair odds
- Available market odds comparison
- Model edge calculations
- Pinnacle closing-line comparison
- Closing Line Value tracking
- Beat-the-close percentage
- Performance analysis by bet timing
- Model versus market Log Loss
- Probability calibration analysis
- Confidence-bucket breakdowns
- Interactive reliability diagram
- Historical forecast and result tracking
- Automated daily forecast scoring

---

## Technology Stack

### Modeling and Data Processing

- Python
- Pandas
- NumPy
- SciPy

### Data Storage

- SQLite
- CSV

### Visualization and Application

- Streamlit
- Plotly

### Data Sources

- MLB Stats API
- Public MLB performance data
- Historical betting-market snapshots
- Pinnacle closing prices

### Deployment

- Railway
- GitHub

---

## Methodology

The model estimates expected First 5 Innings runs for each team using a pitcher-first forecasting process.

The core projection incorporates:

- Rolling pitcher First 5 runs allowed
- Exponentially weighted recent starts
- Team First 5 scoring splits versus right- and left-handed pitchers
- Park factors
- Home-plate umpire tendencies
- Pitcher rest and injury-return context
- Expected lineup handedness

Projected team runs are converted into win probabilities using a Negative Binomial distribution, allowing the model to represent greater scoring variance than a standard Poisson model.

Model probabilities are then compared against market-implied probabilities to identify differences between the forecast and available prices.

Historical forecasts are evaluated using:

- Closing Line Value
- Beat-the-close rate
- Log Loss
- Brier Score
- Calibration error
- Predicted versus actual win rate
- Projected versus actual First 5 runs

---

## Application Pages

### Today’s Games

Displays the current MLB slate with:

- Starting pitchers
- Projected First 5 runs
- Projected First 5 totals
- Model win probabilities
- Model fair odds
- Available market prices
- Estimated model edge

### Performance

Evaluates historical model performance against the market.

The page includes:

- Results for qualified model edges
- Six-hour and two-hour market snapshots
- Average Closing Line Value
- Beat-the-close percentage
- Average projected edge
- Profit and loss at standardized stakes
- Model versus closing-market Log Loss
- Average probability difference from the closing line
- Rolling analysis of whether the model is moving closer to the market

### Calibration

Measures whether the model’s stated confidence matches actual outcomes.

The page includes:

- Overall predicted win rate
- Overall actual win rate
- Weighted calibration error
- Probability confidence buckets
- Sample size by bucket
- Predicted-versus-actual differences
- Interactive reliability diagram
- Identification of overconfident and underconfident forecast ranges

---

## Market Evaluation

Forecast quality is evaluated independently from short-term betting results.

Closing prices provide a useful benchmark because they incorporate the information available to the market before a game begins.

The platform tracks:

- Whether the model beat the closing price
- How model probabilities compare with closing implied probabilities
- Whether model Log Loss outperforms closing-market Log Loss
- Whether probability accuracy is improving over time
- How forecast performance changes at different edge thresholds and bet timings

This allows the project to evaluate both predictive accuracy and market agreement without relying only on wins, losses, or profit.

---

## Latest Updates

- Added Pinnacle closing-line comparisons
- Added six-hour and two-hour market timing analysis
- Added Closing Line Value tracking
- Added beat-the-close percentage
- Added qualified-edge result tables
- Added model versus market Log Loss analysis
- Added rolling model-versus-market performance chart
- Added confidence-bucket calibration analysis
- Added weighted calibration error
- Added interactive reliability diagram
- Added overconfidence and underconfidence classifications
- Expanded historical forecast and market-snapshot tracking
- Improved navigation and dashboard organization

---

## Running Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open:

```text
http://localhost:8501
```

---

## Future Improvements

- Push-adjusted First 5 moneyline probability calculations
- Further Negative Binomial dispersion testing
- Early-season regression to the mean
- Expanded lineup-quality adjustments
- Improved starting-pitcher availability validation
- Automated model-version comparison
- Edge-threshold performance analysis
- Additional market-timing comparisons
- Multi-season historical backtesting
- Exportable model-performance reports

---

## Disclaimer

This project is experimental and continuously evolving.

Forecasts may be affected by small samples, lineup uncertainty, starting-pitcher changes, market movement, and model error. Nothing displayed by the application should be interpreted as financial advice or guaranteed betting value.

---

## Author

**Michael Dunk**

Analytics Developer focused on predictive modeling, sports forecasting, model validation, and public-facing decision-support applications.

LinkedIn:

https://linkedin.com/in/michaeldunkanalytics

GitHub:

https://github.com/ClutchSheets
