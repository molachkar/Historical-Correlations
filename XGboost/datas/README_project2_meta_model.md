# XAUUSD Gold — Meta-Model & Live Signal Project

## What This Project Is

The second layer of the XAUUSD trading system. Built on top of the
base LightGBM model from Project 1. This project has two components:

1. **Meta-model:** A small calibration model trained on the base
   model's out-of-fold (OOF) predictions. Learns when to trust the
   base model's signal and outputs a clean 0-1 confidence probability.

2. **Live feature fetcher:** A script that fetches today's market data,
   computes the same 15 features the base model was trained on, runs
   both models, and outputs a daily go/no-go trading signal.

This README documents the design so you can build it when ready.

---

## Status

| Component | Status |
|-----------|--------|
| Base model (Project 1) | COMPLETE — saved as cv_best_fold_model.pkl |
| Meta-model script | NOT YET BUILT |
| Live fetcher script | NOT YET BUILT |
| Full pipeline test | NOT YET BUILT |

**Build order:** Meta-model first, live fetcher second.
The live fetcher is pointless without the meta-model to interpret
the base model's output cleanly.

---

## Why a Meta-Model

The base model outputs a raw number like `0.00031`. That alone tells
you nothing actionable. You still have to manually decide:
- Is this strong enough to trade?
- Is the current regime one where the model works?
- What is my confidence level right now?

The meta-model answers all three automatically. It is a small
logistic regression or decision tree trained on the OOF predictions
as input, with a binary target: did the base model correctly call
direction yes or no. The output is a probability between 0 and 1.

**Your trading rule becomes:** probability > 0.65 → enter. Below → wait.

No manual interpretation. No pred_z threshold tuning by hand.

---

## What is OOF and Why It Matters (No Leakage)

OOF = Out-Of-Fold predictions.

When the base model was trained using walk-forward CV, each fold
produced predictions on data the model had never seen during training.
Fold 1 trained on 2011-2013, predicted on 2013-2016. The model had
no knowledge of 2013-2016 when making those predictions.

This means the OOF predictions in `cv_predictions_oof.csv` are honest.
They represent what the model would say on truly unseen data — no
optimistic bias, no leakage of future information.

Training the meta-model on these OOF predictions is clean. If instead
you trained the meta-model on the base model's training-set predictions
(in-sample), those predictions would be inflated — the base model
memorized the training data so its in-sample predictions look far
better than they really are. The meta-model would learn to trust a
signal that doesn't exist in real trading.

OOF predictions = what the model actually delivers in practice.

---

## Meta-Model Design

### Input Features

The meta-model takes the base model's OOF prediction plus context:

```
oof_prediction          The raw base model output (float)
pred_z                  OOF prediction normalized vs 252-day history
Market_State            Current regime (categorical)
Macro_Fast              Today's macro value (float)
Macro_Fast_change3      Macro_Fast today minus 3 days ago (transition signal)
abs_pred_z              Absolute value of pred_z (confidence regardless of direction)
```

### Target Variable

```
direction_correct       1 if sign(oof_prediction) == sign(actual_return), else 0
```

### Algorithm

Logistic Regression or shallow Decision Tree (max depth 3-4).
Keep it simple on purpose. The base model already does the heavy
lifting. The meta-model only needs to learn one thing: under what
conditions is the base model's direction call reliable.

### Training Data

The OOF file covers 2013-2024 (4,050 rows). Use walk-forward split
here too — train meta-model on 2013-2020, validate on 2020-2022,
test on 2022-2024. Same discipline as the base model.

---

## Expected Output

After running the full pipeline on any given day, output:

```
Date             : 2025-03-15
Market_State     : bull_neutral
Macro_Fast       : 1.24
Macro_Fast_change: +0.31 (rising — transition signal present)
Base pred_z      : 1.87
Meta confidence  : 0.71
Direction        : LONG
Decision         : ENTER  (confidence > 0.65 threshold)
```

If `Market_State` is `risk_off` or `risk_on`, the regime weight blocks
the signal before it even reaches the meta-model output stage:

```
Date             : 2025-03-16
Market_State     : bull_risk_off
Regime weight    : 0.0
Decision         : WAIT — regime blocked
```

---

## Live Feature Fetcher Design

### Data Sources Needed

| Feature | Source | Notes |
|---------|--------|-------|
| Close_XAUUSD | Yahoo Finance / broker API | XAU=X or XAUUSD=X ticker |
| Close_EURUSD | Yahoo Finance / broker API | EURUSD=X |
| Close_USDJPY | Yahoo Finance / broker API | JPY=X |
| Volume | Yahoo Finance | Same ticker as XAUUSD |
| Macro_Fast | **Your own calculation** | Must match training exactly |
| Market_State | **Your own calculation** | Must match training exactly |

### Computed Features (from raw prices)

Once you have XAUUSD, EURUSD, USDJPY closing prices for the last
30 days, all other features can be computed:

```python
# Log return
df["Log_Returns"] = np.log(df["Close_XAUUSD"] / df["Close_XAUUSD"].shift(1))

# Simple return
df["Close_Returns"] = df["Close_XAUUSD"].pct_change()

# Z-scores (20-day window)
df["LogReturn_ZScore"] = (df["Log_Returns"] - df["Log_Returns"].rolling(20).mean()) \
                          / df["Log_Returns"].rolling(20).std()
df["Return_ZScore"]    = (df["Close_Returns"] - df["Close_Returns"].rolling(20).mean()) \
                          / df["Close_Returns"].rolling(20).std()

# Percentile rank
df["Return_Percentile"] = df["Log_Returns"].rolling(20).rank(pct=True)

# Bollinger middle band
df["BB_Middle"] = df["Close_XAUUSD"].rolling(20).mean()

# MACD signal
ema12 = df["Close_XAUUSD"].ewm(span=12).mean()
ema26 = df["Close_XAUUSD"].ewm(span=26).mean()
macd  = ema12 - ema26
df["MACD_Signal"] = macd.ewm(span=9).mean()

# Distance from all-time high
df["Distance_From_AllTimeHigh"] = df["Close_XAUUSD"] - df["Close_XAUUSD"].cummax()
df["Pct_From_AllTimeHigh"]      = df["Distance_From_AllTimeHigh"] / df["Close_XAUUSD"].cummax()

# Volume percentile
df["Volume_Percentile"] = df["Volume"].rolling(20).rank(pct=True)
```

### Critical: Macro_Fast and Market_State

These two features drive 72% of the base model's predictions.
They **must** be computed in exactly the same way as during training.
Document their exact formulas here before building the live fetcher:

```
Macro_Fast formula:
  [ fill in your calculation here ]

Market_State formula:
  [ fill in your calculation here — what determines bull/bear/sideways
    and what determines risk_on/neutral/risk_off ]
```

If these are computed differently from training, the base model will
receive inputs it was not trained on and its predictions will be wrong
even if all other features are correct.

---

## Full Pipeline Flow (once built)

```
Every trading day at market close:
  1. Fetch last 30 days of XAUUSD, EURUSD, USDJPY closes + volume
  2. Compute all 15 features for today
  3. Load base model: joblib.load("cv_best_fold_model.pkl")
  4. Get base prediction (raw float)
  5. Compute pred_z: normalize vs last 252 days of predictions
  6. Check regime_weight from Market_State
  7. If regime_weight == 0: output WAIT, stop here
  8. Load meta-model, compute meta confidence score
  9. If confidence > 0.65: output ENTER + direction
     If confidence < 0.65: output WAIT
 10. Log everything to daily_signals.csv
```

---

## Files This Project Will Produce

```
meta_model.pkl                  Saved meta-model (logistic regression)
meta_model_results.csv          Precision, recall, accuracy on test period
daily_signals.csv               Running log of every day's signal
live_fetcher.py                 Script to fetch features and run pipeline
meta_model_train.py             Script to train and save the meta-model
```

---

## Key Decisions Still to Make

Before building, answer these:

1. **Meta-model algorithm** — logistic regression (simpler, more stable)
   or decision tree (more interpretable, captures nonlinear combos)?
   Recommendation: start with logistic regression.

2. **Confidence threshold** — 0.65 is a reasonable starting point but
   should be tuned on the validation set. Higher = fewer trades but
   higher quality. Lower = more trades but more noise.

3. **Macro_Fast and Market_State exact formulas** — document these
   before writing a single line of the live fetcher. Without this
   the whole system breaks.

4. **Data source** — Yahoo Finance (free, easy, slight delay) or
   broker API (real-time, requires authentication)?

---

## Relationship to Project 1

```
Project 1 (Base Model)          Project 2 (Meta + Live)
─────────────────────           ───────────────────────
cv_walkforward_v2.py   ──────>  meta_model_train.py
cv_best_fold_model.pkl ──────>  live_fetcher.py
cv_predictions_oof.csv ──────>  meta_model.pkl
feature_guide.txt      ──────>  (reference for feature computation)
```

Project 2 consumes the outputs of Project 1. Never retrain Project 1
unless you are doing a full model refresh with new data. The base
model checkpoint is stable — build on top of it.

---

*Last updated: system design phase — not yet implemented*
*Depends on: cv_best_fold_model.pkl, cv_predictions_oof.csv*
