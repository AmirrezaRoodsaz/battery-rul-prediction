# Interview notes — own this project

Five questions a technical interviewer is most likely to ask about this repo, with crisp answers.
Study the *reasoning*, not just the words.

---

### 1. "Walk me through how you avoid data leakage — how do you know your 9 % isn't cheating?"

Three guards, in order of importance:

1. **The split is defined by the dataset, not by me.** Severson splits 124 cells into 41 train /
   43 primary-test / 40 secondary-test. The secondary test is a *later manufacturing batch* — a real
   distribution shift, not a random sample. I reproduce that split exactly (`src/data/preprocess.py`)
   and never move a cell across it.
2. **Hyperparameters are chosen by 5-fold CV on the training split only** (`GridSearchCV` in
   `src/models/train.py`). The test sets are touched exactly once, at the end, for reporting.
3. **Scaling lives inside the CV pipeline.** `StandardScaler` is part of the `Pipeline`, so it is
   re-fit on each CV fold's training portion — the validation fold never sees the scaler's statistics.
   Fitting the scaler on all of train before CV would be a subtle leak; the pipeline prevents it.

If I wanted to break my own result, I'd look here first — which is exactly why I built it this way.

---

### 2. "Why predict log(cycle life) instead of cycle life directly?"

Cycle life is **right-skewed** (≈150 to ≈2300, long tail of long-lived cells — see the distribution
plot). Three reasons to log-transform the target:

- **Symmetry / variance stabilization:** linear models assume roughly homoscedastic, symmetric
  errors; log makes the target closer to that.
- **Multiplicative error is the natural scale:** being off by 100 cycles matters far more for a
  300-cycle cell than a 2000-cycle one. Modeling log means the model optimizes *relative* error,
  which is also what **MAPE** measures — so training objective and reporting metric agree.
- **It's what the paper does**, so the comparison is apples-to-apples.

I exponentiate predictions back to cycles before computing RMSE/MAE/MAPE, so all reported errors are
in real, interpretable cycle counts.

---

### 3. "What is the ΔQ(V) feature, physically, and why does one number predict so much?"

ΔQ(V) = (discharge capacity vs voltage at cycle 100) − (same at cycle 10), on a fixed 1000-point
voltage grid. It measures **how the shape of the discharge curve changes** over the first 100 cycles.

Physically, different degradation mechanisms (loss of lithium inventory, loss of active material,
impedance growth) deform the voltage curve in characteristic ways. A cell degrading fast shows a
larger, more structured ΔQ(V) **even while its total capacity has barely moved** — that's the key
insight: the *shape change* leads the *capacity loss*. Collapsing ΔQ(V) to its **variance** and
taking log gives a single feature that correlates with log cycle life at **r ≈ −0.93**. That tight
correlation is why a 1-feature linear model is already competitive.

---

### 4. "Your gradient-boosting model has ~0 % training error. Isn't that bad?"

Yes — it's **overfitting**, and I report it openly. With only **41 training cells**, a flexible tree
ensemble can memorize the training set (train MAPE ≈ 0) while its held-out error (9–12 %) is only
marginally better than the regularized linear model's. Two takeaways I'd defend:

- **On small data, model capacity is a liability.** The honest floor — ElasticNet on the variance
  feature — lands within a few percent of the benchmark and generalizes more consistently across
  batches. I'd ship the simpler model in production unless more data justified the complexity.
- **The right metric is held-out, ideally cross-batch.** The secondary test (a new manufacturing
  batch) is where over-tuned models get caught: the model that wins the primary test doesn't always
  win there. That's the number I trust most for "will this work on next quarter's cells?"

---

### 5. "How would you put this into production / what would you do next?"

- **Productionizing:** wrap the trained pipeline (`models/*.joblib`) behind a small service that takes
  a cell's first-100-cycle data, runs the same feature extraction (`src/features`), and returns a
  prediction. The feature code is shared between training and serving, so there's no train/serve skew.
- **Uncertainty:** a point estimate isn't enough for screening decisions. I'd add a quantile or
  conformal-prediction wrapper to return a calibrated interval ("1200 ± 150 cycles, 90 %").
- **More data + drift monitoring:** 124 cells is small. With more cells I'd revisit model capacity,
  and I'd monitor the feature distribution per batch to catch manufacturing drift (the secondary-test
  gap already hints this matters).
- **The stretch tasks:** predict the full SOH-vs-cycle trajectory (not just the endpoint), and the
  short-life/long-life classification framing for fast pass/fail screening.

---

*Supporting numbers and figures: [`results.md`](results.md). Code: [`../src/`](../src).*
