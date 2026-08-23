# RetailLens — Agent Operational Guide & Tracking

## Project Mission
RetailLens is a collaborative machine learning & time-series forecasting project built to predict retail item sales across multiple stores and surface dynamic inventory safety thresholds via a Streamlit dashboard.

---

## Team Roles & Ownership Matrix

| Track | Owner | Scope & Deliverables |
|---|---|---|
| **Data Engineering** | Sandipan Biswas | Ingestion, data cleaning (`src/clean_data.py`), validation, frozen `sales_clean.csv`. |
| **EDA & Visuals** | Soumili Das | Seasonal decomposition, store/item clustering, sales distribution analysis. |
| **Time-Series Forecasting** | Rudra Pratap Singh | Prophet pipeline, SARIMA baseline, 30/60/90d horizons, uncertainty intervals, hyperparameter tuning (`src/models/prophet_model.py`, `src/models/sarima_model.py`, `src/models/hyperparameter_tuner.py`). |
| **Machine Learning & Eval** | Sohel Mallik | Feature engineering (lags/rolling), XGBoost quantile models, SHAP explainability, hybrid residual modeling. |
| **Dashboard & UI** | Jeet Jana | Streamlit application, dynamic inventory reorder point alerts, forecast visualization. |

---

## Agent Instructions & Best Practices

1. **Modular Code Structure**:
   - Keep business logic in `src/` as reusable, typed classes and functions.
   - Notebooks in `notebooks/` should import from `src/` rather than define duplicate logic.
2. **Deterministic & Resilient Execution**:
   - Provide synthetic fallback generators so models can be developed and unit tested even without external raw datasets.
   - Use fixed random seeds for reproducible cross-validation and hyperparameter search.
3. **Data Contract Adherence**:
   - Always adhere to standard schema: `date` (`datetime64`), `store` (`int`), `item` (`int`), `sales` (`float/int`).
   - For Prophet modeling: mapping to `ds` and `y` must preserve chronological order.
4. **Git Commit Hygiene**:
   - Make atomic, descriptive commits following Conventional Commits format:
     - `feat(...)`: New feature or model implementation
     - `fix(...)`: Bug fix
     - `docs(...)`: Documentation updates
     - `test(...)`: Adding or updating test suites
     - `refactor(...)`: Code refactoring without behavior change
5. **Testing & Quality Assurance**:
   - All forecasting modules must have corresponding unit tests in `tests/`.
   - Verify forecast horizons (30, 60, 90 days) and assert confidence interval properties ($y_{lower} \le y_{hat} \le y_{upper}$).

---

## Current Sprint Goal: Time-Series Forecasting Track
- [x] Create `architecture.md` and `agent.md`.
- [ ] Build synthetic data generator & metrics utility.
- [ ] Implement core Prophet forecaster with multi-step forecasts (30/60/90d) and confidence intervals.
- [ ] Implement SARIMA benchmark model.
- [ ] Implement hyperparameter tuner with rolling-origin cross-validation.
- [ ] Implement unified evaluation CLI and comparative reporting.
- [ ] Add unit test suite in `tests/` and verify end-to-end execution.
