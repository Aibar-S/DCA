# Decline Curve Analysis Tool with Economics

This project is a Streamlit web app built for reservoir and petroleum engineers to perform Decline Curve Analysis (DCA) on oil production data. It allows users to fit decline models, forecast production, calculate Estimated Ultimate Recovery (EUR), and perform a simple economic analysis including NPV.

---

## 💻 Features

- Upload or generate synthetic production data
- Fit decline models: Exponential, Harmonic, and Hyperbolic
- Forecast future production rates
- Calculate EUR using Simpson's rule
- Perform simple economics (Revenue, Cost, NPV)
- Download all forecasted data with economics as a CSV

---

## 📂 Project Structure

- `DCA.py` — The Streamlit application with all logic
- `requirements.txt` — Python dependencies
- `README.md` — Project documentation (this file)

---

## 📊 Function Overview

### 1. **Decline Curve Models**

```python
def exponential_decline(t, q_i, D):
```
- \( q(t) = q_i \cdot e^{-Dt} \)
- Inputs: `t` (time), `q_i` (initial rate), `D` (decline rate)

```python
def harmonic_decline(t, q_i, D):
```
- \( q(t) = \frac{q_i}{1 + Dt} \)

```python
def hyperbolic_decline(t, q_i, D, b):
```
- \( q(t) = \frac{q_i}{(1 + bDt)^{1/b}} \)

---

### 2. **Model Fitting**

```python
def fit_model(model_func, t, q, p0):
```
- Fits the chosen model using `scipy.optimize.curve_fit`
- Returns fitted parameters and standard errors

---

### 3. **Forecasting & EUR**

```python
def forecast_production(model_func, t_forecast, *params):
```
- Uses fitted model to forecast production over time

```python
def calculate_eur(t, q):
```
- Uses `scipy.integrate.simpson` to calculate EUR from forecasted rates

---

### 4. **Economic Evaluation**

```python
def economic_analysis(t, q, price_per_bbl=70, cost_per_bbl=15):
```
- Calculates:
  - Revenue = rate × price
  - Cost = rate × operating cost
  - Net cash flow = revenue - cost
  - NPV = discounted sum of net cash flow at 10% rate

---

### 5. **Plotting**

```python
def plot_fit(df, model_name, model_func, params):
```
- Plots actual production data and the fitted model forecast
- Returns the plot and forecasted values

---

## 🔧 Installation

```bash
pip install -r requirements.txt
```

Run the app:

```bash
streamlit run main.py
```

---

## 📤 CSV Export Content

Downloaded CSV includes:
- Time (months)
- Forecasted rate (STB/day)
- Revenue ($)
- Cost ($)
- Net cash flow ($)
- Summary rows:
  - Total Revenue
  - Total Cost
  - Total Net Cash Flow
  - NPV (10%)


