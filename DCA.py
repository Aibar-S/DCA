import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.integrate import simps
import numpy_financial as npf
from io import BytesIO

# -------------------------------
# 1. Decline Curve Models
# -------------------------------

def exponential_decline(t, q_i, D):
    return q_i * np.exp(-D * t)

def harmonic_decline(t, q_i, D):
    return q_i / (1 + D * t)

def hyperbolic_decline(t, q_i, D, b):
    return q_i / np.power(1 + b * D * t, 1 / b)

# -------------------------------
# 2. Fit Models
# -------------------------------

def fit_model(model_func, t, q, p0):
    popt, pcov = curve_fit(model_func, t, q, p0=p0, maxfev=10000)
    return popt, np.sqrt(np.diag(pcov))

# -------------------------------
# 3. Forecasting & EUR
# -------------------------------

def forecast_production(model_func, t_forecast, *params):
    return model_func(t_forecast, *params)

def calculate_eur(t, q):
    return simps(q, x=t)

# -------------------------------
# 4. Economic Evaluation
# -------------------------------

def economic_analysis(t, q, price_per_bbl=70, cost_per_bbl=15):
    revenue = q * price_per_bbl
    cost = q * cost_per_bbl
    net_cash_flow = revenue - cost
    npv = npf.npv(0.1, net_cash_flow)
    return revenue, cost, net_cash_flow, npv

# -------------------------------
# 5. Plotting
# -------------------------------

def plot_fit(df, model_name, model_func, params):
    t = df['time']
    q = df['rate']
    t_forecast = np.linspace(t.min(), t.max() + 60, 200)
    q_forecast = forecast_production(model_func, t_forecast, *params)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(t, q, 'bo', label='Actual')
    ax.plot(t_forecast, q_forecast, 'r-', label=f'{model_name} Fit')
    ax.set_xlabel('Time (months)')
    ax.set_ylabel('Rate (STB/day)')
    ax.set_title(f'Decline Curve Analysis - {model_name}')
    ax.legend()
    ax.grid(True)
    return fig, t_forecast, q_forecast

# -------------------------------
# 6. Streamlit App
# -------------------------------

def main():
    st.title("Decline Curve Analysis Tool with Economics")

    uploaded_file = st.file_uploader("Upload CSV file with 'time' and 'rate' columns", type='csv')
    use_synthetic = st.checkbox("Use synthetic data", value=True)

    if use_synthetic:
        time = np.linspace(0, 60, 61)
        q_i = 1000
        D = 0.05
        b = 0.7
        rate = q_i / np.power(1 + b * D * time, 1 / b)
        df = pd.DataFrame({'time': time, 'rate': rate})
    elif uploaded_file:
        df = pd.read_csv(uploaded_file)
        df.columns = df.columns.str.strip().str.lower()
        if 'time' not in df or 'rate' not in df:
            st.error("CSV must contain 'time' and 'rate' columns")
            return
    else:
        st.warning("Please upload a file or use synthetic data.")
        return

    model_choice = st.selectbox("Choose decline model", ['Exponential', 'Harmonic', 'Hyperbolic'])

    models = {
        'Exponential': (exponential_decline, [1000, 0.05]),
        'Harmonic': (harmonic_decline, [1000, 0.05]),
        'Hyperbolic': (hyperbolic_decline, [1000, 0.05, 0.7])
    }

    model_func, p0 = models[model_choice]
    t = df['time'].values
    q = df['rate'].values

    try:
        params, _ = fit_model(model_func, t, q, p0)
        fig, t_forecast, q_forecast = plot_fit(df, model_choice, model_func, params)
        st.pyplot(fig)
        st.success(f"Fitted parameters: {params}")

        eur = calculate_eur(t_forecast, q_forecast)
        st.info(f"Estimated Ultimate Recovery (EUR): {eur:.2f} STB")

        price = st.number_input("Oil Price ($/bbl)", value=70.0)
        cost = st.number_input("Operating Cost ($/bbl)", value=15.0)

        revenue_arr, cost_arr, net_cf_arr, npv = economic_analysis(t_forecast, q_forecast, price, cost)

        st.write(f"**Total Revenue:** ${revenue_arr.sum():,.2f}")
        st.write(f"**Total Cost:** ${cost_arr.sum():,.2f}")
        st.write(f"**Net Cash Flow:** ${net_cf_arr.sum():,.2f}")
        st.write(f"**NPV (10%):** ${npv:,.2f}")

        forecast_df = pd.DataFrame({
            'time': t_forecast,
            'forecast_rate': q_forecast,
            'revenue ($)': revenue_arr,
            'cost ($)': cost_arr,
            'net_cash_flow ($)': net_cf_arr
        })

        # Add summary row
        summary_row = pd.DataFrame({
            'time': ['TOTAL / NPV'],
            'forecast_rate': [np.nan],
            'revenue ($)': [revenue_arr.sum()],
            'cost ($)': [cost_arr.sum()],
            'net_cash_flow ($)': [net_cf_arr.sum()]
        })

        forecast_df = pd.concat([forecast_df, summary_row], ignore_index=True)
        forecast_df.loc[len(forecast_df)] = ['NPV (10%)', np.nan, np.nan, np.nan, npv]

        csv = forecast_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Forecast + Economics CSV", data=csv, file_name='forecast_economics.csv', mime='text/csv')

    except RuntimeError:
        st.error(f"Fit did not converge for {model_choice}")

if __name__ == '__main__':
    main()
