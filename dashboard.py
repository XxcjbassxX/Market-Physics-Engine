"""Streamlit + Plotly dashboard for Market Physics Engine."""
import streamlit as st
import pandas as pd
import numpy as np

from data_loader import load_stock_data
from derivatives import compute_derivatives
from physics_model import get_dynamics_summary, compute_market_energy
from signal_analysis import detect_turning_points, detect_cycles_fft, get_dominant_cycle_lengths
from diffusion_model import fit_and_simulate
from predictive_experiments import run_predictive_backtest
from visualization import (
    plot_price_interactive, plot_velocity_interactive, plot_acceleration_interactive,
    plot_jerk_interactive, plot_total_energy_interactive, plot_frequency_spectrum_interactive,
    plot_brownian_paths_interactive, plot_backtest_signals_interactive,
)

st.set_page_config(page_title="Market Physics Engine", layout="wide")
st.title("Market Physics Engine")
st.markdown(
    """
    A physics-based financial research platform. Models stock prices as particles in a dynamic system.
    Computes derivatives (velocity, acceleration, jerk), market energy, turning points, dominant cycles,
    Brownian simulations, and predictive signal backtesting.
    """
)

sidebar = st.sidebar
sidebar.header("Configuration")
ticker = sidebar.text_input("Stock Ticker", value="", placeholder="e.g., AAPL, NVDA, TSLA")
period = sidebar.selectbox("Time Range", ["1mo", "3mo", "6mo", "1y", "2y"], index=4)
ma_window = sidebar.selectbox("Moving Average (Potential Energy)", [50, 100], index=0)
show_brownian = sidebar.checkbox("Show Brownian Simulation", value=True)
show_backtest = sidebar.checkbox("Show Predictive Backtest", value=True)
run_btn = sidebar.button("Run Analysis")

if run_btn and ticker:
    try:
        with st.spinner("Loading data and computing physics metrics..."):
            df = load_stock_data(ticker.strip().upper(), period)
            prices, dates = df["Close"].values, df.index
            v, a, j = compute_derivatives(df["Close"])
            k, pot, te = compute_market_energy(prices, v, ma_window=ma_window)
            tp = detect_turning_points(dates, prices, a)
            freqs, power, cycles = detect_cycles_fft(prices, n_dominant=5)
            cycle_prominence = get_dominant_cycle_lengths(prices)
            summary = get_dynamics_summary(v, a)

        st.success(f"Loaded {len(df)} trading days for {ticker}")

        st.subheader("Key Metrics")
        metric_label_style = "font-size: 0.7rem; color: var(--text-color-secondary, rgb(128, 132, 149));"
        metric_value_style = "font-size: 1.5rem; font-weight: 600; color: inherit;"
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.markdown(
            f'<div style="padding: 0.25rem 0;"><div style="{metric_label_style}">Max Velocity</div>'
            f'<div style="{metric_value_style}">${summary["max_velocity"]:.2f}/day</div></div>',
            unsafe_allow_html=True,
        )
        c2.markdown(
            f'<div style="padding: 0.25rem 0;"><div style="{metric_label_style}">Max Acceleration</div>'
            f'<div style="{metric_value_style}">${summary["max_acceleration"]:.2f}/day²</div></div>',
            unsafe_allow_html=True,
        )
        c3.markdown(
            f'<div style="padding: 0.25rem 0;"><div style="{metric_label_style}">'
            '<span title="Average magnitude of market force (acceleration) acting on price." style="cursor: help;">'
            'Avg Force Magnitude &#9432;</span></div>'
            f'<div style="{metric_value_style}">${summary["average_force_magnitude"]:.2f}</div></div>',
            unsafe_allow_html=True,
        )
        c4.markdown(
            f'<div style="padding: 0.25rem 0;"><div style="{metric_label_style}">Force Direction Changes</div>'
            f'<div style="{metric_value_style}">{summary["force_direction_changes"]}</div></div>',
            unsafe_allow_html=True,
        )
        c5.markdown(
            f'<div style="padding: 0.25rem 0;"><div style="{metric_label_style}">Turning Points</div>'
            f'<div style="{metric_value_style}">{len(tp)}</div></div>',
            unsafe_allow_html=True,
        )

        st.subheader("Price (Position) & Turning Points")
        st.plotly_chart(plot_price_interactive(dates, prices, tp, ticker), use_container_width=True)

        st.subheader("Dynamics")
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(plot_velocity_interactive(dates, v), use_container_width=True)
            st.plotly_chart(plot_acceleration_interactive(dates, a), use_container_width=True)
        with col2:
            st.plotly_chart(plot_jerk_interactive(dates, j), use_container_width=True)
            st.plotly_chart(plot_total_energy_interactive(dates, te), use_container_width=True)

        st.markdown(
            '### Frequency Analysis (Dominant Cycles) '
            '<span title="Uses the Fourier transform to convert price data into frequency space, revealing repeating patterns and dominant cycle lengths (e.g., 20-day, 60-day, 120-day)." style="cursor: help;">&#9432;</span>',
            unsafe_allow_html=True,
        )
        if len(freqs) > 0:
            st.plotly_chart(plot_frequency_spectrum_interactive(freqs, power, cycles), use_container_width=True)
            st.dataframe(pd.DataFrame(cycles)[["period_days", "frequency", "power"]].round(4), use_container_width=True)
            st.caption("Prominence of 20d, 60d, 120d cycles:")
            st.dataframe(pd.DataFrame(cycle_prominence), use_container_width=True)

        if show_brownian:
            st.markdown(
                '### Brownian Motion Simulation '
                '<span title="Models price as a random walk (diffusion process). Uses historical volatility to simulate many possible future price paths and compare them to actual movement." style="cursor: help;">&#9432;</span>',
                unsafe_allow_html=True,
            )
            sim_paths, mu, sigma = fit_and_simulate(df["Close"], n_days=min(252, len(prices) // 2), n_paths=50)
            st.plotly_chart(plot_brownian_paths_interactive(dates, prices, sim_paths, 20), use_container_width=True)
            st.caption(f"Drift μ={mu:.6f}, Volatility σ={sigma:.4f} (daily)")

        if show_backtest:
            st.markdown(
                '### Predictive Signal Backtest '
                '<span title="Tests whether physics-derived conditions (e.g., acceleration sign change + high total energy) predict price moves in the next 5 trading days. Evaluated on historical data." style="cursor: help;">&#9432;</span>',
                unsafe_allow_html=True,
            )
            bt = run_predictive_backtest(prices, v, a, k, pot, te, horizon_days=5)
            bt_col1, bt_col2 = st.columns([1, 2])
            with bt_col1:
                st.metric("Signals (accel sign change + high energy)", bt["n_signals"])
                st.metric("Avg 5-day return", f"{bt['avg_return']*100:.2f}%" if bt["n_signals"] > 0 else "—")
                st.metric("Hit rate (accuracy)", f"{bt['hit_rate']*100:.1f}%" if bt["n_signals"] > 0 else "—")
            with bt_col2:
                if bt["n_signals"] > 0 and bt.get("signal_indices"):
                    st.plotly_chart(
                        plot_backtest_signals_interactive(dates, prices, bt["signal_indices"], bt["signal_types"], ticker),
                        use_container_width=True,
                    )

    except Exception as e:
        st.error(str(e))
