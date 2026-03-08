"""Streamlit + Plotly dashboard for Market Physics Engine."""
import io
import streamlit as st
import pandas as pd
import numpy as np

from data_loader import load_stock_data, load_multi_ticker_data
from derivatives import compute_derivatives
from physics_model import get_dynamics_summary, compute_market_energy
from signal_analysis import detect_turning_points
from diffusion_model import fit_and_simulate
from predictive_experiments import run_predictive_backtest
from visualization import (
    plot_price_interactive, plot_velocity_interactive, plot_acceleration_interactive,
    plot_jerk_interactive, plot_total_energy_interactive, plot_phase_space_interactive,
    plot_brownian_paths_interactive, plot_backtest_signals_interactive,
    TALL_CHART_HEIGHT,
)

st.set_page_config(page_title="Market Physics Engine", layout="wide")
PLOTLY_CONFIG = {"scrollZoom": True}
st.title("Market Physics Engine")
st.markdown(
    """
    A physics-based financial research platform. It models stock prices as particles in a dynamic
    system and analyzes their motion using derivatives such as velocity, acceleration, and jerk.
    The platform also calculates market energy, turning points, phase space behavior, Brownian
    simulations, and predictive signal backtesting.

    This system provides an in-depth analysis of any stock you choose and helps explore real-world
    problems at the intersection of finance and physics.

    Built in Python, the platform pulls a large amount of historical market data and presents it
    in a clear, easy-to-understand format.
    """
)

sidebar = st.sidebar
sidebar.header("Configuration")
compare_mode = sidebar.checkbox("Compare tickers", value=False)
ticker = sidebar.text_input("Stock Ticker", value="", placeholder="e.g., AAPL, NVDA, TSLA")
ticker2 = None
if compare_mode:
    ticker2 = sidebar.text_input("Second Ticker", value="", placeholder="e.g., MSFT, GOOGL")

period_opts = ["1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "Custom"]
period = sidebar.selectbox("Time Range", period_opts, index=4)
start_date = end_date = None
if period == "Custom":
    start_date = sidebar.date_input("Start date", value=pd.Timestamp("2022-01-01"))
    end_date = sidebar.date_input("End date", value=pd.Timestamp.today())
    period = None

ma_window = sidebar.selectbox("Moving Average (Potential Energy)", [50, 100], index=0)
show_brownian = sidebar.checkbox("Show Brownian Simulation", value=True)
show_backtest = sidebar.checkbox("Show Predictive Backtest", value=True)
energy_percentile = sidebar.slider(
    "Backtest energy threshold (percentile)", 60, 95, 75, 5,
    help="Only acceleration sign changes when total energy is above this percentile count as signals. Higher = fewer but potentially stronger signals."
)
horizon_days = sidebar.selectbox("Backtest horizon (days)", [5, 10, 20], index=0)
run_btn = sidebar.button("Run Analysis")


def _load_and_compute(ticker_str, per, start, end):
    df = load_stock_data(ticker_str, period=per or "2y", start=str(start) if start else None, end=str(end) if end else None)
    prices = df["Close"].values
    dates = df.index
    v, a, j = compute_derivatives(df["Close"])
    k, pot, te = compute_market_energy(prices, v, ma_window=ma_window)
    tp = detect_turning_points(dates, prices, a)
    summary = get_dynamics_summary(v, a)
    return {"df": df, "prices": prices, "dates": dates, "v": v, "a": a, "j": j, "k": k, "pot": pot, "te": te, "tp": tp, "summary": summary}


def _render_metrics(summary, tp, ticker_label):
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


tickers_to_run = []
if ticker and ticker.strip():
    tickers_to_run.append(ticker.strip().upper())
if compare_mode and ticker2 and ticker2.strip():
    tickers_to_run.append(ticker2.strip().upper())

if run_btn and tickers_to_run:
    try:
        with st.spinner("Loading data and computing physics metrics..."):
            results = {}
            for t in tickers_to_run:
                results[t] = _load_and_compute(t, period, start_date, end_date)

        period_label = f"{start_date} to {end_date}" if period is None else period
        st.success(f"Loaded {len(tickers_to_run)} ticker(s) for {period_label}")

        # Export CSV
        export_data = []
        for t, r in results.items():
            export_data.append({
                "ticker": t, "period": period_label, "n_days": len(r["prices"]),
                "max_velocity": r["summary"]["max_velocity"], "max_acceleration": r["summary"]["max_acceleration"],
                "avg_force": r["summary"]["average_force_magnitude"], "force_changes": r["summary"]["force_direction_changes"],
                "turning_points": len(r["tp"]),
            })
        export_df = pd.DataFrame(export_data)
        csv_buf = io.StringIO()
        export_df.to_csv(csv_buf, index=False)
        st.sidebar.download_button("Download metrics CSV", csv_buf.getvalue(), file_name="market_physics_metrics.csv", mime="text/csv")

        for ticker_label, r in results.items():
            df, prices, dates, v, a, j, k, pot, te, tp, summary = r["df"], r["prices"], r["dates"], r["v"], r["a"], r["j"], r["k"], r["pot"], r["te"], r["tp"], r["summary"]

            st.subheader(ticker_label)

            tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview", "Dynamics", "Phase Space", "Simulations", "Backtest"])

            with tab1:
                with st.expander("Relevance", expanded=False, key=f"rel1_{ticker_label}"):
                    st.markdown(
                        "**What this section does:** The Overview summarizes key physics-derived metrics (velocity, acceleration, force, turning points) and displays price with detected peaks and troughs. "
                        "**Problems it helps solve:** Identifying regime changes, spotting local extrema for timing, and gauging how volatile or dynamic the price has been over the period."
                    )
                _render_metrics(summary, tp, ticker_label)
                st.plotly_chart(plot_price_interactive(dates, prices, tp, ticker_label, height=TALL_CHART_HEIGHT), use_container_width=True, config=PLOTLY_CONFIG)

            with tab2:
                with st.expander("Relevance", expanded=False, key=f"rel2_{ticker_label}"):
                    st.markdown(
                        "**What this section does:** Shows velocity (momentum), acceleration (force), jerk (rate of force change), and total market energy over time. "
                        "**Problems it helps solve:** Understanding momentum shifts, detecting when force reverses (potential turning points), and measuring market \"activity\" or energy for volatility context."
                    )
                col1, col2 = st.columns(2)
                with col1:
                    st.plotly_chart(plot_velocity_interactive(dates, v), use_container_width=True, config=PLOTLY_CONFIG)
                    st.plotly_chart(plot_acceleration_interactive(dates, a), use_container_width=True, config=PLOTLY_CONFIG)
                with col2:
                    st.plotly_chart(plot_jerk_interactive(dates, j), use_container_width=True, config=PLOTLY_CONFIG)
                    st.plotly_chart(plot_total_energy_interactive(dates, te), use_container_width=True, config=PLOTLY_CONFIG)

            with tab3:
                with st.expander("Relevance", expanded=False, key=f"rel3_{ticker_label}"):
                    st.markdown(
                        "**What this section does:** Plots price (x-axis) vs velocity (y-axis), the classical phase space. The trajectory reveals loops (cycles), attractors, and momentum patterns. "
                        "**Problems it helps solve:** Identifying cyclical behavior, visualizing mean reversion vs momentum regimes, and spotting structural patterns that time-series charts may hide."
                    )
                st.markdown(
                    '### Phase Space (Price vs Velocity) '
                    '<span title="Plots price on the x-axis and velocity on the y-axis. This classical physics view reveals the trajectory of the market as a dynamical system—cycles appear as loops, and the pattern can suggest momentum and reversal behavior." style="cursor: help;">&#9432;</span>',
                    unsafe_allow_html=True,
                )
                fig_phase = plot_phase_space_interactive(prices, v, ticker_label, height=TALL_CHART_HEIGHT)
                st.plotly_chart(fig_phase, use_container_width=True, config=PLOTLY_CONFIG)
                try:
                    png_bytes = fig_phase.to_image(format="png")
                    st.download_button(
                        "Download chart as PNG",
                        data=png_bytes,
                        file_name=f"{ticker_label}_phase_space.png",
                        mime="image/png",
                        key=f"phase_png_{ticker_label}",
                    )
                except Exception:
                    pass

            with tab4:
                with st.expander("Relevance", expanded=False, key=f"rel4_{ticker_label}"):
                    st.markdown(
                        "**What this section does:** Simulates many possible future price paths using geometric Brownian motion, fitted to historical drift and volatility. The blue line is actual price; the purple paths show simulations. "
                        "**Problems it helps solve:** Assessing path uncertainty, stress-testing \"what if\" scenarios, and comparing actual movement to a simple random-walk benchmark."
                    )
                if show_brownian:
                    st.markdown(
                        '### Brownian Motion Simulation '
                        '<span title="Models price as a random walk. Uses historical volatility to simulate future paths with real dates on the x-axis." style="cursor: help;">&#9432;</span>',
                        unsafe_allow_html=True,
                    )
                    sim_paths, mu, sigma = fit_and_simulate(df["Close"], n_days=min(252, len(prices) // 2), n_paths=50)
                    st.plotly_chart(plot_brownian_paths_interactive(dates, prices, sim_paths, 20, height=TALL_CHART_HEIGHT), use_container_width=True, config=PLOTLY_CONFIG)
                    st.caption(f"Drift μ={mu:.6f}, Volatility σ={sigma:.4f} (daily)")
                else:
                    st.info("Enable Brownian Simulation in the sidebar.")

            with tab5:
                with st.expander("Relevance", expanded=False, key=f"rel5_{ticker_label}"):
                    st.markdown(
                        "**What this section does:** Backtests whether physics-derived signals (acceleration sign change + high energy) predict forward returns. Reports hit rate, average return, and buy-and-hold benchmark with confidence intervals. "
                        "**Problems it helps solve:** Evaluating if the physics model has predictive value, comparing strategy performance to buy-and-hold, and assessing statistical significance of results."
                    )
                if show_backtest:
                    st.markdown(
                        '### Predictive Signal Backtest '
                        '<span title="Tests whether acceleration sign change plus high total energy predicts price moves. Adjust energy threshold and horizon in the sidebar." style="cursor: help;">&#9432;</span>',
                        unsafe_allow_html=True,
                    )
                    bt = run_predictive_backtest(prices, v, a, k, pot, te, horizon_days=horizon_days, energy_threshold_percentile=energy_percentile)
                    bt_col1, bt_col2 = st.columns([1, 2])
                    with bt_col1:
                        st.metric("Signals (accel sign change + high energy)", bt["n_signals"])
                        st.metric(f"Avg {horizon_days}-day return", f"{bt['avg_return']*100:.2f}%" if bt["n_signals"] > 0 else "—")
                        if bt["n_signals"] > 0:
                            lo, hi = bt["avg_return_ci"]
                            st.caption(f"95% CI: [{lo*100:.2f}%, {hi*100:.2f}%]")
                        st.metric("Hit rate (accuracy)", f"{bt['hit_rate']*100:.1f}%" if bt["n_signals"] > 0 else "—")
                        if bt["n_signals"] > 0:
                            lo, hi = bt["hit_rate_ci"]
                            st.caption(f"95% CI: [{lo*100:.1f}%, {hi*100:.1f}%]")
                        st.metric(f"Avg {horizon_days}-day return (buy & hold)", f"{bt['buy_hold_return']*100:.2f}%")
                    with bt_col2:
                        if bt["n_signals"] > 0 and bt.get("signal_indices"):
                            st.plotly_chart(
                                plot_backtest_signals_interactive(dates, prices, bt["signal_indices"], bt["signal_types"], ticker_label, height=TALL_CHART_HEIGHT),
                                use_container_width=True,
                                config=PLOTLY_CONFIG,
                            )
                else:
                    st.info("Enable Predictive Backtest in the sidebar.")

    except Exception as e:
        st.error(str(e))
