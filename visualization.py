"""Plotly charts."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go

CHART_HEIGHT = 260
TALL_CHART_HEIGHT = 380

def _add_rangeslider(fig):
    fig.update_layout(
        xaxis=dict(rangeslider=dict(visible=True, thickness=0.05)),
        dragmode="zoom",
    )

def plot_price_interactive(dates, prices, turning_points=None, ticker="Stock", show_rangeslider=True, height=None):
    h = height or CHART_HEIGHT
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates,y=prices,mode="lines",name="Price",line=dict(color="#2563eb",width=2)))
    if turning_points is not None and len(turning_points)>0:
        for tp,sym,col in [("peak","triangle-down","#dc2626"),("trough","triangle-up","#059669")]:
            df = turning_points[turning_points["type"]==tp]
            if len(df)>0:
                fig.add_trace(go.Scatter(x=df["date"],y=df["price"],mode="markers",name=tp.title(),marker=dict(symbol=sym,size=12,color=col)))
    fig.update_layout(title=f"{ticker} Price", xaxis_title="Date", yaxis_title="Price ($)", template="plotly_white", hovermode="x unified", height=h)
    if show_rangeslider:
        _add_rangeslider(fig)
    return fig

def plot_phase_space_interactive(prices, velocity, ticker="Stock", height=None):
    """Phase space: price (x) vs velocity (y). Trajectory reveals cycles and attractors."""
    h = height or CHART_HEIGHT
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=prices, y=velocity, mode="lines+markers", line=dict(color="#0ea5e9", width=1.5), marker=dict(size=3), name="Trajectory"))
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    fig.update_layout(
        title=f"{ticker} Phase Space (Price vs Velocity)",
        xaxis_title="Price ($)",
        yaxis_title="Velocity",
        template="plotly_white",
        height=h,
        dragmode="zoom",
    )
    return fig

def plot_velocity_interactive(dates, v, show_rangeslider=True):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=v, mode="lines", line=dict(color="#059669", width=1.5)))
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    fig.update_layout(title="Velocity", xaxis_title="Date", yaxis_title="Velocity", template="plotly_white", height=CHART_HEIGHT)
    if show_rangeslider:
        _add_rangeslider(fig)
    return fig

def plot_acceleration_interactive(dates, a, show_rangeslider=True):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=a, mode="lines", line=dict(color="#7c3aed", width=1.5)))
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    fig.update_layout(title="Acceleration", xaxis_title="Date", yaxis_title="Acceleration", template="plotly_white", height=CHART_HEIGHT)
    if show_rangeslider:
        _add_rangeslider(fig)
    return fig

def plot_jerk_interactive(dates, j, show_rangeslider=True):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=j, mode="lines", line=dict(color="#ea580c", width=1.5)))
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    fig.update_layout(title="Jerk", xaxis_title="Date", yaxis_title="Jerk", template="plotly_white", height=CHART_HEIGHT)
    if show_rangeslider:
        _add_rangeslider(fig)
    return fig

def plot_total_energy_interactive(dates, e, show_rangeslider=True):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=e, mode="lines", line=dict(color="#be185d", width=1.5)))
    fig.update_layout(title="Total Energy", xaxis_title="Date", yaxis_title="Energy", template="plotly_white", height=CHART_HEIGHT)
    if show_rangeslider:
        _add_rangeslider(fig)
    return fig

def plot_brownian_paths_interactive(dates, actual, simulated, n_show=20, height=None):
    """Brownian simulation with real dates on x-axis. Extended dates for future paths."""
    h = height or CHART_HEIGHT
    dates_pd = pd.DatetimeIndex(dates) if not isinstance(dates, pd.DatetimeIndex) else dates
    n_sim = simulated.shape[0]
    future_dates = pd.bdate_range(start=dates_pd[-1] + pd.Timedelta(days=1), periods=n_sim - 1, freq="B")
    sim_dates = list(dates_pd[-1:]) + list(future_dates)
    fig = go.Figure()
    indices = list(np.linspace(0, simulated.shape[1] - 1, min(n_show, simulated.shape[1]), dtype=int))
    for idx, i in enumerate(indices):
        fig.add_trace(go.Scatter(
            x=sim_dates, y=simulated[:, i], mode="lines",
            line=dict(color="rgba(100,100,200,0.3)", width=1),
            name="Simulated paths" if idx == 0 else None,
            showlegend=(idx == 0),
        ))
    fig.add_trace(go.Scatter(x=dates_pd, y=actual, mode="lines", name="Actual", line=dict(color="#2563eb", width=2)))
    fig.update_layout(title="Brownian vs Actual", xaxis_title="Date", yaxis_title="Price ($)", template="plotly_white", height=h)
    _add_rangeslider(fig)
    return fig

def plot_backtest_signals_interactive(dates, prices, signal_indices, signal_types, ticker="Stock", height=None):
    h = height or CHART_HEIGHT
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=prices, mode="lines", name="Price", line=dict(color="#2563eb", width=2)))
    buy = [i for i, t in zip(signal_indices, signal_types) if t == "buy"]
    sell = [i for i, t in zip(signal_indices, signal_types) if t == "sell"]
    if buy:
        fig.add_trace(go.Scatter(x=dates[buy], y=prices[buy], mode="markers", name="Buy", marker=dict(symbol="triangle-up", size=14, color="#059669")))
    if sell:
        fig.add_trace(go.Scatter(x=dates[sell], y=prices[sell], mode="markers", name="Sell", marker=dict(symbol="triangle-down", size=14, color="#dc2626")))
    fig.update_layout(title=f"{ticker} Backtest Signals", xaxis_title="Date", yaxis_title="Price ($)", template="plotly_white", height=h)
    _add_rangeslider(fig)
    return fig
