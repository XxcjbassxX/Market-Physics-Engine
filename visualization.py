"""Plotly charts."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go

def plot_price_interactive(dates, prices, turning_points=None, ticker="Stock"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates,y=prices,mode="lines",name="Price",line=dict(color="#2563eb",width=2)))
    if turning_points is not None and len(turning_points)>0:
        for tp,sym,col in [("peak","triangle-down","#dc2626"),("trough","triangle-up","#059669")]:
            df = turning_points[turning_points["type"]==tp]
            if len(df)>0:
                fig.add_trace(go.Scatter(x=df["date"],y=df["price"],mode="markers",name=tp.title(),marker=dict(symbol=sym,size=12,color=col)))
    fig.update_layout(title=f"{ticker} Price",xaxis_title="Date",yaxis_title="Price ($)",template="plotly_white",hovermode="x unified",height=350)
    return fig

def plot_velocity_interactive(dates,v):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates,y=v,mode="lines",line=dict(color="#059669",width=1.5)))
    fig.add_hline(y=0,line_dash="dash",line_color="gray",opacity=0.5)
    fig.update_layout(title="Velocity",xaxis_title="Date",yaxis_title="Velocity",template="plotly_white",height=300)
    return fig

def plot_acceleration_interactive(dates,a):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates,y=a,mode="lines",line=dict(color="#7c3aed",width=1.5)))
    fig.add_hline(y=0,line_dash="dash",line_color="gray",opacity=0.5)
    fig.update_layout(title="Acceleration",xaxis_title="Date",yaxis_title="Acceleration",template="plotly_white",height=300)
    return fig

def plot_jerk_interactive(dates,j):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates,y=j,mode="lines",line=dict(color="#ea580c",width=1.5)))
    fig.add_hline(y=0,line_dash="dash",line_color="gray",opacity=0.5)
    fig.update_layout(title="Jerk",xaxis_title="Date",yaxis_title="Jerk",template="plotly_white",height=300)
    return fig

def plot_total_energy_interactive(dates,e):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates,y=e,mode="lines",line=dict(color="#be185d",width=1.5)))
    fig.update_layout(title="Total Energy",xaxis_title="Date",yaxis_title="Energy",template="plotly_white",height=300)
    return fig

def plot_frequency_spectrum_interactive(freqs,power,cycles):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=1.0/freqs,y=power,mode="lines",line=dict(color="#0ea5e9",width=1.5)))
    for c in cycles[:5]:
        p = c.get("period_days",0)
        if 0<p<1e6:
            fig.add_vline(x=p,line_dash="dot",annotation_text=f"{p:.0f}d",line_color="gray")
    fig.update_layout(title="Frequency Spectrum",xaxis_title="Period (days)",yaxis_title="Power",template="plotly_white",height=350)
    return fig

def plot_brownian_paths_interactive(dates,actual,simulated,n_show=20):
    fig = go.Figure()
    for i in np.linspace(0,simulated.shape[1]-1,min(n_show,simulated.shape[1]),dtype=int):
        fig.add_trace(go.Scatter(x=np.arange(len(simulated)),y=simulated[:,i],mode="lines",line=dict(color="rgba(100,100,200,0.3)",width=1),showlegend=False))
    fig.add_trace(go.Scatter(x=np.arange(len(actual)),y=actual,mode="lines",name="Actual",line=dict(color="#2563eb",width=2)))
    fig.update_layout(title="Brownian vs Actual",xaxis_title="Day",yaxis_title="Price ($)",template="plotly_white",height=350)
    return fig

def plot_backtest_signals_interactive(dates,prices,signal_indices,signal_types,ticker="Stock"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates,y=prices,mode="lines",name="Price",line=dict(color="#2563eb",width=2)))
    buy = [i for i,t in zip(signal_indices,signal_types) if t=="buy"]
    sell = [i for i,t in zip(signal_indices,signal_types) if t=="sell"]
    if buy:
        fig.add_trace(go.Scatter(x=dates[buy],y=prices[buy],mode="markers",name="Buy",marker=dict(symbol="triangle-up",size=14,color="#059669")))
    if sell:
        fig.add_trace(go.Scatter(x=dates[sell],y=prices[sell],mode="markers",name="Sell",marker=dict(symbol="triangle-down",size=14,color="#dc2626")))
    fig.update_layout(title=f"{ticker} Backtest Signals",xaxis_title="Date",yaxis_title="Price ($)",template="plotly_white",height=350)
    return fig
