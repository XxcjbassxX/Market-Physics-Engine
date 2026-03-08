"""Data acquisition for Market Physics Engine."""
import pandas as pd
import yfinance as yf
from typing import Union, Optional

DEFAULT_TICKERS = ["AAPL", "NVDA", "TSLA"]
DEFAULT_PERIOD = "2y"

def load_stock_data(
    ticker: str,
    period: str = DEFAULT_PERIOD,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    if not ticker or not str(ticker).strip():
        raise ValueError("Ticker symbol cannot be empty.")
    stock = yf.Ticker(str(ticker).strip().upper())
    if start and end:
        df = stock.history(start=start, end=end)
    else:
        df = stock.history(period=period)
    if df.empty:
        raise ValueError(f"No data for ticker '{ticker}'.")
    df = df[["Close"]].dropna().sort_index()
    df.columns = ["Close"]
    return df

def load_multi_ticker_data(
    tickers: Union[list, None] = None,
    period: str = DEFAULT_PERIOD,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> dict:
    tickers = tickers or DEFAULT_TICKERS
    result = {}
    for t in tickers:
        try:
            result[t] = load_stock_data(t, period=period, start=start, end=end)
        except ValueError:
            result[t] = None
    return result
