"""Main entry point."""
import sys
from data_loader import load_stock_data, load_multi_ticker_data, DEFAULT_TICKERS
from derivatives import compute_derivatives
from physics_model import get_dynamics_summary, compute_market_energy
from signal_analysis import detect_turning_points
from predictive_experiments import run_predictive_backtest, backtest_multi_ticker
from visualization import plot_price_interactive

def run(ticker="AAPL", period="2y", out=None):
    df = load_stock_data(ticker, period)
    p, d = df["Close"].values, df.index
    v,a,j = compute_derivatives(df["Close"])
    k,pot,te = compute_market_energy(p,v,50)
    tp = detect_turning_points(d,p,a)
    bt = run_predictive_backtest(p,v,a,k,pot,te,5)
    s = get_dynamics_summary(v,a)
    print("Metrics:", s, "| Turning points:", len(tp), "| Signals:", bt["n_signals"])
    if out:
        plot_price_interactive(d,p,tp,ticker).write_html(out)
        print("Saved to", out)
    return s

if __name__=="__main__":
    t = sys.argv[1] if len(sys.argv)>=2 else "AAPL"
    r = sys.argv[2] if len(sys.argv)>=3 else "2y"
    if t.upper()=="ALL":
        data = load_multi_ticker_data(None,r)
        m = {}
        for tk,df in data.items():
            if df is None: continue
            v,a,j = compute_derivatives(df["Close"])
            k,p,te = compute_market_energy(df["Close"].values,v,50)
            m[tk] = {"prices":df["Close"].values,"velocity":v,"acceleration":a,"kinetic":k,"potential":p,"total_energy":te}
        print(backtest_multi_ticker(m).to_string())
    else:
        run(t,r,"market_physics_report.html")
