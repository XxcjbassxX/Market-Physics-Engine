# Market Physics Engine

**App Link:** https://market-physics-engine-3juvzzyewdfhg8tb2uvxct.streamlit.app/

A physics-based financial research platform. It models stock prices as particles in a dynamic system and analyzes their motion using derivatives such as velocity, acceleration, and jerk. The platform also calculates market energy, turning points, phase space behavior, Brownian simulations, and predictive signal backtesting.

This system provides an in-depth analysis of any stock you choose and helps explore real-world problems at the intersection of finance and physics.

Built in Python, the platform pulls a large amount of historical market data and presents it in a clear, easy-to-understand format.

---

## Features

- **Time range**: 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, or custom date range
- **Compare tickers**: Analyze two tickers side by side
- **Phase space plot**: Price vs velocity (classical physics view)
- **Export**: Download metrics as CSV and charts as PNG
- **Backtest**: Adjustable energy threshold and horizon; buy-and-hold benchmark; 95% confidence intervals

---

## Methodology

### Derivatives
Price is treated as position. Central finite differences give:
- **Velocity** = d(price)/dt (momentum)
- **Acceleration** = d(velocity)/dt (force)
- **Jerk** = d(acceleration)/dt (rate of change of force)

### Energy
- **Kinetic energy** = velocity² (proportional to momentum squared)
- **Potential energy** = (price − moving average)² (deviation from equilibrium)
- **Total energy** = kinetic + potential

### Turning Points
Peaks and troughs are detected where acceleration changes sign (force reversal).

### Phase Space
Plots price on the x-axis and velocity on the y-axis. The trajectory reveals cycles (loops) and momentum patterns.

### Predictive Signals
Signals trigger when acceleration changes sign **and** total energy exceeds a percentile threshold. The backtest evaluates forward returns over a configurable horizon and reports hit rate with bootstrap confidence intervals. Buy-and-hold return provides a benchmark.

### Brownian Simulation
Uses historical drift (μ) and volatility (σ) to simulate possible future price paths as geometric Brownian motion.

---

## Screenshots

*Add screenshots of the dashboard here (e.g., Overview tab, Phase Space, Backtest) to showcase the app.*

---

## Setup (Optional)

1. **Clone the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/market_physics_model.git
   cd market_physics_model
   ```

2. **Create a virtual environment** (optional but recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate   # Linux/Mac
   # or: venv\Scripts\activate   # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the app**
   ```bash
   streamlit run dashboard.py
   ```

5. Open [http://localhost:8501](http://localhost:8501) in your browser.
