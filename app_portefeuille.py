import datetime as dt
import numpy as np
import pandas as pd
import yfinance as yf
import cvxpy as cp
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def load_price_data(tickers, start, end):
    """Download price data from Yahoo Finance"""
    data = yf.download(tickers, start=start, end=end, progress=False)["Close"]
    if isinstance(data, pd.Series):
        data = data.to_frame()
    data = data.dropna()
    return data


def compute_return_stats(prices):
    """Compute annualized returns and covariance"""
    rets = prices.pct_change().dropna()
    mu = rets.mean() * 252
    cov = rets.cov() * 252
    return rets, mu, cov


def make_positive_definite(cov_matrix, min_eigenval=1e-8):
    """Ensure covariance matrix is positive definite"""
    eigvals = np.linalg.eigvalsh(cov_matrix)
    if np.min(eigvals) < min_eigenval:
        adjustment = (min_eigenval - np.min(eigvals)) + 1e-8
        cov_matrix = cov_matrix + np.eye(len(cov_matrix)) * adjustment
    return cov_matrix


def estimate_covariance(returns, method="ledoit"):
    """Estimate covariance matrix with Ledoit-Wolf shrinkage by default"""
    if method == "ledoit":
        lw = LedoitWolf()
        cov_matrix = lw.fit(returns.values).covariance_ * 252
    else:
        cov_matrix = returns.cov().values * 252

    cov_matrix = make_positive_definite(cov_matrix)
    return pd.DataFrame(cov_matrix, index=returns.columns, columns=returns.columns)


# ============================================================================
# PORTFOLIO OPTIMIZATION
# ============================================================================


def optimize_minimum_variance(cov, min_weight=0.0, max_weight=1.0):
    """Minimum Variance Portfolio"""
    n = len(cov)
    w = cp.Variable(n)

    risk = cp.quad_form(w, cov.values)
    objective = cp.Minimize(risk)

    constraints = [cp.sum(w) == 1, w >= min_weight, w <= max_weight]

    problem = cp.Problem(objective, constraints)

    try:
        problem.solve(solver=cp.OSQP, verbose=False)
        if problem.status not in ["optimal", "optimal_inaccurate"]:
            problem.solve(solver=cp.SCS, verbose=False)
    except:
        problem.solve(solver=cp.SCS, verbose=False)

    if w.value is None:
        return None

    weights = np.array(w.value).flatten()
    weights = np.maximum(weights, 0)
    weights = weights / weights.sum()

    return pd.Series(weights, index=cov.index)


def optimize_mean_variance(mu, cov, risk_aversion=2.5, min_weight=0.0, max_weight=1.0):
    """Mean-Variance Portfolio"""
    n = len(mu)
    w = cp.Variable(n)

    ret = mu.values @ w
    risk = cp.quad_form(w, cov.values)

    objective = cp.Maximize(ret - risk_aversion * risk)

    constraints = [cp.sum(w) == 1, w >= min_weight, w <= max_weight]

    problem = cp.Problem(objective, constraints)

    try:
        problem.solve(solver=cp.OSQP, verbose=False)
        if problem.status not in ["optimal", "optimal_inaccurate"]:
            problem.solve(solver=cp.SCS, verbose=False)
    except:
        problem.solve(solver=cp.SCS, verbose=False)

    if w.value is None:
        return None

    weights = np.array(w.value).flatten()
    weights = np.maximum(weights, 0)
    weights = weights / weights.sum()

    return pd.Series(weights, index=mu.index)


def optimize_max_sharpe_scipy(mu, cov, rf=0.0, min_weight=0.0, max_weight=1.0):
    """Maximum Sharpe Ratio using scipy - robust method"""
    n = len(mu)

    def neg_sharpe(w):
        w = np.array(w)
        port_return = np.dot(w, mu.values)
        port_vol = np.sqrt(np.dot(w, np.dot(cov.values, w)))
        if port_vol == 0:
            return 1e10
        return -(port_return - rf) / port_vol

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]

    bounds = tuple((min_weight, max_weight) for _ in range(n))
    w0 = np.array([1.0 / n] * n)

    result = minimize(
        neg_sharpe,
        w0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-9, "maxiter": 1000},
    )

    if not result.success:
        w0 = np.random.dirichlet(np.ones(n))
        result = minimize(
            neg_sharpe,
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-9, "maxiter": 1000},
        )

    if not result.success:
        return None

    weights = result.x
    weights = np.maximum(weights, 0)
    weights = weights / weights.sum()

    return pd.Series(weights, index=mu.index)


def optimize_risk_parity(cov, max_iter=1000, tol=1e-8):
    """Risk Parity Portfolio"""
    n = len(cov)
    cov_matrix = cov.values
    cov_matrix = make_positive_definite(cov_matrix)

    vols = np.sqrt(np.diag(cov_matrix))
    w = 1.0 / vols
    w = w / w.sum()

    for iteration in range(max_iter):
        port_vol = np.sqrt(w @ cov_matrix @ w)

        if port_vol < 1e-10:
            break

        mrc = cov_matrix @ w
        rc = w * mrc / port_vol
        rc = np.maximum(rc, 1e-12)

        target_rc = port_vol / n
        w_new = w * target_rc / rc
        w_new = np.maximum(w_new, 0)
        w_new = w_new / w_new.sum()

        if np.allclose(w, w_new, rtol=tol, atol=tol):
            w = w_new
            break

        w = 0.5 * w + 0.5 * w_new

    return pd.Series(w, index=cov.index)


# ============================================================================
# BLACK-LITTERMAN MODEL
# ============================================================================


def compute_market_implied_returns(cov, market_weights, delta=2.5):
    """Compute equilibrium returns: π = δ * Σ * w_market"""
    pi = delta * cov.values @ market_weights
    return pd.Series(pi, index=cov.index)


def black_litterman_update(pi, cov, P, Q, Omega, tau=0.025):
    """Black-Litterman posterior expected returns"""
    Sigma = cov.values
    pi_vec = pi.values.reshape(-1, 1)
    Q_vec = Q.reshape(-1, 1)

    tau_sigma = tau * Sigma
    tau_sigma_inv = np.linalg.inv(tau_sigma)

    Omega_inv = np.linalg.inv(Omega)

    post_precision = tau_sigma_inv + P.T @ Omega_inv @ P
    post_mean = np.linalg.inv(post_precision) @ (
        tau_sigma_inv @ pi_vec + P.T @ Omega_inv @ Q_vec
    )

    return pd.Series(post_mean.flatten(), index=pi.index)


def build_bl_views(returns, cov, rf, tau, view_dict, view_uncertainty):
    """Construct Black-Litterman model from views"""
    assets = list(cov.index)
    n_assets = len(assets)

    market_weights = np.ones(n_assets) / n_assets
    pi = compute_market_implied_returns(cov, market_weights, delta=2.5)

    if not view_dict or all(v is None or v == 0 for v in view_dict.values()):
        return pi

    active_views = [
        (asset, ret) for asset, ret in view_dict.items() if ret is not None and ret != 0
    ]

    if not active_views:
        return pi

    k = len(active_views)
    P = np.zeros((k, n_assets))
    Q = np.zeros(k)

    for i, (asset, expected_return) in enumerate(active_views):
        asset_idx = assets.index(asset)
        P[i, asset_idx] = 1.0
        Q[i] = expected_return

    Omega = np.eye(k) * (view_uncertainty**2)
    mu_bl = black_litterman_update(pi, cov, P, Q, Omega, tau)

    return mu_bl


def get_smart_bl_views(tickers, mu_sample):
    """Generate smart Black-Litterman views based on asset characteristics"""
    views = {}

    # Asset classification
    us_tech = ["AAPL", "MSFT", "GOOGL", "NVDA", "META", "AMZN", "TSLA"]
    us_value = ["JPM", "BAC", "WFC", "JNJ", "PG", "KO", "PEP", "XOM", "CVX"]
    international = ["EWJ", "EWG", "EWU", "MCHI", "EEM"]
    bonds = ["TLT", "AGG", "BND", "LQD", "HYG"]
    commodities = ["GLD", "SLV", "DBC", "USO"]

    for ticker in tickers:
        sample_return = mu_sample.get(ticker, 0.08)

        if ticker in us_tech:
            views[ticker] = sample_return * 1.15
        elif ticker in us_value:
            views[ticker] = sample_return * 0.95
        elif ticker in international:
            views[ticker] = sample_return * 1.05
        elif ticker in bonds:
            views[ticker] = 0.03
        elif ticker in commodities:
            views[ticker] = 0.05
        else:
            views[ticker] = sample_return * 1.0

    return views


# ============================================================================
# PERFORMANCE METRICS
# ============================================================================


def portfolio_performance(weights, mu, cov, rf=0.0):
    """Calculate portfolio return, volatility, and Sharpe ratio"""
    w = weights.values
    ret = float(mu.values @ w)
    vol = float(np.sqrt(w @ cov.values @ w))
    sharpe = (ret - rf) / vol if vol > 1e-10 else 0.0
    return ret, vol, sharpe


def calculate_risk_contributions(weights, cov):
    """
    Calculate marginal and percentage risk contributions for each asset
    Returns:
        - marginal_risk: Marginal Contribution to Risk (MCR) for each asset
        - risk_contrib: Contribution to Risk (CR) for each asset
        - pct_contrib: Percentage contribution to total risk
    """
    w = weights.values
    cov_mat = cov.values

    portfolio_var = w @ cov_mat @ w
    portfolio_vol = np.sqrt(portfolio_var)

    if portfolio_vol < 1e-10:
        return (
            pd.Series(0, index=weights.index),
            pd.Series(0, index=weights.index),
            pd.Series(0, index=weights.index),
        )

    # Marginal Contribution to Risk (MCR): ∂σ/∂w_i = (Σw)_i / σ
    marginal_risk = (cov_mat @ w) / portfolio_vol

    # Contribution to Risk (CR): w_i * MCR_i
    risk_contrib = w * marginal_risk

    # Percentage contribution to total risk
    pct_contrib = (risk_contrib / portfolio_vol) * 100

    return (
        pd.Series(marginal_risk, index=weights.index),
        pd.Series(risk_contrib, index=weights.index),
        pd.Series(pct_contrib, index=weights.index),
    )


def calculate_var_cvar(returns, weights, confidence_level=0.95, method="historical"):
    """
    Calculate Value at Risk (VaR) and Conditional Value at Risk (CVaR)

    Args:
        returns: DataFrame of asset returns
        weights: Series of portfolio weights
        confidence_level: Confidence level (default 95%)
        method: 'historical' or 'parametric'

    Returns:
        var: Value at Risk (positive number representing loss)
        cvar: Conditional Value at Risk (Expected Shortfall)
    """
    # Calculate portfolio returns
    portfolio_returns = (returns @ weights.values).values

    if method == "historical":
        # Historical VaR: empirical quantile
        var = -np.percentile(portfolio_returns, (1 - confidence_level) * 100)

        # CVaR: mean of returns below VaR threshold
        cvar = -portfolio_returns[portfolio_returns <= -var].mean()

    else:  # parametric (assumes normal distribution)
        mean_return = portfolio_returns.mean()
        std_return = portfolio_returns.std()

        # Parametric VaR using normal distribution
        from scipy import stats

        z_score = stats.norm.ppf(confidence_level)
        var = -(mean_return - z_score * std_return)

        # Parametric CVaR
        cvar = -(
            mean_return - std_return * stats.norm.pdf(z_score) / (1 - confidence_level)
        )

    # Annualize (daily to annual)
    var_annual = var * np.sqrt(252)
    cvar_annual = cvar * np.sqrt(252)

    return var_annual, cvar_annual


def compute_efficient_frontier_extended(
    mu, cov, min_weight=0.0, max_weight=1.0, n_points=200
):
    """
    Compute EXTENDED efficient frontier covering MUCH wider range
    This creates a more complete frontier line
    """
    n = len(mu)

    # Find minimum variance portfolio
    min_var_weights = optimize_minimum_variance(cov, min_weight, max_weight)
    if min_var_weights is None:
        return np.array([]), np.array([])

    min_ret, min_vol, _ = portfolio_performance(min_var_weights, mu, cov)

    # Maximum feasible return - EXTENDED
    max_single_ret = mu.max()
    max_feasible_ret = max_single_ret * max_weight + mu.sort_values(
        ascending=False
    ).iloc[1:].mean() * (1 - max_weight)

    # EXTENDED RANGE: Cover from 80% of min to 150% of max
    target_returns = np.linspace(min_ret * 0.8, max_feasible_ret * 1.5, n_points)

    frontier_vols = []
    frontier_rets = []

    for target_ret in target_returns:
        w = cp.Variable(n)

        risk = cp.quad_form(w, cov.values)
        objective = cp.Minimize(risk)

        constraints = [
            cp.sum(w) == 1,
            mu.values @ w >= target_ret,
            w >= min_weight,
            w <= max_weight,
        ]

        problem = cp.Problem(objective, constraints)

        try:
            problem.solve(solver=cp.OSQP, verbose=False, max_iter=15000)
            if problem.status not in ["optimal", "optimal_inaccurate"]:
                problem.solve(solver=cp.SCS, verbose=False, max_iters=15000)
        except:
            continue

        if w.value is not None and problem.status in ["optimal", "optimal_inaccurate"]:
            weights_opt = np.array(w.value).flatten()
            weights_opt = np.maximum(weights_opt, 0)
            if weights_opt.sum() > 0:
                weights_opt = weights_opt / weights_opt.sum()

                port_ret = float(mu.values @ weights_opt)
                port_vol = float(np.sqrt(weights_opt @ cov.values @ weights_opt))

                frontier_rets.append(port_ret)
                frontier_vols.append(port_vol)

    return np.array(frontier_vols), np.array(frontier_rets)


def generate_diverse_random_portfolios(mu, cov, n_portfolios=15000):
    """
    Generate random portfolios with WIDER distribution
    Uses different strategies to cover more of the risk-return space
    """
    n_assets = len(mu)
    random_rets = []
    random_vols = []

    # Strategy 1: Pure random Dirichlet (30%)
    for _ in range(int(n_portfolios * 0.3)):
        weights = np.random.dirichlet(np.ones(n_assets))
        port_ret = mu.values @ weights
        port_vol = np.sqrt(weights @ cov.values @ weights)
        random_rets.append(port_ret)
        random_vols.append(port_vol)

    # Strategy 2: Concentrated portfolios (30%) - higher variance
    for _ in range(int(n_portfolios * 0.3)):
        # Favor 1-3 assets heavily
        n_concentrated = np.random.randint(1, min(4, n_assets + 1))
        weights = np.zeros(n_assets)
        selected = np.random.choice(n_assets, n_concentrated, replace=False)
        random_weights = np.random.random(n_concentrated)
        weights[selected] = random_weights / random_weights.sum()

        port_ret = mu.values @ weights
        port_vol = np.sqrt(weights @ cov.values @ weights)
        random_rets.append(port_ret)
        random_vols.append(port_vol)

    # Strategy 3: High variance portfolios (20%)
    for _ in range(int(n_portfolios * 0.2)):
        # Exponential distribution - more extreme weights
        weights = np.random.exponential(scale=2.0, size=n_assets)
        weights = weights / weights.sum()

        port_ret = mu.values @ weights
        port_vol = np.sqrt(weights @ cov.values @ weights)
        random_rets.append(port_ret)
        random_vols.append(port_vol)

    # Strategy 4: Low variance portfolios (20%)
    for _ in range(int(n_portfolios * 0.2)):
        # More balanced - Dirichlet with high concentration
        weights = np.random.dirichlet(np.ones(n_assets) * 10)
        port_ret = mu.values @ weights
        port_vol = np.sqrt(weights @ cov.values @ weights)
        random_rets.append(port_ret)
        random_vols.append(port_vol)

    return np.array(random_vols), np.array(random_rets)


# ============================================================================
# BACKTESTING
# ============================================================================


def rolling_backtest(
    returns, cov_method, models_config, window_years=3, rebalance_months=3
):
    """Rolling window out-of-sample backtest"""
    dates = returns.index
    start_date = dates[0]
    end_date = dates[-1]

    oos_start = start_date + pd.DateOffset(years=window_years)

    if oos_start >= end_date:
        return pd.DataFrame(), {}

    rebal_dates = []
    current_date = oos_start
    while current_date <= end_date:
        valid_dates = dates[dates >= current_date]
        if len(valid_dates) == 0:
            break
        rebal_dates.append(valid_dates[0])
        current_date = current_date + pd.DateOffset(months=rebalance_months)

    model_names = [
        name for name, config in models_config.items() if config.get("enabled", True)
    ]
    backtest_returns = pd.DataFrame(index=dates, columns=model_names, dtype=float)

    for i, rebal_date in enumerate(rebal_dates):
        train_start = rebal_date - pd.DateOffset(years=window_years)
        train_data = returns.loc[(dates >= train_start) & (dates < rebal_date)]

        if len(train_data) < 60:
            continue

        mu_sample = train_data.mean() * 252
        cov_est = estimate_covariance(train_data, method=cov_method)

        mu_bl = None
        bl_config = models_config.get("BL Mean-Variance", {})
        if bl_config.get("enabled", False):
            mu_bl = build_bl_views(
                train_data,
                cov_est,
                bl_config["rf"],
                bl_config["tau"],
                bl_config["views"],
                bl_config["uncertainty"],
            )

        if i < len(rebal_dates) - 1:
            hold_end = rebal_dates[i + 1] - pd.Timedelta(days=1)
        else:
            hold_end = end_date

        hold_data = returns.loc[(dates >= rebal_date) & (dates <= hold_end)]

        if hold_data.empty:
            continue

        portfolios = {}

        if models_config.get("Min Variance", {}).get("enabled", False):
            cfg = models_config["Min Variance"]
            portfolios["Min Variance"] = optimize_minimum_variance(
                cov_est, cfg["min_weight"], cfg["max_weight"]
            )

        if models_config.get("Mean-Variance", {}).get("enabled", False):
            cfg = models_config["Mean-Variance"]
            portfolios["Mean-Variance"] = optimize_mean_variance(
                mu_sample,
                cov_est,
                cfg["risk_aversion"],
                cfg["min_weight"],
                cfg["max_weight"],
            )

        if models_config.get("Max Sharpe", {}).get("enabled", False):
            cfg = models_config["Max Sharpe"]
            mu_for_sharpe = mu_bl if mu_bl is not None else mu_sample
            portfolios["Max Sharpe"] = optimize_max_sharpe_scipy(
                mu_for_sharpe, cov_est, cfg["rf"], cfg["min_weight"], cfg["max_weight"]
            )

        if models_config.get("Risk Parity", {}).get("enabled", False):
            portfolios["Risk Parity"] = optimize_risk_parity(cov_est)

        if (
            models_config.get("BL Mean-Variance", {}).get("enabled", False)
            and mu_bl is not None
        ):
            cfg = models_config["BL Mean-Variance"]
            portfolios["BL Mean-Variance"] = optimize_mean_variance(
                mu_bl,
                cov_est,
                cfg["risk_aversion"],
                cfg["min_weight"],
                cfg["max_weight"],
            )

        for name, weights in portfolios.items():
            if weights is not None:
                backtest_returns.loc[hold_data.index, name] = (
                    hold_data @ weights.values
                ).values

    # Calculate stats
    stats = {}
    for col in backtest_returns.columns:
        series = backtest_returns[col].dropna()
        if not series.empty:
            ann_ret = (1 + series.mean()) ** 252 - 1
            ann_vol = series.std() * np.sqrt(252)
            sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
            cum = (1 + series).cumprod()
            max_dd = (cum / cum.cummax() - 1).min()

            stats[col] = {
                "Return (%)": ann_ret * 100,
                "Volatility (%)": ann_vol * 100,
                "Sharpe": sharpe,
                "Max DD (%)": max_dd * 100,
            }

    return backtest_returns, stats


# ============================================================================
# VISUALIZATION
# ============================================================================


def plot_efficient_frontier_complete(
    frontier_vols, frontier_rets, portfolios, mu, cov, rf
):
    """
    Plot efficient frontier with BLUE color, EXTENDED range,
    and WIDESPREAD Monte Carlo background
    """
    fig = go.Figure()

    # Generate DIVERSE random portfolios - WIDER distribution
    random_vols, random_rets = generate_diverse_random_portfolios(
        mu, cov, n_portfolios=15000
    )

    # Plot Monte Carlo with NO HOVER (static background) - MORE SPREAD OUT
    fig.add_trace(
        go.Scatter(
            x=random_vols * 100,
            y=random_rets * 100,
            mode="markers",
            name="Monte Carlo Portfolios",
            marker=dict(size=2, color="lightgray", opacity=0.25),
            hoverinfo="skip",  # No hover - static background
            showlegend=True,
        )
    )

    # Efficient Frontier - BLUE COLOR, EXTENDED
    if len(frontier_vols) > 0:
        fig.add_trace(
            go.Scatter(
                x=frontier_vols * 100,
                y=frontier_rets * 100,
                mode="lines",
                name="Efficient Frontier",
                line=dict(width=5, color="blue"),  # BLUE instead of black
                hovertemplate="<b>Efficient Frontier</b><br>Risk: %{x:.2f}%<br>Return: %{y:.2f}%<extra></extra>",
            )
        )

    # Capital Market Line (CML)
    max_sharpe_portfolio = portfolios.get("Max Sharpe")
    if max_sharpe_portfolio is not None and rf is not None:
        ret_sharpe, vol_sharpe, sharpe_ratio = portfolio_performance(
            max_sharpe_portfolio, mu, cov, rf
        )

        # Extend CML line further
        max_vol_plot = (
            max(
                frontier_vols.max() if len(frontier_vols) > 0 else vol_sharpe,
                random_vols.max() if len(random_vols) > 0 else vol_sharpe,
                vol_sharpe,
            )
            * 1.2
        )

        cml_vols = np.linspace(0, max_vol_plot, 100)
        cml_rets = rf + sharpe_ratio * cml_vols

        fig.add_trace(
            go.Scatter(
                x=cml_vols * 100,
                y=cml_rets * 100,
                mode="lines",
                name=f"CML (Sharpe={sharpe_ratio:.2f})",
                line=dict(dash="dash", width=3, color="red"),
                hovertemplate="<b>Capital Market Line</b><br>Risk: %{x:.2f}%<br>Return: %{y:.2f}%<extra></extra>",
            )
        )

        # Risk-free rate point
        fig.add_trace(
            go.Scatter(
                x=[0],
                y=[rf * 100],
                mode="markers+text",
                name="Risk-Free Rate",
                text=["Rf"],
                textposition="top center",
                marker=dict(size=12, color="green", symbol="star"),
                hovertemplate="<b>Risk-Free Rate</b><br>Return: %{y:.2f}%<extra></extra>",
            )
        )

        # Tangency (Max Sharpe) point
        fig.add_trace(
            go.Scatter(
                x=[vol_sharpe * 100],
                y=[ret_sharpe * 100],
                mode="markers+text",
                name="Max Sharpe Ratio (CML Tangency)",
                text=["Tangency"],
                textposition="top center",
                marker=dict(size=14, color="red", symbol="diamond"),
                hovertemplate="<b>Max Sharpe (Tangency)</b><br>Risk: %{x:.2f}%<br>Return: %{y:.2f}%<br>Sharpe: "
                + f"{sharpe_ratio:.2f}<extra></extra>",
            )
        )

    # Plot other portfolios
    colors = {
        "Min Variance": "darkgreen",
        "Mean-Variance": "darkblue",
        "Risk Parity": "purple",
        "BL Mean-Variance": "cyan",
    }

    for name, weights in portfolios.items():
        if weights is None or name == "Max Sharpe":
            continue

        ret, vol, sharpe = portfolio_performance(weights, mu, cov, rf if rf else 0)

        fig.add_trace(
            go.Scatter(
                x=[vol * 100],
                y=[ret * 100],
                mode="markers+text",
                name=name,
                text=[name.split()[0]],
                textposition="top center",
                marker=dict(size=11, color=colors.get(name, "orange")),
                hovertemplate=f"<b>{name}</b><br>Risk: %{{x:.2f}}%<br>Return: %{{y:.2f}}%<br>Sharpe: {sharpe:.2f}<extra></extra>",
            )
        )

    fig.update_layout(
        title="Portfolio Optimization: Efficient Frontier & Allocation Methods",
        xaxis_title="Risk (Standard Deviation %)",
        yaxis_title="Expected Return (%)",
        template="plotly_white",
        height=650,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
        ),
    )

    return fig


def plot_asset_movements(prices):
    """Plot normalized asset price movements"""
    normalized = prices / prices.iloc[0] * 100

    fig = go.Figure()

    for col in normalized.columns:
        fig.add_trace(
            go.Scatter(
                x=normalized.index,
                y=normalized[col],
                mode="lines",
                name=col,
                line=dict(width=2),
            )
        )

    fig.update_layout(
        title="Asset Price Movements (Normalized to 100)",
        xaxis_title="Date",
        yaxis_title="Normalized Price",
        template="plotly_white",
        height=500,
        hovermode="x unified",
    )

    return fig


def plot_correlation_heatmap(returns):
    """Plot correlation heatmap"""
    corr = returns.corr()

    fig = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.columns,
            colorscale="RdBu",
            zmid=0,
            text=corr.values,
            texttemplate="%{text:.2f}",
            textfont={"size": 10},
            colorbar=dict(title="Correlation"),
        )
    )

    fig.update_layout(
        title="Asset Correlation Matrix",
        template="plotly_white",
        height=500,
        xaxis={"side": "bottom"},
    )

    return fig


def plot_returns_distribution(returns):
    """Plot distribution of returns"""
    fig = go.Figure()

    for col in returns.columns:
        fig.add_trace(
            go.Histogram(x=returns[col] * 100, name=col, opacity=0.7, nbinsx=50)
        )

    fig.update_layout(
        title="Distribution of Daily Returns",
        xaxis_title="Daily Return (%)",
        yaxis_title="Frequency",
        template="plotly_white",
        height=400,
        barmode="overlay",
    )

    return fig


def plot_weights_allocations(portfolios_dict):
    """Plot portfolio allocations as stacked bar chart"""
    if not portfolios_dict:
        return go.Figure()

    data = []
    for name, w in portfolios_dict.items():
        if w is None:
            continue
        data.append({"Portfolio": name, **{k: v * 100 for k, v in w.items()}})

    if not data:
        return go.Figure()

    df = pd.DataFrame(data)
    assets = [col for col in df.columns if col != "Portfolio"]

    fig = go.Figure()
    for asset in assets:
        fig.add_trace(
            go.Bar(
                name=asset,
                y=df["Portfolio"],
                x=df[asset],
                orientation="h",
                text=df[asset].apply(lambda x: f"{x:.1f}%" if x > 2 else ""),
                textposition="inside",
            )
        )

    fig.update_layout(
        barmode="stack",
        title="Portfolio Allocations (%)",
        xaxis_title="Weight (%)",
        yaxis_title="Portfolio",
        template="plotly_white",
        height=400,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
    )

    return fig


def plot_risk_contributions(portfolios_dict, cov):
    """Plot risk contributions for each portfolio"""
    if not portfolios_dict:
        return go.Figure()

    data = []
    for name, w in portfolios_dict.items():
        if w is None:
            continue
        _, _, pct_contrib = calculate_risk_contributions(w, cov)
        data.append({"Portfolio": name, **{k: v for k, v in pct_contrib.items()}})

    if not data:
        return go.Figure()

    df = pd.DataFrame(data)
    assets = [col for col in df.columns if col != "Portfolio"]

    fig = go.Figure()
    for asset in assets:
        fig.add_trace(
            go.Bar(
                name=asset,
                y=df["Portfolio"],
                x=df[asset],
                orientation="h",
                text=df[asset].apply(lambda x: f"{x:.1f}%" if abs(x) > 2 else ""),
                textposition="inside",
            )
        )

    fig.update_layout(
        barmode="stack",
        title="Risk Contributions by Asset (%)",
        xaxis_title="Risk Contribution (%)",
        yaxis_title="Portfolio",
        template="plotly_white",
        height=400,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
    )

    return fig


# ============================================================================
# STREAMLIT APP
# ============================================================================

st.set_page_config(page_title="Portfolio Optimizer", layout="wide", page_icon="📊")

# Initialize session state - FIX: Add run_analysis flag
if "backtest_results" not in st.session_state:
    st.session_state.backtest_results = None
if "backtest_stats" not in st.session_state:
    st.session_state.backtest_stats = None
if "run_analysis" not in st.session_state:
    st.session_state.run_analysis = False
if "optimization_done" not in st.session_state:
    st.session_state.optimization_done = False

st.title("📊 Portfolio Optimization: Efficient Frontier & Allocation Methods")

st.markdown(
    """
**Professional Portfolio Optimization Tool** featuring:
- **Diversified Multi-Asset Portfolio** (Stocks, Bonds, International, Commodities)
- Mean-Variance Optimization (Markowitz)
- Maximum Sharpe Ratio (Tangency Portfolio with CML)
- Minimum Variance Portfolio
- Risk Parity
- Black-Litterman Model with Smart Default Views
- Out-of-Sample Rolling Backtest
"""
)

# Sidebar Configuration
st.sidebar.header("⚙️ Configuration")

with st.sidebar.expander("📈 Market Data", expanded=True):
    default_tickers = "AAPL,MSFT,JPM,JNJ,GLD,TLT,EEM,DBC"

    tickers_input = st.text_input(
        "Tickers (comma-separated)",
        default_tickers,
        help="""
        Diversified Portfolio:
        - AAPL, MSFT: US Tech (Growth)
        - JPM: Financials
        - JNJ: Healthcare (Defensive)
        - GLD: Gold (Safe Haven)
        - TLT: Long-term Bonds
        - EEM: Emerging Markets
        - DBC: Commodities
        """,
    )
    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", dt.date(2010, 1, 1))
    with col2:
        end_date = st.date_input("End Date", dt.date(2020, 1, 1))

st.sidebar.markdown("### 🎯 Portfolio Models")

enable_minvar = st.sidebar.checkbox("Minimum Variance", value=True)
enable_meanvar = st.sidebar.checkbox("Mean-Variance", value=True)
enable_sharpe = st.sidebar.checkbox("Max Sharpe", value=True)
enable_riskparity = st.sidebar.checkbox("Risk Parity", value=True)
enable_bl = st.sidebar.checkbox("Black-Litterman", value=True)

with st.sidebar.expander("🔧 Optimization Parameters"):
    global_min_weight = st.slider("Min Weight (%)", 0, 20, 0) / 100
    global_max_weight = st.slider("Max Weight (%)", 10, 100, 40) / 100
    risk_aversion = st.slider("Risk Aversion λ", 0.5, 10.0, 2.5, 0.5)
    risk_free_rate = st.number_input("Risk-Free Rate (%)", value=2.75, step=0.25) / 100

bl_views = {}
if enable_bl:
    with st.sidebar.expander("🔮 Black-Litterman Views"):
        use_smart_views = st.checkbox(
            "Use Smart Default Views",
            value=True,
            help="Automatically generate views based on asset class",
        )
        tau = st.slider("Tau (uncertainty)", 0.01, 0.10, 0.025, 0.005)
        view_uncertainty = st.slider("View Uncertainty (%)", 1.0, 20.0, 8.0) / 100

        if not use_smart_views:
            st.markdown("**Custom Expected Returns (% per year):**")
            for ticker in tickers[:6]:
                view_val = st.number_input(
                    f"{ticker}", value=0.0, step=1.0, key=f"view_{ticker}"
                )
                if view_val != 0:
                    bl_views[ticker] = view_val / 100

# FIX: Use button callback to set state
if st.sidebar.button("🚀 Run Analysis", type="primary"):
    st.session_state.run_analysis = True
    st.session_state.optimization_done = False
    st.session_state.backtest_results = None
    st.session_state.backtest_stats = None

if st.session_state.run_analysis:
    with st.spinner("Loading data..."):
        try:
            prices = load_price_data(tickers, start_date, end_date)

            if prices.empty:
                st.error("No data downloaded. Check tickers and dates.")
                st.stop()

            returns, mu_sample, cov_sample = compute_return_stats(prices)
            cov = estimate_covariance(returns, method="ledoit")

            st.success(f"✅ Loaded {len(tickers)} assets, {len(prices)} days of data")

        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            st.stop()

    # Build models configuration
    models_config = {}

    if enable_minvar:
        models_config["Min Variance"] = {
            "enabled": True,
            "min_weight": global_min_weight,
            "max_weight": global_max_weight,
        }

    if enable_meanvar:
        models_config["Mean-Variance"] = {
            "enabled": True,
            "risk_aversion": risk_aversion,
            "min_weight": global_min_weight,
            "max_weight": global_max_weight,
        }

    if enable_sharpe:
        models_config["Max Sharpe"] = {
            "enabled": True,
            "rf": risk_free_rate,
            "min_weight": global_min_weight,
            "max_weight": global_max_weight,
        }

    if enable_riskparity:
        models_config["Risk Parity"] = {"enabled": True}

    # Black-Litterman with smart views
    mu_bl = None
    if enable_bl:
        if use_smart_views or not bl_views:
            bl_views = get_smart_bl_views(tickers, mu_sample)

        mu_bl = build_bl_views(
            returns, cov, risk_free_rate, tau, bl_views, view_uncertainty
        )
        models_config["BL Mean-Variance"] = {
            "enabled": True,
            "risk_aversion": risk_aversion,
            "min_weight": global_min_weight,
            "max_weight": global_max_weight,
            "rf": risk_free_rate,
            "tau": tau,
            "views": bl_views,
            "uncertainty": view_uncertainty,
        }

    # Optimize portfolios
    if not st.session_state.optimization_done:
        with st.spinner("Optimizing portfolios..."):
            portfolios = {}

            if enable_minvar:
                portfolios["Min Variance"] = optimize_minimum_variance(
                    cov, global_min_weight, global_max_weight
                )

            if enable_meanvar:
                portfolios["Mean-Variance"] = optimize_mean_variance(
                    mu_sample, cov, risk_aversion, global_min_weight, global_max_weight
                )

            if enable_sharpe:
                mu_for_sharpe = mu_bl if mu_bl is not None else mu_sample
                portfolios["Max Sharpe"] = optimize_max_sharpe_scipy(
                    mu_for_sharpe,
                    cov,
                    risk_free_rate,
                    global_min_weight,
                    global_max_weight,
                )

            if enable_riskparity:
                portfolios["Risk Parity"] = optimize_risk_parity(cov)

            if enable_bl and mu_bl is not None:
                portfolios["BL Mean-Variance"] = optimize_mean_variance(
                    mu_bl, cov, risk_aversion, global_min_weight, global_max_weight
                )

            # Store in session state
            st.session_state.portfolios = portfolios
            st.session_state.mu_for_frontier = mu_bl if mu_bl is not None else mu_sample
            st.session_state.cov = cov
            st.session_state.models_config = models_config
            st.session_state.optimization_done = True

    # Load from session state
    portfolios = st.session_state.portfolios
    mu_for_frontier = st.session_state.mu_for_frontier
    cov = st.session_state.cov
    models_config = st.session_state.models_config

    # Compute EXTENDED efficient frontier
    with st.spinner("Computing extended efficient frontier..."):
        frontier_vols, frontier_rets = compute_efficient_frontier_extended(
            mu_for_frontier, cov, global_min_weight, global_max_weight, n_points=200
        )

    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(
        ["📊 Optimization", "📈 Data Analysis", "💼 Allocations", "🔄 Backtest"]
    )

    with tab1:
        st.header("Efficient Frontier Analysis")

        fig = plot_efficient_frontier_complete(
            frontier_vols,
            frontier_rets,
            portfolios,
            mu_for_frontier,
            cov,
            risk_free_rate,
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Portfolio Statistics")
        stats = []
        for name, weights in portfolios.items():
            if weights is not None:
                ret, vol, sharpe = portfolio_performance(
                    weights, mu_for_frontier, cov, risk_free_rate
                )
                var, cvar = calculate_var_cvar(
                    returns, weights, confidence_level=0.95, method="historical"
                )
                stats.append(
                    {
                        "Portfolio": name,
                        "Return (%)": f"{ret*100:.2f}",
                        "Volatility (%)": f"{vol*100:.2f}",
                        "Sharpe Ratio": f"{sharpe:.2f}",
                        "VaR 95% (%)": f"{var*100:.2f}",
                        "CVaR 95% (%)": f"{cvar*100:.2f}",
                    }
                )

        st.dataframe(pd.DataFrame(stats), hide_index=True, use_container_width=True)

    with tab2:
        st.header("Data Analysis & Asset Behavior")

        st.subheader("📈 Asset Price Movements")
        fig_movements = plot_asset_movements(prices)
        st.plotly_chart(fig_movements, use_container_width=True)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📊 Correlation Matrix")
            fig_corr = plot_correlation_heatmap(returns)
            st.plotly_chart(fig_corr, use_container_width=True)

        with col2:
            st.subheader("📉 Returns Distribution")
            fig_dist = plot_returns_distribution(returns)
            st.plotly_chart(fig_dist, use_container_width=True)

        st.subheader("📋 Asset Statistics")
        stats_df = pd.DataFrame(
            {
                "Asset": mu_sample.index,
                "Ann. Return (%)": (mu_sample.values * 100).round(2),
                "Ann. Volatility (%)": (np.sqrt(np.diag(cov.values)) * 100).round(2),
                "Sharpe Ratio": (
                    (mu_sample.values - risk_free_rate) / np.sqrt(np.diag(cov.values))
                ).round(2),
            }
        )
        st.dataframe(stats_df, hide_index=True, use_container_width=True)

    with tab3:
        st.header("Portfolio Allocations & Risk Analysis")

        # Portfolio Allocations
        st.subheader("💼 Weight Allocations")
        fig_alloc_bar = plot_weights_allocations(portfolios)
        st.plotly_chart(fig_alloc_bar, use_container_width=True)

        allocation_data = {}
        for name, weights in portfolios.items():
            if weights is not None:
                allocation_data[name] = weights.values * 100

        # Risk Contributions
        st.subheader("⚠️ Risk Contributions by Asset")
        st.markdown(
            """
        Risk contribution shows how much each asset contributes to the total portfolio risk.
        Unlike weight allocation, this reveals which assets are driving portfolio volatility.
        """
        )

        fig_risk_contrib = plot_risk_contributions(portfolios, cov)
        st.plotly_chart(fig_risk_contrib, use_container_width=True)

        # Detailed risk contribution table
        risk_contrib_data = {}
        for name, weights in portfolios.items():
            if weights is not None:
                _, _, pct_contrib = calculate_risk_contributions(weights, cov)
                risk_contrib_data[name] = pct_contrib.values

        if risk_contrib_data:
            df_risk_contrib = pd.DataFrame(
                risk_contrib_data, index=list(portfolios.values())[0].index
            )
            st.subheader("📊 Detailed Risk Contributions (%)")
            st.dataframe(
                df_risk_contrib.style.format("{:.2f}%"), use_container_width=True
            )

        # VaR and CVaR Analysis
        st.subheader("📉 Value at Risk (VaR) & Conditional VaR (CVaR)")
        st.markdown(
            """
        - **VaR (95%)**: Maximum expected loss in 95% of scenarios (annualized)
        - **CVaR (95%)**: Expected loss when VaR threshold is exceeded (tail risk)
        """
        )

        col1, col2 = st.columns(2)
        with col1:
            confidence_level = st.slider("Confidence Level", 0.90, 0.99, 0.95, 0.01)
        with col2:
            var_method = st.selectbox("Method", ["historical", "parametric"])

        var_stats = []
        for name, weights in portfolios.items():
            if weights is not None:
                var, cvar = calculate_var_cvar(
                    returns, weights, confidence_level, var_method
                )
                ret, vol, sharpe = portfolio_performance(
                    weights, mu_for_frontier, cov, risk_free_rate
                )
                var_stats.append(
                    {
                        "Portfolio": name,
                        "Return (%)": f"{ret*100:.2f}",
                        "Volatility (%)": f"{vol*100:.2f}",
                        f"VaR {int(confidence_level*100)}% (%)": f"{var*100:.2f}",
                        f"CVaR {int(confidence_level*100)}% (%)": f"{cvar*100:.2f}",
                        "CVaR/VaR Ratio": f"{(cvar/var) if var > 0 else 0:.2f}",
                    }
                )

        if var_stats:
            df_var = pd.DataFrame(var_stats)
            st.dataframe(df_var, hide_index=True, use_container_width=True)

            st.info(
                f"""
            **Interpretation:**
            - With {int(confidence_level*100)}% confidence, losses will not exceed VaR
            - CVaR shows the average loss in the worst {int((1-confidence_level)*100)}% of cases
            - Higher CVaR/VaR ratio indicates fatter tails (more extreme risk)
            """
            )

    with tab4:
        st.header("Out-of-Sample Backtest")

        col1, col2 = st.columns(2)
        with col1:
            window_years = st.slider(
                "Calibration Window (years)", 1, 5, 3, key="bt_window"
            )
        with col2:
            rebal_months = st.slider(
                "Rebalancing Frequency (months)", 1, 12, 3, key="bt_rebal"
            )

        # FIX: Use a unique key for the button and check state properly
        run_backtest = st.button("Run Backtest", type="primary", key="run_backtest_btn")

        if run_backtest:
            with st.spinner("Running backtest... This may take a minute."):
                bt_returns, bt_stats = rolling_backtest(
                    returns, "ledoit", models_config, window_years, rebal_months
                )

                # Store in session state
                st.session_state.backtest_results = bt_returns
                st.session_state.backtest_stats = bt_stats

        # Display results if they exist in session state
        if (
            st.session_state.backtest_results is not None
            and not st.session_state.backtest_results.empty
        ):
            bt_returns = st.session_state.backtest_results
            bt_stats = st.session_state.backtest_stats

            if bt_returns.notna().any().any():
                cum_returns = (1 + bt_returns).cumprod()

                fig_bt = go.Figure()
                for col in cum_returns.columns:
                    if cum_returns[col].notna().any():
                        fig_bt.add_trace(
                            go.Scatter(
                                x=cum_returns.index,
                                y=cum_returns[col],
                                mode="lines",
                                name=col,
                                line=dict(width=2),
                            )
                        )

                fig_bt.update_layout(
                    title="Cumulative Returns (Out-of-Sample)",
                    xaxis_title="Date",
                    yaxis_title="Cumulative Return",
                    height=500,
                    template="plotly_white",
                    hovermode="x unified",
                )

                st.plotly_chart(fig_bt, use_container_width=True)

                if bt_stats:
                    st.subheader("Backtest Performance Statistics")
                    stats_rows = []
                    for model, metrics in bt_stats.items():
                        stats_rows.append(
                            {
                                "Portfolio": model,
                                "Return (%)": f"{metrics['Return (%)']:.2f}",
                                "Volatility (%)": f"{metrics['Volatility (%)']:.2f}",
                                "Sharpe": f"{metrics['Sharpe']:.2f}",
                                "Max DD (%)": f"{metrics['Max DD (%)']:.2f}",
                            }
                        )

                    st.dataframe(
                        pd.DataFrame(stats_rows),
                        hide_index=True,
                        use_container_width=True,
                    )
            else:
                st.warning(
                    "⚠️ Not enough data for backtest. Try reducing the calibration window or using a longer data period."
                )
        else:
            st.info("👆 Click 'Run Backtest' to see out-of-sample performance")

else:
    st.info("👈 Configure parameters and click **Run Analysis**")

    # st.markdown("""
    # ### 📚 Quick Guide

    # **Default Portfolio:**
    # - **AAPL, MSFT**: US Tech (Growth & Innovation)
    # - **JPM**: Financials (Economic Sensitivity)
    # - **JNJ**: Healthcare (Defensive)
    # - **GLD**: Gold (Safe Haven, Inflation Hedge)
    # - **TLT**: Long-term Treasury Bonds (Safety, Negative Correlation)
    # - **EEM**: Emerging Markets (Diversification)
    # - **DBC**: Commodities Index (Real Asset Exposure)

    # **Features:**
    # - ✅ **Extended Efficient Frontier (BLUE)** - Complete risk/return spectrum
    # - ✅ **Wide Monte Carlo Distribution** - Covers full feasible region
    # - ✅ **Capital Market Line (CML)** - Leverage/Lending line
    # - ✅ **Smart BL Views** - Asset-class based expectations
    # - ✅ **Static Monte Carlo** - Clean background visualization
    # - ✅ **Comprehensive Analysis** - Price movements, correlations, distributions
    # - ✅ **Persistent Backtest** - Results preserved during navigation
    # """)
