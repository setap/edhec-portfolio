import pandas as pd
import yfinance as yf
from scipy.stats import norm
import numpy as np
import scipy.stats


def drawdown(return_series: pd.Series):
    wealth_index = 1000 * (1 + return_series).cumprod()
    prevous_peaks = wealth_index.cummax()
    drawdowns = (wealth_index - prevous_peaks) / prevous_peaks
    return pd.DataFrame({
        'Wealth': wealth_index,
        'Previous Peaks': prevous_peaks,
        'Drawdown': drawdowns
    })


def download_yahoo_stocks(stocks: str, atr: str):
    df = yf.download(stocks, period='2y', interval='1d')
    df = df.dropna()
    return df[atr]


def semideviation(r):
    is_negative = r < 0
    return r[is_negative].std(ddof=0)


def var_historic(r, level=5):
    """
    Histoic VaR
    """
    if isinstance(r, pd.DataFrame):
        return r.agg(var_historic, level=level)
    elif isinstance(r, pd.Series):
        return -np.percentile(r, level)
    else:
        raise TypeError('Expected r is Series of DataFrame')


def var_gaussian(r, level=5, modified=False):
    """
    Returns parametric Gaussian VaR of Series or DataFrames
    If "modified" is True, then the modified VaR is returned,
    using the Cornish-Fisher modification
    """
    z = norm.ppf(level/100)

    if modified:
        s = skewness(r)
        k = kurtosis(r)
        z = (z + (z**2 - 1)*s/6 + (z**3 - 3*z)*(k - 3)/24 - (2*z**3 - 5*z)*(s**2)/36)

    return -(r.mean() + z * r.std(ddof=0))


def skewness(r):
    """
    Alternative to scipy.stats.skew()
    Compute of skewness of the supplied Series or DataFrame
    Return float or Series
    """
    demeaned_r = r - r.mean()
    sigma_r = r.std(ddof=0)
    exp = (demeaned_r**3).mean()
    return exp/sigma_r**3


def kurtosis(r):
    """
    Alternative to scipy.stats.kurtosis()
    Compute of skewness of the supplied Series or DataFrame
    Return float or Series
    """
    demeaned_r = r - r.mean()
    sigma_r = r.std(ddof=0)
    exp = (demeaned_r**4).mean()
    return exp/sigma_r**4


def is_normal(r, level=.01):
    """
    Applies the Jarque-Bera test to determine if a Series is normal or not
    Test is applied at the 1% level by default
    Return True if the hypothesis of normality is accepted, False otherwise
    """
    statistic, p_value = scipy.stats.jarque_bera(r)
    return p_value > level


def cvar_historic(r, level=5):
    """
    Compute the Conditional VaR of Series or DataFrame
    """
    if isinstance(r, pd.Series):
        is_beond_var = r <= var_historic(r, level)
        return -r[is_beond_var].mean()
    elif isinstance(r, pd.DataFrame):
        return pd.agg(cvar_historic, level=level)
    else:
        raise TypeError('Expected r to be a Series or DataFrame')


def get_ffme_returns():
    """
    Load the Fama-French Dataset for the returns of the Top and Bottom Deciles by MarketCap
    """
    me_m = pd.read_csv("data/Portfolios_Formed_on_ME_monthly_EW.csv", header=0, index_col=0, na_values=-99.99)
    rets = me_m[['Lo 20', 'Hi 20']]
    rets = rets / 100
    rets.index = pd.to_datetime(rets.index, format='%Y%m').to_period('M')
    return rets


def get_hfi_returns():
    """
    Load the Fama-French Dataset for the returns of the Top and Bottom Deciles by MarketCap
    """
    hfi = pd.read_csv("data/edhec-hedgefundindices.csv", header=0, index_col=0, parse_dates=True)
    hfi = hfi / 100
    hfi.index = hfi.index.to_period('M')
    return hfi


def get_ind_returns():
    """
    Load and format the Ken French 30 Indastry Portfolios Value Weighted Monthly Returns
    """
    ind = pd.read_csv('../data/ind30_m_vw_rets.csv', header=0, parse_dates=True, index_col=0) / 100
    ind.index = pd.to_datetime(ind.index, format='%Y%m').to_period('M')
    ind.columns = ind.columns.str.strip()
    return ind


def get_ind_size():
    """
    """
    ind = pd.read_csv('../data/ind30_m_size.csv', header=0, parse_dates=True, index_col=0)
    ind.index = pd.to_datetime(ind.index, format='%Y%m').to_period('M')
    ind.columns = ind.columns.str.strip()
    return ind


def get_ind_nfirms():
    """
    """
    ind = pd.read_csv('../data/ind30_m_nfirms.csv', header=0, parse_dates=True, index_col=0)
    ind.index = pd.to_datetime(ind.index, format='%Y%m').to_period('M')
    ind.columns = ind.columns.str.strip()
    return ind

def annualize_rets(r, periods_per_year):
    """
    Annualizes a set of returns
    """
    compounded_growth = (1 + r).prod()
    n_periods = r.shape[0]
    return compounded_growth ** (periods_per_year / n_periods) - 1


def annualize_vol(r, periods_per_year):
    """
    Annualizes the vol of a set of returns
    """
    return r.std() * (periods_per_year ** 0.5)


def sharp_ratio(r, riskfree_rate, periods_per_year):
    """
    Computes annualized sharp ratio of a set or returns
    """
    rf_per_period = (1 + riskfree_rate) ** (1 / periods_per_year) - 1
    excess_ret = r - rf_per_period
    ann_ex_ret = annualize_rets(excess_ret, periods_per_year)
    ann_vol = annualize_vol(r, periods_per_year)
    return ann_ex_ret / ann_vol


def portfolio_return(weights, returns):
    """
    Weight to Returns
    """
    return weights.T @ returns


def portfolio_vol(weights, cov_matrix):
    """
    Weights to Volatile
    """
    return (weights.T @ cov_matrix @ weights) ** 0.5


def plot_ef2(n_points, er, cov):
    """
    Plots the 2-asset efficient frontier
    """
    if er.shape[0] != 2 or cov.shape[0] != 2:
        raise ValueError("plot_ef2 can plot only 2-asset frontiers")
    weights = [np.array([w, 1 - w]) for w in np.linspace(0, 1, n_points)]
    rets = [portfolio_return(w, er) for w in weights]
    vols = [portfolio_vol(w, cov) for w in weights]
    ef = pd.DataFrame({"Returns": rets, "Volatility": vols})
    return ef.plot.line(x="Volatility", y="Returns", style=".-")


from scipy.optimize import minimize


def minimize_vol(target_return, er, cov):
    """
    target_return -> W
    """
    n = er.shape[0]
    init_guess = np.repeat(1 / n, n)
    bounds = ((0.0, 1.0),) * n
    return_is_target = {
        "type": "eq",
        "args": (er,),
        "fun": lambda weights, er: target_return - portfolio_return(weights, er)
    }
    weights_sum_to_1 = {
        "type": "eq",
        "fun": lambda weights: np.sum(weights) - 1
    }
    result = minimize(portfolio_vol,
                      init_guess,
                      args=(cov,),
                      method="SLSQP",
                      options={"disp": False},
                      constraints=(return_is_target, weights_sum_to_1),
                      bounds=bounds)
    return result.x


def optimal_weights(n_points, er, cov):
    """
    :return: list of weights to run the optimizer on to minimize the vol
    """
    target_ret = np.linspace(er.min(), er.max(), n_points)
    weights = [minimize_vol(tr, er, cov) for tr in target_ret]
    return weights


def gmv(cov):
    """
    Return the weights for th Global Minimum Vol porfolio,
    given the covariance matrix
    """
    n = cov.shape[0]
    return msr(0, np.repeat(1, n), cov)


def plot_ef(n_points, er, cov, show_cml=False, riskfree_rate=0, style='.-', show_eqw=False, show_gmv=False):
    """
    Plots the N-asset efficient frontier
    """
    weights = optimal_weights(n_points, er, cov)
    rets = [portfolio_return(w, er) for w in weights]
    vols = [portfolio_vol(w, cov) for w in weights]
    ef = pd.DataFrame({"Returns": rets, "Volatility": vols})
    ax = ef.plot.line(x="Volatility", y="Returns", style=style)
    if show_eqw:
        # Show Equally Weighted Portfolio
        n = er.shape[0]
        w_eqw = np.repeat(1 / n, n)
        er_eqw = portfolio_return(w_eqw, er)
        vol_eqw = portfolio_vol(w_eqw, cov)
        ax.plot([vol_eqw], [er_eqw], color="goldenrod", marker='o', markersize=10)
    if show_gmv:
        # Show Global Minimum Variance
        n = er.shape[0]
        w_gmv = gmv(cov)
        er_gmv = portfolio_return(w_gmv, er)
        vol_gmv = portfolio_vol(w_gmv, cov)
        ax.plot([vol_gmv], [er_gmv], color="midnightblue", marker='o', markersize=10)
    if show_cml:
        ax.set_xlim(left=0)
        w_msr = msr(riskfree_rate, er, cov)
        r_msr = portfolio_return(w_msr, er)
        vol_msr = portfolio_vol(w_msr, cov)
        cml_x = [0, vol_msr]
        cml_y = [riskfree_rate, r_msr]
        ax.plot(cml_x, cml_y, color="green", marker='o', linestyle="dashed")
        ax.axvline(x=vol_msr, ls=':', lw=1)
    return ax


def msr(riskfree_return, er, cov):
    """
    Returns the weigths of the portfolio that gives you the maximum sharp ratio
    given the riskfree rate and expected returns and a covariance matrix
    """
    n = er.shape[0]
    init_guess = np.repeat(1 / n, n)
    bounds = ((0.0, 1.0),) * n
    weights_sum_to_1 = {
        "type": "eq",
        "fun": lambda weights: np.sum(weights) - 1
    }

    def negative_sharp_ratio(weights, riskfree_return, er, cov):
        """
        Returns the negative of the sharp ratio for given weights
        """
        r = portfolio_return(weights, er)
        vol = portfolio_vol(weights, cov)
        return -(r - riskfree_return) / vol

    result = minimize(negative_sharp_ratio,
                      init_guess,
                      args=(riskfree_return, er, cov,),
                      method="SLSQP",
                      options={"disp": False},
                      constraints=(weights_sum_to_1),
                      bounds=bounds)
    return result.x
