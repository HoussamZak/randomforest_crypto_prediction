import ta as ta
import pandas as pd 
import numpy as np 

#-- Kaufmann Adaptive Moving Average function

def kaufman_ad_movavg(df, col, n):
    """
    Function to calculate the Kaufman Adaptive Mov Avg for a selected column
    in a dataframe, results added to a new column kama_{n}

    Parameters: 
    df: pandas.DataFrame
    col: str
    n: int , window period as integer for Kama calculation

    Return:
    df_copy: pandas.DataFrame
    new dataframe with kama_{n} column.
    """
    df_copy = df.copy()
    df_copy[f"kama_{n}"] = ta.momentum.KAMAIndicator(df_copy[col], n).kama()
    return df_copy

#-- Kaufmann Adaptive Moving Average function

def kama_mkt_regime(df, col, n , m):
    """
    Function to calculate the Kaufman Adaptive MovAvg to detect market regime.

    Parameters:
    df: pd.DataFrame with price data and other metrics
    col: str, column name on which to apply kama mkt regime
    n: int, period length for first KAMA calculation
    m: int, period length for second KAMA

    Returns:
    df: pd.DataFrame with 2 added column kama_diff and mkt_trend
    """

    df = (kaufman_ad_movavg(df, col, n))
    df = (kaufman_ad_movavg(df, col, m))

    df["kama_diff"] = df[f"kama_{m}"] - df[f"kama_{n}"]
    df["mkt_trend"] = -1
    df.loc[0 < df["kama_diff"], "mkt_trend"] = 1
    return df


#-- Volatility YangZhang Estimator

def moving_yngzng_estimator(df, window_size = 7):
    """
    Function to calculate yangzhang volatility estimator over a rolling window in 
    a dataframe containing ohlc prices, Yang-Zhang is another volatiity estimation 
    method that aims to improve accuracy by including open and close prices

    Parameters:
    df: pd.DataFrame, dataset containing ohlc and other metrics 

    Returns:
    volatility: float, in new column 
    """
    def yngzng_estimator(df):
        N = len(window)

        term_1 = np.log(window['high'] / window['close']) * np.log(window['high'] / window['open'])
        term_2 = np.log(window['low'] / window['close']) * np.log(window['low'] / window['open'])
        sum = np.sum(term_1 + term_2)
        volatility =  np.sqrt(sum / N)
        return volatility
    
    df_copy = df.copy()
    #-- empty series to store moving volatility
    roll_vola = pd.Series(dtype = 'float64')

    for i in range( window_size, len(df)):
        window = df_copy.loc[df_copy.index[i - window_size]: df_copy.index[i]]
        volatility = yngzng_estimator(window)
        roll_vola.at[df_copy.index[i]] = volatility

    #-- add the moving volatility to original dataframe
    df_copy['rolling_vola_yng_zng'] = roll_vola
    return df_copy


#-- VWAP volume weighted average price 

def vwap(df):
    df_vwap = df.copy()
    df_vwap['volXclose'] = df_vwap['close'] * df_vwap['volume']
    df_vwap['cum_vol'] = df_vwap['volume'].cumsum()
    df_vwap['cum_volXclose'] = (df_vwap['volume'] * df_vwap['high'] + df_vwap['close']/3).cumsum()
    df_vwap['vwap'] = df_vwap['cum_volXclose'] / df_vwap['cum_vol']
    # df_vwap['vwap'] = df_vwap.fillna(0)
    return df_vwap

#-- Hurst Exponent

def hurst_expo_dyn(df):

    power = 10 
    n = 2 ** power

    #-- arrays initialization

    hurst = np.array([])
    tstats = np.array([])
    pvalues = np.array([])

    #-- data 
    datum = df
    prices = np.array(datum[1:])
    returns = np.array(datum)[1:] / np.array(datum)[:-1] - 1
    #-- calculation of rolling Hurst exponent 

    for t in np.arange(n, len(returns) + 1):
        data = returns[t - n: t]
        X = np.arange(2, power + 1)
        Y = np.array([])
        for p in X:
            m = 2 ** p
            s = 2 ** (power - p)
            rs_array = np.array([])

            for i in np.arange(0, s):
                subsample = data[i * m : (i+1) * m]
                mean = np.average(subsample)
                deviate = np.cumsum(subsample - mean)
                difference = max(deviate) - min(deviate)
                stdev = np.std(subsample) 
                rescaled_range = difference/stdev
                rs_array = np.append(rs_array, rescaled_range)
            
            Y = np.append(Y, np.log2(np.average(rs_array)))
        reg = sm.OLS(Y, sm.add_constant(X))
        res = reg.fit()
        hurst = res.params[1]
        tstat = (res.params[1] - 0.5)/res.bse[1]
        pvalue = 2 * (1 - sps.t.cdf(abs(tstat), res.df_resid))
        hursts = np.append(hursts, hurst)
        tstats = np.append(tstats, tstat)
        pvalues = np.append(pvalues, pvalue)
        # return hurst
        datum['hurstxp'] = hurst
    return datum