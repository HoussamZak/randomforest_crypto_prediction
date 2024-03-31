import python_ta as ta

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

    df_copy = df.copy()
    df_copy = (kaufman_ad_movavg(df_copy, col, n))
    df_copy = (kaufman_ad_movavg(df_copy, col, m))

    df_copy["kama_diff"] = df_copy[f"kama_{m}"] - df[f"kama_{n}"]
    df_copy["mkt_trend"] = -1
    df_copy.loc[0 < df['kama_diff'], "mkt_trend"] = 1
    return df_copy


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

    #-- add the moving volatility to original dataframe
    df_copy['rolling_vola_yng_zng'] = roll_vola
    return df_copy