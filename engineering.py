import pandas as pd
import numpy as np
import datetime
import math
import yfinance as yf

def get_ticker_data(ticker, start, end):
    stock = yf.Ticker(ticker)
    historical_data = stock.history(start=start, end=end)
    return historical_data

## --------------------------------------------------------------------------------Price-Based Features--------------------------------------------------------------------------------------------------------------- 

### Log-Returns
def log_returns(data, window):
    data[f"Log Returns_t-{window}"] = np.log(data["Close"]) - np.log(data["Close"].shift(window))

    for i in range(window):
        if i == 0:
            data[f"Log Returns_t-{window}"][i] = np.log(data["Close"][i]) - np.log(data["Open"][i])
        else:
            data[f"Log Returns_t-{window}"][i] = np.log(data["Close"][i]) - np.log(data["Close"][0])
    return data

### Returns
def rate_of_change(data, window):
    data[f"ROC_t-{window}"] = (data["Close"] - data["Close"].shift(window)) / data["Close"].shift(window)

    for i in range(window):
        if i == 0:
            data[f"ROC_t-{window}"][i] = (data["Close"][i] - data["Open"][i]) / data["Open"][i]
        else:
            data[f"ROC_t-{window}"][i] = (data["Close"][i] - data["Close"][0]) / data["Close"][0]
    return data

### SMA & EMA
def moving_averages(data, window):
    price_changes = data[f"ROC_t-1"]

    ema = np.zeros(len(data))
    sma = np.zeros(len(data))

    ema_sd = np.zeros(len(data))
    sma_sd = np.zeros(len(data))

    # for the first window observation the exponential moving average 
    # is eqaul to the simple moving average up to the ith observation
    for i in range(window):
        ema[i] = price_changes[:i+1].mean()
        sma[i] = price_changes[:i+1].mean()
        if i == 0:
            ema_sd[i] = 0
            sma_sd[i] = 0
        else:
            ema_sd[i] = price_changes[:i+1].std()
            sma_sd[i] =  price_changes[:i+1].std()
   
    alpha = 2 / (window + 1)
    for i in range(window, len(data)):
        ema[i] = alpha * price_changes[i] + (1 - alpha) * ema[i - 1]
        sma[i] = price_changes[i - window:i].mean()

        ema_sd[i] = alpha * (price_changes[i] - ema[i])**2 + (1 - alpha) * ema_sd[i - 1]
        sma_sd[i] = price_changes[i - window:i].std()
    data[f"EMA_{window}"] = ema
    data[f"SMA_{window}"] = sma
    data[f"EMSD_{window}"] = ema_sd
    data[f"SMSD_{window}"] = sma_sd
    return data

### Price change relative to price change of S&P 500
def chnage_to_benchmark(data, window, benchmark:str):
    data_start_date = datetime.datetime(2000, 1, 1)
    data_end_date = datetime.datetime(2025, 2, 28)
    benchmark_data = pd.read_csv("data\^GSPC.csv")
    benchmark_price_chnages = benchmark_data["Close"] - benchmark_data["Close"].shift(window)
    price_changes = data["Close"] - data["Close"].shift(window)

    for i in range(window):
        if i == 0:
            price_changes[i] = data["Close"][i] - data["Open"][i]
            benchmark_price_chnages = benchmark_data["Close"][i] - benchmark_data["Open"][i]
        else:
            price_changes[i] = data["Close"][i] - data["Close"][0]
            benchmark_price_chnages = benchmark_data["Close"][i] - benchmark_data["Close"][0]

    relative_chnage = price_changes / benchmark_price_chnages
    data[f"Price Relative to {benchmark}_{window}"] = relative_chnage
    return data



###---------------------------------------------------------------------------------Volatility & Risk-Based Features----------------------------------------------------------------------------------- 

## Bollinger Bands
def bollinger_bands(data,  window, k=2):
    upper_band = data[f"SMA_{window}"]  + k * data[f"SMSD_{window}"]
    lower_band = data[f"SMA_{window}"]  - k * data[f"SMSD_{window}"]

    data[f"Bollinger Bands Width_{window}"] = ((upper_band - lower_band) / data[f"SMA_{window}"]) * 100
    return data


## Average True Range ATR
def average_true_range(data, window):
    true_range = np.zeros(len(data))
    for i in range(len(data)):
        if i == 0:
            high_low = data['High'][i] - data['Low'][i]
            high_pr_close = abs(data['High'][i] - data['Open'][i])
            low_pr_close = abs(data['Low'][i] - data['Open'][i])
            true_range[i] = max([high_low, high_pr_close, low_pr_close])
        else:
            high_low = data['High'][i] - data['Low'][i]
            high_pr_close = abs(data['High'][i] - data['Close'][i - 1])
            low_pr_close = abs(data['Low'][i] - data['Close'][i - 1])
            true_range[i] = max([high_low, high_pr_close, low_pr_close])
    if 'TR' not in list(data.columns):
        data['TR'] = true_range
    data[f'ATR_{window}'] = data['TR'].rolling(window=window).mean()
    for i in range(window):
        data[f'ATR_{window}'][i] = data['TR'][:i+1].mean()

    data[f'Normalized_ATR_{window}'] = (data[f'ATR_{window}'] - data[f'ATR_{window}'].shift(1)) / data[f'ATR_{window}'].shift(1)
    data[f'Normalized_ATR_{window}'][0] = 0
    return data


###---------------------------------------------------------Liquidity & Volume-Based Features-------------------------------------------------------------------------------------------------

## On.Balance-Volume OBV
def on_balance_volume(data):
    on_volume_balance = np.zeros(len(data))
    for i in range(len(data)):
        if i == 0:
            if data["Close"][i] > data["Open"][i]:
                on_volume_balance[i] = data["Volume"][i]
            elif data["Close"][i] < data["Open"][i]:
                on_volume_balance[i] = -data["Volume"][i]
            else:
                on_volume_balance[i] = 0
        else:
            if data["Close"][i] > data["Close"][i - 1]:
                on_volume_balance[i] = on_volume_balance[i - 1] + data["Volume"][i]
            elif data["Close"][i] < data["Close"][i - 1]:
                on_volume_balance[i] = on_volume_balance[i - 1] - data["Volume"][i]
            else:
                on_volume_balance[i] = on_volume_balance[i - 1]

    data["On Balance Volume"] = on_volume_balance
    return data


##  Volume Change Weighted Avg Return
def VWAP(data, window):
    price_changes = data[f"ROC_t-1"]
    price_volume = np.zeros(len(data))
    price_volume = price_changes * data["Volume"]

    vwap = np.zeros(len(data))
    for i in range(len(data)):
        if i < window:
            vwap[i] = np.sum(price_volume[: i+1]) / np.sum(data["Volume"][: i+1])
        else:
            vwap[i] = np.sum(price_volume[i - window: i+1]) / np.sum(data["Volume"][i - window: i+1])

    data[f"VWAP_{window}"] = vwap
    return data


## Relative Volume
def relative_volume(data, window):
    data[f"Relative Volume_{window}"] = data["Volume"] /  data["Volume"].rolling(window=window).mean()
    for i in range(window):
        data[f"Relative Volume_{window}"][i] = data["Volume"][i] /  data["Volume"][: i+1].mean()
    return data

###--------------------------------------------------------------------Trend & Momentum Indicators-------------------------------------------------------------------------------------------


## Signal and MACD
def compute_macd(data, short_window=6, long_window=13, signal_window=5):
    # Compute short and long EMAs
    short_ema = data[f"ROC_t-1"].ewm(span=short_window, adjust=False).mean()
    long_ema = data[f"ROC_t-1"].ewm(span=long_window, adjust=False).mean()

    # Compute MACD Line
    data[f'MACD_{short_window}-{long_window}'] = short_ema - long_ema

    # Compute Signal Line (9-day EMA of MACD)
    data[f'Signal_Line_{short_window}-{long_window}-{signal_window}'] = data[f'MACD_{short_window}-{long_window}'].ewm(span=signal_window, adjust=False).mean()
    return data


## RSI
def rsi(data, window):
    gain = data[f"ROC_t-1"].apply(lambda x: x if x > 0 else 0)
    loss = data[f"ROC_t-1"].apply(lambda x: -x if x < 0 else 0)

    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()

    rs = avg_gain / avg_loss

    data[f"RSI_{window}"] = 100 - (100 / (1 + rs))
    return data


## Stochastic Oscillator and the Smoothed Stochastic Oscillator
def stochastic_oscillator(data, window):
    low_min = data["Low"].rolling(window=window, min_periods=1).min()
    high_max = data["High"].rolling(window=window, min_periods=1).max()

    data[f"Stochastic Oscillator_{window}"] = ((data["Close"] - low_min) / (high_max - low_min)) * 100
    data[f"Smoothed Stochastic Oscillator_{window}"] = data[f"Stochastic Oscillator_{window}"].rolling(window=3, min_periods=1).mean()
    return data


## Average Directional Index 
def directional_index(data, window):
    true_range = np.maximum(data["High"] - data["Low"], 
                            np.maximum(abs(data["High"] - data["Close"].shift(1)),
                                    abs(data["Low"] - data["Close"].shift(1))))

    plus_dm = np.where((data["High"] - data["High"].shift(1)) > (data["Low"].shift(1) - data["Low"]), data["High"] - data["High"].shift(1), 0)
    minus_dm = np.where((data["Low"].shift(1) - data["Low"]) > (data["High"] - data["High"].shift(1)), data["Low"].shift(1) - data["Low"], 0)

    plus_dm = pd.Series(plus_dm, index=data.index)
    minus_dm = pd.Series(minus_dm, index=data.index)
    true_range = pd.Series(true_range, index=data.index)

    dx_vector = pd.Series(0, index=data.index)
    for i in range(window):
        if i == 0:
            dx_vector[i] = 0
        else:
            plus_di = 100 * (plus_dm[:i+1].mean() / true_range[:i+1].mean())
            minus_di = 100 * (minus_dm[:i+1].mean() / true_range[:i+1].mean())

            dx_vector[i] = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di))
        
    plus_di = 100 * (plus_dm.rolling(window=window).mean() / true_range.rolling(window=window).mean())
    minus_di = 100 * (minus_dm.rolling(window=window).mean() / true_range.rolling(window=window).mean())

    dx_vector[window:] = (100 * (abs(plus_di - minus_di) / (plus_di + minus_di))).iloc[window:]

    data[f"DX_{window}"] = dx_vector
    data[f"ADX_{window}"] = dx_vector.ewm(span=window, adjust=False).mean()
    return data


##-----------------------------------------------------------------------------Date Indicator-------------------------------------------------------------------------------------------------

# Day of the week
def day_of_week(data):
    data["Date"] = data["Date"].str[:10]
    data.index = np.arange(len(data))
    data["Date"] = pd.to_datetime(data["Date"])
    
    # add the day of the week
    data["Weekday"] = data["Date"].dt.day_name()
    return data

##-------------------------------------------------------------------Movement and future returns-----------------------------------------------------------------------------------------------

## Stock Movement
def stock_movement(data, window):
    weekly_direction = pd.Series(0, index=np.arange(len(data)))
    for i in range(len(data)):
        if i + window >= len(data):
            weekly_direction[i] = np.sign(data['Close'][len(data) - 1] - data['Close'][i], )
        else:
            weekly_direction[i] = np.sign(data['Close'][i + window] - data['Close'][i])

    weekly_direction[weekly_direction == 0] = 1
    weekly_direction[weekly_direction == -1] = 0 
    return weekly_direction


## Stock Returns
def movement_returns(df, window):
    returns = pd.Series(0, index=np.arange(len(df)))

    for i in range(len(df)):
        if len(df) - i <= window:
            returns[i] = (df["Close"][len(df)-1] - df["Close"][i]) / df["Close"][i]
        else:
            returns[i] = (df["Close"][i+window] - df["Close"][i]) / df["Close"][i]

    return returns



##--------------------------------------------------------------------Lagged Features----------------------------------------------------------------------------------------------------------------------------------------

def lagged_feature(data, feature, period):
    data[f"Lagged_{feature}_t-{period}"] = data[f"{feature}"].shift(period)
    data[f"Lagged_{feature}_t-{period}"][0] = 0
    for i in range(period):
        if i == 0:
            data[f"Lagged_{feature}_t-{period}"][0] = 0
        else:
            data[f"Lagged_{feature}_t-{period}"][i] = (data[f"{feature}"].shift(i))[i]
    
    return data




if __name__ == "__main__":
    data_start_date = datetime.datetime(2000, 1, 1)
    data_end_date = datetime.datetime(2025, 2, 28)
    data = get_ticker_data("AAPL", data_start_date, data_end_date)
    data = rate_of_change(data, 1)
    print(data)
    data = lagged_feature(data, "ROC_t-1", 3)
    print(data)

