# This trader was created with assistance from freeCodeCamp.org, stackoverflow.com, and chatGPT
# In total though, the code was written by myself: Kyle Shanahan. And meant to be used for educational purposes only.
# Please do not redistribute. Thank you. -Kyle Shanahan

# Descriptionm of the program:
# -pyTrader takes data from the S&P 500 index and uses a machine learning model to predict stock price movement.
# The tool takes and analyzes CURRENT data and the parameters of the model to predict the stock price movement are from 2024 and on.
# In order to alleviate a stock from prior months that may have fallen out of the S & P 500 from ending up in the created portfolio.
# -Calculates monthly returns and volatility, grouping the stocks into similar asset clusters.

# Packages used:
# -pandas -pandas_ta -numpy -matplotlib -pandas_datareader -datetime -sklearn -PyPortfolioOpt
# !!! Ensure you have all packages installed for the program to run correctly !!!


from statsmodels.regression.rolling import RollingOLS
# Import statements
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader as pdr
import pandas_ta as ta
from datetime import datetime, timedelta



# This section is for getting downloading data from Yahoo Finance
#----------------------------------------------------------------------------------------------------------
pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')


# Get list of S&P 500 tickers from Wikipedia
SnP500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
SnP500['Symbol'] = SnP500['Symbol'].str.replace(".", "-")
symbolList = SnP500['Symbol'].unique.tolist()

# Range of data to be used is from (Current Date) and (4 years prior to Current Date)
startTime = (datetime.now() - timedelta(days=365*4)).date()
endTime = dt.datetime.now()
# Get data from Yahoo Finance using ONLY adjusted close price "['Adj Close']"
# .stack() stacks data on top of each other
yFdata = yf.download(tickers =symbolList,start = startTime,end = endTime)['Adj Close']
yFdata.stack()
yFdata.index.names = ['Date', 'Ticker']
yFdata.columns = yFdata.str.Lower()
#----------------------------------------------------------------------------------------------------------

# This section is for calculations
#----------------------------------------------------------------------------------------------------------
# ******** DEVELOPER NOTE: Need to determine what 'np' is and what to change it to *************************
# calculate Garman-Klass volatility
yFdata['Gar_Klass_Volatility'] = ((np.log(yFdata['high'] - np.log(yFdata['low']))**2)/2 \
                                  - (2 * np.log(2)-1)*(np.log(yFdata['Adj Close'])-np.log(yFdata['open']))**2) 
# calculate RSI using pamda_ta
yFdata['tickerGroup'] = yFdata.groupby(level=1)['Adj Close'].transform(lambda x: pandas_ta.rsi(close=x, length=20))
#----------------------------------------------------------------------------------------------------------

# Check to see if data was imported correctly, using Apple and plot it as a graph
yFdata.xs('AAPL', level=1)['rsi'].plot()


# Supplying the log of daily close prices to the bollinger bands function
# ----------- Commented out test line of code --------------------------------
# ta.bbands(close = yFdata.xs('AAPL', level=1)['adj close'], length = 20)


# ** Bollinger Bands are a technical analysis tool developed by John Bollinger. 
# They consist of a middle band being an N-period simple moving average (SMA), an upper band at K times an N-period standard deviation above the middle band,
# and a lower band at K times an N-period standard deviation below the middle band. **
# ** Bollinger Bands are used in finance to measure volatility and identify market trends. 
# It's important to use Bollinger Bands in conjunction with other technical analysis tools and indicators. They are not foolproof, and false signals can occur. **
# Calculate for the LOW band, then MID band, then HIGH band:
yFdata['bb_low'] = yFdata.groupby(level=1)['adj close'].transform(lambda x: pandas_ta.bbands(close=np.log1(x), length = 20).iloc[:,0])
yFdata['bb_mid'] = yFdata.groupby(level=1)['adj close'].transform(lambda x: pandas_ta.bbands(close=np.log1(x), length=20).iloc[:, 1])
yFdata['bb_high'] = yFdata.groupby(level=1)['adj close'].transform(lambda x: pandas_ta.bbands(close=np.log1(x), length=20).iloc[:, 2])

