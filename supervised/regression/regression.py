import quandl
import math
import numpy as np
from sklearn import preprocessing, model_selection, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pylab as plt
from matplotlib import style
import datetime
import pickle

style.use('ggplot')

# Configure api key
quandl.ApiConfig.api_key = "8vrZXAw5S_pd2Szvfjym"

# Get the pandas dataframe
df = quandl.get("WIKI/GOOGL")
# Set df to the columns we want
df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]
# We define our own feature columns in the dataframe based on existing columns
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100

# Set the dataframe to only include the following columns
df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]

# The label column
forecast_col = 'Adj. Close'
# We can't work with NaN data, so replace them
df.fillna('-99999', inplace=True)

# The number of days into the future we want to predict
forecast_out = int(math.ceil(0.1*len(df))) # forecast 10% out of df

# The label is forecast_col shifted up. I.e. each row has the value of forecast_col forecast_out days in the future as its label
df['label'] = df[forecast_col].shift(-forecast_out)

# Define our feature set X and our labels y
X = np.array(df.drop(['label'], 1))
X = preprocessing.scale(X)
X_lately = X[-forecast_out:]
X = X[:-forecast_out]

# Because we shifted the column, the last forecast_out rows have no data in the label column, so we remove those rows from the dataset
df.dropna(inplace=True)
y = np.array(df['label'])

# shuffle our data and split into training and testing sets
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

# Perform a linear regression on the data
# clf = LinearRegression(n_jobs=10) # svm.SVR()
# clf.fit(X_train, y_train)
#
# with open('linearregression.pickle', 'wb') as f:
#     pickle.dump(clf, f)

pickle_in = open('linearregression.pickle', 'rb')
clf = pickle.load(pickle_in)

accuracy = clf.score(X_test, y_test)

forecast_set = clf.predict(X_lately)
print(forecast_set, accuracy, forecast_out)
df['Forecast'] = np.nan

last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns) - 1)] + [i]

df['Adj. Close'].plot()
df['Forecast'].plot()

plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()