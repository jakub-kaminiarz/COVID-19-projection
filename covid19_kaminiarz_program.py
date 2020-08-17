"""covid19_kaminiarz_program.py

Dependiencies
"""
import pandas as pd
import geopandas
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from datetime import timedelta
from sklearn import metrics

sns.set_style('whitegrid')

# Linear Regression
from sklearn.linear_model import LinearRegression
import matplotlib.dates as mdates
import warnings

warnings.filterwarnings('ignore')

# MLP Regression
from sklearn.neural_network import MLPRegressor
import matplotlib.dates as mdates
import warnings

warnings.filterwarnings('ignore')

# Neural Networks
import math
from keras.models import Sequential
from keras.layers import Dense
from statistics import mean

df = pd.read_csv('covid_19_activity.csv')
df.head(10)

df.tail(10)

df.info()

df.describe()

"""Dropping unnecessery column and cleaning datas"""

# comment 'df.drop(...) line after first program run' 
df.drop(['county_name', 'county_fips_number', 'province_state_name', 'data_source_name', 'country_alpha_2_code'],
        axis=1, inplace=True)
df.rename(columns={'country_short_name': 'country',
                   'continent_name': 'continent',
                   'report_date': 'date',
                   'people_positive_new_cases_count': 'positive_per_day',
                   'people_positive_cases_count': 'positive_total',
                   'people_death_new_count': 'death_per_day',
                   'people_death_count': 'death_total',
                   'country_alpha_3_code': 'iso_a3'}, inplace=True)
df.date = pd.to_datetime(df.date)
df.sort_values(['continent', 'country', 'date'], axis=0, inplace=True)
df.head(10)

df.tail(10)

"""EDA"""

# Number of countries in dataset
no_country = df.country.unique().size
print(f'Number of countries in dataset: {no_country}')

# Range of reported dates
min_data = df.date.min()
max_data = df.date.max()
print(f'Data was reported from {min_data} to {max_data}')

# # below lines were created because United States apeares in dataframe multiple times (different cities and states)
temp_df = df.groupby(['country', 'continent'], as_index=False).agg({"positive_per_day": "sum"})
temp_death_df = df.groupby(['country', 'continent'], as_index=False).agg({"death_per_day": "sum"})
temp_df.sort_values(['continent', 'country'], axis=0, inplace=True)
temp_death_df.sort_values(['continent', 'country'], axis=0, inplace=True)
temp_df.rename(columns={'positive_per_day': 'positive_total'}, inplace=True)
temp_death_df.rename(columns={'death_per_day': 'death_total'}, inplace=True)
print(temp_df.head())

# average positive cases in each continents
continents = temp_df.continent.unique()
for continent in continents:
    temp2_df = temp_df.loc[temp_df.continent == continent]
    print('-' * 80)
    print(f'average number of positive cases in {continent}:  {temp2_df.positive_total.mean():.2f}')

temp3_df = df.groupby(['country', 'continent'], as_index=False).agg({"death_per_day": "sum"})
temp3_df.sort_values(['continent', 'country'], axis=0, inplace=True)
temp3_df.rename(columns={'death_per_day': 'death_total'}, inplace=True)

# average death count in each continents:
for continent in continents:
    temp2_df = temp3_df.loc[temp3_df.continent == continent]
    print('-' * 800)
    print(f'average number of fatalities in {continent}:  {temp2_df.death_total.mean():.2f}')

# max values of positive cases in each continents:
for continent in continents:
    temp2_df = temp_df.loc[temp_df.continent == continent]
    print('-' * 800)
    print(
        f'Maximum number of positive cases in {continent}:  {temp2_df.positive_total.max()}   {temp2_df["country"].loc[temp2_df.positive_total == temp2_df.positive_total.max()].unique()}')

# max values of fatalities in each continents:
for continent in continents:
    temp2_df = temp3_df.loc[temp3_df.continent == continent]
    print('-' * 800)
    print(
        f'Maximum number of fatalities in {continent}:  {temp2_df.death_total.max()}   {temp2_df["country"].loc[temp2_df.death_total == temp2_df.death_total.max()].unique()}')

"""Data visualisation on map - world"""

# construction of object 'world' and Antarctica removal
world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
world = world[world.name != "Antarctica"]
# changing the column name from COUNTRY_ALPHA_3_CODE to iso_3 for dataframe joining
vis_temp_df = df[['iso_a3', 'positive_per_day']]
# grouping by 3 letter codes so i.e. America instead of a few thousand lines has just one with the sum of cases
vis_temp_df = vis_temp_df.groupby('iso_a3', as_index=False).agg({"positive_per_day": "sum"})
vis_world_df = pd.merge(vis_temp_df, world, on='iso_a3', how='inner', sort=True)
fig, ax = plt.subplots(1, 1, figsize=(15, 6))
gdf = geopandas.GeoDataFrame(vis_world_df)
gdf.plot(column='positive_per_day', edgecolor=u'gray', cmap='OrRd', ax=ax, legend=True,
         legend_kwds={'label': "PEOPLE POSITIVE CASES",
                      'orientation': "horizontal"})
plt.title('PEOPLE POSITIVE CASES ACROSS THE WORLD ( before 2020-07-07 )', fontsize=20)
plt.show()

"""Data visualisation on map - europe"""

# temp dataframe
vis_eu_temp_df = df[['iso_a3', 'positive_per_day']].loc[df.continent == 'Europe']
vis_eu_temp_df = vis_eu_temp_df.groupby('iso_a3', as_index=False).agg({"positive_per_day": "sum"})
vis_europe_df = pd.merge(vis_eu_temp_df, world, on='iso_a3', how='inner', sort=True)
fig, ax = plt.subplots(1, 1, figsize=(15, 6))
gdf = geopandas.GeoDataFrame(vis_europe_df)
gdf.plot(column='positive_per_day', edgecolor=u'gray', cmap='OrRd', ax=ax, legend=True,
         legend_kwds={'label': "PEOPLE POSITIVE CASES",
                      'orientation': "horizontal"})
plt.title('PEOPLE POSITIVE CASES IN EUROPE ( before 2020-07-07 )', fontsize=20)
plt.show()

# Histogram of positive cases across the world
plt.figure(figsize=(15, 6))
hist = sns.distplot(temp_df['positive_total'], bins=50)
plt.title('Distribiution of positive cases across the world', fontsize=20)
plt.show()

# Histogram of fatalities per day across the world
plt.figure(figsize=(15, 6))
hist = sns.distplot(temp_death_df['death_total'], bins=50)
plt.title('Distribiution of fatalities across the world', fontsize=20)
plt.show()

# histogram of positive cases in Europe
plt.figure(figsize=(15, 6))
hist = sns.distplot(temp_df['positive_total'].loc[temp_df['continent'] == 'Europe'], bins=50)
plt.title('Distribiution of positive cases in Europe', fontsize=20)
plt.show()

# histogram of fatalities in Europe
plt.figure(figsize=(15, 6))
hist = sns.distplot(temp_death_df['death_total'].loc[temp_death_df['continent'] == 'Europe'], bins=50)
plt.title('Distribiution of fatalities in Europe', fontsize=20)
plt.show()

# regression of new positive cases per day across the world
temp4_df = df.groupby('date', as_index=False).agg({"positive_per_day": "sum"})
temp4_df.date = pd.to_datetime(temp4_df.date)
plt.figure(figsize=(15, 6))
regplot = sns.regplot(x=np.arange(len(temp4_df['positive_per_day'])), y=temp4_df['positive_per_day'])
ticklabels = [item.strftime('%b %d') for item in temp4_df.date[::int(len(temp4_df.date) / 8)]]
regplot.xaxis.set_major_formatter(ticker.FixedFormatter(ticklabels))
plt.gcf().autofmt_xdate()
plt.title('Distribiution of new positive cases per day across the world')
regplot.set_xticklabels(ticklabels)
plt.show()

# regression of new positive cases per day in Europe
temp5_df = df[df.continent == 'Europe'].groupby('date', as_index=False).agg({"positive_per_day": "sum"})
temp5_df.date = pd.to_datetime(temp5_df.date)
plt.figure(figsize=(15, 6))
regplot = sns.regplot(x=np.arange(len(temp5_df)), y=temp5_df['positive_per_day'])
ticklabels = [item.strftime('%b %d') for item in temp5_df.date[::int(len(temp5_df.date) / 8)]]
regplot.xaxis.set_major_formatter(ticker.FixedFormatter(ticklabels))
plt.gcf().autofmt_xdate()
plt.title('Distribiution of new positive cases per day in Europe')
regplot.set_xticklabels(ticklabels)
plt.show()

"""Poland"""


# Regression of data in Poland

def regression_Poland(date, column):
    date = pd.to_datetime(date)
    df.date = pd.to_datetime(df.date)
    df.sort_values(by='date', inplace=True)
    plt.figure(figsize=(15, 6))
    plt.plot(df[(df.country == 'Poland') & (df.date < date)].date,
             df[(df.country == 'Poland') & (df.date < date)][column], label='before')
    plt.plot(df[(df.country == 'Poland') & (df.date >= date)].date,
             df[(df.country == 'Poland') & (df.date >= date)][column], label='after')
    plt.legend()
    plt.xlabel('Day', fontsize=15)
    plt.ylabel('Cases', fontsize=15)
    plt.title(f'Regression of {column} cases in Poland before and after {date}', fontsize=18)
    plt.show()


regression_Poland('2020-07-01', 'positive_per_day')

regression_Poland('2020-07-01', 'death_per_day')

regression_Poland('2020-07-01', 'positive_total')

regression_Poland('2020-07-01', 'death_total')

"""Linear Regression and MLP projection"""

df_poland = df[df.country == 'Poland']

annotate_day = '2020-07-01'
text_location = '2020-07-03'
fig = plt.figure(figsize=(15, 6))
ax = fig.add_subplot(1, 1, 1)
myFmt = mdates.DateFormatter('%d-%m')
ax.xaxis.set_major_formatter(myFmt)
plt.plot(df_poland.date, df_poland.positive_total)
plt.title('Total number of positive cases in Poland', fontsize=20)
plt.xlabel('Day', fontsize=15)
plt.ylabel('Cases', fontsize=15)
plt.grid(True)
plt.show()


# Positive cases prediction
# For better results, data from 2020-04-01 was considered as training data
def projection(regressor, start_date='2020-04-01', end_date='2020-07-01', start_day_index=90, prediction_term=7,
               column='positive_total'):
    X = np.array(df_poland[(df_poland.date > start_date) & (df_poland.date < end_date)].date)
    y = np.array(df_poland[column][(df_poland.date > start_date) & (df_poland.date < end_date)])
    day_numbers = []
    for i in range(1, len(X) + 1):
        day_numbers.append([i])
    X = day_numbers
    if regressor.casefold() == 'linear regression':
        model = LinearRegression()
    elif regressor.casefold() == 'mlp':
        model = MLPRegressor(solver='lbfgs', learning_rate='adaptive')
    model.fit(X, y)
    # row with index 74 is the equivalent of 2020-06-14 from wich we launch prediction process
    start_predict_date = start_day_index
    prediction_term = prediction_term

    X_test = [[i] for i in range(start_predict_date, start_predict_date + prediction_term - 1)]
    y_pred = model.predict(X_test)
    y_true = np.array(df_poland[column][(df_poland.date >= end_date)])
    plt.figure(figsize=(15, 6))
    plt.text(len(X + X_test) + 3 - 2, y_pred[-2], f'Prediction: {int(y_pred[-2])}', fontsize=15)
    plt.plot(X, df_poland[column][(df_poland.date > start_date) & (df_poland.date < end_date)], label='Train')
    plt.plot(X_test, y_pred, label='Predict')
    temp_list = X + X_test
    x_axis_labels = ['2020-04-01', '2020-05-01', '2020-06-01', '2020-07-01']
    plt.xticks(temp_list[::30], x_axis_labels)
    plt.title(f'Projection of {column} cases in Poland after 1st July and prediction for 7th July ({regressor})',
              fontsize=18)
    plt.xlabel('Day', fontsize=15)
    plt.ylabel(f'{column}', fontsize=15)
    plt.legend()
    plt.grid(True)
    plt.show()
    print(f'RMSE of {regressor} model for {column}:  {np.sqrt(metrics.mean_squared_error(y_true, y_pred)):.2f}')


projection(regressor='linear regression', column='positive_total')

projection(regressor='linear regression', column='death_total')

projection(regressor='linear regression', column='positive_per_day')

projection(regressor='linear regression', column='death_per_day')

projection(regressor='MLP', column='positive_total')

projection(regressor='MLP', column='death_total')

projection(regressor='MLP', column='positive_per_day')

projection(regressor='MLP', column='death_per_day')


"""Deep Learning with Multilayer Perceptron model"""
def add_dimmension(dataframe, window_width=1):
    dataX, dataY = [], []
    for i in range(len(dataframe) - window_width):
        a = dataframe[i:(i + window_width), 0]
        dataX.append(a)
        dataY.append(dataframe[i + window_width, 0])
    return np.array(dataX), np.array(dataY)


def nn_model(column, window_width=10, optimizer='adam', activation='relu', epochs=1000, batch_size=30):
    # testing smaller dataset from '2020-04-01' (uncoment if necessary)
    # dataset = df_poland[column][df_poland.date >= '2020-04-01']
    dataset = df_poland[column]
    dataset = dataset.values.reshape(len(dataset), 1)
    dataset = dataset.astype('float32')
    # split into train and test sets 70% train 30% test
    train_size = int(len(dataset) * 0.7)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]
    # reshape dataset window 10 samples width
    trainX, trainY = add_dimmension(train, window_width)
    testX, testY = add_dimmension(test, window_width)
    # create and fit Multilayer Perceptron model
    model = Sequential()
    model.add(Dense(16, input_dim=window_width, activation=activation))
    model.add(Dense(32, activation=activation))
    model.add(Dense(8, activation=activation))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer=optimizer)

    model.fit(trainX, trainY, epochs=epochs, batch_size=batch_size, verbose=0)
    trainScore = model.evaluate(trainX, trainY, verbose=0)
    print('Train Score: %.2f MSE (%.2f RMSE)' % (trainScore, math.sqrt(trainScore)))
    testScore = model.evaluate(testX, testY, verbose=0)
    print('Test Score: %.2f MSE (%.2f RMSE)' % (testScore, math.sqrt(testScore)))
    # generate predictions for training
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)
    # shift train predictions for plotting
    trainPredictPlot = np.empty_like(dataset)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[window_width:len(trainPredict) + window_width, :] = trainPredict
    # shift test predictions for plotting
    testPredictPlot = np.empty_like(dataset)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[-len(testPredict):, :] = testPredict
    # future predictions
    prediction_range = 7
    future_df = testX[-2:]
    for i in range(prediction_range):
        recent_pred = model.predict(future_df[-1].reshape(1, window_width))
        future_df = np.vstack([future_df, [future_df[-1][-9], future_df[-1][-8], future_df[-1][-7], future_df[-1][-6],
                                           future_df[-1][-5], future_df[-1][-4], future_df[-1][-3], future_df[-1][-2],
                                           future_df[-1][-1], float(recent_pred)]])
    future_dataset = np.vstack([dataset, future_df[-prediction_range:, -1].reshape(prediction_range, 1)])
    futurePredictPlot = np.empty_like(future_dataset)
    futurePredictPlot[:, :] = np.nan
    futurePredictPlot[-prediction_range:, :] = future_df[-prediction_range:, -1].reshape(prediction_range, 1)
    return column, dataset, trainPredictPlot, testPredictPlot, futurePredictPlot, future_dataset, prediction_range


def plot_prediction(column, dataset, trainPredictPlot, testPredictPlot, futurePredictPlot, future_dataset,
                    prediction_range):
    # plot baseline and predictions
    fig = plt.figure(figsize=(15, 6))
    plt.text(len(dataset) + prediction_range + 3, futurePredictPlot[-1], f'Prediction: {int(futurePredictPlot[-1])}',
             fontsize=15)
    date_range = pd.date_range(start="2019-12-30", end='2020-06-27').strftime('%Y-%m-%d')
    plt.plot(dataset, label='dataset')
    plt.plot(trainPredictPlot, label='train predict')
    plt.plot(testPredictPlot, label='test predict')
    plt.plot(futurePredictPlot, label='future predict')
    plt.xticks(range(len(future_dataset))[::30], date_range[::30])
    plt.title(f'Projection of {column} cases in Poland after 1st July and prediction for 7th July (Nueral Network)',
              fontsize=18)
    plt.xlabel('Day', fontsize=15)
    plt.ylabel(f'{column}', fontsize=15)
    fig.autofmt_xdate()
    plt.legend(loc='best')
    plt.show()


column, dataset, trainPredictPlot, testPredictPlot, futurePredictPlot, future_dataset, prediction_range = nn_model(
    'positive_total', batch_size=30, epochs=1000)
plot_prediction(column, dataset, trainPredictPlot, testPredictPlot, futurePredictPlot, future_dataset, prediction_range)

column, dataset, trainPredictPlot, testPredictPlot, futurePredictPlot, future_dataset, prediction_range = nn_model(
    'positive_per_day', batch_size=30, epochs=10)
plot_prediction(column, dataset, trainPredictPlot, testPredictPlot, futurePredictPlot, future_dataset, prediction_range)

column, dataset, trainPredictPlot, testPredictPlot, futurePredictPlot, future_dataset, prediction_range = nn_model(
    'death_total', epochs=500, batch_size=15)
plot_prediction(column, dataset, trainPredictPlot, testPredictPlot, futurePredictPlot, future_dataset, prediction_range)

column, dataset, trainPredictPlot, testPredictPlot, futurePredictPlot, future_dataset, prediction_range = nn_model(
    'death_per_day', epochs=5, batch_size=10)
plot_prediction(column, dataset, trainPredictPlot, testPredictPlot, futurePredictPlot, future_dataset, prediction_range)
