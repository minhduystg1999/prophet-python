import numpy as np
import pandas as pd
from scipy import stats
import plotly.graph_objects as go
from fbprophet import Prophet
from fbprophet.plot import add_changepoints_to_plot
from fbprophet.diagnostics import cross_validation
from fbprophet.diagnostics import performance_metrics
from fbprophet.plot import plot_cross_validation_metric


df = pd.read_csv('example_wp_log_peyton_manning.csv')

print(df)

df.plot()

df.columns = ['ds', 'y']
df['ds'] = pd.to_datetime(df['ds'])

prediction_size = 20
train_df = df[:-prediction_size]

p = Prophet()
p.fit(train_df)

future = p.make_future_dataframe(periods=prediction_size, freq='D')
forecast = p.predict(future)

p.plot(forecast)
p.plot_components(forecast)

def make_comparison_dataframe(initial, forecast):
    return forecast.set_index('ds')[['yhat', 'yhat_lower', 'yhat_upper']].join(initial.set_index('ds'))

cmp_df = make_comparison_dataframe(df, forecast)
#print(cmp_df.tail(20))

def calculate_forecast_errors(df, prediction_size):    
    df = df.copy()
    df['e'] = df['y'] - df['yhat']
    df['p'] = 100 * df['e'] / df['y']
    predicted_part = df[-prediction_size:]
    error_mean = lambda error_name: np.mean(np.abs(predicted_part[error_name]))
    return {'MAPE': error_mean('p'), 'MAE': error_mean('e')}

for err_name, err_value in calculate_forecast_errors(cmp_df, prediction_size).items():
    print(err_name, err_value)

#Real predict
p2 = Prophet()
p2.fit(df)
future2 = p2.make_future_dataframe(periods=365, freq='D')
forecast2 = p2.predict(future2)
print(forecast2[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])
p2.plot(forecast2)
p2.plot_components(forecast2)


