#importing libraries
import math
import numpy as np
from scipy.optimize import minimize
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import warnings
warnings.filterwarnings('ignore')

#importing data
confirmed_csv = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_US.csv')
death_csv = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_US.csv')

confirmed_table = confirmed_csv.melt(id_vars=["UID", "iso2", "iso3", "code3", "FIPS", "Admin2", "Province_State", "Country_Region","Lat","Long_","Combined_Key"], var_name="Date", value_name="Confirmed").fillna('').drop(["UID", "iso2", "iso3", "code3", "FIPS","Country_Region","Lat","Long_"], axis=1)
death_table = death_csv.melt(id_vars=["UID", "iso2", "iso3", "code3", "FIPS", "Admin2", "Province_State", "Country_Region","Lat","Long_","Combined_Key"], var_name="Date", value_name="Deaths").fillna('').drop(["UID", "iso2", "iso3", "code3", "FIPS","Country_Region","Lat","Long_"], axis=1)

full_table = confirmed_table.merge(death_table)

full_table['Date'] = pd.to_datetime(full_table['Date'])

#summing up the counties in each state
def get_state(state):
    if full_table[full_table['Province_State'] == state]['Admin2'].nunique() > 1:
        state_table = full_table[full_table['Province_State'] == state]
        summed = pd.pivot_table(state_table, values = ['Confirmed', 'Deaths'],index='Date', aggfunc=sum)
        state_df = pd.DataFrame(summed.to_records())
        return state_df.set_index('Date')[['Confirmed', 'Deaths']]
    df = full_table[(full_table['Province_State'] == state) 
                & (full_table['Admin2'].isin(['', state]))]
    return df.set_index('Date')

def model(N, a, alpha, t):
    return N * (1 - math.e ** (-a * t)) ** alpha

def model_loss(params):
    N, a, alpha = params
    r = 0
    for t in range(len(df)):
        r += (model(N, a, alpha, t) - df.iloc[t, model_index]) ** 2
    return math.sqrt(r)

state = 'New Jersey'
df = get_state(state)
print(df.tail(10))

#confirmed cases
model_index = 0

opt_confirmed = minimize(model_loss, x0=np.array([200000, 0.05, 15]), method='Nelder-Mead', tol=1e-5).x
print("Confirmed Parameters:")
print(opt_confirmed)

#deaths
model_index = 1
opt_deaths = minimize(model_loss, x0=np.array([200000, 0.05, 15]), method='Nelder-Mead', tol=1e-5).x
print("Death Parameters: ")
print(opt_deaths)

model_x = []
for t in range(len(df)):
    model_x.append([df.index[t], model(*opt_confirmed, t), model(*opt_deaths, t)])
model_sim = pd.DataFrame(model_x, dtype=int)
model_sim.set_index(0, inplace=True)
model_sim.columns = ['Model-Confirmed', 'Model-Deaths']

plot_color = ['#0000FF55', '#FF000055', '#0000FF99', '#FF000099']

pd.concat([model_sim, df], axis=1).plot(color = plot_color)

start_date = df.index[0]
n_days = len(df) + 30
extended_model_x = []

for t in range(n_days):
    extended_model_x.append([start_date + datetime.timedelta(days=t), model(*opt_confirmed, t), model(*opt_deaths, t)])

extended_model_sim = pd.DataFrame(extended_model_x, dtype=int)
extended_model_sim.set_index(0, inplace=True)
extended_model_sim.columns = ['Model-Confirmed', 'Model-Deaths']

plot_color = ['#0000FF55', '#FF000055', '#0000FF99', '#FF000099']

pd.concat([extended_model_sim, df], axis=1).plot(color = plot_color)
print('NJ COVID-19 Prediction')
print(extended_model_sim.tail())
plt.show()



#compare model success between states
states = ['Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California', 'Colorado', 'Connecticut', 'Delaware', 'Florida', 'Georgia', 'Hawaii', 'Idaho', 'Illinois', 'Indiana', 'Iowa', 'Kansas', 'Kentucky', 'Louisiana', 'Maine', 'Maryland', 'Massachusetts', 'Michigan', 'Minnesota', 'Mississippi', 'Missouri', 'Montana', 'Nebraska', 'Nevada', 'New Hampshire', 'New Jersey', 'New Mexico', 'New York', 'North Carolina', 'North Dakota', 'Ohio', 'Oklahoma', 'Oregon', 'Pennsylvania', 'Rhode Island', 'South Carolina', 'South Dakota', 'Tennessee', 'Texas', 'Utah', 'Vermont', 'Virginia', 'Washington', 'West Virginia', 'Wisconsin', 'Wyoming']
table = []

for state in states:
    df = get_state(state)
    model_index = 0
    opt_confirmed = minimize(model_loss, x0=np.array([200000, 0.05, 15]), method='Nelder-Mead', tol=1e-5).x
    model_index = 1
    opt_deaths = minimize(model_loss, x0=np.array([200000, 0.05, 15]), method='Nelder-Mead', tol=1e-5).x
    model_state = []
    error_confirmed = 0
    error_deaths = 0
    for t in range(len(df)):
        t_confirmed = model(*opt_confirmed, t)
        t_deaths = model(*opt_deaths, t)
        error_confirmed += (t_confirmed - df.iloc[t,0])**2
        error_deaths += (t_deaths - df.iloc[t,1])**2
    error_confirmed = math.sqrt(error_confirmed/df.iloc[len(df)-1,0])
    error_deaths = math.sqrt(error_deaths/df.iloc[len(df)-1,1])
    print(state)
    print(error_confirmed)
    print(error_deaths)
    table.append([state, error_confirmed, error_deaths])

state_errors = pd.DataFrame(table)
state_errors.set_index(0, inplace=True)
state_errors.columns = ['Confirmed Error', 'Death Error']
state_errors.to_csv('new doc.csv')


