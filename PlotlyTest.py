# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 09:10:13 2021

@author: Jiwoo Ahn
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 22:09:46 2020

@author: Jiwoo Ahn
"""

# Import relevant libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import chart_studio
import chart_studio.plotly as py
import plotly.express as px
import plotly.graph_objects as go
from plotly.offline import download_plotlyjs, init_notebook_mode,  plot
import plotly.io as pio
from plotly.subplots import make_subplots

# Plotly credentials
username = 'j-ahn'
api_key = 'eoNF8mZbtyci47iTLmaq'
chart_studio.tools.set_credentials_file(username=username,api_key=api_key)

# Pull data from John Hopkins University and organise into dataframe 
df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')

# Curve fitting Global COVID-19 Cases

def logistic(t, a, b, c, d):
    return c + (d - c)/(1 + a * np.exp(- b * t))

def exponential(t, a, b, c):
    return a * np.exp(b * t) + c
    
def doubling(t):
    return (1+np.log(2)/2)**t

def plotCases(dataframe, column, countries, start_date, curvefit, forecast):

    #fig = go.Figure()
    fig = make_subplots(rows=1,cols=2,subplot_titles=('Total Cases','New Cases'))
    fig.update_layout(template='plotly_white')
    fig.update_layout(title='Queensland Covid-19 Dashboard',title_font_size=30,title_x=0.5)     
    fig.update_layout(legend={'title':'Legend','bordercolor':'black','borderwidth':1})
    fig.update_layout(legend_title_font=dict(family="Arial, Tahoma, Helvetica",size=16,color="#404040"))
    fig.update_layout(
        font=dict(
            family="Arial, Tahoma, Helvetica",
            size=14,
            color="#404040"
        ))

    #clean_chart_format(fig)
    
    PSM_ColorMap = [(0,0,0),
                (27/256,38/256,100/256),
                (245/256,130/256,100/256),
                (134/256,200/256,230/256),
                (210/256,210/256,185/256),
                (74/256,93/256,206/256),
                (249/256,180/256,161/256),
                (16/256,23/256,60/256),
                (194/256,50/256,13/256),
                (37/256,136/256,181/256),
                (144/256,144/256,93/256)]
    c_index = 0
    
    for country in countries:
        
        c_index = c_index + 1 
        
        co = dataframe[dataframe[column] == country].iloc[:,4:].T.sum(axis = 1)
        co = pd.DataFrame(co)
        co.columns = ['Cases']
        co['date'] = co.index
        co['date'] = pd.to_datetime(co['date'])  
        mask = (co['date'] >= start_date)
        co = co.loc[mask]
        co['Cases'] = co['Cases'] - co['Cases'][0]
        
        y = np.array(co['Cases'])
        x = np.arange(y.size)
        date = co['date']
        
        x2 = np.arange(y.size+forecast)
        
        date2 = pd.date_range(date[0],freq='1d',periods=len(date)+forecast)
        
        fig.add_trace(go.Scatter(x=date,y=y,mode='markers',name='Confirmed Cases',marker_color='rgba(27,38,100,.8)'),row=1,col=1)

        # Logistic regression -----------------------------------------------------------------------
        lpopt, lpcov = curve_fit(logistic, x, y, maxfev=10000)
        lerror = np.sqrt(np.diag(lpcov))
        # for logistic curve at half maximum, slope = growth rate/2. so doubling time = ln(2) / (growth rate/2)
        ldoubletime = np.log(2)/(lpopt[1]/2)
        # standard error
        ldoubletimeerror = 1.96 * ldoubletime * np.abs(lerror[1]/lpopt[1])
        
        # calculate R^2
        residuals = y - logistic(x, *lpopt)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        logisticr2 = 1 - (ss_res / ss_tot)  
        
        #if logisticr2 > 0.95:
        fig.add_trace(go.Scatter(x=date2,y=logistic(x2, *lpopt), mode='lines', name="Logistic (r2={0}) Doubling Time = {1}±{2} days".format(round(logisticr2,2),round(ldoubletime,2),round(ldoubletimeerror,2)),line_color='rgba(245,130,100,.8)',line_shape='spline',line_dash='dash'),row=1,col=1)
        # -----------------------------------------------------------------------
        
        
        # Exponential regression--------------------------------------------------------------------
        epopt, epcov = curve_fit(exponential, x, y, bounds=([0.99,0,-100],[1.01,0.9,100]), maxfev=10000)
        eerror = np.sqrt(np.diag(epcov))
        
        # for exponential curve, slope = growth rate. so doubling time = ln(2) / growth rate
        edoubletime = np.log(2)/epopt[1]
        # standard error
        edoubletimeerror = 1.96 * edoubletime * np.abs(eerror[1]/epopt[1])
        
        # calculate R^2
        residuals = y - exponential(x, *epopt)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        expr2 = 1 - (ss_res / ss_tot)
        
        #if expr2 > 0.95:
        fig.add_trace(go.Scatter(x=date2,y=exponential(x2, *epopt), mode='lines', name="Exponential (r2={0}) Doubling Time = {1}±{2} days".format(round(expr2,2),round(edoubletime,2),round(edoubletimeerror,2)),line_color='rgba(134,200,230,.8)',line_shape='spline',line_dash='dash'),row=1,col=1)
        # --------------------------------------------------------------------

        # Calculations for new cases
        delta = np.diff(co['Cases'])
        fig.add_trace(go.Scatter(x=y[1:],y=delta,mode='lines',name='New Daily Cases',line_color='rgba(210,210,185,.8)'),row=1,col=2)
        
        dbl_cases = 2**(x2/2)
        dbl_delta = 0.5*np.log(2)*np.exp((np.log(2)*x2)/2)
        fig.add_trace(go.Scatter(x=dbl_cases,y=dbl_delta,mode='lines',name='2 Day Doubling Time',line = {'color':'black','dash':'dash'}),row=1,col=2)
        
        fig.update_xaxes(title_text='Date',row=1,col=1)
        fig.update_yaxes(title_text='Total confirmed cases since {0}'.format(start_date),row=1,col=1)
        
        fig.update_xaxes(title_text='Total confirmed cases since {0}'.format(start_date),range=[0,np.log10(max(y)+100)],type="log",row=1,col=2)
        fig.update_yaxes(title_text='New daily cases',type="log",range=[0,np.log10(max(delta)+100)],row=1,col=2)
        
    return fig
    
    
# plotCases(dataframe, column, countries, days since)
#AusStates = ['New South Wales','Victoria','Queensland','Western Australia','South Australia', 'Tasmania', 'Australian Capital Territory']
AusStates = ['Queensland']
fig = plotCases(df, 'Province/State', AusStates, '2021-01-01', True, 3)
#py.plot(fig, filename = 'AusCovid-19', auto_open=True)

pio.write_html(fig,file='index.html',auto_open=True)