import plotly.plotly as py
import plotly.graph_objs as go
import plotly.figure_factory as FF
import matplotlib.pyplot as plt


import pandas_datareader as pdr
import pandas_datareader.data as web
from yql.api import YQL
import datetime
from datetime import datetime
from dateutil.relativedelta import relativedelta
from datetime import *; from dateutil.relativedelta import *
import calendar
import pandas as pd
import zipfile
import os
import glob
import numpy as np
from html5lib.treebuilders.etree_lxml import tostring
from __builtin__ import True
from numpy import NaN, nan

#from datetime import datetime

def extractFile(zipFile, dirToExt):
    zip_ref = zipfile.ZipFile(zipFile, 'r')
    print 'Starting to extract zip file :' ,zipFile
    zip_ref.extractall(dirToExt)
    zip_ref.printdir()
    print 'Finished to extract zip'
    zip_ref.close()
    return ;

def createUVXYfile(sourceDirFiles, splitFile, newFileName, year):
   
    allfiles = sorted(glob.glob(os.path.join(sourceDirFiles,"*.csv")))
  
    np_array_list = []
    splitRatio = pd.read_csv(splitFile,index_col=None, header=0)
    print (splitRatio)
    for file_ in allfiles:   
        
        df = pd.read_csv(file_,index_col=None, header=0)   
        df = df.loc[(df['Underlying'] == 'UVXY') & (df['Type'] == 'C')] # get only UVXY call options
        
        strDate = file_.partition('options_')[2].rpartition('.csv')[0] # extract date from file name
        print 'Append file content: ' ,file_, '; for date: ', strDate
        
        df['Date'] = pd.to_datetime(strDate,format="%Y%m%d") # add 'Date' column
        df['OrigUnderlyingPrice'] = df['UnderlyingPrice'] # add 'Date' column
        df['Expiry'] = pd.to_datetime(df['Expiry'],format="%Y%m%d") # convert 'Expiry' to Date type
           
        for t in range(0,len(splitRatio)):
            splitDate = splitRatio.loc[t,'Date']
            intRatio = splitRatio.loc[t,'Ratio']
            
            print '--------------'

            print 'Split Date: ', splitDate, ' ', 'Split Ratio: ', intRatio
            print '--------------'
                
            df.loc[df['Date']  >= splitDate, 'UnderlyingPrice'] = (df.loc[df['Date']  >= splitDate, 'UnderlyingPrice']/intRatio).round(2)
          

        np_array_list.append(df.as_matrix())
        
    
    comb_np_array = np.vstack(np_array_list)
    big_frame = pd.DataFrame(comb_np_array)
    
    big_frame.columns = ["Underlying","UnderlyingPrice","Expiry","Type","Strike","Last","Bid","Ask","Volume","OpenInterest","Date","OrigUnderlyingPrice"]
    
    big_frame.to_csv(newFileName + "_" + str(year) + ".csv", index=False,header=True)
    print 'Finished to create UVXYfile: ', newFileName
    return;

def listFiles(path):
    # returns a list of names (with extension, without full path) of all files 
    # in folder path
    files = []
    for name in os.listdir(path):
        if os.path.isfile(os.path.join(path, name)):
            files.append(name)
    return files 
#ibm = pdr.get_data_yahoo(symbols='^VIX', start=datetime(2014, 1, 1), end=datetime(2014, 12, 31))
#print(ibm['Adj Close'])
def getHistPrice():
    df = pd.read_csv("/home/ori/Downloads/2014_temp/output_VIX.csv")
#print(df.loc[(df["Underlying"]=="VIX")])
    
    for i, row in df.iterrows():  #i: dataframe index; row: each row in series format
        if row['Underlying']=="VIX":
            #data[row['feature']]=data[row['feature']].astype(np.object)
            print 'VIX'
   # elif row['Underlying']=="UVXY":
        #data[row['feature']]=data[row['feature']].astype(np.float)
      #  print 'UVXY'
#print df.dtypes
    return;

#if __name__ == '__main__':
def getHistPrice1():
    df = pd.read_csv("/home/orifux/Downloads/2014_temp/output_UVXY_VIX.csv")
    print(df.loc[(df['Underlying'] == 'VIX') & ( pd.to_datetime(df['Date']) == '2014-12-31')])
    return;

def createVIXfile(sourceDirFiles, newFileName, year):
    
    #get files from source directory
    allfiles = sorted(glob.glob(os.path.join(sourceDirFiles,"*.csv")))

    np_array_list = []

    #get vix prices from fred
    start = datetime.datetime(year, 1, 1)
    end = datetime.datetime(year, 12, 31)
    vixHist= web.DataReader("VIXCLS", 'fred', start, end)

    for file_ in allfiles:
        # open file
        df = pd.read_csv(file_,index_col=None, header=0)
        
        # get only VIX CALL options data
        df = df.loc[(df['Underlying'] == 'VIX') & (df['Type'] == 'C')]
        
        # extract date from file name
        strDate = pd.to_datetime(file_.partition('options_')[2].rpartition('.csv')[0],format="%Y%m%d")
        
        #get VIX price for the current date
        tmpPrice = vixHist.ix[strDate]['VIXCLS']
        if not(np.isnan (tmpPrice))==True:
            vixPrice = tmpPrice
        print 'Append file content: ' ,file_, '; for date: ', strDate, '; Vix Price:',vixPrice
        
        # add 'Date' column 
        df['Date'] = strDate
        
        # convert 'Expiry' to date type
        df['Expiry'] = pd.to_datetime(df['Expiry'],format="%Y%m%d")
        
        # save the original value
        df['OrigUnderlyingPrice'] = df['UnderlyingPrice']
        
        # set VIX price 
        df.loc[df['Date'] == strDate, 'UnderlyingPrice'] = vixPrice
        
        # append the data to the array
        np_array_list.append(df.as_matrix())    
    
    comb_np_array = np.vstack(np_array_list)
    big_frame = pd.DataFrame(comb_np_array)
    
    big_frame.columns = ["Underlying","UnderlyingPrice","Expiry","Type","Strike","Last","Bid","Ask","Volume","OpenInterest","Date","OrigUnderlyingPrice"]
    
    big_frame.to_csv(newFileName + "_" + str(year) + ".csv", index=False,header=True)
    print 'Finished to create VIXfile: ', newFileName
    
    return;
def plotMy(fileName):
    
    df = pd.read_csv(fileName)

    #df_external_source = FF.create_table(df.head())
    #py.iplot(df_external_source, filename='df-external-source-table')
    df['Date'] = pd.to_datetime(df['Date'])
    #df = df.set_index('Date')
    df = df.sort_values('Date', ascending=True)
    #plt.plot(df.index, df['UnderlyingPrice'])
    plt.xticks(rotation='vertical')
    plt.plot_date(df['Date'], df['UnderlyingPrice']) 
    plt.gcf().autofmt_xdate()
    plt.show() # Depending on whether you use IPython or interactive mode, etc.
#    trace = go.Scatter(x = df['Date'], y = df['UnderlyingPrice'],name='UVXY Prices')
 #   layout = go.Layout(title='UVXY Prices over time (2014)',
#                   plot_bgcolor='rgb(230, 230,230)', 
#                   showlegend=True)
#    fig = go.Figure(data=[trace], layout=layout)

 #   py.iplot(fig, filename='uvxy-stock-prices')
    return;
def plot1():
    fig, ax1 = plt.subplots()
    df = pd.read_csv('/home/orifux/Downloads/outputFiles/output.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    t = pd.to_datetime(df['Date'])
    s1 = df['UnderlyingPrice']
    
    
    ax1.set_xlabel('time (s)')
    # Make the y-axis label, ticks and tick labels match the line color.
    ax1.set_ylabel('exp', color='b')
    ax1.tick_params('y', colors='b')
    ax1.plot(t, s1, 'b-')
    
    ax2 = ax1.twinx()
    dr = pd.read_csv('/home/orifux/Downloads/outputFiles/outputVIX_2014.csv')
    #dr['Date'] = pd.to_datetime(dr['Date'])
    #t = pd.to_datetime(dr['Date'])
    s2 = dr['UnderlyingPrice']
    
    ax2.plot(t, s2, 'r.')
    ax2.set_ylabel('sin', color='r')
    ax2.tick_params('y', colors='r')
    
    fig.tight_layout()
    
    plt.show()
    return;

def calc(UVXYscrFile, VIXscrFile):
    
    np_array_list = []
    # open files
    df_uvxy = pd.read_csv(UVXYscrFile)
    

    df_uvxy.set_index(['Expiry'])
    
    df_vix = pd.read_csv(VIXscrFile)
    
    #temp = df.groupby(['Date']).mean()   
    
    
    temp = pd.DataFrame() # with 0s rather than NaNs
    
    dateList=df_uvxy.Date.unique()
    for element in np.asarray(dateList).flat:
        
        #print df[df.Date==element].UnderlyingPrice.item()
        try:
            currPrice_uvxy = df_uvxy.loc[df_uvxy.Date==element,'UnderlyingPrice'].values[0]
        except:
            print "currPrice_uvxy no index at", element
        try:    
            currOrigPrice_uvxy = df_uvxy.loc[df_uvxy.Date==element,'OrigUnderlyingPrice'].values[0]
        except:
            print "currOrigPrice_uvx no index at", element
            
        try:
            currPrice_vix = df_vix.loc[df_vix.Date==element,'UnderlyingPrice'].values[0]
        except:
            print "currPrice_vix no index at", element
            
        try:
            currExpiry_uvxy = df_uvxy.loc[df_uvxy.Date==element,'Expiry'].values[0]
        except:
            print "currPrice_vix no index at", element
        
        six_months = pd.to_datetime(element) + relativedelta(months=+6)               
        print element, currPrice_uvxy, currPrice_vix, six_months    
        if np.isnan(currPrice_vix) == True:
           print 'Priceeeeeee:', currPrice_vix
        
        currOptDate = pd.date_range(pd.to_datetime(element) +relativedelta(months=5), periods=3, freq='1M')
        print currOptDate[0]
        
        leftb =pd.to_datetime(currOptDate[0])
        rigthb = pd.to_datetime(currOptDate[1])
        print leftb , rigthb
        #print df_uvxy.query('@leftb < Expiry< @rigthb')
        print 'exp:' ,currExpiry_uvxy, 'cala Exp:', df_uvxy.query('2014-01-01< Expiry < 2016-01-01')
  #      df = df[(df['date'] > '2000-6-1') & (df['date'] <= '2000-6-10')]
        
        
        #currOpt_uvxy = df_uvxy.loc[df_uvxy.Date==element,'UnderlyingPrice'].values[0]       
        a = np.array([element,currPrice_uvxy,currOrigPrice_uvxy,currPrice_vix])
        np_array_list.append(a)    
    
    comb_np_array = np.vstack(np_array_list)
    big_frame = pd.DataFrame(comb_np_array)
    
    big_frame.columns = ["Date", "UVXYPrice", "UVXYorigPrice", "VIXPrice"]
    
    big_frame.to_csv('/home/orifux/Downloads/outputFiles/test.csv', index=False,header=True)
    print 'Finished to create VIXfile: ', 'test.csv'

 
    return;


#getHistPrice1()
#extractFile('/home/orifux/Downloads/ODN-2014-test.zip', '/home/orifux/Downloads/ODN-2014-test1')
#createUVXYfile('/home/orifux/Downloads/ODN-2014-test1', '/home/orifux/Downloads/inputFile/splitRatio.csv', '/home/orifux/Downloads/outputFiles/outputUVXY',2014)
#createUVXYfile('/home/orifux/Downloads/2014', '/home/orifux/Downloads/inputFile/splitRatio.csv', '/home/orifux/Downloads/outputFiles/outputUVXY.csv')
#createVIXfile('/home/orifux/Downloads/2014', '/home/orifux/Downloads/outputFiles/outputVIX', 2014)
#plotMy('/home/orifux/Downloads/outputFiles/output.csv')
#plot1()

calc('/home/orifux/Downloads/outputFiles/output.csv','/home/orifux/Downloads/outputFiles/outputVIX_2014.csv')


    