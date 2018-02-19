
# coding: utf-8

# # Part A: 

# In[11]:

from pandas_datareader import data as web
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
import numpy as np
from __future__ import division
import matplotlib.dates as dates


# In[1]:

get_ipython().system(u'pip install pandoc')


# In[12]:

# Retrieve data


# In[13]:

Retail_list=['WMT','BBY','TGT','COST','GNC']    # walmart, bestbuy,target, costco, GNC
IT_list=['AAPL','MSFT','AMZN','GOOG','FB']    # apple, microsoft, amazon, google, facebook
Bank_list=['C','JPM','WFC','HSBC','GS']     # citi, jp morgan, wells fargo, hsbc, goldman sachs
Auto_list=['GM','Ford','TSLA','RACE','HMC']  # gm, ford, tesla, ferrari, honda

start_date=datetime(2006,10,1)
end_date=datetime(2017,10,1)


# In[14]:

df=pd.DataFrame()
for stock in Retail_list+IT_list+Bank_list+Auto_list:
    df[stock]=web.DataReader(stock,'yahoo',start_date,end_date)['Adj Close']


# In[15]:

df.head()


# In[16]:


outside=['Retail','Retail','Retail','Retail','Retail','IT','IT','IT','IT','IT','Bank','Bank','Bank','Bank','Bank','Auto','Auto','Auto','Auto','Auto']
inside=Retail_list+IT_list+Bank_list+Auto_list

hier_index=list(zip(outside,inside))


df.columns=pd.MultiIndex.from_tuples(hier_index)


# In[17]:

df.columns.names=['Industry','Stock']


# In[18]:

df.head()


# In[19]:

#visualize the patterns of the stocks


# In[20]:

industry_list=set(outside)
for industry in industry_list:
    df[industry].plot()
    plt.title('Price Pattern of '+str(industry)+' Industry')
    plt.legend(loc=0)
    plt.ylabel('Price')
    plt.xlabel('Year')
plt.tight_layout()


# In[21]:

#calculate the days of up and down for each stock in each year


# In[22]:

pct_df=pd.DataFrame()
for stock,columns in zip(inside,df) :
    pct_df[stock]=df[columns].pct_change()
    
    
def up_down(x):
    
    if x>0.0:
        return "up"
    elif x<0.0:
        return "down"
    


# In[23]:

pct_df.head()


# In[24]:

ud_df=pd.DataFrame()
for stock in inside:
    ud_df[stock]=pct_df[stock].map(up_down)
    


# In[25]:

ud_df.head()


# In[33]:

up_down_df=ud_df.groupby(ud_df.index.year).describe()


# In[65]:

year=sorted(set(ud_df.index.year))
up_down_df.head()


# In[41]:

up_down_df['WMT'].loc[2006]['unique']


# In[50]:

for company in inside:
    print '\n'
    for y in year:
        if up_down_df[company].loc[y]['top']!='Nan':
            print str(company)+' in ' + str(y)+ ' ' + str(up_down_df[company].loc[y]['top'])+             ' is : '+ str(up_down_df[company].loc[y]['freq'])
            if up_down_df[company].loc[y]['top']=='up':
                print str(company)+' in '+str(y)+' down is: '+str(up_down_df[company].loc[y]['count']-up_down_df[company].loc[y]['freq'])
            else:
                print str(company)+' in '+str(y)+' up is: '+str(up_down_df[company].loc[y]['count']-up_down_df[company].loc[y]['freq'])
                


# In[85]:

# calculate the max ,min, mean, median of the stock log return for each year and visualize it


# In[51]:

log_df=pd.DataFrame()
for stock,columns in zip(inside,df) :
    log_df[stock]=np.log(df[columns]/df[columns].shift(1))

    


# In[52]:

log_df.head()


# In[58]:

for y in sorted(set(log_df.index.year)):
    print "Year of "+str(y)
    print log_df[str(y)].describe()
    print "\n\n"


# In[57]:

log_summary=log_df.groupby(log_df.index.year).describe()
log_summary.head(15)


# In[69]:

for company in inside:
    
    ma=[]
    mi=[]
    me=[]
    mean=[]
    for y in year:
        ma.append(log_summary[company].loc[y]['max'])
        mi.append(log_summary[company].loc[y]['min'])
        me.append(log_summary[company].loc[y]['50%'])
        mean.append(log_summary[company].loc[y]['mean'])
        d={'max':ma,'min':mi,'median':me,'mean':mean}
    summary=pd.DataFrame(d,columns=d.keys(),index=year)
    summary.plot()
    plt.title(company)
    plt.legend(loc=0)


# In[196]:

fig,axes=plt.subplots(4,5,figsize=(15,10))

plt.tight_layout()
for i in range(4):
    for j in range(5):
        axes[i,j].plot(log_df.index,log_df[inside[i*5+j]],lw=1)
        axes[i,j].set_title(" "+inside[i*5+j])
        axes[i,j].xaxis.set_major_formatter(dates.DateFormatter('\n-%y'))
        axes[i,j].set_xlabel("year")
        axes[i,j].set_ylim(-0.4,0.5)
fig.autofmt_xdate()


# In[125]:

# compare the stock price patterns of these companines via visualization


# In[234]:

fig,axes=plt.subplots(4,5,figsize=(15,10))


for i in range(4):
    for j in range(5):
        axes[i,j].plot(df[outside[i*5+j],inside[i*5+j]],lw=1)
        axes[i,j].set_title(" "+inside[i*5+j])
        axes[i,j].xaxis.set_major_formatter(dates.DateFormatter('\n-%y'))
        axes[i,j].set_xlabel("Year")
        axes[i,j].set_ylabel("Price")
fig.autofmt_xdate()
plt.tight_layout()


# In[141]:

# compare the volatilities of the stock and draw your conclusion


# In[160]:

# standard deviation of bar chart
log_df.std().plot(kind='barh')


# # Conclusion:
# While comparing the standard deviation of log return of each stock. We can notice that Ford has the largest volatility. It seems retail industry has a lower volatility compared with other industries. That is because retail industry is insensitive to the economics fluctuation.

# In[161]:

# give your suggestion about these investment


# In[170]:

# calculate the sharp ratio for each stock
# we assume risk free rate equals 0.


risk_free_rate=0.0

for stock,mean,std in zip(log_df.mean().index,log_df.mean(), log_df.std()):
    print "Sharp ratio of "+str(stock)+" is "+str(mean/std)


# In[182]:

(log_df.mean()/log_df.std()).plot(kind='bar')


# From the picture above we can see Race (Ferrari) has the largest sharp ratio, which means Ferrari has a large mean log return and small volatility. Thus we can inveset in Ferrari for high profit with low risk. 

# In[237]:

# calculate the cumulative retrun
fig,axes=plt.subplots(4,5,figsize=(15,10))


for i in range(4):
    for j in range(5):
        axes[i,j].plot(df[outside[i*5+j],inside[i*5+j]]['2017']/df[outside[i*5+j],inside[i*5+j]]['2017'].iloc[0],lw=1)
        axes[i,j].set_title(" "+inside[i*5+j])
        axes[i,j].set_ylim(0.5,2)
        axes[i,j].xaxis.set_major_formatter(dates.DateFormatter('\n-%m'))
        axes[i,j].set_xlabel("Date")
        axes[i,j].set_ylabel("Cumulative Return of 2017")
fig.autofmt_xdate()
plt.tight_layout()


# As we can see from the trend line for each stock in 2017, BBY,AAPL,AMZN,FB,TSLA and RACE have a good trend.
# Thus we can also consider these stocks.

# In[238]:

(log_df['2017'].mean()/log_df['2017'].std()).plot(kind='bar')


# From the chart above we can see FB and RACE has a large sharp ratio . We can obtain high profit with low risk by investing these two stocks.

# # Part B

# In[1]:

pwd


# In[4]:

option=pd.read_csv('NBOption.csv')


# In[5]:

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


# In[6]:

option.head(20)


# In[7]:

ask_price=option['Ask']
bid_price=option['Bid']
option_price=option['LastPrice']



# In[8]:

get_ipython().magic(u'matplotlib notebook')


# In[10]:


fig=plt.figure(figsize=(10,5))
ax=Axes3D(fig)
surf=ax.scatter(ask_price, bid_price,option_price,alpha=0.2)

# set x,y,z label
ax.set_xlabel('ask price')
ax.set_ylabel('bid price')
ax.set_zlabel('option price')



# In[14]:

# write a module to find k-25 nearest neighbors of an option on behalf of 
# Euclidean, correlation and Manhattan distances


# In[39]:

k=25
df=option
target_index=0
parameter=['Ask','Bid','LastPrice']


# In[164]:


def KNN(k,df,target_index,parameter):
    distance_dic={}
    for i in df.index:
        distance=0
        cor_dis=1-np.corrcoef(df.iloc[target_index][parameter].get_values().tolist(),df.iloc[i][parameter].get_values().tolist())[0,1]
        Euc_dis=((df.iloc[target_index][parameter]-df.iloc[i][parameter])**2).sum()**(1/2)
        Man_dis=abs((df.iloc[target_index][parameter]-df.iloc[i][parameter]).sum())
        
        distance=cor_dis+Euc_dis+Man_dis
        
        distance_dic[i]=distance
        
    nearest_25=sorted(distance_dic.iteritems(),key=lambda (k,v): (v,k))[:25]
    nearest_25={i[0]:i[1] for i in nearest_25}  
    return nearest_25


# In[180]:

n_25=KNN(k,df,998,parameter)


# In[181]:

option.iloc[n_25.keys()]


# In[151]:

#compare the volatility/implied volatility values among the option and its neighnor


# In[152]:

option.iloc[target_index]['Volatility']/option.iloc[target_index]['ImpliedVolatility']


# In[155]:

for i in n_25.keys():
    print option.iloc[i]['Volatility']/option.iloc[i]['ImpliedVolatility']


# In[182]:

for num in range(1,11):
    target_index=np.random.randint(len(option))
    k=25
    df=option
    parameter=['Ask','Bid','LastPrice']
    n_25=KNN(k,df,target_index,parameter)
    
    target=option.iloc[target_index]['Volatility']/option.iloc[target_index]['ImpliedVolatility']
    print str(num)+ " Sample"
    print "Target index is: "+ str(target_index)
    print "Volatility over ImpliedVolatility of target is: "+str(target)
    neighbor=[]
    print "The index of "+str(k)+" nearest neighbors are: "+str(n_25.keys())
    mean=0
    for i in n_25.keys():
        mean+=(option.iloc[i]['Volatility']/option.iloc[i]['ImpliedVolatility'])
    mean= mean/k
    print "Mean Volatility over ImpliedVolatility of the 25 neighbors is: "+str(mean) 
    
    print "\n"


# From the result we can see that 8 outcomes are good while 2 have extreme values.
# KNN method is susceptible to the noise data.
# 
# 
# Since KNN are used to predict qualitative data , its can't be used to predict quantitative data such as volatility and implied volatility. However, if we categorize the Volatility over ImpliedVolatility , we can use KNN to predict category.
