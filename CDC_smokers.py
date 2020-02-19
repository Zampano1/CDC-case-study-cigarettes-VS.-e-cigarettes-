#!/usr/bin/env python
# coding: utf-8

# In[567]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import chain
import statsmodels.stats.multitest as smm
from scipy.stats import chi2_contingency


# In[443]:


ASCfile = pd.read_csv("F://downloads//LLCP2017.csv")
ASCfile


# In[444]:


ASCfile.replace(to_replace=r'^\s*$',value=0,regex=True,inplace=True)
ASCfile = ASCfile.drop(columns=['DLYOTHER'])
ASCfile = pd.DataFrame(pd.DataFrame(ASCfile,dtype=np.float),dtype=np.int)


# In[445]:


smnum = len(ASCfile[(ASCfile['SMOKDAY2'].isin([1,2]))])
smtotal = len(ASCfile[(ASCfile['SMOKDAY2'].isin([1,2,3,7,9]))])
ecnum = len(ASCfile[(ASCfile['ECIGNOW'].isin([1,2]))])
ectotal = len(ASCfile[(ASCfile['ECIGNOW'].isin([1,2,3,7,9]))])
print("There are", smnum, "people claiming they are currently smoking cigarettes, occupying",'%.2f%%' % (100*smnum/smtotal),"of the people who answered this question.")
print("There are", ecnum, "people claiming they are currently using e-cigarettes, occupying",'%.2f%%' % (100*ecnum/ectotal),"of the people who answered this question.")


# In[446]:


smevery = len(ASCfile[(ASCfile['SMOKDAY2'] == 1)])
ecevery = len(ASCfile[(ASCfile['ECIGNOW'] == 1)])
print("For smokers,", smevery,"people claim they smoke every day, taking",'%.2f%%' % (100*smevery/smnum) ,"of the group of smokers.")
print("For e-cigarettes users,", ecevery,"people claim they use e-cigarettes or other electronic vaping products every day, taking",'%.2f%%' % (100*ecevery/ecnum) ,"of the group of users.")


# In[447]:


def general_smoker_statistics():
    data = {'Smoke (every day)':[len(ASCfile[(ASCfile['SMOKDAY2']==1) & (ASCfile['ECIGNOW']==1)]),                                 len(ASCfile[(ASCfile['SMOKDAY2']==1) & (ASCfile['ECIGNOW']==2)]),                                 len(ASCfile[(ASCfile['SMOKDAY2']==1) & (ASCfile['ECIGNOW']==3)])],
           'Smoke (some days)':[len(ASCfile[(ASCfile['SMOKDAY2']==2) & (ASCfile['ECIGNOW']==1)]),\
                                 len(ASCfile[(ASCfile['SMOKDAY2']==2) & (ASCfile['ECIGNOW']==2)]),\
                                 len(ASCfile[(ASCfile['SMOKDAY2']==2) & (ASCfile['ECIGNOW']==3)])],
           'Smoke (not at all)':[len(ASCfile[(ASCfile['SMOKDAY2']==3) & (ASCfile['ECIGNOW']==1)]),\
                                 len(ASCfile[(ASCfile['SMOKDAY2']==3) & (ASCfile['ECIGNOW']==2)]),\
                                 len(ASCfile[(ASCfile['SMOKDAY2']==3) & (ASCfile['ECIGNOW']==3)])]}
    df = pd.DataFrame(data)
    df.index = ['E-cigarettes (every day)','E-cigarettes (some days)','E-cigarettes (not at all)']
    df['Col_sum'] = df.apply(lambda x: x.sum(), axis=1)
    df.loc['Row_sum'] = df.apply(lambda x: x.sum())
    return df
    
table_1 = general_smoker_statistics()
cm = sns.light_palette("green", as_cmap=True)
table_1.style.background_gradient(cmap=cm)


# In[448]:


def tb1_percentage_sm():
    dff = pd.DataFrame(table_1.iloc[3,:].copy()).T
    tb2 = pd.DataFrame(np.repeat(dff.values,4,axis=0))
    tb2.columns = table_1.columns
    tb2.index = table_1.index
    tb = (table_1/tb2)
    for t in table_1.columns:
        tb[t] = tb[t].apply(lambda x: format(x, '.2%'))
    return tb

def tb1_percentage_ec():
    dff = pd.DataFrame(table_1.iloc[:,3])
    tb3 = pd.DataFrame(np.repeat(dff.values,4,axis=1))
    tb3.columns = table_1.columns
    tb3.index = table_1.index
    tb = (table_1/tb3)
    for t in table_1.columns:
        tb[t] = tb[t].apply(lambda x: format(x, '.2%'))
    return tb


# In[449]:


table_2 = tb1_percentage_sm()
table_2


# In[450]:


table_3 = tb1_percentage_ec()
table_3


# In[454]:


# From thepercentage tables above, we found the simultaneous use of cigarettes and e-cigarettes is common among respondents.
# Among daily smokers, 96.73% of people stated they only use e-cigarettes occasionally or never use it before.
# Among daily e-cigarettes users, 81.51% of people stated they only smoke occasionally or never smoke.
# These percentages is great to let us split the respondents into two independent and non-overlapping groups to study the 
# demographics and prevelance characteristics respectively.

smokers = ASCfile[(ASCfile['SMOKDAY2']==1) & (ASCfile['ECIGNOW'].isin([2,3]))]
print(smokers.shape)
ec_users = ASCfile[(ASCfile['ECIGNOW']==1) & (ASCfile['SMOKDAY2'].isin([2,3]))]
print(ec_users.shape)
dataset = smokers.assign(Category = 1).append(ec_users.assign(Category = 2))
dataset


# In[559]:


# Data handling for visualization
dataset.loc[(dataset['CHILDREN']>=4) & (dataset['CHILDREN']<88),'CHILDREN'] = 4
# Conclude people with 4 or 4+ children into one single group
dataset.loc[dataset['DIABETE3']==2,'DIABETE3'] = 22
dataset.loc[dataset['DIABETE3']==3,'DIABETE3'] = 2
# Ignore data entries that are about people's special health status only happened during pregnance
dataset.loc[dataset['PREDIAB1']==2,'PREDIAB1'] = 22
dataset.loc[dataset['PREDIAB1']==3,'PREDIAB1'] = 2
# Ignore data entries that are about people's special health status only happened during pregnance
dataset.loc[dataset['BPHI2MR']==2,'BPHI2MR'] = 22
dataset.loc[dataset['BPHI2MR']==3,'BPHI2MR'] = 2
# Ignore data entries that are about people's special health status only happened during pregnance
# dataset.loc[dataset['INCOME2']==2,'INCOME2'] = 3
# dataset.loc[dataset['INCOME2']==4,'INCOME2'] = 5


# In[458]:


# Integrate weight data 
dataset.loc[((dataset['WEIGHT2']>=50) & (dataset['WEIGHT2']<=80)) | ((dataset['WEIGHT2']>=9000) & (dataset['WEIGHT2']<=9036)),'WEIGHT2'] = 1
dataset.loc[((dataset['WEIGHT2']>80) & (dataset['WEIGHT2']<=120)) | ((dataset['WEIGHT2']>9036) & (dataset['WEIGHT2']<=9054)),'WEIGHT2'] = 2
dataset.loc[((dataset['WEIGHT2']>120) & (dataset['WEIGHT2']<=160)) | ((dataset['WEIGHT2']>9054) & (dataset['WEIGHT2']<=9072)),'WEIGHT2'] = 3
dataset.loc[((dataset['WEIGHT2']>160) & (dataset['WEIGHT2']<=200)) | ((dataset['WEIGHT2']>9072) & (dataset['WEIGHT2']<=9090)),'WEIGHT2'] = 4
dataset.loc[((dataset['WEIGHT2']>200) & (dataset['WEIGHT2']<=240)) | ((dataset['WEIGHT2']>9090) & (dataset['WEIGHT2']<=9109)),'WEIGHT2'] = 5
dataset.loc[((dataset['WEIGHT2']>240) & (dataset['WEIGHT2']<=700)) | ((dataset['WEIGHT2']>9109) & (dataset['WEIGHT2']<9999)),'WEIGHT2'] = 6


# In[459]:


# Integrate height data 
dataset.loc[((dataset['HEIGHT3']>=200) & (dataset['HEIGHT3']<=500)) | ((dataset['HEIGHT3']>=9000) & (dataset['HEIGHT3']<=9152)),'HEIGHT3'] = 1
dataset.loc[((dataset['HEIGHT3']>500) & (dataset['HEIGHT3']<=525)) | ((dataset['HEIGHT3']>9152) & (dataset['HEIGHT3']<=9160)),'HEIGHT3'] = 2
dataset.loc[((dataset['HEIGHT3']>525) & (dataset['HEIGHT3']<=550)) | ((dataset['HEIGHT3']>9160) & (dataset['HEIGHT3']<=9167)),'HEIGHT3'] = 3
dataset.loc[((dataset['HEIGHT3']>550) & (dataset['HEIGHT3']<=575)) | ((dataset['HEIGHT3']>9167) & (dataset['HEIGHT3']<=9175)),'HEIGHT3'] = 4
dataset.loc[((dataset['HEIGHT3']>575) & (dataset['HEIGHT3']<=600)) | ((dataset['HEIGHT3']>9175) & (dataset['HEIGHT3']<=9182)),'HEIGHT3'] = 5
dataset.loc[((dataset['HEIGHT3']>600) & (dataset['HEIGHT3']<=625)) | ((dataset['HEIGHT3']>9182) & (dataset['HEIGHT3']<=9190)),'HEIGHT3'] = 6
dataset.loc[((dataset['HEIGHT3']>625) & (dataset['HEIGHT3']<=999)) | ((dataset['HEIGHT3']>9190) & (dataset['HEIGHT3']<9999)),'HEIGHT3'] = 7


# In[558]:


# Replace the numbers appeared in the dataset with their real names for visualization conducted later
variable_category_SEX = {1:'Male',2:'Female',3:'Refused'}
# Gender info?
variable_category_MARITAL= {0:'Not asked or Missing',1:'Married ',2:'Divorced',3:'Widowed',4:'Separated',5:'Never married',
                            6:'Unmarried Couple',9:'Refused'}
# Marital info?
variable_category_EDUCA= {0:'Not asked or Missing',1:'kindergarten',2:'Grades 1 through 8',3:'Grades 9 through 11',
                          4:'Grade 12 or GED',5:'College 1 year to 3',6:'College 4 years or more',9:'Refused'}
# Highest education?
variable_category_RENTHOM1= {0:'Not asked or Missing',1:'Own',2:'Rent',3:'Other arrangement',7:'Don’t know/Not Sure',
                             9:'Refused'}  
# Do you own or rent your home? 
variable_category_NUMHHOL2= {0:'Not asked or Missing',1:'Yes',2:'No',7:'Don’t know/Not sure',9:'Refused'}  
#More than one phone number?
variable_category_CPDEMO1A= {0:'Not asked or Missing',1:'Yes',2:'No',7:'Don’t know/Not sure',9:'Refused'} 
#have a cell phone?
variable_category_VETERAN3= {0:'Not asked or Missing',1:'Yes',2:'No',7:'Don’t know/Not sure',9:'Refused'}
# Have you ever served on active duty in the United States Armed Forces, either in the regular military or in a
# National Guard or military reserve unit?
variable_category_EMPLOY1= {0:'Not asked or Missing',1:'Employed for wages',2:'Self-employed',
                            3:'Out of work for 1 year or more',
                            4:'Out of work for less than 1 year',5:'A homemaker',6:'A student',7:'Retired',8:'Unable to work',
                            9:'Refused'}
# Employment situation?
variable_category_CHILDREN= {0:'Not asked or Missing',1:'One',2:'Two',3:'Three',4:'Four or more',88:'None',99:'Refused'} 
# How many children less than 18 years of age live in your household?
variable_category_INCOME2= {0:'Not asked or Missing ',1:'Less than $10,000',2:'Less than $15,000',3:'Less than $20,000',
                            4:'Less than $25,000',5:'Less than $35,000',
                            6:'Less than $50,000',7:'Less than $75,000',8:'$75,000 or more',77:'Don’t know/Not sure',
                            99:'Refused '}
# : Is your annual household income from all sources?
variable_category_INTERNET= {0:'Not asked or Missing',1:'Yes',2:'No',7:'Don’t know/Not sure',9:'Refused'}
# Have you used the internet in the past 30 days?
variable_category_WEIGHT2= {0:'Not asked or Missing',1:'lower than 80lb',2:'80-120lb',3:'120-160lb',4:'160-200lb',5:'200-240lb',
                            6:'240lb or heavier',
                            7777:'Don’t know/Not sure',9999:'Refused'}
# How much do you weigh without shoes?
variable_category_HEIGHT3= {0:'Not asked or Missing',1:'lower than 152cm',2:'152-160cm',3:'160-167cm',4:'167-175cm',
                            5:'175-182cm',6:'182-190cm',7:'190cm or higher',7777:'Don’t know/Not sure ',9999:'Refused'}
# How tall are you without shoes? 
variable_category_DEAF= {0:'Not asked or Missing',1:'Yes',2:'No',7:'Don’t know/Not sure',9:'Refused'}
# Are you deaf or do you have serious difficulty hearing?
variable_category_BLIND= {0:'Not asked or Missing',1:'Yes',2:'No',7:'Don’t know/Not sure',9:'Refused'}
# Are you blind or do you have serious difficulty seeing, even when wearing glasses?
variable_category_DECIDE= {0:'Not asked or Missing',1:'Yes',2:'No',7:'Don’t know/Not sure',9:'Refused'}
# Because of a physical, mental, or emotional condition, do you have serious difficulty concentrating,
# remembering, or making decisions?
variable_category_DIFFWALK= {0:'Not asked or Missing',1:'Yes',2:'No',7:'Don’t know/Not sure',9:'Refused'}
# Do you have serious difficulty walking or climbing stairs?
variable_category_DIFFDRES= {0:'Not asked or Missing',1:'Yes',2:'No',7:'Don’t know/Not sure',9:'Refused'}
# Do you have difficulty dressing or bathing?
variable_category_DIFFALON= {0:'Not asked or Missing',1:'Yes',2:'No',7:'Don’t know/Not sure',9:'Refused'}
#Because of a physical, mental, or emotional condition, do you have difficulty doing errands alone such as
# visiting a doctor´s office or shopping?
variable_category_PREGNANT= {0:'Not asked or Missing',1:'Yes',2:'No',7:'Don’t know/Not sure',9:'Refused'}
# To your knowledge, are you now pregnant?
variable_category__PRACE1= {0:'Missing',1:'White',2:'Black or African American',3:'American Indian or Alaskan Native',
                            4:'Asian',5:'Native Hawaiian or other Pacific Islander',6:'Other race',7:'No preferred race',
                            77:'Don’t know/Not sure',99:'Refused'}
# Preferred race?
variable_category__HISPANC= {1:'Hispanic, Latino/a, or Spanish origin',2:'Not of Hispanic, Latino/a, or Spanish origin',
           9:'Don´t Know, Refused or Missing'}
# Ethnicity?
variable_category__AGE_G= {1:'Age 18 to 24',2:'Age 25 to 34',3:'Age 35 to 44',4:'Age 45 to 54',
                           5:'Age 55 to 64',6:'Age 65 or older'}
# Age group?


# In[624]:


# Define a function to generate bar charts for demographics variables
def return_bar_chart():
    variable = input("Enter a variable name:")
    variable_category = eval("variable_category_"+variable)
    plt_title = "Percentage of Respondents' "+ variable+ " in Generalized smokers' Community"

    if variable == "PREGNANT":
        s = (dataset[dataset['SEX']==2].pivot_table(
        index= 'Category', columns= variable, values= '_STATE', aggfunc= 'count'))
    else:
        s = (dataset.pivot_table(
        index= 'Category', columns= variable, values= '_STATE', aggfunc= 'count'))
    
    s.index = ['Smokers','E-cigarettes users']
    s.rename(columns= variable_category, inplace = True)
    row, col = s.shape
    s = s.fillna(0)

    drop_col = []
    for j in range(col):
        if (s.iloc[0,j]< 100) & (s.iloc[1,j]< 20):
            drop_col.append(j)
    ss=s.copy()
    s.drop(s.columns[drop_col], axis= 1, inplace= True)
    print(s)
    print("Due to the small percentage,",ss.columns[drop_col],"are not shown in the chart.")
    
    ax = (s.div(s.sum(1), axis= 0)).plot(kind= 'bar',figsize=(15,6),width= 0.8,edgecolor= None)
    plt.legend(labels=s.columns,fontsize= 12,loc= [1,0.2],framealpha= 0.3)
    plt.title(plt_title,fontsize=14, pad= 25)

    plt.xticks(fontsize= 12,rotation = 0)
    for spine in plt.gca().spines.values():
        spine.set_visible(False)
    plt.yticks([])

    for p in ax.patches:
        width, height = p.get_width(), p.get_height()
        x, y = p.get_xy() 
        ax.annotate('{:.2%}'.format(height), (x, y + height + 0.01))


# In[658]:


return_bar_chart()


# In[642]:


# Define a function to generate pie charts for demographics variables
def return_pie_chart():
    variable = input("Enter a variable name:")
    variable_category = eval("variable_category_"+variable)
    plt_title = "Percentage of Respondents' "+ variable+ " in Generalized smokers' Community"

    if variable == "PREGNANT":
        s = (dataset[dataset['SEX']==2].pivot_table(
            index= 'Category', columns= variable, values= '_STATE', aggfunc= 'count'))
    else:
        s = (dataset.pivot_table(
            index= 'Category', columns= variable, values= '_STATE', aggfunc= 'count'))
    
    s.index = ['Smokers','E-cigarettes users']
    s.rename(columns= variable_category, inplace = True)
    row, col = s.shape
    s = s.fillna(0)

    drop_col = []
    for j in range(col):
        if (s.iloc[0,j]< 100) & (s.iloc[1,j]< 20):
            drop_col.append(j)
    ss=s.copy()
    s.drop(s.columns[drop_col], axis= 1, inplace= True)
    print(s)
    print("Due to the small percentage,",ss.columns[drop_col],"are not shown in the chart.")


    fig = plt.figure()
    axs = (s.div(s.sum(1), axis= 0)).T.plot(kind= 'pie',subplots=True,figsize=(10,4.7), autopct='%1.2f%%')
    axs[0].get_legend().remove()
    axs[1].legend(labels=s.columns, loc = [1,0.4])
    plt.title(plt_title,fontsize=14, pad= 25,loc ='right')
    axs[0].set_ylabel('Smokers',fontsize= 12)
    axs[1].set_ylabel('E-cigarettes users',fontsize= 12)


# In[659]:


return_pie_chart()


# In[596]:


# Do statistics tests for specific variables of the dataset
study_list = []
p_value = []
concatenated = chain(range(31,32),range(33,34),range(35,36),range(38,39),range(40,52),range(53,55),range(104,105),
                     range(106,109),range(110,114),range(120,126),range(127,130),range(131,143),range(145,148),
                     range(149,150),range(165,168),range(170,171),range(173,174),range(175,181),range(186,193),
                     range(196,198),range(217,219),range(220,222),range(226,227),range(229,230),range(238,241),
                    range(243,245))
for i in concatenated:
    m = dataset.iloc[:,[357,i]]
    variable_name = dataset.columns[i]
    a = len(m[(m['Category'] == 1) & (m[variable_name] == 1)])
    b = len(m[(m['Category'] == 1) & (m[variable_name] == 2)])
    c = len(m[(m['Category'] == 2) & (m[variable_name] == 1)])
    d = len(m[(m['Category'] == 2) & (m[variable_name] == 2)])
    table = np.array([[a, b],[c, d]])
    study_list.append(variable_name)
    p_value.append(chi2_contingency(table)[1])
    
summary = pd.DataFrame(p_value)


# In[597]:


# Conduct the method of FDR correction to adjust P-values because I conducted hypothesis tests for multiple times
rej, pval_corr = smm.multipletests(p_value, alpha=0.05, method='fdr_bh')[:2]
summary['a'] = pval_corr
summary['b'] = rej
summary.index = study_list
summary.columns = ['P_value','Adjusted_p_value','Rejection']
summary.sort_values(by="Adjusted_p_value" , ascending=True)
# All variables retrieved from the original dataset that are used for hypothesis test are ranked according to theie P-values 

