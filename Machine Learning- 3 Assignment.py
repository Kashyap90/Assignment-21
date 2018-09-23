
# coding: utf-8

# In[1]:


# I decided to treat this as a classification problem by creating a new binary variable affair
# (did the woman have at least one affair?) and trying to predict the classification for each woman.


# In[2]:


# Dataset:

#The dataset I chose is the affairs dataset that comes with Statsmodels. It was derivedfrom a survey of women in 1974 by 
#Redbook magazine, in which married women were asked about their participation in extramarital affairs. 
#More information about the study is available in a 1978 paper from the Journal of Political Economy.


# In[3]:


# Description of Variables:

#The dataset contains 6366 observations of 9 variables:
#rate_marriage: woman's rating of her marriage (1 = very poor, 5 = very good)
#age: woman's age
#yrs_married: number of years married
#children: number of children
#religious: woman's rating of how religious she is (1 = not religious, 4 = strongly religious)
#educ: level of education (9 = grade school, 12 = high school, 14 = some college, 16 =
#college graduate, 17 = some graduate school, 20 = advanced degree)
#occupation: woman's occupation (1 = student, 2 = farming/semi-skilled/unskilled, 3 =
#"white collar", 4 = teacher/nurse/writer/technician/skilled, 5 = managerial/business, 6 =professional with advanced degree)

#occupation_husb: husband's occupation (same coding as above)
#affairs: time spent in extra-marital affairs
#Code to loading data and modules


# In[5]:


# importing libraries:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from patsy import dmatrices
from sklearn import metrics
import statsmodels.api as sm


# In[7]:


# Read data:

dta = sm.datasets.fair.load_pandas().data


# In[8]:


dta.head()


# In[9]:


# Whether any null value is present or not:

dta.isnull().sum()


# In[10]:


# Briefing the data type:

dta.info()


# In[11]:


# Add "affair" column: 1 represents having affairs, 0 represents not:

dta['affairs'] = (dta['affairs']>0).astype(int)
dta.head()


# In[12]:


dta.columns


# In[13]:


# Data Description:

dta.describe(include='all')


# In[14]:


# Cross-tabulation of data:

print(pd.crosstab(dta['affairs'], dta['rate_marriage'], rownames=['affairs']))
pd.crosstab(dta['affairs'], dta['rate_marriage'], rownames=['affairs']).plot(kind='bar')


# In[15]:


print(pd.crosstab(dta['affairs'], dta['age'], rownames=['affairs']))
pd.crosstab(dta['affairs'], dta['age'], rownames=['affairs']).plot(kind='bar')


# In[16]:


print(pd.crosstab(dta['affairs'], dta['yrs_married'], rownames=['affairs']))
pd.crosstab(dta['affairs'], dta['yrs_married'], rownames=['affairs']).plot(kind='bar')


# In[17]:


print(pd.crosstab(dta['affairs'], dta['children'], rownames=['affairs']))
pd.crosstab(dta['affairs'], dta['children'], rownames=['affairs']).plot(kind='bar')


# In[18]:


print(pd.crosstab(dta['affairs'], dta['religious'], rownames=['affairs']))
pd.crosstab(dta['affairs'], dta['religious'], rownames=['affairs']).plot(kind='bar')


# In[19]:


print(pd.crosstab(dta['affairs'], dta['educ'], rownames=['affairs']))
pd.crosstab(dta['affairs'], dta['educ'], rownames=['affairs']).plot(kind='bar')


# In[20]:


print(pd.crosstab(dta['affairs'], dta['occupation'], rownames=['affairs']))
pd.crosstab(dta['affairs'], dta['occupation'], rownames=['affairs']).plot(kind='bar')


# In[21]:


print(pd.crosstab(dta['affairs'], dta['occupation_husb'], rownames=['affairs']))
pd.crosstab(dta['affairs'], dta['occupation_husb'], rownames=['affairs']).plot(kind='bar')


# In[22]:


dta.groupby('affairs').describe()


# In[24]:


# create dataframes with an intercept column and dummy variables for occupation and occupation_husb:

y, x = dmatrices('affairs ~ rate_marriage + age + yrs_married + children + religious                     + educ + C(occupation) + C(occupation_husb)',dta, return_type="dataframe")


# In[25]:


x.columns


# In[26]:


y.columns


# In[27]:


x = x.rename(columns={'C(occupation)[T.2.0]' :'occ_2',
                        'C(occupation)[T.3.0]':'occ_3',
                        'C(occupation)[T.4.0]':'occ_4',
                        'C(occupation)[T.5.0]':'occ_5',
                        'C(occupation)[T.6.0]':'occ_6',
                        'C(occupation_husb)[T.2.0]':'occ_husb_2',
                        'C(occupation_husb)[T.3.0]':'occ_husb_3',
                        'C(occupation_husb)[T.4.0]':'occ_husb_4',
                        'C(occupation_husb)[T.5.0]':'occ_husb_5',
                        'C(occupation_husb)[T.6.0]':'occ_husb_6'})

x.head(2)


# In[29]:


# Flatten y into a 1-D array:

y = np.ravel(y)
y


# In[30]:


# There are 2 ways to apply Logistic Regression:

# 1. With use of Statsmodel.Logit function
# 2. With use of SKLearn.LogisticRegression


# In[31]:


# With use of Statsmodel.Logit function:


# In[32]:


type(x)


# In[33]:


type(y)


# In[35]:


# Fit model:

logit = sm.Logit(y, x)
result = logit.fit()


# In[36]:


# Model Summary:

result.summary2()


# In[37]:


# With use of SKLearn.LogisticRegression


# In[38]:


# Fit model:

model = LogisticRegression()
model = model.fit(x, y)


# In[39]:


# Model Accuracy:

model.score(x, y)


# In[40]:


# what percentage had affairs?

y.mean()


# In[41]:


# Examining the coefficients:

for el in zip(x.columns, np.transpose(model.coef_).tolist()):
    print(el)


# In[42]:


# Model Evaluation Using a Validation Set:

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)


# In[44]:


x_train.shape, x_test.shape, y_test.shape, y_test.shape


# In[45]:


model2 = LogisticRegression()
model2.fit(x_train, y_train)


# In[46]:


# Predict class labels for the test set:

predicted = model2.predict(x_test)
predicted


# In[47]:


# Generate class probabilities:

probs = model2.predict_proba(x_test)
probs


# In[48]:


# Generate evaluation metrics:

print(metrics.accuracy_score(y_test, predicted))
print(metrics.roc_auc_score(y_test, probs[:, 1]))


# In[49]:


# We can check the confusion matrix and a classification report with the other metrics:


# In[50]:


print(metrics.confusion_matrix(y_test, predicted))
print(metrics.classification_report(y_test, predicted))


# In[51]:


# Model Evaluation Using Cross-Validation

scores = cross_val_score(LogisticRegression(), x, y, scoring='accuracy', cv=10)
scores, scores.mean()

