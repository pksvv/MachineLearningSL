#!/usr/bin/env python
# coding: utf-8

# # Performing Linear Regression on Advertising dataset

# ## Why do you use Regression Analysis?
# 
# Regression analysis estimates the relationship between two or more variables. 
# 

# In[1]:


# import libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from math import sqrt

# this allows plots to appear directly in the notebook
get_ipython().run_line_magic('matplotlib', 'inline')


# Let's take a look at some data, ask some questions about that data, and then use Linear regression to answer those questions.

# In[2]:


# read data into a DataFrame
data = pd.read_csv('Advertising.csv', index_col=0)
data.head()
data.columns = ['TV','Radio','Newspaper','Sales']


#  **Indepenent variables**
# - TV: Advertising dollars spent on TV for a single product in a given market (in thousands of dollars)
# - Radio: Advertising dollars spent on Radio
# - Newspaper: Advertising dollars spent on Newspaper
# 
# **Target Variable **
# - Sales: sales of a single product in a given market (in thousands of widgets)

# In[3]:


# print the shape of the DataFrame
data.shape


# In[4]:


# visualize the relationship between the features and the response using scatterplots
fig, axs = plt.subplots(1, 3, sharey=True)
data.plot(kind='scatter', x='TV', y='Sales', ax=axs[0], figsize=(16, 8))
data.plot(kind='scatter', x='Radio', y='Sales', ax=axs[1])
data.plot(kind='scatter', x='Newspaper', y='Sales', ax=axs[2])


# ## Questions About the Advertising Data
# 
# On the basis of this data, how should you spend advertising money in the future?
# These general questions might lead you to more specific questions:
# 
# 1. Is there a relationship between ads and sales?
# 2. How strong is that relationship?
# 3. Which ad types contribute to sales?
# 4. What is the effect of each ad type of sales?
# 5. Given ad spending, can sales be predicted?
# 
# Exploring these questions below.

# In[5]:


# create X and y
#taking only one variable for now
feature_cols = ['TV']
X = data[feature_cols]
y = data.Sales

# follow the usual sklearn pattern: import, instantiate, fit
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X, y)

# print intercept and coefficients
print(lm.intercept_)
print(lm.coef_)


# ## Interpreting Model Coefficients
# 
# How do you interpret the TV coefficient ($\beta_1$)?
# - A "unit" increase in TV ad spending was **associated with** a 0.047537 "unit" increase in Sales.
# - Or more clearly: An additional $1,000 spent on TV ads was **associated with** an increase in sales of 47.537 widgets.
# 
# Note that if an increase in TV ad spending was associated with a **decrease** in sales, $\beta_1$ would be **negative**.

# ## Using the Model for Prediction
# 
# Let's say that there was a new market where the TV advertising spend was **$50,000**. How would you predict the sales in that market?
# 
# $$y = \beta_0 + \beta_1x$$
# $$y = 7.032594 + 0.047537 \times 50$$

# In[6]:


# manually calculate the prediction
7.032594 + 0.047537*50


# Thus, you would predict Sales of **9,409 widgets** in that market.

# In[7]:


# you have to create a DataFrame since the Statsmodels formula interface expects it
X_new = pd.DataFrame({'TV': [50]})
X_new.head()


# In[8]:


# use the model to make predictions on a new value
lm.predict(X_new)


# ## Plotting the Least Squares Line
# 
# Let's make predictions for the **smallest and largest observed values of x**, and then use the predicted values to plot the least squares line:

# In[9]:


# create a DataFrame with the minimum and maximum values of TV
X_new = pd.DataFrame({'TV': [data.TV.min(), data.TV.max()]})
X_new.head()


# In[10]:


# make predictions for those x values and store them
preds = lm.predict(X_new)
preds


# In[11]:


# first, plot the observed data
data.plot(kind='scatter', x='TV', y='Sales')

# then, plot the least squares line
plt.plot(X_new, preds, c='red', linewidth=2)


# ## Confidence in the Model
# 
# **Question:** Is linear regression a high bias/low variance model, or a low bias/high variance model?
# 
# **Answer:** It's a High bias/low variance model. Under repeated sampling, the line will stay roughly in the same place (low variance), but the average of those models won't do a great job capturing the true relationship (high bias). Note that low variance is a useful characteristic when you don't have a lot of training data.
# 
# A closely related concept is **confidence intervals**. Statsmodels calculate 95% confidence intervals for your model coefficients, which are interpreted as follows: If the population from which this sample was drawn was **sampled 100 times**, approximately **95 of those confidence intervals** would contain the "true" coefficient.

# In[12]:


import statsmodels.formula.api as smf
lm = smf.ols(formula='Sales ~ TV', data=data).fit()
lm.conf_int()


# Keep in mind that you only have a **single sample of data**, and not the **entire population of data**. The "true" coefficient is either within this interval or it isn't, but there's no way to actually know.
# You estimate the coeffeicient with the data you have and indicate uncertainity about the estimate by giving a range that the co-efficient is probably within
# 
# Note that using 95% confidence intervals is just a convention. You can create 90% confidence intervals (which will be more narrow), 99% confidence intervals (which will be wider), or whatever intervals you like.

# ## Hypothesis Testing and p-values
# 
# Closely related to confidence intervals is **hypothesis testing**. Generally speaking, you start with a **null hypothesis** and an **alternative hypothesis** (that is opposite the null). Then, you check whether the data supports **rejecting the null hypothesis** or **failing to reject the null hypothesis**.
# 
# (Note that "failing to reject" the null is not the same as "accepting" the null hypothesis. The alternative hypothesis may indeed be true, except that you just don't have enough data to show that.)
# 
# As it relates to model coefficients, here is the conventional hypothesis test:
# - **null hypothesis:** There is no relationship between TV ads and Sales (and thus $\beta_1$ equals zero)
# - **alternative hypothesis:** There is a relationship between TV ads and Sales (and thus $\beta_1$ is not equal to zero)
# 
# How to test this hypothesis? Intuitively,  reject the null (and thus believe the alternative) if the 95% confidence interval **does not include zero**. Conversely, the **p-value** represents the probability that the coefficient is actually zero:

# In[13]:


# print the p-values for the model coefficients
lm.pvalues


# If the 95% confidence interval **includes zero**, the p-value for that coefficient will be **greater than 0.05**. If the 95% confidence interval **does not include zero**, the p-value will be **less than 0.05**. Thus, a p-value less than 0.05 is one way to decide whether there is likely a relationship between the feature and the response. (Again, using 0.05 as the cutoff is just a convention.)
# 
# In this case, the p-value for TV is far less than 0.05, and so there is a relationship between TV ads and Sales.
# Generally the p-value is ignored for the intercept.

# ## How Well Does the Model Fit the data?
# 
# The most common way to evaluate the overall fit of a linear model is by the **R-squared** value. R-squared is the **proportion of variance explained**, meaning the proportion of variance in the observed data that is explained by the model, or the reduction in error over the **null model**. (The null model just predicts the mean of the observed response, and thus it has an intercept and no slope.)
# 
# R-squared is between 0 and 1, and higher is better because it means that more variance is explained by the model. Here's an example of what R-squared "looks like":

# <img src="images/08_r_squared.png">

# You can see that the **blue line** explains some of the variance in the data (R-squared=0.54), the **green line** explains more of the variance (R-squared=0.64), and the **red line** fits the training data even further (R-squared=0.66). (Does the red line look like it's overfitting?)
# 
# Let's calculate the R-squared value for the simple linear model:

# In[14]:


# print the R-squared value for the model
lm.rsquared


# Is that a "good" R-squared value? It's hard to say. The threshold for a good R-squared value depends widely on the domain. Therefore, it's most useful as a tool for **comparing different models**.

# ## Multiple Linear Regression
# 
# Simple linear regression can easily be extended to include multiple features. This is called **multiple linear regression**:
# 
# $y = \beta_0 + \beta_1x_1 + ... + \beta_nx_n$
# 
# Each $x$ represents a different feature, and each feature has its own coefficient. In this case:
# 
# $y = \beta_0 + \beta_1 \times TV + \beta_2 \times Radio + \beta_3 \times Newspaper$
# 
# Let's use Statsmodels to estimate these coefficients:

# In[15]:



# create X and y
feature_cols = ['TV', 'Radio', 'Newspaper']
X = data[feature_cols]
y = data.Sales

#create train and test split
from sklearn import model_selection
xtrain,xtest,ytrain,ytest = model_selection.train_test_split(X,y,test_size=0.3,random_state=42)


# In[16]:


#without using train and test split dataset
lm = LinearRegression()
lm.fit(X, y)

# print intercept and coefficients
print(lm.intercept_)
print(lm.coef_)


# In[17]:


#using train, test datasets

lm = LinearRegression()
lm.fit(xtrain, ytrain)

# print intercept and coefficients
print(lm.intercept_)
print(lm.coef_)


#predictions  on test dataset
predictions = lm.predict(xtest)
print(sqrt(mean_squared_error(ytest, predictions)))


# How to interpret these coefficients? For a given amount of Radio and Newspaper ad spending, an **increase of $1000 in TV ad spending** was associated with an **increase in Sales of 45.765 widgets**.
# 
# A lot of the information to review piece-by-piece is available in the model summary output:

# In[18]:


lm = smf.ols(formula='Sales ~ TV + Radio + Newspaper', data=data).fit()
lm.conf_int()
lm.summary()


# What are a few key things you learn from this output?
# 
# - TV and Radio have significant **p-values**, whereas Newspaper does not. Thus, reject the null hypothesis for TV and Radio (that there is no association between those features and Sales), and fail to reject the null hypothesis for Newspaper.
# - TV and Radio ad spending are both **positively associated** with Sales, whereas Newspaper ad spending was **slightly negatively associated** with Sales. (However, this is irrelevant since as you have failed to reject the null hypothesis for Newspaper.)
# - This model has a higher **R-squared** (0.897) than the previous model, which means that this model provides a better fit to the data than a model that only includes TV.

# ## Feature Selection
# 
# How do you decide **what features have to be included** in a linear model? Here's one idea:
# - Try different models, and only keep predictors in the model if they have small p-values.
# - Check whether the R-squared value goes up when you add new predictors.
# 
# What are the **drawbacks** in this approach?
# - Linear models rely upon a lot of **assumptions** (such as the features being independent), and if those assumptions are violated (which they usually are), R-squared and p-values are less reliable.
# - Using a p-value cutoff of 0.05 means that if you add 100 predictors to a model that are **pure noise**, 5 of them (on average) will still be counted as significant.
# - R-squared is susceptible to **overfitting**, and thus there is no guarantee that a model with a high R-squared value will generalize. Below is an example:

# In[19]:


# only include TV and Radio in the model
lm = smf.ols(formula='Sales ~ TV + Radio', data=data).fit()
lm.rsquared


# In[20]:


# add Newspaper to the model (which has no association with Sales)
lm = smf.ols(formula='Sales ~ TV + Radio + Newspaper', data=data).fit()
lm.rsquared


# **R-squared will always increase as you add more features to the model**, even if they are unrelated to the response. Thus, selecting the model with the highest R-squared is not a reliable approach for choosing the best linear model.
# 
# There is alternative to R-squared called **adjusted R-squared** that penalizes model complexity (to control for overfitting), but it generally [under-penalizes complexity](http://scott.fortmann-roe.com/docs/MeasuringError.html).
# 
# So is there a better approach to feature selection? **Cross-validation.** It provides a more reliable estimate of out-of-sample error, and thus is a better way to choose which of your models will best **generalize** to out-of-sample data. There is extensive functionality for cross-validation in scikit-learn, including automated methods for searching different sets of parameters and different models. Importantly, cross-validation can be applied to any model, whereas the methods described above only apply to linear models.

# ## Handling Categorical Predictors with Two Categories
# 
# Up until now, all the predictors have been numeric. What if one of the predictors was categorical?
# 
# Let's create a new feature called **Size**, and randomly assign observations to be **small or large**:

# In[21]:


import numpy as np

# set a seed for reproducibility
np.random.seed(12345)

# create a Series of booleans in which roughly half are True
nums = np.random.rand(len(data))
mask_large = nums > 0.5

# initially set Size to small, then change roughly half to be large
data['Size'] = 'small'
data.loc[mask_large, 'Size'] = 'large'
data.head()


# For scikit-learn, you need to represent all data **numerically**. If the feature only has two categories, you can simply create a **dummy variable** that represents the categories as a binary value:

# In[22]:


# create a new Series called IsLarge
data['IsLarge'] = data.Size.map({'small':0, 'large':1})
data.head()


# Let's redo the multiple linear regression and include the **IsLarge** predictor:

# In[23]:


# create X and y
feature_cols = ['TV', 'Radio', 'Newspaper', 'IsLarge']
X = data[feature_cols]
y = data.Sales

# instantiate, fit
lm = LinearRegression()
lm.fit(X, y)

# print coefficients
zip(feature_cols, lm.coef_)


# How do you interpret the **IsLarge coefficient**? For a given amount of TV/Radio/Newspaper ad spending, being a large market was associated with an average **increase** in Sales of 57.42 widgets (as compared to a Small market, which is called the **baseline level**).
# 
# What if you had reversed the 0/1 coding and created the feature 'IsSmall' instead? The coefficient would be the same, except it would be **negative instead of positive**. As such, your choice of category for the baseline does not matter, all that changes is your **interpretation** of the coefficient.

# ## Handling Categorical Predictors with More than Two Categories
# 
# Let's create a new feature called **Area**, and randomly assign observations to be **rural, suburban, or urban**:

# In[24]:


# set a seed for reproducibility
np.random.seed(123456)

# assign roughly one third of observations to each group
nums = np.random.rand(len(data))
mask_suburban = (nums > 0.33) & (nums < 0.66)
mask_urban = nums > 0.66
data['Area'] = 'rural'
data.loc[mask_suburban, 'Area'] = 'suburban'
data.loc[mask_urban, 'Area'] = 'urban'
data.head()


# You have to represent Area numerically, but  can't simply code it as 0=rural, 1=suburban, 2=urban because that would imply an **ordered relationship** between suburban and urban (and thus urban is somehow "twice" the suburban category).
# 
# Instead, create **another dummy variable**:

# In[25]:


# create three dummy variables using get_dummies, then exclude the first dummy column
area_dummies = pd.get_dummies(data.Area, prefix='Area').iloc[:, 1:]

# concatenate the dummy variable columns onto the original DataFrame (axis=0 means rows, axis=1 means columns)
data = pd.concat([data, area_dummies], axis=1)
data.head()


# Here is how you interpret the coding:
# - **rural** is coded as Area_suburban=0 and Area_urban=0
# - **suburban** is coded as Area_suburban=1 and Area_urban=0
# - **urban** is coded as Area_suburban=0 and Area_urban=1
# 
# Why do you only need **two dummy variables, not three?** Because two dummies capture all of the information about the Area feature, and implicitly defines rural as the baseline level. (In general, if you have a categorical feature with k levels, you create k-1 dummy variables.)
# 
# If this is confusing, think about why you only needed one dummy variable for Size (IsLarge), not two dummy variables (IsSmall and IsLarge).
# 
# Let's include the two new dummy variables in the model:

# In[26]:


# create X and y
feature_cols = ['TV', 'Radio', 'Newspaper', 'IsLarge', 'Area_suburban', 'Area_urban']
X = data[feature_cols]
y = data.Sales

# instantiate, fit
lm = LinearRegression()
lm.fit(X, y)

# print coefficients
print(feature_cols, lm.coef_)


# How do you interpret the coefficients?
# - Holding all other variables fixed, being a **suburban** area it is associated with an average **decrease** in Sales of 106.56 widgets (as compared to the baseline level, which is rural).
# - Being an **urban** area it is associated with an average **increase** in Sales of 268.13 widgets (as compared to rural).
# 
# **A final note about dummy encoding:** If you have categories that can be ranked (strongly disagree, disagree, neutral, agree, strongly agree), you can potentially use a single dummy variable and represent the categories numerically (1, 2, 3, 4, 5).
