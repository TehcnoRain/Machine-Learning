#!/usr/bin/env python
# coding: utf-8

# # Task 7: AutoFeatureSelector Tool
# ## This task is to test your understanding of various Feature Selection methods outlined in the lecture and the ability to apply this knowledge in a real-world dataset to select best features and also to build an automated feature selection tool as your toolkit
# 
# ### Use your knowledge of different feature selector methods to build an Automatic Feature Selection tool
# - Pearson Correlation
# - Chi-Square
# - RFE
# - Embedded
# - Tree (Random Forest)
# - Tree (Light GBM)

# ### Dataset: FIFA 19 Player Skills
# #### Attributes: FIFA 2019 players attributes like Age, Nationality, Overall, Potential, Club, Value, Wage, Preferred Foot, International Reputation, Weak Foot, Skill Moves, Work Rate, Position, Jersey Number, Joined, Loaned From, Contract Valid Until, Height, Weight, LS, ST, RS, LW, LF, CF, RF, RW, LAM, CAM, RAM, LM, LCM, CM, RCM, RM, LWB, LDM, CDM, RDM, RWB, LB, LCB, CB, RCB, RB, Crossing, Finishing, Heading, Accuracy, ShortPassing, Volleys, Dribbling, Curve, FKAccuracy, LongPassing, BallControl, Acceleration, SprintSpeed, Agility, Reactions, Balance, ShotPower, Jumping, Stamina, Strength, LongShots, Aggression, Interceptions, Positioning, Vision, Penalties, Composure, Marking, StandingTackle, SlidingTackle, GKDiving, GKHandling, GKKicking, GKPositioning, GKReflexes, and Release Clause.

# In[319]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as ss
from collections import Counter
import math
from scipy import stats
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


# In[320]:


player_df = pd.read_csv("fifa19.csv")


# In[321]:


numcols = ['Overall', 'Crossing','Finishing',  'ShortPassing',  'Dribbling','LongPassing', 'BallControl', 'Acceleration','SprintSpeed', 'Agility',  'Stamina','Volleys','FKAccuracy','Reactions','Balance','ShotPower','Strength','LongShots','Aggression','Interceptions']
catcols = ['Preferred Foot','Position','Body Type','Nationality','Weak Foot']


# In[322]:


player_df = player_df[numcols+catcols]


# In[323]:


traindf = pd.concat([player_df[numcols], pd.get_dummies(player_df[catcols])],axis=1)
features = traindf.columns

traindf = traindf.dropna()


# In[324]:


traindf = pd.DataFrame(traindf,columns=features)


# In[325]:


y = traindf['Overall']>=87
X = traindf.copy()
del X['Overall']


# In[326]:


X.head()


# In[327]:


len(X.columns)


# ### Set some fixed set of features

# In[328]:


feature_name = list(X.columns)
# no of maximum features we need to select
num_feats=30


# ## Filter Feature Selection - Pearson Correlation

# ### Pearson Correlation function

# In[329]:


def cor_selector(X, y,num_feats):
    # Your code goes here (Multiple lines)
    # Calculate Pearson correlation coefficients between features and target variable
    cor_target = []
    for feature in X.columns:
        cor = np.corrcoef(X[feature], y)[0, 1]
        cor_target.append((feature, abs(cor)))

    # Sort features by correlation in descending order
    cor_target.sort(key=lambda x: x[1], reverse=True)

    # Select the top 'num_feats' features based on correlation
    cor_feature = [feature[0] for feature in cor_target[:num_feats]]

    # Create a boolean mask for selected features
    cor_support = [True if feature in cor_feature else False for feature in X.columns]
    # Your code ends here
    return cor_support, cor_feature


# In[330]:


cor_support, cor_feature = cor_selector(X, y,num_feats)
print(str(len(cor_feature)), 'selected features')


# ### List the selected features from Pearson Correlation

# In[331]:


cor_feature


# ## Filter Feature Selection - Chi-Sqaure

# In[332]:


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import MinMaxScaler


# ### Chi-Squared Selector function

# In[333]:


def chi_squared_selector(X, y, num_feats):
    # Your code goes here (Multiple lines)
    # Initialize the SelectKBest feature selector with the chi-squared test
    chi2_selector = SelectKBest(score_func=chi2, k=num_feats)
    
    # Fit the selector to the data and transform the features
    X_new = chi2_selector.fit_transform(X, y)
    
    # Get the boolean mask of selected features
    chi_support = chi2_selector.get_support()
    
    # Get the names of selected features
    chi_feature = X.columns[chi_support].tolist()
    # Your code ends here
    return chi_support, chi_feature


# In[334]:


chi_support, chi_feature = chi_squared_selector(X, y,num_feats)
print(str(len(chi_feature)), 'selected features')


# ### List the selected features from Chi-Square 

# In[335]:


chi_feature


# ## Wrapper Feature Selection - Recursive Feature Elimination

# In[336]:


from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler


# ### RFE Selector function

# In[337]:


def rfe_selector(X, y, num_feats):
    # Your code goes here (Multiple lines)
    
    # Initialize the RFE feature selector with an estimator
    estimator = LogisticRegression(solver='liblinear', max_iter=1000)  # You can choose an appropriate estimator for your problem
    rfe_selector = RFE(estimator)
    
    # Set the number of features to select
    rfe_selector = rfe_selector.set_params(n_features_to_select=num_feats)
    
    # Fit the selector to the data and transform the features
    X_new = rfe_selector.fit_transform(X, y)
    
    # Get the boolean mask of selected features
    rfe_support = rfe_selector.support_
    
    # Get the names of selected features
    rfe_feature = X.columns[rfe_support].tolist()
    
    # Your code ends here
    return rfe_support, rfe_feature


# In[338]:


rfe_support, rfe_feature = rfe_selector(X, y,num_feats)
print(str(len(rfe_feature)), 'selected features')


# ### List the selected features from RFE

# In[339]:


rfe_feature


# ## Embedded Selection - Lasso: SelectFromModel

# In[340]:


from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler


# In[341]:


def embedded_log_reg_selector(X, y, num_feats):
    # Your code goes here (Multiple lines)
    
    # Step 1: Initialize the logistic regression model
    model = LogisticRegression(solver='liblinear', max_iter=1000)  # Adjust max_iter as needed

    # Step 2: Initialize the feature selection method
    feature_selector = SelectFromModel(estimator=model, max_features=num_feats)

    # Step 3: Fit the feature selector to your data
    feature_selector.fit(X, y)  # Replace X and y with your data

    # Step 4: Get the boolean mask of selected features
    embedded_lr_support = feature_selector.get_support()

    # Step 5: Get the names of selected features
    embedded_lr_feature = X.columns[embedded_lr_support].tolist()

    
    # Your code ends here
    return embedded_lr_support, embedded_lr_feature


# In[342]:


embedded_lr_support, embedded_lr_feature = embedded_log_reg_selector(X, y, num_feats)
print(str(len(embedded_lr_feature)), 'selected features')


# In[343]:


embedded_lr_feature


# ## Tree based(Random Forest): SelectFromModel

# In[344]:


from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier


# In[345]:


def embedded_rf_selector(X, y, num_feats):
    # Your code goes here (Multiple lines)
    # Step 1: Initialize the Random Forest classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)

    # Step 2: Initialize the feature selection method
    feature_selector = SelectFromModel(estimator=model, max_features=num_feats)

    # Step 3: Fit the feature selector to your data
    feature_selector.fit(X, y)

    # Step 4: Get the boolean mask of selected features
    embedded_rf_support = feature_selector.get_support()

    # Step 5: Get the names of selected features
    embedded_rf_feature = X.columns[embedded_rf_support].tolist()
    
    # Your code ends here
    return embedded_rf_support, embedded_rf_feature


# In[346]:


embedder_rf_support, embedder_rf_feature = embedded_rf_selector(X, y, num_feats=24)
print(str(len(embedder_rf_feature)), 'selected features')


# In[347]:


embedder_rf_feature


# ## Tree based(Light GBM): SelectFromModel

# In[348]:


from sklearn.feature_selection import SelectFromModel
from lightgbm import LGBMClassifier


# In[349]:


def embedded_lgbm_selector(X, y, num_feats):
    # Your code goes here (Multiple lines)
    
    # Step 1: Initialize the LightGBM classifier
    model = LGBMClassifier(n_estimators=100, random_state=42)

    # Step 2: Initialize the feature selection method
    feature_selector = SelectFromModel(estimator=model, max_features=num_feats)

    # Step 3: Fit the feature selector to your data
    feature_selector.fit(X, y)

    # Step 4: Get the boolean mask of selected features
    embedded_lgbm_support = feature_selector.get_support()

    # Step 5: Get the names of selected features
    embedded_lgbm_feature = X.columns[embedded_lgbm_support].tolist()
    
    # Your code ends here
    return embedded_lgbm_support, embedded_lgbm_feature


# In[350]:


embedded_lgbm_support, embedded_lgbm_feature = embedded_lgbm_selector(X, y, num_feats=30)
print(str(len(embedded_lgbm_feature)), 'selected features')


# In[351]:


embedded_lgbm_feature


# ## Putting all of it together: AutoFeatureSelector Tool

# In[352]:


# Put all selection together
feature_selection_df = pd.DataFrame({'Feature': feature_name, 'Pearson': cor_support, 'Chi-2': chi_support, 'RFE': rfe_support, 'Logistic': embedded_lr_support,
                                    'Random Forest': embedder_rf_support, 'LightGBM': embedded_lgbm_support})

# Count the selected times for each feature
feature_selection_df['Total'] = np.sum(feature_selection_df[['Pearson', 'Chi-2', 'RFE', 'Logistic', 'Random Forest', 'LightGBM']], axis=1)

# Display the top 'num_feats'
feature_selection_df = feature_selection_df.sort_values(['Total', 'Feature'], ascending=False)
feature_selection_df.index = range(1, len(feature_selection_df) + 1)
feature_selection_df.head(num_feats)


# ## Can you build a Python script that takes dataset and a list of different feature selection methods that you want to try and output the best (maximum votes) features from all methods?

# In[353]:


def preprocess_dataset(dataset_path):
    # Load the dataset
    player_df = pd.read_csv(dataset_path)

    numcols = ['Overall', 'Crossing', 'Finishing', 'ShortPassing', 'Dribbling', 'LongPassing', 'BallControl', 'Acceleration', 'SprintSpeed', 'Agility', 'Stamina', 'Volleys', 'FKAccuracy', 'Reactions', 'Balance', 'ShotPower', 'Strength', 'LongShots', 'Aggression', 'Interceptions']
    catcols = ['Preferred Foot', 'Position', 'Body Type', 'Weak Foot']

    # Exclude features with 'Nationality' in their names
    exclude_features = [col for col in player_df.columns if 'Nationality' in col]
    selected_numcols = [col for col in numcols if col not in exclude_features]

    # Select relevant columns
    player_df = player_df[selected_numcols + catcols]

    # Concatenate numerical features and one-hot encoded categorical features
    traindf = pd.concat([player_df[selected_numcols], pd.get_dummies(player_df[catcols])], axis=1)
    features = traindf.columns

    # Drop rows with missing values
    traindf = traindf.dropna()

    # Create X (features) and y (target)
    y = traindf['Overall'] >= 87
    X = traindf.copy()
    del X['Overall']

    # Number of maximum features to select
    num_feats = 30

    return X, y, num_feats


# In[354]:


def autoFeatureSelector(dataset_path, methods=[]):
    # Preprocessing
    X, y, num_feats = preprocess_dataset(dataset_path)

    # Initialize empty lists to store selected features from each method
    selected_features = []

    # Run feature selection methods and collect selected features
    if 'pearson' in methods:
        cor_support, cor_feature = cor_selector(X, y, num_feats)
        selected_features.extend(cor_feature)
    if 'chi-square' in methods:
        chi_support, chi_feature = chi_squared_selector(X, y, num_feats)
        selected_features.extend(chi_feature)
    if 'rfe' in methods:
        rfe_support, rfe_feature = rfe_selector(X, y, num_feats)
        selected_features.extend(rfe_feature)
    if 'log-reg' in methods:
        embedded_lr_support, embedded_lr_feature = embedded_log_reg_selector(X, y, num_feats)
        selected_features.extend(embedded_lr_feature)
    if 'rf' in methods:
        embedded_rf_support, embedded_rf_feature = embedded_rf_selector(X, y, num_feats)
        selected_features.extend(embedded_rf_feature)
    if 'lgbm' in methods:
        embedded_lgbm_support, embedded_lgbm_feature = embedded_lgbm_selector(X, y, num_feats)
        selected_features.extend(embedded_lgbm_feature)

    # Combine all the selected features into a single list
    combined_features = list(set(selected_features))

    # Count the number of selected features
    num_selected_features = len(combined_features)

    return combined_features, num_selected_features


# In[355]:


best_features, num_best_features = autoFeatureSelector(dataset_path="fifa19.csv", methods=['pearson', 'chi-square', 'rfe', 'log-reg', 'rf', 'lgbm'])
print("Best Features:", best_features)
print("Number of Best Features:", num_best_features)


# ### Last, Can you turn this notebook into a python script, run it and submit the python (.py) file that takes dataset and list of methods as inputs and outputs the best features
