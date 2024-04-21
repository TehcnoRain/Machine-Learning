#!/usr/bin/env python
# coding: utf-8

# # Task 10 : Benchmark Top ML Algorithms
# 
# This task tests your ability to use different ML algorithms when solving a specific problem.
# 

# ### Dataset
# Predict Loan Eligibility for Dream Housing Finance company
# 
# Dream Housing Finance company deals in all kinds of home loans. They have presence across all urban, semi urban and rural areas. Customer first applies for home loan and after that company validates the customer eligibility for loan.
# 
# Company wants to automate the loan eligibility process (real time) based on customer detail provided while filling online application form. These details are Gender, Marital Status, Education, Number of Dependents, Income, Loan Amount, Credit History and others. To automate this process, they have provided a dataset to identify the customers segments that are eligible for loan amount so that they can specifically target these customers.
# 
# Train: https://raw.githubusercontent.com/subashgandyer/datasets/main/loan_train.csv
# 
# Test: https://raw.githubusercontent.com/subashgandyer/datasets/main/loan_test.csv

# ## Task Requirements
# ### You can have the following Classification models built using different ML algorithms
# - Decision Tree
# - KNN
# - Logistic Regression
# - SVM
# - Random Forest
# - Any other algorithm of your choice

# ### Use GridSearchCV for finding the best model with the best hyperparameters

# - ### Build models
# - ### Create Parameter Grid
# - ### Run GridSearchCV
# - ### Choose the best model with the best hyperparameter
# - ### Give the best accuracy
# - ### Also, benchmark the best accuracy that you could get for every classification algorithm asked above

# #### Your final output will be something like this:
# - Best algorithm accuracy
# - Best hyperparameter accuracy for every algorithm
# 
# **Table 1 (Algorithm wise best model with best hyperparameter)**
# 
# Algorithm   |     Accuracy   |   Hyperparameters
# - DT
# - KNN
# - LR
# - SVM
# - RF
# - anyother
# 
# **Table 2 (Best overall)**
# 
# Algorithm    |   Accuracy    |   Hyperparameters
# 
# 

# ### Submission
# - Submit Notebook containing all saved ran code with outputs
# - Document with the above two tables

# In[8]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# In[9]:


# URLs for the training and test datasets
train_url = "https://raw.githubusercontent.com/subashgandyer/datasets/main/loan_train.csv"
test_url = "https://raw.githubusercontent.com/subashgandyer/datasets/main/loan_test.csv"

# Load the training and test datasets
train_df = pd.read_csv(train_url)
test_df = pd.read_csv(test_url)


# In[10]:


# Convert 'Loan_Status' to 1 and 0
train_df['Loan_Status'] = train_df['Loan_Status'].map({'Y': 1, 'N': 0})

# Identify categorical columns
categorical_cols = train_df.select_dtypes(include=['object']).columns

# Fill missing values in numeric columns with the mean
numeric_cols = ['LoanAmount', 'Loan_Amount_Term']
train_df[numeric_cols] = train_df[numeric_cols].fillna(train_df[numeric_cols].mean())

# Fill missing values in categorical columns with the mode
train_df[categorical_cols] = train_df[categorical_cols].fillna(train_df[categorical_cols].mode().iloc[0])


# In[11]:


# Check for null values in the training dataset
print("Null Values in Training Dataset:")
print(train_df.isnull().sum())

# Check for null values in the test dataset
print("\nNull Values in Test Dataset:")
print(test_df.isnull().sum())


# In[12]:


# Impute missing values in numeric columns with the mean
numeric_cols = ['LoanAmount', 'Loan_Amount_Term']
train_df[numeric_cols] = train_df[numeric_cols].fillna(train_df[numeric_cols].mean())

# Impute missing values in categorical columns with the mode
categorical_cols_to_impute = ['Gender', 'Dependents', 'Self_Employed', 'Credit_History']
train_df[categorical_cols_to_impute] = train_df[categorical_cols_to_impute].fillna(train_df[categorical_cols_to_impute].mode().iloc[0])


# In[13]:


# Check for null values in the training dataset
print("Null Values in Training Dataset:")
print(train_df.isnull().sum())

# Check for null values in the test dataset
print("\nNull Values in Test Dataset:")
print(test_df.isnull().sum())


# In[15]:


# Encode categorical features
encoder = OneHotEncoder(drop='first', sparse=False)
X_train_encoded = pd.DataFrame(encoder.fit_transform(train_df[categorical_cols]))

# Combine encoded features with remaining columns of train_df
X_train = pd.concat([train_df.drop(columns=categorical_cols), X_train_encoded], axis=1)

# Convert all feature names to strings
X_train.columns = X_train.columns.astype(str)

# Define the target variable (y_train) and the feature matrix (X_train)
y_train = train_df['Loan_Status']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Create and train a Decision Tree classifier
dt_classifier = DecisionTreeClassifier(random_state=42)
dt_classifier.fit(X_train, y_train)

# Predict on the test set
dt_predictions = dt_classifier.predict(X_test)

# Evaluate the model
dt_accuracy = accuracy_score(y_test, dt_predictions)
dt_report = classification_report(y_test, dt_predictions)
dt_confusion = confusion_matrix(y_test, dt_predictions)

print(f"Decision Tree Accuracy: {dt_accuracy}")
print("Decision Tree Classification Report:\n", dt_report)
print("Decision Tree Confusion Matrix:\n", dt_confusion)


# In[18]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

# Create and train a KNN classifier
knn_classifier = KNeighborsClassifier()
knn_classifier.fit(X_train, y_train)

# Predict on the test set
knn_predictions = knn_classifier.predict(X_test)

# Evaluate the KNN model
knn_accuracy = accuracy_score(y_test, knn_predictions)
knn_report = classification_report(y_test, knn_predictions)
knn_confusion = confusion_matrix(y_test, knn_predictions)

print(f"KNN Accuracy: {knn_accuracy}")
print("KNN Classification Report:\n", knn_report)
print("KNN Confusion Matrix:\n", knn_confusion)


# In[22]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Create and train a Logistic Regression classifier
logistic_classifier = LogisticRegression(random_state=42)
logistic_classifier.fit(X_train, y_train)

# Predict on the test set
logistic_predictions = logistic_classifier.predict(X_test)

# Evaluate the Logistic Regression model
logistic_accuracy = accuracy_score(y_test, logistic_predictions)
logistic_report = classification_report(y_test, logistic_predictions)
logistic_confusion = confusion_matrix(y_test, logistic_predictions)

print(f"Logistic Regression Accuracy: {logistic_accuracy}")
print("Logistic Regression Classification Report:\n", logistic_report)
print("Logistic Regression Confusion Matrix:\n", logistic_confusion)


# In[24]:


from sklearn.svm import SVC

# Create and train an SVM classifier
svm_classifier = SVC(random_state=42)
svm_classifier.fit(X_train, y_train)

# Predict on the test set
svm_predictions = svm_classifier.predict(X_test)

# Evaluate the SVM model
svm_accuracy = accuracy_score(y_test, svm_predictions)
svm_report = classification_report(y_test, svm_predictions)
svm_confusion = confusion_matrix(y_test, svm_predictions)

print(f"SVM Accuracy: {svm_accuracy}")
print("SVM Classification Report:\n", svm_report)
print("SVM Confusion Matrix:\n", svm_confusion)


# In[25]:


from sklearn.ensemble import RandomForestClassifier

# Create and train a Random Forest classifier
rf_classifier = RandomForestClassifier(random_state=42)
rf_classifier.fit(X_train, y_train)

# Predict on the test set
rf_predictions = rf_classifier.predict(X_test)

# Evaluate the Random Forest model
rf_accuracy = accuracy_score(y_test, rf_predictions)
rf_report = classification_report(y_test, rf_predictions)
rf_confusion = confusion_matrix(y_test, rf_predictions)

print(f"Random Forest Accuracy: {rf_accuracy}")
print("Random Forest Classification Report:\n", rf_report)
print("Random Forest Confusion Matrix:\n", rf_confusion)


# In[26]:


from sklearn.ensemble import GradientBoostingClassifier

# Create and train a Gradient Boosting classifier
gb_classifier = GradientBoostingClassifier(random_state=42)
gb_classifier.fit(X_train, y_train)

# Predict on the test set
gb_predictions = gb_classifier.predict(X_test)

# Evaluate the Gradient Boosting model
gb_accuracy = accuracy_score(y_test, gb_predictions)
gb_report = classification_report(y_test, gb_predictions)
gb_confusion = confusion_matrix(y_test, gb_predictions)

print(f"Gradient Boosting Accuracy: {gb_accuracy}")
print("Gradient Boosting Classification Report:\n", gb_report)
print("Gradient Boosting Confusion Matrix:\n", gb_confusion)


# In[32]:


from sklearn.metrics import make_scorer
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import make_scorer, f1_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

# Define a function to perform hyperparameter tuning
def tune_hyperparameters(model, param_grid, X_train, y_train):
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring=make_scorer(f1_score, pos_label=1))
    grid_search.fit(X_train, y_train)
    return grid_search.best_params_

# Define the hyperparameter grid for Decision Tree
param_grid_dt = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 10, 20, 30, 40],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Encode categorical features
encoder = OneHotEncoder(drop='first', sparse=False)
X_train_encoded = pd.DataFrame(encoder.fit_transform(train_df[categorical_cols]))

# Combine encoded features with remaining columns of train_df
X_train = pd.concat([train_df.drop(columns=categorical_cols), X_train_encoded], axis=1)

# Convert all feature names to strings
X_train.columns = X_train.columns.astype(str)

# Define the target variable (y_train) and the feature matrix (X_train)
y_train = train_df['Loan_Status']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Tune hyperparameters for Decision Tree
best_params_dt = tune_hyperparameters(DecisionTreeClassifier(random_state=42), param_grid_dt, X_train, y_train)
print("Best Hyperparameters for Decision Tree:", best_params_dt)

# Create and train a Decision Tree classifier with the best hyperparameters
dt_classifier = DecisionTreeClassifier(random_state=42, **best_params_dt)
dt_classifier.fit(X_train, y_train)

# Define the hyperparameter grid for Random Forest
param_grid_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30, 40],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

# Tune hyperparameters for Random Forest
best_params_rf = tune_hyperparameters(RandomForestClassifier(random_state=42), param_grid_rf, X_train, y_train)
print("Best Hyperparameters for Random Forest:", best_params_rf)

# Create and train a Random Forest classifier with the best hyperparameters
rf_classifier = RandomForestClassifier(random_state=42, **best_params_rf)
rf_classifier.fit(X_train, y_train)

# Define the hyperparameter grid for Gradient Boosting
param_grid_gb = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 4, 5],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Tune hyperparameters for Gradient Boosting
best_params_gb = tune_hyperparameters(GradientBoostingClassifier(random_state=42), param_grid_gb, X_train, y_train)
print("Best Hyperparameters for Gradient Boosting:", best_params_gb)

# Create and train a Gradient Boosting classifier with the best hyperparameters
gb_classifier = GradientBoostingClassifier(random_state=42, **best_params_gb)
gb_classifier.fit(X_train, y_train)

# Define the hyperparameter grid for SVM
param_grid_svm = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf', 'poly'],
    'degree': [2, 3, 4]
}

# Tune hyperparameters for SVM
best_params_svm = tune_hyperparameters(SVC(random_state=42), param_grid_svm, X_train, y_train)
print("Best Hyperparameters for SVM:", best_params_svm)

# Create and train a Support Vector Machine classifier with the best hyperparameters
svm_classifier = SVC(random_state=42, **best_params_svm)
svm_classifier.fit(X_train, y_train)

# Define the hyperparameter grid for Logistic Regression
param_grid_logistic = {
    'C': [0.1, 1, 10],
    'penalty': ['l1', 'l2', 'elasticnet', 'none'],
    'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
}

# Tune hyperparameters for Logistic Regression
best_params_logistic = tune_hyperparameters(LogisticRegression(random_state=42), param_grid_logistic, X_train, y_train)
print("Best Hyperparameters for Logistic Regression:", best_params_logistic)

# Create and train a Logistic Regression classifier with the best hyperparameters
logistic_classifier = LogisticRegression(random_state=42, **best_params_logistic)
logistic_classifier.fit(X_train, y_train)

# Make predictions with the models
dt_predictions = dt_classifier.predict(X_test)
rf_predictions = rf_classifier.predict(X_test)
gb_predictions = gb_classifier.predict(X_test)
svm_predictions = svm_classifier.predict(X_test)
logistic_predictions = logistic_classifier.predict(X_test)

# Evaluate Decision Tree, Random Forest, Gradient Boosting, SVM, and Logistic Regression models
dt_hyperparameter_accuracy = accuracy_score(y_test, dt_predictions)
dt_hyperparameter_report = classification_report(y_test, dt_predictions)
rf_hyperparameter_accuracy = accuracy_score(y_test, rf_predictions)
rf_hyperparameter_report = classification_report(y_test, rf_predictions)
gb_hyperparameter_accuracy = accuracy_score(y_test, gb_predictions)
gb_hyperparameter_report = classification_report(y_test, gb_predictions)
svm_hyperparameter_accuracy = accuracy_score(y_test, svm_predictions)
svm_hyperparameter_report = classification_report(y_test, svm_predictions)
logistic_hyperparameter_accuracy = accuracy_score(y_test, logistic_predictions)
logistic_hyperparameter_report = classification_report(y_test, logistic_predictions)

# Print or display the evaluation results for all models
print("Decision Tree Hyperparameter Accuracy:", dt_hyperparameter_accuracy)
print("Decision Tree Hyperparameter Classification Report:\n", dt_hyperparameter_report)
print("\nRandom Forest Hyperparameter Accuracy:", rf_hyperparameter_accuracy)
print("Random Forest Hyperparameter Classification Report:\n", rf_hyperparameter_report)
print("\nGradient Boosting Hyperparameter Accuracy:", gb_hyperparameter_accuracy)
print("Gradient Boosting Hyperparameter Classification Report:\n", gb_hyperparameter_report)
print("\nSVM Hyperparameter Accuracy:", svm_hyperparameter_accuracy)
print("SVM Hyperparameter Classification Report:\n", svm_hyperparameter_report)
print("\nLogistic Regression Hyperparameter Accuracy:", logistic_hyperparameter_accuracy)
print("Logistic Regression Hyperparameter Classification Report:\n", logistic_hyperparameter_report)


# In[36]:


# Define the hyperparameter grid for K-Nearest Neighbors (KNN)
param_grid_knn = {
    'n_neighbors': [3, 5, 7, 9],
    'weights': ['uniform', 'distance'],
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
}

# Tune hyperparameters for K-Nearest Neighbors (KNN)
best_params_knn = tune_hyperparameters(KNeighborsClassifier(), param_grid_knn, X_train, y_train)
print("Best Hyperparameters for K-Nearest Neighbors (KNN):", best_params_knn)

# Create and train a K-Nearest Neighbors (KNN) classifier with the best hyperparameters
knn_classifier = KNeighborsClassifier(**best_params_knn)
knn_classifier.fit(X_train, y_train)

# Make predictions with the KNN model
knn_predictions = knn_classifier.predict(X_test)

# Evaluate the KNN model
knn_accuracy = accuracy_score(y_test, knn_predictions)
knn_report = classification_report(y_test, knn_predictions)

# Compute the hyperparameter accuracy for K-Nearest Neighbors (KNN)
knn_hyperparameter_accuracy = f1_score(y_test, knn_predictions, pos_label=1)

# Update the algorithm_results dictionary with KNN results
algorithm_results["K-Nearest Neighbors"] = {
    "accuracy": knn_accuracy,
    "hyperparameters": best_params_knn,
    "hyperparameter_accuracy": knn_hyperparameter_accuracy
}

# Print or display the evaluation results for KNN
print("\nK-Nearest Neighbors (KNN) Accuracy:", knn_accuracy)
print("K-Nearest Neighbors (KNN) Classification Report:\n", knn_report)
print("K-Nearest Neighbors (KNN) Hyperparameter Accuracy:", knn_hyperparameter_accuracy)


# In[37]:


# Define a dictionary to store accuracy, hyperparameters, and hyperparameter accuracy for each algorithm
algorithm_results = {
    "Decision Tree": {
        "accuracy": dt_accuracy,
        "hyperparameters": best_params_dt,
        "hyperparameter_accuracy": dt_hyperparameter_accuracy
    },
    "Random Forest": {
        "accuracy": rf_accuracy,
        "hyperparameters": best_params_rf,
        "hyperparameter_accuracy": rf_hyperparameter_accuracy
    },
    "Gradient Boosting": {
        "accuracy": gb_accuracy,
        "hyperparameters": best_params_gb,
        "hyperparameter_accuracy": gb_hyperparameter_accuracy
    },
    "Support Vector Machine": {
        "accuracy": svm_accuracy,
        "hyperparameters": best_params_svm,
        "hyperparameter_accuracy": svm_hyperparameter_accuracy
    },
    "Logistic Regression": {
        "accuracy": logistic_accuracy,
        "hyperparameters": best_params_logistic,
        "hyperparameter_accuracy": logistic_hyperparameter_accuracy
    },
    "K-Nearest Neighbors": {
        "accuracy": knn_accuracy,
        "hyperparameters": best_params_knn,
        "hyperparameter_accuracy": knn_hyperparameter_accuracy
    }
    # Add other algorithms and their results as needed
}

# Find the algorithm with the highest accuracy
best_algorithm = max(algorithm_results, key=lambda x: algorithm_results[x]["accuracy"])

# Find the algorithm with the highest hyperparameter accuracy
best_hyperparameter_algorithm = max(algorithm_results, key=lambda x: algorithm_results[x]["hyperparameter_accuracy"])

# Print the table for algorithm-wise best models with hyperparameters and hyperparameter accuracy
print("Table 1 (Algorithm-wise best model with hyperparameters and hyperparameter accuracy)\n")
print("Algorithm                | Accuracy | Hyperparameters | Hyperparameter Accuracy")
print("-" * 80)
for algorithm, results in algorithm_results.items():
    accuracy = results["accuracy"]
    hyperparameters = results["hyperparameters"]
    hyperparameter_accuracy = results["hyperparameter_accuracy"]
    print(f"{algorithm:<25} | {accuracy:.4f} | {hyperparameters} | {hyperparameter_accuracy:.4f}")

# Print the table for the best overall model based on accuracy
print("\nTable 2 (Best overall based on accuracy)\n")
print("Algorithm                | Accuracy | Hyperparameters | Hyperparameter Accuracy")
print("-" * 80)
best_accuracy = algorithm_results[best_algorithm]["accuracy"]
best_hyperparameters = algorithm_results[best_algorithm]["hyperparameters"]
best_hyperparameter_accuracy = algorithm_results[best_algorithm]["hyperparameter_accuracy"]
print(f"{best_algorithm:<25} | {best_accuracy:.4f} | {best_hyperparameters} | {best_hyperparameter_accuracy:.4f}")

# Print the table for the best overall model based on hyperparameter accuracy
print("\nTable 3 (Best overall based on hyperparameter accuracy)\n")
print("Algorithm                | Accuracy | Hyperparameters | Hyperparameter Accuracy")
print("-" * 80)
best_hyperparameter_accuracy_hyperparameters = algorithm_results[best_hyperparameter_algorithm]["hyperparameters"]
print(f"{best_hyperparameter_algorithm:<25} | {best_hyperparameter_accuracy:.4f} | {best_hyperparameter_accuracy_hyperparameters}")


# In[38]:


# Define a dictionary to store accuracy, hyperparameters, and hyperparameter accuracy for each algorithm
algorithm_results = {
    "Decision Tree": {
        "accuracy": dt_accuracy,
        "hyperparameters": best_params_dt,
        "hyperparameter_accuracy": dt_hyperparameter_accuracy
    },
    "Random Forest": {
        "accuracy": rf_accuracy,
        "hyperparameters": best_params_rf,
        "hyperparameter_accuracy": rf_hyperparameter_accuracy
    },
    "Gradient Boosting": {
        "accuracy": gb_accuracy,
        "hyperparameters": best_params_gb,
        "hyperparameter_accuracy": gb_hyperparameter_accuracy
    },
    "Support Vector Machine": {
        "accuracy": svm_accuracy,
        "hyperparameters": best_params_svm,
        "hyperparameter_accuracy": svm_hyperparameter_accuracy
    },
    "Logistic Regression": {
        "accuracy": logistic_accuracy,
        "hyperparameters": best_params_logistic,
        "hyperparameter_accuracy": logistic_hyperparameter_accuracy
    },
    "K-Nearest Neighbors": {
        "accuracy": knn_accuracy,
        "hyperparameters": best_params_knn,
        "hyperparameter_accuracy": knn_hyperparameter_accuracy
    }
    # Add other algorithms and their results as needed
}

# Find the algorithm with the highest accuracy
best_accuracy_algorithm = max(algorithm_results, key=lambda x: algorithm_results[x]["accuracy"])

# Find the algorithm with the highest hyperparameter accuracy
best_hyperparameter_algorithm = max(algorithm_results, key=lambda x: algorithm_results[x]["hyperparameter_accuracy"])

# Function to print a formatted table
def print_table(title, data):
    print(f"{title}\n")
    print("Algorithm                | Accuracy | Hyperparameters | Hyperparameter Accuracy")
    print("-" * 80)
    for algorithm, results in data.items():
        accuracy = results["accuracy"]
        hyperparameters = results["hyperparameters"]
        hyperparameter_accuracy = results["hyperparameter_accuracy"]
        print(f"{algorithm:<25} | {accuracy:.4f} | {hyperparameters} | {hyperparameter_accuracy:.4f}")

# Print the tables
print_table("Table 1 (Algorithm-wise best model with hyperparameters and hyperparameter accuracy)", algorithm_results)
print("\n")
print_table("Table 2 (Best overall based on accuracy)", {best_accuracy_algorithm: algorithm_results[best_accuracy_algorithm]})
print("\n")
print_table("Table 3 (Best overall based on hyperparameter accuracy)", {best_hyperparameter_algorithm: algorithm_results[best_hyperparameter_algorithm]})


# In[ ]:




