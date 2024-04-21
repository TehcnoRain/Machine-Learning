#!/usr/bin/env python
# coding: utf-8

# In[25]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, classification_report


# In[26]:


# Load the dataset
url = "https://raw.githubusercontent.com/subashgandyer/datasets/main/great_customers.csv"
data = pd.read_csv(url)


# In[27]:


# Check for missing values
missing_values = data.isnull().sum()

# Handle missing values (you can choose an appropriate strategy)
# For example, filling missing numeric values with the mean and missing categorical values with the most frequent category
data['age'].fillna(data['age'].mean(), inplace=True)
data['salary'].fillna(data['salary'].mean(), inplace=True)
data['education_rank'].fillna(data['education_rank'].median(), inplace=True)
data['tea_per_year'].fillna(data['tea_per_year'].median(), inplace=True)
data['coffee_per_year'].fillna(data['coffee_per_year'].median(), inplace=True)
data['mins_beerdrinking_year'].fillna(0, inplace=True)
data['mins_exercising_year'].fillna(0, inplace=True)
data['works_hours'].fillna(data['works_hours'].median(), inplace=True)

# Check for missing values again after handling them
missing_values_after = data.isnull().sum()
data.to_csv("cleaned_data.csv", index=False)


# In[28]:


df = pd.read_csv("cleaned_data.csv")


# In[29]:


df.info


# In[30]:


print(data.columns)


# In[31]:


# Define features and target variable
X = data.drop(columns=['great_customer_class'])
y = data['great_customer_class']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[32]:


# Define preprocessing steps
numeric_features = ['age', 'mins_beerdrinking_year', 'mins_exercising_year', 'works_hours', 'tea_per_year', 'coffee_per_year']
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),  # Replace missing values with median
    ('scaler', StandardScaler())  # Standardize features
])

categorical_features = ['workclass', 'education_rank', 'marital-status', 'occupation', 'race', 'sex']
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),  # Replace missing values with most frequent
    ('onehot', OneHotEncoder(handle_unknown='ignore'))  # One-hot encode categorical variables
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Define the classifiers you want to use
classifier1 = RandomForestClassifier(random_state=42)
classifier2 = GradientBoostingClassifier(random_state=42)
classifier3 = LogisticRegression(random_state=42)

# Create a voting classifier
voting_classifier = VotingClassifier(estimators=[
    ('rf', classifier1),
    ('gb', classifier2),
    ('lr', classifier3)
], voting='soft')  # Use 'soft' voting for probabilities

# Create a pipeline with preprocessing and the voting classifier
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', voting_classifier)
])

# Fit the pipeline on the training data
pipeline.fit(X_train, y_train)

# Make predictions on the test data
y_pred = pipeline.predict(X_test)

# Evaluate the model (you can replace this with your preferred evaluation metrics)
from sklearn.metrics import accuracy_score, classification_report

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Print the results
print(f'Accuracy: {accuracy}')
print(f'Classification Report:\n{report}')


# In[ ]:


Print("Random Forest has the best accuracy")


# In[ ]:


# Import necessary libraries
import numpy as np  # Add this import
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import OneHotEncoder  # Import OneHotEncoder

# Perform one-hot encoding for categorical features
categorical_features = ['workclass', 'marital-status', 'occupation', 'race', 'sex']
encoder = OneHotEncoder(sparse=False)
X_categorical_encoded = encoder.fit_transform(X[categorical_features])

# Combine the encoded categorical features with the numerical features
X_encoded = np.hstack((X_categorical_encoded, X.drop(columns=categorical_features).values))

# Define X and y (your feature matrix and target variable)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Perform feature selection (e.g., using SelectKBest with ANOVA F-statistic)
k_best = SelectKBest(score_func=f_classif, k=5)
X_train_selected = k_best.fit_transform(X_train, y_train)
X_test_selected = k_best.transform(X_test)

# Define and train the models
models = {
    'Random Forest': RandomForestClassifier(random_state=42),
    'Support Vector Machine': SVC(kernel='linear', random_state=42),
    'Logistic Regression': LogisticRegression(random_state=42),
    'Naive Bayes': GaussianNB(),
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5)
}

for model_name, model in models.items():
    model.fit(X_train_selected, y_train)
    y_pred = model.predict(X_test_selected)
    
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    print(f"Model: {model_name}")
    print(f"Accuracy: {accuracy}")
    print("Classification Report:")
    print(report)
    print("="*50)


# In[ ]:


# Import necessary libraries
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import OneHotEncoder

# Perform one-hot encoding for categorical features
categorical_features = ['workclass', 'marital-status', 'occupation', 'race', 'sex']
encoder = OneHotEncoder(sparse=False)
X_categorical_encoded = encoder.fit_transform(X[categorical_features])

# Combine the encoded categorical features with the numerical features
X_encoded = np.hstack((X_categorical_encoded, X.drop(columns=categorical_features).values))

# Define X and y (your feature matrix and target variable)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Perform feature selection (e.g., using SelectKBest with ANOVA F-statistic)
k_best = SelectKBest(score_func=f_classif, k=5)
X_train_selected = k_best.fit_transform(X_train, y_train)
X_test_selected = k_best.transform(X_test)

# Define and train the models
models = {
    'Random Forest': RandomForestClassifier(random_state=42),
    'Support Vector Machine': SVC(kernel='linear', random_state=42),
    'Logistic Regression': LogisticRegression(random_state=42),
    'Naive Bayes': GaussianNB(),
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5)
}

for model_name, model in models.items():
    model.fit(X_train_selected, y_train)
    y_pred = model.predict(X_test_selected)
    
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    print(f"Model: {model_name}")
    print(f"Accuracy: {accuracy}")
    print("Classification Report:")
    print(report)
    print("="*50)


# In[ ]:


# Train and evaluate the Random Forest model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_predictions)
print("Random Forest Accuracy:", rf_accuracy)

# Train and evaluate the Support Vector Machine (SVM) model
svm_model = SVC(random_state=42)
svm_model.fit(X_train, y_train)
svm_predictions = svm_model.predict(X_test)
svm_accuracy = accuracy_score(y_test, svm_predictions)
print("SVM Accuracy:", svm_accuracy)

# Train and evaluate the Logistic Regression model
lr_model = LogisticRegression(random_state=42)
lr_model.fit(X_train, y_train)
lr_predictions = lr_model.predict(X_test)
lr_accuracy = accuracy_score(y_test, lr_predictions)
print("Logistic Regression Accuracy:", lr_accuracy)

# Train and evaluate the Naive Bayes model
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
nb_predictions = nb_model.predict(X_test)
nb_accuracy = accuracy_score(y_test, nb_predictions)
print("Naive Bayes Accuracy:", nb_accuracy)

# Train and evaluate the K-Nearest Neighbors (KNN) model
knn_model = KNeighborsClassifier()
knn_model.fit(X_train, y_train)
knn_predictions = knn_model.predict(X_test)
knn_accuracy = accuracy_score(y_test, knn_predictions)
print("K-Nearest Neighbors Accuracy:", knn_accuracy)


# In[ ]:





# In[ ]:




