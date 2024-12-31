#!/usr/bin/env python
# coding: utf-8

# 

# 

# # Perform Exploratory Data Analysis.

# ### Step 1: Load the Data

# In[1]:


import pandas as pd
# To read the .csv file 
df = pd.read_csv('Crop_recommendation.csv')
df


# ### Step 2: Summary Statistics

# In[2]:


# Calculate basic statistics
summary_stats = df.describe()
print(summary_stats)


# ### Step 3: Data Visualization - Histograms

# In[3]:


import matplotlib.pyplot as plt

# Plot histograms
df.hist(figsize=(10, 8))
plt.tight_layout()
plt.show()


# ### Step 4: Data Visualization - Pairplot

# In[4]:


import seaborn as sns
import matplotlib.pyplot as plt

# Scatter plots for numerical features vs. label
plt.figure(figsize=(12, 8))
sns.pairplot(df, hue='label')
plt.show()


# ### Step 5: Data Cleaning - Missing Values

# In[5]:


# Check for missing values
missing_values = df.isnull().sum()
print(missing_values)


# ### Step 6: Data Cleaning - Outlier Detection

# In[6]:


# Outlier detection (box plots)
plt.figure(figsize=(4, 2))
sns.boxplot(df=df)
plt.xticks(rotation=35)
plt.show()


# # Perform Pre-processing.

# ### Step 1: Load the dataset

# In[7]:


print("First few rows of the dataset:")
print(df.head())


# In[8]:


df.describe()


# ### Step 2: Separate features and labels

# In[9]:


X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]  # Numerical features
y = df['label']

# Display the first few rows of features (X) and labels (y)
print("\nFeatures (X):")
print(X.head())
print("\nLabels (y):")
print(y.head())


# ### Step 3: Data Splitting

# In[10]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Display the shapes of training and testing sets
print("\nTraining set shape (X_train, y_train):", X_train.shape, y_train.shape)
print("Testing set shape (X_test, y_test):", X_test.shape, y_test.shape)


# ### Step 4: Feature Scaling:

# In[11]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Display the first few rows of scaled training and testing data
print("Scaled Training Data:")
print(X_train_scaled[:5])
print("\nScaled Testing Data:")
print(X_test_scaled[:5])


# ### Step 5: Handling Missing Values

# In[12]:


from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train_scaled)
X_test_imputed = imputer.transform(X_test_scaled)

# Display the first few rows of imputed training and testing data
print("\nImputed Training Data:")
print(X_train_imputed[:5])
print("\nImputed Testing Data:")
print(X_test_imputed[:5])


# ### Step 6: Encoding Categorical Variables (if applicable)

# In[13]:


crop_dict = {
    'rice': 1,
    'maize': 2,
    'jute': 3,
    'cotton': 4,
    'coconut': 5,
    'papaya': 6,
    'orange': 7,
    'apple': 8,
    'muskmelon': 9,
    'watermelon': 10,
    'grapes': 11,
    'mango': 12,
    'banana': 13,
    'pomegranate': 14,
    'lentil': 15,
    'blackgram': 16,
    'mungbean': 17,
    'mothbeans': 18,
    'pigeonpeas': 19,
    'kidneybeans': 20,
    'chickpea': 21,
    'coffee': 22
}
df['crop_num']=df['label'].map(crop_dict)
print(df.head())


# ### Step 7: Dimensionality Reduction:

# In[14]:


import numpy as np

def manual_pca(X, n_components):
    X_centered = X - np.mean(X, axis=0)
    cov_matrix = np.cov(X_centered, rowvar=False)
    
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]
    top_eigenvectors = sorted_eigenvectors[:, :n_components]
    
    X_pca = np.dot(X_centered, top_eigenvectors)
    variance_ratio = sorted_eigenvalues[:n_components] / np.sum(sorted_eigenvalues)
    
    return X_pca, variance_ratio

X_train_pca, variance_ratio = manual_pca(X_train_imputed, n_components=7)
print("Variance Ratio :", variance_ratio)


# 
# Each of these preprocessing steps plays a crucial role in ensuring the quality and effectiveness of the machine learning model. By carefully preprocessing the data, we can improve the model's accuracy, robustness, and generalization capabilities.
# 

# # Perform feature selection and feature generation.

# ### Step 1: Correlation Matrix and Covariance Matrix

# In[15]:


print(df.columns)


# In[16]:


numerical_columns = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']  
numerical_data = df[numerical_columns]

print("Data Types:")
print(numerical_data.dtypes)

print("\nFirst Few Rows:")
print(numerical_data.head())


# In[17]:


correlation_matrix = numerical_data.corr()
covariance_matrix = numerical_data.cov()

print("Correlation Matrix:")
print(correlation_matrix)

print("\nCovariance Matrix:")
print(covariance_matrix)


# In[18]:


import seaborn as sns
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", annot_kws={"size": 10}, ax=axes[0])
axes[0].set_title('Correlation Matrix')

sns.heatmap(covariance_matrix, annot=True, cmap='coolwarm', fmt=".2f", annot_kws={"size": 10}, ax=axes[1])
axes[1].set_title('Covariance Matrix')

plt.tight_layout()
plt.show()


# ### R - square

# In[19]:


from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

X = numerical_data[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
y = df['crop_num']

model = LinearRegression().fit(X, y)

r_square = model.score(X, y)

print("\nR-square value:", r_square)


# In[20]:


x_values = numerical_data[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
y_values = model.predict(x_values)

# Plot the regression line and actual data
plt.figure(figsize=(20, 25))
plt.plot(y_values, y, color='red', label='Linear Regression Line')
plt.scatter(y_values, y, alpha=0.5, label='Actual Data')
plt.title('Linear Regression Line of label vs. Predicted label')
plt.xlabel('Predicted label')
plt.ylabel('label')
plt.legend()
plt.grid(True)
plt.show() 


# ### Step 2: Feature Generation

# In[21]:


df['temperature_humidity_cross'] = df['temperature'] + df['humidity']
df['ph_rainfall_cross'] = df['ph'] + df['rainfall']

print(df.head())


# # Machine learning Algorithms

# In[22]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score

lr = LinearRegression()

lr_model = LinearRegression().fit(X, y)
lr_model.fit(X, y)
accuracy_lr = lr_model.score(X, y)


# In[23]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

knn_model = KNeighborsClassifier()
knn_model.fit(X_train, y_train)
y_pred_knn = knn_model.predict(X_test)

accuracy_knn = accuracy_score(y_test, y_pred_knn)


# In[24]:


from sklearn.tree import DecisionTreeClassifier

dt_model = DecisionTreeClassifier(random_state=54)

dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)

accuracy_dt = accuracy_score(y_test, y_pred_dt)


# In[25]:


from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()

nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
y_pred_nb = nb_model.predict(X_test)

accuracy_nb = accuracy_score(y_test, y_pred_nb)


# In[26]:


from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier()

rfc.fit(X_train_imputed, y_train)
ypred_rfc = rfc.predict(X_test_imputed)

accuracy_rfc = accuracy_score(y_test, ypred_rfc)


# In[27]:


# Print accuracies
print("RandomForestClassifier Accuracy:", accuracy_rfc)
print("Linear Regression Accuracy:", accuracy_lr)
print("kNN Accuracy:", accuracy_knn)
print("Naive Bayes Accuracy:", accuracy_nb)
print("Decision Tree Accuracy:", accuracy_dt)


# ###  Plot for Accuracy of Machine Learning Models 

# In[28]:


import matplotlib.pyplot as plt

accuracies = []
accuracies.append(accuracy_lr)
accuracies.append(accuracy_rfc)
accuracies.append(accuracy_knn)
accuracies.append(accuracy_nb)
accuracies.append(accuracy_dt)


models = ['Linear Regression','Random Forest Classifier' , 'Naive Bayes', 'Decision Tree','kNN']

plt.figure(figsize=(10, 6))
plt.bar(models, accuracies, color=['blue', 'green', 'orange', 'red','purple'])
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.title('Accuracy of Machine Learning Models')
plt.ylim(0, 1) 
plt.show()


# # Recommendation System

# In[29]:


import numpy as np
import warnings

warnings.filterwarnings("ignore")

def recommendation(N, P, k, temperature, humidity, ph, rainfall):
    
    features = np.array([[N, P, k, temperature, humidity, ph, rainfall]])
    scaled_features = scaler.transform(features)
    prediction = rfc.predict(scaled_features).reshape(1, -1)
    
    return prediction[0] 

inputs = input("Enter values for N(Nitrogen), P(Phosphorus), k(Potassium), temperature, humidity, ph, and rainfall separated by spaces: ")
N,P,k,temperature,humidity,ph,rainfall = map(float, inputs.split())
predict = recommendation(N, P, k, temperature, humidity, ph, rainfall)

crop_dict = {
    'rice': 1,
    'maize': 2,
    'jute': 3,
    'cotton': 4,
    'coconut': 5,
    'papaya': 6,
    'orange': 7,
    'apple': 8,
    'muskmelon': 9,
    'watermelon': 10,
    'grapes': 11,
    'mango': 12,
    'banana': 13,
    'pomegranate': 14,
    'lentil': 15,
    'blackgram': 16,
    'mungbean': 17,
    'mothbeans': 18,
    'pigeonpeas': 19,
    'kidneybeans': 20,
    'chickpea': 21,
    'coffee': 22
}

if predict[0] in crop_dict:
    best_crop = predict[0]
    print(f"{best_crop} is the best crop to be cultivated.")
else:
    print("Crop growth is not possible under the given conditions.")


# In[30]:


from sklearn.naive_bayes import GaussianNB

nb_model = GaussianNB()
nb_model.fit(X_train, y_train)

def recommendation(N, P, k, temperature, humidity, ph, rainfall):
    features = np.array([[N, P, k, temperature, humidity, ph, rainfall]])
    scaled_features = scaler.transform(features)
    prediction = nb_model.predict(scaled_features).reshape(1, -1)
    return prediction[0] 

inputs = input("Enter values for N(Nitrogen), P(Phosphorus), k(Potassium), temperature, humidity, ph, and rainfall separated by spaces: ")
N, P, k, temperature, humidity, ph, rainfall = map(float, inputs.split())
predict = recommendation(N, P, k, temperature, humidity, ph, rainfall)

crop_dict = {
    'rice': 1,
    'maize': 2,
    'jute': 3,
    'cotton': 4,
    'coconut': 5,
    'papaya': 6,
    'orange': 7,
    'apple': 8,
    'muskmelon': 9,
    'watermelon': 10,
    'grapes': 11,
    'mango': 12,
    'banana': 13,
    'pomegranate': 14,
    'lentil': 15,
    'blackgram': 16,
    'mungbean': 17,
    'mothbeans': 18,
    'pigeonpeas': 19,
    'kidneybeans': 20,
    'chickpea': 21,
    'coffee': 22
}

if predict[0] in crop_dict:
    best_crop = predict[0]
    print(f"{best_crop} is the best crop to be cultivated.")
else:
    print("Crop growth is not possible under the given conditions.")

