#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  4 00:04:27 2023

@author: candilsiz
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score


data = pd.read_csv('falldetection_dataset.csv') # 566×306 feature matrix #

# Separate features and labels
head = np.array(data.columns.tolist())
header = head[2:]
feature = data.iloc[:,2:].values 
feature = np.array(feature)

label = np.array(data["F"]) 

# Features #
features = np.vstack((header, feature))

# Labels #
labels = np.insert(label, 0, "F")

####    PART A    ####

# Perform PCA to extract the top two principal components
pca = PCA(n_components=2)
principal_components = pca.fit_transform(features)
variance_ratio = pca.explained_variance_ratio_

print(f"Variance captured by PC1 and PC2: {variance_ratio}")

# Run k-means clustering on the principal components
n_clusters = 2  # Change the number of clusters as needed
kmeans = KMeans(n_clusters=n_clusters, n_init = "auto", random_state=42)
cluster_labels = kmeans.fit_predict(principal_components)


matching_elements = 0
for i in range(566):
    if cluster_labels[i] == 1 and labels[i] == "F" :
        matching_elements +=1
    if cluster_labels[i] == 0 and labels[i] == "NF" :
        matching_elements +=1
        
total_elements = len(cluster_labels)

# Calculate the percentage overlap
overlap = (matching_elements / total_elements) * 100
print(f"Percentage overlap between clusters and action labels: {overlap}%")

# Visualize the clusters
plt.scatter(principal_components[:, 0], principal_components[:, 1], c=cluster_labels)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Clustering Results')
plt.show()


####    PART B    ####

# Convert string elements to int elements
for i in range(566):
    if labels[i] == "F":
        labels[i] = 1
    if labels[i] == "NF":
        labels[i] = 0

# # training/validation/testing sets (e.g., 70%, 15%,15%) #

# featureTrain : (396,306)  ,  # labelTrain : (396,)
# featureTest : (85,306)   ,   # labelTest : (85,)
# featureValidation : (85,306)  , # labelValidation(85,)

#Train: 396, Validation: 85, Test: 85
labelTrain = labels[:396].tolist()
labelTest = labels[396:481].tolist()
labelValidation = labels[481:].tolist()

featureTrain = features[:396,:].tolist()
featureTest = features[396:481,:].tolist()
featureValidation = features[481:,:].tolist()

# Convert string elements to float elements
featureTrain = [[float(num) for num in sublist] for sublist in featureTrain]
featureTest = [[float(num) for num in sublist] for sublist in featureTest]
featureValidation = [[float(num) for num in sublist] for sublist in featureValidation]

####    Support Vector Machine Classification    ####

SVM = SVC()

svm_param_grid = {'C': [10, 100, 1000], 'gamma': [0.01, 0.05, 0.0001]}

# Perform grid search to with different parameters
SVM_gridsearch = GridSearchCV(SVM, svm_param_grid, cv=3)
SVM_gridsearch.fit(featureTrain, labelTrain)
best_SVM = SVM_gridsearch.best_estimator_

# Evaluate the best SVM classifier on the validation set
svm_val_predictions = best_SVM.predict(featureValidation)
svm_val_accuracy = accuracy_score(labelValidation, svm_val_predictions)

#  Validation Accuracy
print("Validation Accuracy (SVM):", svm_val_accuracy)

# Evaluate the best classifiers on the testing set
svm_test_predictions = best_SVM.predict(featureTest)
svm_test_accuracy = accuracy_score(labelTest, svm_test_predictions)

# Testing accuracies
print("Testing Accuracy (SVM):", svm_test_accuracy)


####    Multi-Layer Perceptron Classification    ####

MLP = MLPClassifier()

mlp_param_grid = {'hidden_layer_sizes': [(100,), (50, 50), (50, 100, 50)], 
                  'alpha': [0.0001, 0.001, 0.01]}

# Perform grid search to with different parameters
MLP_gridsearch = GridSearchCV(MLP, mlp_param_grid, cv=3)
MLP_gridsearch.fit(featureTrain, labelTrain)
best_MLP = MLP_gridsearch.best_estimator_

# Evaluate the best MLP classifier on the validation set
mlp_val_predictions = best_MLP.predict(featureValidation)
mlp_val_accuracy = accuracy_score(labelValidation, mlp_val_predictions)

#  Validation Accuracy
print("Validation Accuracy (MLP):", mlp_val_accuracy)

mlp_test_predictions = best_MLP.predict(featureTest)
mlp_test_accuracy = accuracy_score(labelTest, mlp_test_predictions)

# Testing accuracies
print("Testing Accuracy (MLP):", mlp_test_accuracy)



