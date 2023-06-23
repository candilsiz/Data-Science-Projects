#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 21:37:03 2023

@author: candilsiz
"""

import numpy as np
import pandas as pd
from skmultiflow.data import DataStream
from skmultiflow.data import AGRAWALGenerator, SEAGenerator
from skmultiflow.trees import HoeffdingTreeClassifier
#from skmultiflow.meta import AdaptiveRandomForest, StreamingRandomPatches
from skmultiflow.lazy import SAMKNNClassifier
from skmultiflow.drift_detection import ADWIN
from skmultiflow.evaluation import EvaluatePrequential
import matplotlib.pyplot as plt
from skmultiflow.meta import AdaptiveRandomForestClassifier
from skmultiflow.meta import StreamingRandomPatchesClassifier, DynamicWeightedMajorityClassifier


# Define Classifiers

#arf = AdaptiveRandomForestClassifier(random_state=1)
#samknn = SAMKNNClassifier()
srp = StreamingRandomPatchesClassifier()
#dwm = DynamicWeightedMajorityClassifier()

# ---- SEA DATASET ---- #

file = "SEAGenerator.csv"

SEA = pd.read_csv(file)

target = SEA['target']
target = np.array(target)

feature = SEA["att_num_1"]
feature = np.array(feature)

stream = DataStream(feature,target)

# ---- AGRAWAL DATASET ---- #

# file = "AGRAWALGenerator.csv"

# AGRAWAL = pd.read_csv(file)

# target = AGRAWAL['target']
# target = np.array(target)

# feature = AGRAWAL["salary"]
# feature = np.array(feature)

# stream = DataStream(feature,target)


# ---- SPAM DATASET ---- #

# file = "spam.csv"

# spam = pd.read_csv(file)

# target = spam['target']
# target = np.array(target)   # len = 6123


# feature = spam["13"]
# feature = np.array(feature) # len = 6123


# stream = DataStream(feature,target)

# ----ELECTRIC DATASET---- #

# file = "electric.csv"

# electric = pd.read_csv(file)

# target = electric['target']
# target = np.array(target)    # len = 45312

# feature = electric["feat_3"]
# feature = np.array(feature)  # len = 45312

# stream = DataStream(feature,target)


# ----------------------------------------------- ##

# Set the evaluator
evaluator = EvaluatePrequential(max_samples=100000,
                                pretrain_size=1000, 
                                max_time=10000,
                                metrics=['accuracy'],
                                output_file='results_SRP_SEA.csv')

# Run evaluation
result = evaluator.evaluate(stream=stream, model=srp, model_names=['SRP'])


# Load the results file
results_df = pd.read_csv('results_SRP_SEA.csv', skiprows=5)

# Plot the accuracy
plt.figure(figsize=(10,5))
plt.plot(results_df['id'], results_df['current_acc_[SRP]'])
plt.xlabel('Samples')
plt.ylabel('Accuracy')
plt.title('Overall Accuracy of SRP model (SEA)')
plt.show()

# Calculate the sliding window size
window_size = int(len(results_df) / 20)

# Calculate the prequential accuracy values
prequential_accuracy = []
for i in range(20):
    start_index = i * window_size
    end_index = (i + 1) * window_size
    accuracy = results_df['current_acc_[SRP]'][start_index:end_index].mean()
    prequential_accuracy.append(accuracy)

# Plot the prequential accuracy values over time
plt.figure(figsize=(10, 5))
plt.plot(range(20), prequential_accuracy)
plt.xlabel('Dataset')
plt.ylabel('Prequential Accuracy')
plt.title('Prequential Accuracy of SRP model (SEA)')
plt.show()


# ENSEMBLE with base learner: HoeffdingTreeClassifier , drift_detector: ADWIN

class DriftAwareEnsemble:
    def __init__(self, base_learner, drift_detector, n_learners=10):
        self.baseLearners = [base_learner() for _ in range(n_learners)]
        self.drift_detectors = [drift_detector() for _ in range(n_learners)]
        self.prediction_weights = np.ones(n_learners)

    def partial_fit(self, X, y, classes=None):
        for i, (learner, drift_detector) in enumerate(zip(self.baseLearners, self.drift_detectors)):
            learner.partial_fit(X, y, classes=classes)
            y_pred = learner.predict(X)
            for j in range(len(y)):
                drift_detector.add_element(y[j] - y_pred[j])
                if drift_detector.detected_change():
                    self.baseLearners[i] = HoeffdingTreeClassifier()  # Reset the model
                    self.baseLearners[i].partial_fit(X, y, classes=classes)  # Refit the model
        return self
    
    def predict(self, X):
        predictions = np.array([learner.predict(X) for learner in self.baseLearners])
        print(f"Predictions: {predictions}")
        return np.argmax(np.bincount(predictions.astype('int64')), axis=0)  # Vote-based prediction


file1 = "SEAGenerator.csv"
SEA = pd.read_csv(file1)
target = SEA['target'].values
feature = SEA["att_num_1"].values.reshape(-1, 1)
SEA_stream = DataStream(feature, target)

# Initialize the model
model = DriftAwareEnsemble(HoeffdingTreeClassifier, ADWIN, n_learners=10)

evaluator = EvaluatePrequential(pretrain_size=1, 
                                max_samples=100000, 
                                show_plot=False, 
                                metrics=['accuracy'], 
                                output_file='results_Ensembler_SEA.csv')

# Run evaluation
evaluator.evaluate(stream=SEA_stream, model=model, model_names=['Ensemble']);

# Load the results file
results_df = pd.read_csv('results_Ensembler_SEA.csv', skiprows=5)

# Plot the accuracy
plt.figure(figsize=(10,5))
plt.plot(results_df['id'], results_df['current_acc_[Ensemble]'])
plt.xlabel('Samples')
plt.ylabel('Accuracy')
plt.title('Overall Accuracy of Ensemble model (SEA)')
plt.show()

# Calculate the sliding window size
window_size = int(len(results_df) / 20)

# Calculate the prequential accuracy values
prequential_accuracy = []
for i in range(20):
    start_index = i * window_size
    end_index = (i + 1) * window_size
    accuracy = results_df['current_acc_[Ensemble]'][start_index:end_index].mean()
    prequential_accuracy.append(accuracy)

# Plot the prequential accuracy values over time
plt.figure(figsize=(10, 5))
plt.plot(range(20), prequential_accuracy)
plt.xlabel('Dataset')
plt.ylabel('Prequential Accuracy')
plt.title('Prequential Accuracy of Ensemble model (SEA)')
plt.show()


