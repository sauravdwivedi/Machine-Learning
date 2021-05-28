"""Task description:

The file task_data.csv contains an example data set that has been artificially
generated. The set consists of 400 samples where for each sample there are 10
different sensor readings available. The samples have been divided into two
classes where the class label is either 1 or -1. The class labels define to what
particular class a particular sample belongs.

Your task is to rank the sensors according to their importance/predictive power
with respect to the class labels of the samples. Your solution should be a
Python script or a Jupyter notebook file that generates a ranking of the sensors
from the provided CSV file. The ranking should be in decreasing order where the
first sensor is the most important one."""

import pandas as pd
import numpy as np
import csv
import io
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def impt_features(x,y):
    """Method to sort features by their importance"""
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3)
    #Feature Selection using Random Forest Classifier
    rfc = RandomForestClassifier()
    rfc.fit(x_train, y_train)
    #Dataframe for features sorted by RandomForestClassifier
    df_imp = pd.DataFrame({'Feature': x.columns, \
        'Importance': rfc.feature_importances_}) \
        .sort_values('Importance', ascending = False) \
        .reset_index(drop = True)
    list_feats = df_imp
    return list_feats


if __name__ == '__main__':
    df = pd.read_csv('task_data.csv')
    #Convert class_lable to binary values
    label = {-1.0: 0, 1.0: 1}
    df['class_label'] = df['class_label'].map(label)
    #Select numerical features
    features = ['class_label', 'sensor0', 'sensor1', 'sensor2', 'sensor3',
        'sensor4', 'sensor5', 'sensor6', 'sensor7', 'sensor8', 'sensor9']
    #Chose target variable and features
    x = df[features].drop("class_label", axis=1)
    y = df[features]["class_label"]
    result = impt_features(x,y)
    file = open('ranked_list_of_sensors.txt', 'w')
    file.write(str(result) + '\n')
    file.close()
    print("Output written in 'ranked_list_of_sensors.txt'")
