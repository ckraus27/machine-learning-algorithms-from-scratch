import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import numpy as np
import matplotlib.pyplot as plt
import random

df = pd.read_csv("datasets/car.csv")

def entropy(df, target):
    
    counts = df[target].value_counts()
    ent = 0

    for count in counts:
        x = count / counts.sum()
        ent -= x * np.log2(x)
    
    return ent

def dt(df, features):


    if(len(features) == 0):
        return df["class"].value_counts().idxmax()
    
    if(len(df["class"].unique()) == 1):
        return df["class"].iloc[0]
    
    ent_init = entropy(df, "class")
    max_gain = -1
    max_feature = None

    for feature in features:
        ent = 0
        total = len(df)
        
        for value, group in df.groupby(feature):
            weight = len(group) / total
            ent += weight * entropy(group, "class")
        
        information_gain = ent_init - ent
        
        if (information_gain > max_gain):
            max_gain = information_gain
            max_feature = feature
    
    tree = {max_feature: {}}
    
    remaining_features = features.copy()
    remaining_features.remove(max_feature)

    for value, group in df.groupby(max_feature):
        tree[max_feature][value] = dt(group, remaining_features)
    
    return tree

def predict(tree, car):

    if type(tree) != dict:
        return tree
    
    feature = next(iter(tree))
    value = car[feature]

    if value not in tree[feature]:
        return None

    return predict(tree[feature][value], car)

def accuracy(y_true, y_pred):
    correct = 0
    for i in range(0, len(y_pred)):
        if(y_true.iloc[i] == y_pred[i]):
            correct += 1
    return correct / len(y_pred)

train_accuracy = []
test_accuracy = []

for i in range(0, 100):
    
    df = shuffle(df, random_state = i)

    train, test = train_test_split(df, test_size = 0.2, random_state = i)

    features = {"buying_price", "maintenance_price", "number_doors", "capacity", "luggage_boot_size", "safety_level"}

    tree = dt(train, features)

    train_pred = []
    for index, car in train.iterrows():
        train_pred.append(predict(tree, car))

    test_pred = []
    for index, car in test.iterrows():
        test_pred.append(predict(tree, car))
    
    train_accuracy.append(accuracy(train["class"], train_pred))
    test_accuracy.append(accuracy(test["class"], test_pred))

train_mean = np.mean(train_accuracy)  
train_std = np.std(train_accuracy) 
test_mean = np.mean(test_accuracy)
test_std = np.std(test_accuracy)
print(train_mean, train_std, test_mean, test_std) 

plt.figure()
plt.hist(train_accuracy)
plt.xlabel("(Accuracy)")
plt.ylabel("(Accuracy Frequency on Training Data)")
plt.savefig("training_accuracy_dt.png")
plt.close()

plt.figure()
plt.hist(test_accuracy)
plt.xlabel("(Accuracy)")
plt.ylabel("(Accuracy Frequency on Testing Data)")
plt.savefig("testing_accuracy_dt.png")
plt.close()