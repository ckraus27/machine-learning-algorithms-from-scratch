import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("datasets/wdbc.csv")


def find_neighbors(X_train, X_predict, training = False):

    all_neighbors = []

    for r in range(0, len(X_predict)):
        point1 = X_predict[r]
        distances = {}
        for row in range(0, len(X_train)):
            if training and row == r:
                continue
            point2 = X_train[row]
            dist = 0
            for col in range(0, X_train.shape[1]):
                dist += (point1[col] - point2[col]) ** 2
            dist = dist ** 0.5
            distances[row] = dist

        lowest_keys = sorted(distances, key = distances.get)
        all_neighbors.append(lowest_keys)
    
    return all_neighbors

def predict(k, neighbors, y_train):

    predictions = []

    for lowest_keys in neighbors:
        one_cnt = 0
        zero_cnt = 0

        for key in lowest_keys[:k]:
            if(y_train.iloc[key] == 0):
                zero_cnt += 1
            if (y_train.iloc[key] == 1):
                one_cnt += 1

        if (zero_cnt > one_cnt):
            prediction = 0
        else:
            prediction = 1
        
        predictions.append(prediction) 
    
    return predictions

def accuracy(y, pred):
    correct = 0

    for i in range(0, len(y)):
        if (y.iloc[i] == pred[i]):
            correct += 1
        
    return correct / len(y)
    
ks = []

for i in range(0, 26):
    ks.append(2 * i + 1)

accuracies_train_set = []
accuracies_test_set = []
accuracies_test_set_no_scale = []

for i in range(0, 26):
    accuracies_train_set.append([])
    accuracies_test_set.append([])
    accuracies_test_set_no_scale.append([])

for i in range(0, 20):
    
    df = shuffle(df, random_state = i)

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = i)
        
    X_test_no_scale = X_test.copy()
    X_train_no_scale = X_train.copy()
    X_test_no_scale = X_test_no_scale.to_numpy()
    X_train_no_scale = X_train_no_scale.to_numpy()

    train_max = X_train.max()
    train_min = X_train.min()

    X_train = (X_train - train_min) / (train_max - train_min)
    X_test = (X_test - train_min) / (train_max - train_min)
        
    X_train = X_train.to_numpy()
    X_test = X_test.to_numpy()

    train_neighbors = find_neighbors(X_train, X_train, training = True)
    test_neighbors = find_neighbors(X_train, X_test)
    test_neighbors_no_scale = find_neighbors(X_train_no_scale, X_test_no_scale)
    k = 1
    index = 0
    
    while (k < 52):

        accuracies_train_set[index].append(accuracy(y_train, predict(k, train_neighbors, y_train)))
        accuracies_test_set[index].append(accuracy(y_test, predict(k, test_neighbors, y_train)))
        accuracies_test_set_no_scale[index].append(accuracy(y_test, predict(k, test_neighbors_no_scale, y_train)))
        k += 2
        index += 1
    
train_means = []
train_sds = []
for accuracies in accuracies_train_set:
    train_means.append(np.mean(accuracies))
    train_sds.append(np.std(accuracies))
    
test_means = []
test_sds = []
for accuracies in accuracies_test_set:
    test_means.append(np.mean(accuracies))
    test_sds.append(np.std(accuracies))

test_means_no_scale = []
test_sds_no_scale = []
for accuracies in accuracies_test_set_no_scale:
    test_means_no_scale.append(np.mean(accuracies))
    test_sds_no_scale.append(np.std(accuracies))

plt.figure()
plt.errorbar(ks, train_means, yerr = train_sds, fmt = 'o-', capsize = 5)
plt.xlabel("(Value of k)")
plt.ylabel("Accuracy over training data")
plt.title("k-NN Accuracy vs k using training data")
plt.savefig("training_accuracy.png")
plt.close()

plt.figure()
plt.errorbar(ks, test_means, yerr = test_sds, fmt = 'o-', capsize = 5)
plt.xlabel("(Value of k)")
plt.ylabel("Accuracy over testing data")
plt.title("k-NN Accuracy vs k using testing data")
plt.savefig("testing_accuracy.png")
plt.close()

plt.figure()
plt.errorbar(ks, test_means_no_scale, yerr = test_sds_no_scale, fmt = 'o-', capsize = 5)
plt.xlabel("(Value of k)")
plt.ylabel("Accuracy over testing data")
plt.title("k-NN Accuracy vs k using testing data without scaling")
plt.savefig("testing_accuracy_no_scaling.png")
plt.close()

