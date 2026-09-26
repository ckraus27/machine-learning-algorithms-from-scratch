import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import numpy as np
import matplotlib.pyplot as plt
import random
from pandas.api.types import is_numeric_dtype
import math

wdbc_df = pd.read_csv("datasets/wdbc.csv")
loan_df = pd.read_csv("datasets/loan.csv")
raisin_df = pd.read_csv("datasets/raisin.csv")
titanic_df = pd.read_csv("datasets/titanic.csv")

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def make_nn(layout):
    neurons = []
    for i in range(0, len(layout)):
        if i == len(layout) - 1:
            neurons.append(np.ones(layout[i]))
        else:
            neurons.append(np.ones(1 + layout[i]))

    weights = []
    for i in range(1, len(layout)):
        weights.append(np.random.uniform(-1, 1, (layout[i], 1 + layout[i - 1])))

    #print("layout:", layout)
    #print("neurons:", [len(x) for x in neurons])
    #print("weights:", [x.shape for x in weights])

    return neurons, weights

def forward(neurons, weights, inputs):
    neurons[0][0] = 1
    for n in range(1, len(neurons[0])):
        neurons[0][n] = inputs[n - 1]

    for i in range(1, len(neurons)):
        if i == len(neurons) - 1:
            start = 0
        else:
            start = 1
        for j in range(start, len(neurons[i])):
            neurons[i][j] = sigmoid(np.dot(neurons[i-1], weights[i-1][j-start]))

    return neurons, weights

def error_nn(neurons, weights, output):
    errors = []
    for n in neurons:
        errors.append(n.copy())
    errors[len(errors) - 1] = neurons[len(neurons) - 1] - output

    for k in range(len(errors) - 2, 0, -1):
        for i in range(1, len(errors[k])):
            sum = 0
            if (k + 1 == len(errors) - 1):
                for j in range(0, len(errors[k + 1])):
                    sum += errors[k + 1][j] * weights[k][j][i]
            else:
                for j in range(1, len(errors[k+1])):
                    sum += errors[k+1][j] * weights[k][j-1][i]
            errors[k][i] = neurons[k][i] * (1 - neurons[k][i]) * sum

    return errors

def get_gradients_sum(neurons, weights, errors, gradients):
    for l in range(0, len(weights)):
        for t in range(0, len(weights[l])):
            for f in range(0, len(weights[l][t])):
                if (l < len(neurons) - 2):
                    gradients[l][t][f] += neurons[l][f] * errors[l + 1][t + 1]
                else:
                    gradients[l][t][f] += neurons[l][f] * errors[l + 1][t]
    
    return gradients

def adjust_weights(weights, learning_rate, ld, n, gradients):
    for l in range(len(gradients)):
        for t in range(len(gradients[l])):
            for f in range(len(gradients[l][t])):
                gradients[l][t][f] /= n

                if f != 0:
                    gradients[l][t][f] += (ld/n) * weights[l][t][f]
    
    for l in range(len(weights)):
        for t in range(len(weights[l])):
            for f in range(len(weights[l][t])):
                weights[l][t][f] -= learning_rate * gradients[l][t][f]

    return weights

def cost(weights, n, y_true, y_pred, ld):
    c = 0
    for i in range(n):
        if np.isscalar(y_true[i]):
            c += -1 * y_true[i] * math.log(y_pred[i])
            c-= (1 - y_true[i]) * math.log(1 - y_pred[i])
        else: 
            for j in range(len(y_true[i])):
                c += -1 * y_true[i][j] * math.log(y_pred[i][j])
                c-= (1 - y_true[i][j]) * math.log(1 - y_pred[i][j])

    c /= n

    r = 0
    for l in range(len(weights)):
        for t in range(len(weights[l])):
            for f in range(1, len(weights[l][t])):
                r += weights[l][t][f] * weights[l][t][f]

    r *= ld / (2 * n)

    c += r
    return c

def train_nn(train, neurons, weights, learning_rate, ld):
    
    for i in range(500):           
        y_trues = []
        gradients = []
        for weight in weights:
            gradients.append(weight.copy())
        for l in range(len(gradients)):
            for t in range(len(gradients[l])):
                for f in range(len(gradients[l][t])):
                    gradients[l][t][f] = 0
        for index, instance in train.iterrows():
            inputs = instance.iloc[:-1].to_numpy()
            y_trues.append(instance.iloc[-1])
            neurons, weights = forward(neurons, weights, inputs)
            errors = error_nn(neurons, weights, y_trues[-1])
            gradients = get_gradients_sum(neurons, weights, errors, gradients)

        weights = adjust_weights(weights, learning_rate, ld, len(train), gradients)

        y_preds = []
        for index, instance in train.iterrows():
            inputs = instance.iloc[:-1].to_numpy()
            neurons, weights = forward(neurons, weights, inputs)
            y_preds.append(neurons[-1][0])
            
        j_cost = cost(weights, len(train), y_trues, y_preds, ld)

    return weights

def evaluation(y_true, y_pred):
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for i in range(0, len(y_pred)):
        if(y_true.iloc[i] == 1 and y_pred[i] == 1):
            tp += 1
        elif(y_true.iloc[i] == 0 and y_pred[i] == 0):
            tn += 1
        elif (y_true.iloc[i] == 0 and y_pred[i] == 1):
            fp += 1
        elif (y_true.iloc[i] == 1 and y_pred[i] == 0):
            fn += 1

    if (tp + tn + fp + fn == 0):
        accuracy = 0
    else:
        accuracy = (tp + tn) / (tp + tn + fp + fn)

    if(tp + fp) == 0:
        precision = 0
    else:
        precision = tp / (tp + fp)

    if (tp + fn) == 0:
        recall = 0
    else:
        recall = tp / (tp + fn)

    if (precision + recall) == 0:
        f1_score = 0
    else:
        f1_score = 2 * ((precision * recall) / (precision + recall))
    return accuracy, f1_score

def normalize(train, test):
    for col in train.columns[:-1]:

        if train[col].dtype == bool:
            continue

        min = train[col].min()
        max = train[col].max()

        if max != min:
            train[col] = (train[col] - min) / (max - min)
            test[col] = (test[col] - min) / (max - min)
        else:
            train[col] = 0.5
            test[col] = 0.5

    return train, test


def k_cross_validation_evaluation(df, k, layout,learning_rate, ld):

    df = df.dropna()
    target = df.columns[-1]
    target_values = df[target]
    df = df.drop(columns = [target])
    categorical_attributes = df.select_dtypes(exclude = np.number).columns
    df = pd.get_dummies(df, columns = categorical_attributes)
    df[target] = target_values

    print("shape: ", df.shape)

    accuracies = []
    f1_scores = []

    pos = df[df.iloc[:, -1] == 1]
    neg = df[df.iloc[:, -1] == 0]

    pos = shuffle(pos, random_state = 42)
    neg = shuffle(neg, random_state = 42)

    pos_folds = np.array_split(pos.index, k)
    neg_folds = np.array_split(neg.index, k)

    for i in range(0, k):
        test = pd.concat([pos.loc[pos_folds[i]], neg.loc[neg_folds[i]]], ignore_index=True)
        train = pd.DataFrame()
        for j in range(0, k):
            if j != i:
                train = pd.concat([train, pos.loc[pos_folds[j]], neg.loc[neg_folds[j]]], ignore_index=True)

        test = shuffle(test, random_state=i)
        train = shuffle(train, random_state=i)

        train, test = normalize(train, test)
        neurons, weights = make_nn(layout)

        weights = train_nn(train, neurons, weights, learning_rate, ld)
        y_preds = []
        for index, instance in test.iterrows():
            inputs = instance.iloc[:-1].to_numpy()
            neurons, weights = forward(neurons, weights, inputs)
            if (neurons[-1][0] >= 0.5):
                y_preds.append(1)
            else:
                y_preds.append(0)

        accuracy, f1_score = evaluation(test.iloc[:, -1], y_preds)
        accuracies.append(accuracy)
        f1_scores.append(f1_score)

    return np.mean(accuracies), np.mean(f1_scores)

def get_learning_curve(df, layout, learning_rate, ld, sizes):
    df = df.dropna()
    target = df.columns[-1]
    target_values = df[target]
    df = df.drop(columns = [target])
    categorical_attributes = df.select_dtypes(exclude = np.number).columns
    df = pd.get_dummies(df, columns = categorical_attributes)
    df[target] = target_values
    
    pos = df[df.iloc[:, -1] == 1]
    neg = df[df.iloc[:, -1] == 0]
    
    pos = shuffle(pos, random_state = 42)
    neg = shuffle(neg, random_state = 42)

    pos_train = pos.iloc[int(len(pos) * 0.2):]
    pos_test = pos.iloc[:int(len(pos) * 0.2)]

    neg_train = neg.iloc[int(len(neg) * 0.2):]
    neg_test = neg.iloc[:int(len(neg) * 0.2)]

    train = pd.concat([pos_train, neg_train], ignore_index = True)
    test = pd.concat([pos_test, neg_test], ignore_index = True)

    train = shuffle(train, random_state = 42)
    test = shuffle(test, random_state = 42)

    train, test = normalize(train, test)

    costs = []
    for size in sizes:
        train_subset = train.iloc[0:size].copy()

        neurons, weights = make_nn(layout)

        weights = train_nn(train_subset, neurons, weights, learning_rate, ld)

        y_true = []
        y_pred = []

        for index, instance in test.iterrows():
            y_true.append(instance.iloc[-1])
            inputs = instance.iloc[:-1].to_numpy()

            neurons, weights = forward(neurons, weights, inputs)
            y_pred.append(neurons[-1][0])
        j = cost(weights, len(test), y_true, y_pred, ld)
        costs.append(j)

    return sizes, costs



'''

sizes, cost = get_learning_curve(wdbc_df, [30, 20, 1], 0.5, 0.5, [5, 10, 20, 50, 100])
print("sizes: ", sizes)
print("cost: ", cost)
plt.figure()
plt.plot(sizes, cost, marker = 'o')
plt.title("Learning Curve where lambda = 0.5, learning_rate = 0.5, and m = 500 as the stopping criteria")
plt.xlabel("number of training samples")
plt.ylabel("J cost")
plt.savefig("learning_curve.png")
plt.close()
'''


layouts_wdbc = [[30, 2, 1], [30, 10, 1], [30, 20, 1], [30, 5, 10, 1], [30, 10, 5, 1], [30, 20, 5, 1], [30, 10, 15, 1]]
layouts_loan = [[20, 2, 1], [20, 10, 1], [20, 20, 1], [20, 5, 10, 1], [20, 10, 5, 1], [20, 20, 5, 1], [20, 10, 15, 1]]
learning_rates = [0.5]
lds = [0.5]

'''
for layout in layouts_wdbc:
    for learning_rate in learning_rates:
        for ld in lds:
            accuracy, f1 = k_cross_validation_evaluation(wdbc_df, 5, layout, learning_rate, ld)
            print(f"layout, learning rate, ld with 100 iterations:  {layout}, {learning_rate}, {ld}")
            print(f"accuracy = {accuracy:.3f}, F1 score = {f1:.3f}")  

for layout in layouts_loan:
    for learning_rate in learning_rates:
        for ld in lds:
            accuracy, f1 = k_cross_validation_evaluation(loan_df, 5, layout, learning_rate, ld)
            print(f"layout, learning rate, ld with 100 iterations:  {layout}, {learning_rate}, {ld}")
            print(f"accuracy = {accuracy:.3f}, F1 score = {f1:.3f}")        


'''
sizes, cost = get_learning_curve(loan_df, [20, 20, 1], 0.5, 0.5, [5, 10, 20, 50, 100])
print("sizes: ", sizes)
print("cost: ", cost)
plt.figure()
plt.plot(sizes, cost, marker = 'o')
plt.title("Learning Curve where lambda = 0.5, learning_rate = 0.5, and m = 500 as the stopping criteria")
plt.xlabel("number of training samples")
plt.ylabel("J cost")
plt.savefig("learning_curve_loan.png")
plt.close()

sizes, cost = get_learning_curve(raisin_df, [7, 20, 1], 0.5, 0.5, [5, 10, 20, 50, 100])
print("sizes: ", sizes)
print("cost: ", cost)
plt.figure()
plt.plot(sizes, cost, marker = 'o')
plt.title("Learning Curve where lambda = 0.5, learning_rate = 0.5, and m = 500 as the stopping criteria")
plt.xlabel("number of training samples")
plt.ylabel("J cost")
plt.savefig("learning_curve_rasin_dataset.png")
plt.close()



'''
layouts_raisin = [[7, 2, 1], [7, 10, 1], [7, 20, 1], [7, 5, 10, 1], [7, 10, 5, 1], [7, 20, 5, 1], [7, 10, 15, 1]]
for layout in layouts_raisin:
    for learning_rate in learning_rates:
        for ld in lds:
            accuracy, f1 = k_cross_validation_evaluation(raisin_df, 5, layout, learning_rate, ld)
            print(f"layout, learning rate, ld with 100 iterations:  {layout}, {learning_rate}, {ld}")
            print(f"accuracy = {accuracy:.3f}, F1 score = {f1:.3f}") 



layouts_titanic = [[5, 2, 1], [7, 10, 1], [7, 20, 1], [7, 5, 10, 1], [7, 10, 5, 1], [7, 20, 5, 1], [7, 10, 15, 1]]
for layout in layouts_titanic:
    for learning_rate in learning_rates:
        for ld in lds:
            accuracy, f1 = k_cross_validation_evaluation(titanic_df, 5, layout, learning_rate, ld)
            print(f"layout, learning rate, ld with 50 iterations:  {layout}, {learning_rate}, {ld}")
            print(f"accuracy = {accuracy:.3f}, F1 score = {f1:.3f}") 

'''
