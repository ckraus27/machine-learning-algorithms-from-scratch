import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import numpy as np
import matplotlib.pyplot as plt
import random
from pandas.api.types import is_numeric_dtype

wdbc_df = pd.read_csv("wdbc.csv")
loan_df = pd.read_csv("loan.csv")

#print(wdbc_df.dtypes)
#print(loan_df.dtypes)

def entropy(df, target):
    
    counts = df[target].value_counts()
    ent = 0

    for count in counts:
        x = count / counts.sum()
        ent -= x * np.log2(x)
    
    return ent

def dt(df, features, min_gain):

    #features = list(df.columns[:-1])
    target  = df.columns[-1]
    m = int(np.sqrt(len(features)))
    random_features = random.sample(features, m)
    
    if(len(df.iloc[:, -1].unique()) == 1):
        return df.iloc[0, -1]
    
    ent_init = entropy(df, df.columns[-1])
    max_gain = -1
    max_feature = None

    for feature in random_features:
        ent = 0
        total = len(df)

        if is_numeric_dtype(df[feature]) == False:

            for value, group in df.groupby(feature):
                weight = len(group) / total
                ent += weight * entropy(group, target)
        
            information_gain = ent_init - ent

        else:
            left = df[df[feature] <= df[feature].mean()]
            right = df[df[feature] > df[feature].mean()]
            left_weight = len(left) / total
            right_weight = len(right) / total
            ent = left_weight * entropy(left, target) + right_weight * entropy(right, target)
            information_gain = ent_init - ent

        
        if (information_gain > max_gain):
            max_gain = information_gain
            max_feature = feature

            if is_numeric_dtype(df[max_feature]):
                threshold = df[max_feature].mean()
                best_left = left
                best_right = right

    if max_gain <= min_gain:
        return df[target].value_counts().idxmax()

    if is_numeric_dtype(df[max_feature]) == False:
    
        tree = {max_feature: {}}

        for value, group in df.groupby(max_feature):
            tree[max_feature][value] = dt(group, features, min_gain)
    else:
        tree = {max_feature: {}}
        tree[max_feature]["threshold"] = threshold
        tree[max_feature]["left"] = dt(best_left, features, min_gain)
        tree[max_feature]["right"] = dt(best_right, features, min_gain)
    
    return tree

def predict(tree, instance):

    if type(tree) != dict:
        return tree
    
    feature = next(iter(tree))
    value = instance[feature]

    if "threshold" in tree[feature]:
        if value <= tree[feature]["threshold"]:
            return predict(tree[feature]["left"], instance)
        else:
            return predict(tree[feature]["right"], instance)
    else:
        if value not in tree[feature]:
            return None


    return predict(tree[feature][value], instance)

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
    return accuracy, precision, recall, f1_score

def make_forest(df, n_trees, min_gain):
    forest = []
    for i in range(0, n_trees):
        bootstrap_df = df.sample(n = len(df), replace = True, random_state=i)
        tree = dt(bootstrap_df, list(bootstrap_df.columns[:-1]), min_gain)
        forest.append(tree)

    return forest

def vote(forest, instance):
    cnt = 0
    for tree in forest:
        if (predict(tree, instance) == 1):
            cnt += 1

    if cnt > (len(forest) / 2):
        return 1
    return 0

def k_cross_validation_evaluation(df, k, n_trees, min_gain):
    accuracies = []
    precisions = []
    recalls = []
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

        forest = make_forest(df = train, n_trees = n_trees, min_gain = min_gain)
        y_pred = []
        for index, instance in test.iterrows():
            y_pred.append(vote(forest, instance))

        accuracy, precision, recall, f1_score = evaluation(test.iloc[:, -1], y_pred)
        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1_score)

    return np.mean(accuracies), np.mean(precisions), np.mean(recalls), np.mean(f1_scores)

n_trees = [1, 5, 10, 20, 30, 40, 50]
min_gain = [0.001]
wdbc_accuracies = []
wdbc_precisions = []
wdbc_recalls = []
wdbc_f1_scores = []
loan_accuracies = []
loan_precisions = []
loan_recalls = []
loan_f1_scores = []


for n_tree in n_trees:
    for gain in min_gain:
        wdbc_accuracy, wdbc_precision, wdbc_recall, wdbc_f1_score = k_cross_validation_evaluation(df=wdbc_df, k=5, n_trees=n_tree, min_gain = gain)
        loan_accuracy, loan_precision, loan_recall, loan_f1_score = k_cross_validation_evaluation(df=loan_df, k=5, n_trees=n_tree, min_gain = gain)

        wdbc_accuracies.append(wdbc_accuracy)
        wdbc_precisions.append(wdbc_precision)
        wdbc_recalls.append(wdbc_recall)
        wdbc_f1_scores.append(wdbc_f1_score)
        loan_accuracies.append(loan_accuracy)
        loan_precisions.append(loan_precision)
        loan_recalls.append(loan_recall)
        loan_f1_scores.append(loan_f1_score)

        print(f"n_tree: {n_tree} with  the minimum_gain criterion: {gain} has an accuracy, precision, recall, and f1 score of {wdbc_accuracy:.3f}, {wdbc_precision:.3f}, {wdbc_recall:.3f}, {wdbc_f1_score:.3f} using the wdbc dataset and an accuracy, precision, recall, and f1 score of {loan_accuracy:.3f}, {loan_precision:.3f}, {loan_recall:.3f}, {loan_f1_score:.3f} using the loan dataset.")


#figure 1
plt.figure()
plt.plot(n_trees, wdbc_accuracies, marker = 'o')
plt.title("Accuracy using different number of trees on wdbc dataset")
plt.xlabel("number of trees")
plt.ylabel("Accuracy")
plt.savefig("wdbc_accuracy.png")
plt.close()

#figure 2
plt.figure()
plt.plot(n_trees, wdbc_precisions, marker = 'o')
plt.title("Precision using different number of trees on wdbc dataset")
plt.xlabel("number of trees")
plt.ylabel("Precision")
plt.savefig("wdbc_precision.png")
plt.close()

#figure 3
plt.figure()
plt.plot(n_trees, wdbc_recalls, marker = 'o')
plt.title("Recall using different number of trees on wdbc dataset")
plt.xlabel("number of trees")
plt.ylabel("Recall")
plt.savefig("wdbc_recall.png")
plt.close()

#figure 4
plt.figure()
plt.plot(n_trees, wdbc_f1_scores, marker = 'o')
plt.title("F1 score using different number of trees on wdbc dataset")
plt.xlabel("number of trees")
plt.ylabel("F1 score")
plt.savefig("wdbc_f1_score.png")
plt.close()

#figure 5
plt.figure()
plt.plot(n_trees, loan_accuracies, marker = 'o')
plt.title("Accuracy using different number of trees on loan dataset")
plt.xlabel("number of trees")
plt.ylabel("Accuracy")
plt.savefig("loan_accuracy.png")
plt.close()

#figure 6
plt.figure()
plt.plot(n_trees, loan_precisions, marker = 'o')
plt.title("Precision using different number of trees on loan dataset")
plt.xlabel("number of trees")
plt.ylabel("Precision")
plt.savefig("loan_precision.png")
plt.close()

#figure 7
plt.figure()
plt.plot(n_trees, loan_recalls, marker = 'o')
plt.title("Recall using different number of trees on loan dataset")
plt.xlabel("number of trees")
plt.ylabel("Recall")
plt.savefig("loan_recall.png")
plt.close()

#figure 8
plt.figure()
plt.plot(n_trees, loan_f1_scores, marker = 'o')
plt.title("F1 score using different number of trees on loan dataset")
plt.xlabel("number of trees")
plt.ylabel("F1 score")
plt.savefig("loan_f1_score.png")
plt.close()