from utils import load_training_set, load_test_set
import math
import matplotlib.pyplot as plt

def train(pos_train, neg_train, vocab):

    cnt_pos = {}
    cnt_neg = {}

    for review in pos_train:
        for word in review:
            if word not in cnt_pos:
                cnt_pos[word] = 0
            cnt_pos[word] += 1
    
    for review in neg_train:
        for word in review:
            if word not in cnt_neg:
                cnt_neg[word] = 0
            cnt_neg[word] += 1
    
    return cnt_pos, cnt_neg, len(pos_train), len(neg_train), len(vocab), sum(cnt_pos.values()), sum(cnt_neg.values())


    
def predict(review, cnt_pos, cnt_neg, num_pos_rev, num_neg_rev, num_vocab_words, num_pos_words, num_neg_words, a):

    pr_pos = num_pos_rev / (num_pos_rev + num_neg_rev)
    pr_neg = num_neg_rev / (num_pos_rev + num_neg_rev)

    #question 2
    pr_pos_given_review = math.log(pr_pos)
    pr_neg_given_review = math.log(pr_neg)
    
    #question 1
    #pr_pos_given_review = pr_pos
    #pr_neg_given_review = pr_neg

    for word in review:
        if word not in cnt_pos and word not in cnt_neg:
            continue

        if word not in cnt_pos:
            #question 2
            pr_pos_given_review += math.log(a / (num_pos_words + a * num_vocab_words))
            
            #question 1
            #pr_pos_given_review = 0
        else:
            #question 2
            pr_pos_given_review += math.log((cnt_pos[word] + a) / (num_pos_words + a * num_vocab_words))
            
            #question 1
            #pr_pos_given_review *= (cnt_pos[word]) / (num_pos_words)

        if word not in cnt_neg:
            #question 2
            pr_neg_given_review += math.log(a / (num_neg_words + a * num_vocab_words))
            
            #question 1
            #pr_neg_given_review = 0
        else:
            #question 2
            pr_neg_given_review += math.log((cnt_neg[word] + a) / (num_neg_words + a * num_vocab_words))
            
            #question 1
            #pr_neg_given_review *= (cnt_neg[word]) / (num_neg_words)
    
    #print(f"pr_pos_given_review: {pr_pos_given_review} vs pr_neg_given_review: {pr_neg_given_review}")
    if (pr_pos_given_review > pr_neg_given_review):
        return "positive"
    return "negative"

if __name__ == '__main__':
    
    #question 3
    '''
    percentage_positive_instances_train = 1.0
    percentage_negative_instances_train = 1.0
    '''
    
    #question 4
    '''
    percentage_positive_instances_train = 0.3
    percentage_negative_instances_train = 0.3   
    '''

    #question 6
    percentage_positive_instances_train = 0.1
    percentage_negative_instances_train = 0.5

    percentage_positive_instances_test = 1.0
    percentage_negative_instances_test = 1.0

    (pos_train, neg_train, vocab) = load_training_set(percentage_positive_instances_train, percentage_negative_instances_train)
    (pos_test, neg_test) = load_test_set(percentage_positive_instances_test, percentage_negative_instances_test)

    print("Number of positive training instances:", len(pos_train))
    print("Number of negative training instances:", len(neg_train))
    print("Number of positive test instances:", len(pos_test))
    print("Number of negative test instances:", len(neg_test))

    cnt_pos, cnt_neg, num_pos_rev, num_neg_rev, num_vocab_words, num_pos_words, num_neg_words = train(pos_train, neg_train, vocab)

    tp = tn = fp = fn = 0

    #question 3, 4, 6   
    for review in pos_test:
        if (predict(review, cnt_pos, cnt_neg, num_pos_rev, num_neg_rev, num_vocab_words, num_pos_words, num_neg_words, 10) == "positive"):
            tp += 1
        else:
            fn += 1
    
    for review in neg_test:
        if (predict(review, cnt_pos, cnt_neg, num_pos_rev, num_neg_rev, num_vocab_words, num_pos_words, num_neg_words, 10) == "negative"):
            tn += 1
        else:
            fp += 1
    
    accuracy = (tp + tn) / (len(pos_test) + len(neg_test))
    if (tp + fp) == 0:
        precision = 0
    else:
        precision = tp / (tp + fp)
    
    if (tp + fn) == 0:
        recall = 0
    else:
        recall = tp / (tp + fn)
            
    print(f"accuracy: {accuracy}, precision: {precision}, recall: {recall}")
    print(f"tp: {tp}, tn: {tn}, fp: {fp}, fn: {fn}")

#question 2
'''
    alphas = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
    accuracies = []

    for a in alphas:
        tp = tn = fp = fn = 0
        
        for review in pos_test:
            if (predict(review, cnt_pos, cnt_neg, num_pos_rev, num_neg_rev, num_vocab_words, num_pos_words, num_neg_words, a) == "positive"):
                tp += 1
            else:
                fn += 1
    
        for review in neg_test:
            if (predict(review, cnt_pos, cnt_neg, num_pos_rev, num_neg_rev, num_vocab_words, num_pos_words, num_neg_words, a) == "negative"):
                tn += 1
            else:
                fp += 1
    
        accuracy = (tp + tn) / (len(pos_test) + len(neg_test))
        
        if a == 1:
            if (tp + fp) == 0:
                precision = 0
            else:
                precision = tp / (tp + fp)
    
            if (tp + fn) == 0:
                recall = 0
            else:
                recall = tp / (tp + fn)
            
            print(f"accuracy: {accuracy}, precision: {precision}, recall: {recall}")
            print(f"tp: {tp}, tn: {tn}, fp: {fp}, fn: {fn}")
        accuracies.append(accuracy)
    
    plt.figure()
    plt.plot(alphas, accuracies, marker = 'o')
    plt.title("Accuracy using each alpha")
    plt.xlabel("Alpha")
    plt.ylabel("Accuracy")
    plt.xscale('log')
    plt.savefig("testing_accuracy.png")
    plt.close()
'''
    