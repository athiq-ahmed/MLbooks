# Evalutaion metrics
def accuracy(y_true, y_pred):
    #initialize a simple counter for correct predictions
    correct_counter = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == yp:
            correct_counter += 1
    return correct_counter/len(y_true)

from sklearn import metrics
l1 = [0,1,1,1,0,0,0,1]
l2 = [0,1,0,1,0,1,0,0]
metrics.accuracy_score(l1,l2)

def true_positives(y_true, y_pred):
    tp = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == 1 and yp == 1:
            tp += 1
    return tp

def true_negatives(y_true, y_pred):
    tn = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == 0 and yp == 0:
            tn +=1
    return tn

def false_positives(y_true, y_pred): # Type 1 error
    fp = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == 0 and yp == 1:
            fp +=1
    return fp

def false_negatives(y_true, y_pred): # Type 2 error
    fn = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == 1 and yp == 0:
            fn +=1
    return fn

l1 = [0,1,1,1,0,0,0,1]
l2 = [0,1,0,1,0,1,0,0]

true_positives(l1, l2)
true_negatives(l1, l2)
false_positives(l1, l2)
false_negatives(l1, l2)

# Accuracy_score = (TP+TN)/(TP+TN+FP+FN)
def accuracy_v2(y_true, y_pred):
    tp = true_positives(y_true, y_pred)
    tn = true_negatives(y_true, y_pred)
    fp = false_positives(y_true, y_pred)
    fn = false_negatives(y_true, y_pred)
    accuracy_score = (tp+tn)/(tp+tn+fp+fn)
    return accuracy_score

l1 = [0,1,1,1,0,0,0,1]
l2 = [0,1,0,1,0,1,0,0]
accuracy_v2(l1, l2)
metrics.accuracy_score(l1,l2)

# Precision: TP/(TP+FP)
"""
Let’s say we make a new model on the new skewed dataset and our model correctly
identified 80 non-pneumothorax out of 90 and 8 pneumothorax out of 10. Thus, we
identify 88 images out of 100 successfully. The accuracy is, therefore, 0.88 or 88%.  B
ut, out of these 100 samples, 10 non-pneumothorax images are misclassified as  having pneumothorax
and 2 pneumothorax are misclassified as not having  pneumothorax.

Thus, we have:  
- TP: 8  
- TN: 80  
- FP: 10  
- FN: 2  

So, our precision is 8/ (8 + 10) = 0.444. 
This means our model is correct 44.4%  times when it’s trying to identify positive samples (pneumothorax).

"""


def precision(y_true, y_pred):
    tp = true_positives(y_true, y_pred)
    fp = false_positives(y_true, y_pred)
    precision = tp/(tp+fp)
    return precision

l1 = [0,1,1,1,0,0,0,1]
l2 = [0,1,0,1,0,1,0,0]
precision(l1, l2) # This means our model is correct 67% times when its trying to identify positive samples

# Recall: TP/(TP+FN)
"""
In the above case recall is 8/ (8 + 2) = 0.80. This means our model identified 80%  of positive samples correctly. 

"""

def recall(y_true, y_pred):
    tp = true_positives(y_true, y_pred)
    fn = false_negatives(y_true, y_pred)
    recall = tp/(tp+fn)
    return recall

l1 = [0,1,1,1,0,0,0,1]
l2 = [0,1,0,1,0,1,0,0]
recall(l1, l2) # This means our model identified 50%  of positive samples correctly.

# Precision recall curve
y_true = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0]
y_pred = [0.02638412, 0.11114267, 0.31620708,
          0.0490937, 0.0191491, 0.17554844,
          0.15952202, 0.03819563, 0.11639273,
          0.079377, 0.08584789, 0.39095342,
          0.27259048, 0.03447096, 0.04644807,
          0.03543574, 0.18521942, 0.05934905,
          0.61977213, 0.33056815]
precisions = []
recalls = []
thresholds = [0.0490937, 0.05934905, 0.079377,  0.08584789, 0.11114267,
              0.11639273,  0.15952202, 0.17554844, 0.18521942,  0.27259048,
              0.31620708, 0.33056815,  0.39095342, 0.61977213]

# for every threshold, calculate predictions in binary
# and append calculated precisions and recalls
# to their respective lists
for i in thresholds:
    temp_prediction = [1 if x >=i else 0 for x in y_pred]
    p = precision(y_true, temp_prediction)
    r = recall(y_true, temp_prediction)
    precisions.append(p)
    recalls.append(r)

import matplotlib.pyplot as plt
plt.figure(figsize=(7,7))
plt.plot(recalls,precisions)
plt.xlabel('Recall', fontsize=15)
plt.ylabel('Precision', fontsize=15)
plt.show()

# F1 score
"""
Simple weighted harmonic mean of precision and recall
F1 = 2PR/(P+R)
F1 = 2TP/(2TP+FP+FN)

"""

def f1(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    score = 2 * p * r /(p+r)
    return score

y_true = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0]
y_pred = [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0]
f1(y_true, y_pred)

metrics.f1_score(y_true,y_pred)


# True positive rate/recall/sensitivity = TP/(TP+FN)
def tpr(y_true, y_pred):
    return recall(y_true, y_pred)

# False positive rate = FP/(TN+FP)
"""
1-FPR = Specificity or True negative rate
"""
def fpr(y_true, y_pred):
    fp = false_positives(y_true, y_pred)
    tn = true_negatives(y_true, y_pred)
    return fp/(tn+fp)

# Calcuate TPR and FPR
tpr_list = []
fpr_list = []

#actual targets
y_true = [0, 0, 0, 0, 1, 0, 1,  0, 0, 1, 0, 1, 0, 0, 1]

#predicted probabilities of a sample being 1
y_pred = [0.1, 0.3, 0.2, 0.6, 0.8, 0.05,  0.9, 0.5, 0.3, 0.66, 0.3, 0.2,  0.85, 0.15, 0.99]

#handmade thresholds
thresholds = [0, 0.1, 0.2, 0.3, 0.4, 0.5,  0.6, 0.7, 0.8, 0.85, 0.9, 0.99, 1.0]

# loop over thresholds
for thresh in thresholds:
    # calcuate predictions for a given threshold
    temp_pred = [1 if x >= thresh else 0 for x in y_pred]
    # calculate tpr
    temp_tpr = tpr(y_true, temp_pred)
    # calculate fpr
    temp_fpr = fpr(y_true, temp_pred)
    tpr_list.append(temp_tpr)
    fpr_list.append(temp_fpr)

len(thresholds)
len(tpr_list)
len(fpr_list)

import pandas as pd
df = pd.DataFrame({'threshold': thresholds,
                   'tpr': tpr_list,
                   'fpr': fpr_list})
df

plt.figure()
plt.title('Receiver Operating Curve (ROC)')
plt.fill_between(fpr_list, tpr_list, alpha=0.4)
plt.plot(fpr_list, tpr_list)
plt.xlim(0,1.0)
plt.ylim(0,1.0)
plt.xlabel('FPR', fontsize=15)
plt.ylabel('TPR', fontsize=15)
plt.show()

# Area under ROC
from sklearn import metrics
y_true = [0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1]
y_pred = [0.1, 0.3, 0.2, 0.6, 0.8, 0.05, 0.9, 0.5, 0.3, 0.66, 0.3, 0.2, 0.85, 0.15, 0.99]
metrics.roc_auc_score(y_true, y_pred)

# How to choose the best threshold
"""
High threshold indicates less of False positives and more of False negatives and vice versa
"""
#empty lists to store true positive and false positive values
tp_list = []
fp_list = []

#actual targets
y_true = [0, 0, 0, 0, 1, 0, 1,  0, 0, 1, 0, 1, 0, 0, 1]

#predicted probabilities of a sample being 1
y_pred = [0.1, 0.3, 0.2, 0.6, 0.8, 0.05,  0.9, 0.5, 0.3, 0.66, 0.3, 0.2,  0.85, 0.15, 0.99]

#some handmade thresholds
thresholds = [0, 0.1, 0.2, 0.3, 0.4, 0.5,  0.6, 0.7, 0.8, 0.85, 0.9, 0.99, 1.0]

# loop over thresholds
for thresh in thresholds:
    # calculate predictions for a given threshold
    temp_pred = [1 if x >= thresh else 0 for x in y_pred]
    # calculate tp
    temp_tp = true_positives(y_true, temp_pred)
    # calculate fp
    temp_fp = false_positives(y_true, temp_pred)
    tp_list.append(temp_tp)
    fp_list.append(temp_fp)

df = pd.DataFrame({'threshold': thresholds,
                   'tp': tp_list,
                   'fp': fp_list})
df

plt.figure()
plt.title('Area under Receiver Operating Curve')
plt.fill_between(fpr_list, tpr_list, alpha=0.4)
plt.plot(fp_list, tp_list)
plt.xlim(0,1.0)
plt.ylim(0,1.0)
plt.xlabel('FP', fontsize=15)
plt.ylabel('TP', fontsize=15)
plt.show()

# log loss = -1.0 * (target * log(prediction)) + (1-target) * (log(1-prediction))
import numpy as np
def log_loss(y_true, y_proba):
    epsilon = 1e-15
    loss = []
    for yt, yp in zip(y_true, y_proba):
        # adjust probability
        # 0 gets converted to 1e-15
        # 1 gets converted to 1-1e-15
        yp = np.clip(yp, epsilon, 1-epsilon)
        # calculate loss for one sample
        temp_loss = -1.0 * (yt * np.log(yp) + (1-yt) * np.log(1-yp))
        loss.append(temp_loss)
    return np.mean(loss)

y_true =  [0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1]
y_proba = [0.1, 0.3, 0.2, 0.6, 0.8, 0.05, 0.9, 0.5, 0.3, 0.66, 0.3, 0.2,0.85, 0.15, 0.99]

log_loss(y_true, y_proba)

from sklearn import metrics
metrics.log_loss(y_true, y_proba)

# Metrics for multi-class classification
"""
- Macro averaged precision: calculate precision for all classes individually and then average them
- Micro averaged precision: calculate class wise true positive and false  positive and then use that to 
    calculate overall precision
- Weighted precision: same as macro but in this case, it is weighted average  depending on the number 
    of items in each class

"""

# Macro averaged precision
def macro_precision(y_true, y_pred):
    # find the number of classes
    num_classes = len(np.unique(y_true))
    # Initialize precision to zero
    precision = 0
    # Loop over all classes
    for class_ in range(num_classes):
        # all classes except current are considered negative
        temp_true = [1 if p == class_ else 0 for p in y_true]
        temp_pred = [1 if p == class_ else 0 for p in y_pred]

        # calculate true positive for current class
        tp = true_positives(temp_true, temp_pred)

        # calculate false positive for current class
        fp = false_positives(temp_true, temp_pred)

        # calculate precision for current class
        temp_precision = tp / (tp+fp)

        # keep adding precision for all classes
        precision += temp_precision

    # calculate and return average average precision for all classes
    precision /= num_classes
    return  precision


# Micro averaged precision
def micro_precision(y_true, y_pred):
    num_classes = len(np.unique(y_true))
    tp = 0
    fp = 0
    for class_ in range(num_classes):
        # all classes except current are considered negative
        temp_true = [1 if p == class_ else 0 for p in y_true]
        temp_pred = [1 if p == class_ else 0 for p in y_pred]

        # calculate true positive for current class and update overall tp
        tp += true_positives(temp_true, temp_pred)

        # calculate false positive for current class and update overall fp
        fp += false_positives(temp_true, temp_pred)

    # calculate and return overall precision
    precision = tp / (tp + fp)
    return precision

from collections import Counter
import numpy as np

def weighted_precision(y_true, y_pred):
    num_classes = len(np.unique(y_true))

    # create class:sample count dictionary
    # #it looks something like this:  #{0: 20, 1:15, 2:21}
    class_counts = Counter(y_true)

    precision = 0

    for class_ in range(num_classes):
        # all classes except current are considered negative
        temp_true = [1 if p == class_ else 0 for p in y_true]
        temp_pred = [1 if p == class_ else 0 for p in y_pred]

        # calculate true positive for current class
        tp = true_positives(temp_true, temp_pred)

        # calculate false positive for current class
        fp = false_positives(temp_true, temp_pred)

        # calculate precision for current class
        temp_precision = tp / (tp + fp)

        # multiple precision with count of samples in class
        weighted_precision = class_counts[class_] * temp_precision

        # add to overall precision
        precision += weighted_precision

        #calculate overall precision by dividing by  #total number of samples
        overall_precision = precision/ len(y_true)
    return  overall_precision


from sklearn import metrics
y_true = [0, 1, 2, 0, 1, 2, 0, 2, 2]
y_pred = [0, 2, 1, 0, 2, 1, 0, 0, 2]
macro_precision(y_true, y_pred)
metrics.precision_score(y_true, y_pred, average="macro")

micro_precision(y_true, y_pred)
metrics.precision_score(y_true, y_pred, average="micro")

weighted_precision(y_true, y_pred)
metrics.precision_score(y_true, y_pred, average="weighted")

from collections import Counter
import numpy as np

def weighted_f1(y_true, y_pred):
    num_classes = len(np.unique(y_true))

    # create class:sample count dictionary
    # #it looks something like this:  #{0: 20, 1:15, 2:21}
    class_counts = Counter(y_true)

    f1 = 0

    for class_ in range(num_classes):
        # all classes except current are considered negative
        temp_true = [1 if p == class_ else 0 for p in y_true]
        temp_pred = [1 if p == class_ else 0 for p in y_pred]

        # calculate precision and recall for each class
        p = precision(temp_true, temp_pred)
        r = recall(temp_true, temp_pred)

        # calculate f1 for each class
        if p + r != 0:
            temp_f1 = 2 * p * r / (p+r)
        else:
            temp_f1 = 0

        # multiply f1 with count of samples in class
        weighted_f1 = class_counts[class_] * temp_f1

        # add to f1 precision
        f1 += weighted_f1

    # calculate overall f1 by dividing by total number of samples
    overall_f1 = f1 / len(y_true)
    return overall_f1


from sklearn import metrics
y_true = [0, 1, 2, 0, 1, 2, 0, 2, 2]
y_pred = [0, 2, 1, 0, 2, 1, 0, 0, 2]
weighted_f1(y_true, y_pred)
metrics.f1_score(y_true,y_pred, average = "weighted")


# confusion matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics

y_true = [0, 1, 2, 0, 1, 2, 0, 2, 2]
y_pred = [0, 2, 1, 0, 2, 1, 0, 0, 2]
cm = metrics.confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10,10))
cmap = sns.cubehelix_palette(50, hue=0.05, rot=0, light=0.9, dark=0, as_cmap=True)
sns.set(font_scale=2.5)
sns.heatmap(cm ,annot=True, cmap=cmap, cbar=False)
plt.xlabel('Actual Labels', fontsize=20)
plt.ylabel('Predicted Labels', fontsize=20)

# Metrics for multi-label classification
# Precision at K (P@K)
""""
If you have a list of original classes for a given  sample and list of predicted classes for the same, 
precision is defined as the number  of hits in the predicted list considering only top-k predictions, 
divided by k. 
"""

def pk(y_true, y_pred, k):
    # if k is 0, return 0. we should never have this as k is always >=1
    if k==0:
        return 0
    y_pred = y_pred[:k] # we are interested only in top k predictions
    pred_set = set(y_pred) # convert predictions to set
    true_set = set(y_true) # convert actual values to set
    common_values = pred_set.intersection(true_set)  # find common values
    return len(common_values)/len(y_pred[:k])

# Average precision at k (AP@K)
"""
AP@k is calculated using P@k.  For example, if we have to calculate AP@3, 
we calculate AP@1, AP@2 and AP@3  and then divide the sum by 3. 
"""
def apk(y_true, y_pred, k):
    pk_values = [] #initialize p@k list of values
    for i in range(1, k+1):  # loop over all k. from 1 to k+1
        pk_values.append(pk(y_true, y_pred, i)) # calculate pk@i and append to list

    if len(pk_values)==0: # if we have no values in the list, return 0
        return 0
    else:
        return  sum(pk_values)/len(pk_values)

y_true = [[1, 2, 3], [0, 2], [1], [2, 3], [1, 0], [] ]
y_pred = [[0, 1, 2], [1], [0, 2, 3], [2, 3, 4, 0],[0, 1, 2],[0] ]
for i in range(len(y_true)):
    for j in range(1,4):
        print(f""" 
            y_true={y_true[i]},
            y_pred={y_pred[i]},  
            AP@{j}={apk(y_true[i], y_pred[i], k=j)} """ )
        y_true = [1, 2, 3], y_pred = [0, 1, 2], AP @ 1 = 0.0
        y_true = [1, 2, 3], y_pred = [0, 1, 2], AP @ 2 = 0.25
        y_true = [1, 2, 3], y_pred = [0, 1, 2], AP @ 3 = 0.38888888888888884..

# Mean average precision at k (MAP@k)
"""
Its an average of AP@K
"""
def mapk(y_true, y_pred, k):
    """  This function calculates mean avg precision at k  for a single sample
    :param y_true: list of values, actual classes
    :param y_pred: list of values, predicted classes
    :return: mean avg precision at a given value k
    """

    apk_values = [] #initialize empty list for apk values
    for i in range(len(y_true)):  #loop over all samples
        apk_values.append(apk(y_true[i], y_pred[i], k=k))
    return sum(apk_values)/len(apk_values)

y_true = [[1, 2, 3],[0, 2], [1], [2, 3],[1, 0], []]
y_pred = [[0, 1, 2],[1], [0, 2, 3], [2, 3, 4, 0],[0, 1, 2],[0]]
mapk(y_true, y_pred, k=1)
mapk(y_true, y_pred, k=2)
mapk(y_true, y_pred, k=3)
mapk(y_true, y_pred, k=4)

# Different implementation of p@k, Ap@k
def apk(actual, predicted, k=10):
    if len(predicted)>k:
        predicted=predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1
            score += num_hits/(i+1.0)
        if not actual:
            return 0.0
    return score/min(len(actual), k)

# Metrics for regression
"""
Error = True value - Predicted value

Absolute Error = Abs(True value - Predicted value)

Mean absolute error = mean(errors)
"""

# Mean absolute error
import numpy as np
def mean_absolute_error(y_true, y_pred):
    error = 0
    for yt, yp in zip(y_true, y_pred):
        error += np.abs(yt-yp)
    return error/len(y_true)

# mean square error
def mean_square_error(y_true, y_pred):
    error = 0
    for yt, yp in zip(y_true, y_pred):
        error += (yt-yp) ** 2
    return error/len(y_true)

# Squared logarithmic error(SLE)
# If we take the mean of SLE across all samples then it is mean squared logarithmic error (msle)
import numpy as np
def mean_squared_log_error(y_true, y_pred):
    error = 0
    for yt, yp in zip(y_true, y_pred):
        error += (np.log(1+yt) - np.log(1+yp)) ** 2
    return error/len(y_true)

# Percentage error: ((True value - Predicted value)/True value) * 100
# Mean percentage error
def mean_percentage_error(y_true, y_pred):
    error = 0
    for yt, yp in zip(y_true, y_pred):
        error += (yt - yp) / yt
    return error/len(y_true)

# Mean absolute percentage error (MAPE)
def mean_abs_percentage_error(y_true, y_pred):
    error = 0
    for yt, yp in zip(y_true, y_pred):
        error += np.abs(yt - yp) / yt
    return error/len(y_true)

# R square
import numpy as np
def r2(y_true, y_pred):
    mean_true_value = np.mean(y_true)
    numerator = 0
    denominator = 0
    for yt,yp in zip(y_true, y_pred):
        numerator += (yt - yp)** 2
        denominator += (yt - mean_true_value) ** 2
        ratio = numerator/denominator
    return 1-ratio

# mean absoulte error
def mae_np(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

# Quadratic weighted kappa / QWK / Cohen's Kappa
from sklearn import metrics
y_true = [1, 2, 3, 1, 2, 3, 1, 2, 3]
y_pred = [2, 1, 3, 1, 2, 3, 3, 1, 2]
metrics.cohen_kappa_score(y_true, y_pred, weights="quadratic")
metrics.accuracy_score(y_true, y_pred)

# Matthew's correlation coeffecient (MCC)
def mcc(y_true, y_pred):
    tp = true_positives(y_true, y_pred)
    tn = true_negatives(y_true, y_pred)
    fp = false_positives(y_true, y_pred)
    fn = false_negatives(y_true, y_pred)

    numerator = (tp * tn) - (fp * fn)
    denominator = (
        (tp+fp) *
        (fn+tn) *
        (fp+tn) *
        (tp+fn)
    )
    denominator = denominator ** 0.5
    return numerator/denominator

