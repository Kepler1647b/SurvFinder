from torchvision import transforms
import random
import os
import numpy as np
import torch
import Utils.config_lympho as CONFIG
import itertools
import matplotlib.pyplot as plt
import scipy
from sklearn.metrics import roc_curve, roc_auc_score, auc
import math
from torch.utils.data.sampler import BatchSampler


from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from scipy import interp
from sklearn.metrics import average_precision_score

n_classes = 3

path_add_256 = '/home/21/zihan/Storage/lympho/patch/lym_0.5ol_20x_256'
path_add_512 = '/home/21/zihan/Storage/lympho/patch/lym_0.5ol_20x_512'

def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


class Random90Rotation():
    def __init__(self, degrees=[0, 90, 180, 270]):
        self.degrees = degrees

    def __call__(self, im):
        degree = random.sample(self.degrees, k=1)[0]
        return im.rotate(degree)



transform_train = transforms.Compose([
    transforms.RandomCrop(224),
    Random90Rotation(),
    transforms.RandomVerticalFlip(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomApply([transforms.ColorJitter(brightness=0.8)], p=0.8),
    transforms.RandomApply([transforms.ColorJitter(contrast=0.8)], p=0.8),
    transforms.RandomApply([transforms.ColorJitter(saturation=0.8, hue=0.2)], p=0.8),

    transforms.ToTensor(),
    transforms.Normalize(
        mean=CONFIG.normMean, std=CONFIG.normStd
    ),
])

transform_train_lym3 = transforms.Compose([
    # transforms.Resize(224),
    transforms.Resize(512),
    Random90Rotation(),
    transforms.RandomVerticalFlip(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomApply([transforms.ColorJitter(brightness=0.8)], p=0.8),
    transforms.RandomApply([transforms.ColorJitter(contrast=0.8)], p=0.8),
    transforms.RandomApply([transforms.ColorJitter(saturation=0.8, hue=0.2)], p=0.8),

    transforms.ToTensor(),
    transforms.Normalize(
        mean=CONFIG.normMean, std=CONFIG.normStd
    ),
])

transform_train_512 = transforms.Compose([
    transforms.RandomRotation(360),
    transforms.CenterCrop(224),
    Random90Rotation(),
    transforms.RandomVerticalFlip(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomApply([transforms.ColorJitter(brightness=0.8)], p=0.8),
    transforms.RandomApply([transforms.ColorJitter(contrast=0.8)], p=0.8),
    transforms.RandomApply([transforms.ColorJitter(saturation=0.8, hue=0.2)], p=0.8),

    transforms.ToTensor(),
    transforms.Normalize(
        mean=CONFIG.normMean, std=CONFIG.normStd
    ),
])

transform_valid = transforms.Compose([
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=CONFIG.normMean, std=CONFIG.normStd
    ),
])

transform_valid_lym3 = transforms.Compose([
    # transforms.Resize(224),
    transforms.Resize(512),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=CONFIG.normMean, std=CONFIG.normStd
    ),
])

transform_valid_512 = transforms.Compose([
    transforms.RandomRotation(360),    
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=CONFIG.normMean, std=CONFIG.normStd
    ),
])

transform_test = transforms.Compose([
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=CONFIG.normMean, std=CONFIG.normStd
    ),
])


def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    # https://blog.csdn.net/qq_32425195/article/details/101537049
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    fig = plt.figure(figsize=(12, 6))
    plt.title(title)
    for plot_index in range(2):
        if plot_index == 1:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        plt.subplot(1, 2, plot_index + 1)
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=15);
        plt.yticks(tick_marks, classes)

        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i,
                     cm[i, j] if plot_index == 0 else "{:.2f}%".format(cm[i, j] * 100),
                     horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

        plt.ylabel('True label');
        plt.xlabel('Predicted label')
        plt.tight_layout()
    return fig


def print_conf_matrix(confusion_matrix, classes):
    first_line = [' ']
    first_line.extend(classes)
    first_line = map(lambda s: s.ljust(20), first_line)

    print(''.join(first_line))
    all_acc = 0
    i = 0
    for row, _class in zip(confusion_matrix, classes):
        row_pretty = [_class]
        row_pretty.extend(
            ['{:>5.2f}%({:>5.0f}/{:<5.0f})'.format(number * 100 / sum(row), number, sum(row)) for number in row])
        row_pretty = map(lambda s: s.ljust(20), row_pretty)
        all_acc += (row[i] / sum(row))
        i += 1
        print(''.join(row_pretty))
    return all_acc / len(classes)


def make_weights_for_balanced_classes(label_list, nclasses):
    count = [0] * nclasses
    for item in label_list:
        count[item] += 1
    weight_per_class = [0.] * nclasses
    N = float(sum(count))
    print(N)

    for i in range(nclasses):
        print(i)
        weight_per_class[i] = N / float(count[i])
    print(count)
    print(weight_per_class)
    weight = [0] * len(label_list)
    for idx, val in enumerate(label_list):
        weight[idx] = weight_per_class[val]
    return weight, weight_per_class


def eer_threshold(all_labels, all_values):
    fpr, tpr, threshold = roc_curve(all_labels, all_values, pos_label=1)
    min_val = 999
    min_i = 0
    for i in range(len(fpr)):
        val = abs(fpr[i] + tpr[i] - 1)
        if val < min_val:
            min_val = val
            min_i = i
    print(threshold[min_i], fpr[min_i], tpr[min_i])
    return threshold[min_i]

def bootstrap_auc(all_labels, all_values, n_bootstraps=1000):
    rng_seed = 1  # control reproducibility
    bootstrapped_scores = []

    rng = np.random.RandomState(rng_seed)
    for i in range(n_bootstraps):
        # bootstrap by sampling with replacement on the prediction indices
        indices = rng.randint(0, len(all_values), len(all_values))
        if len(np.unique(all_labels[indices])) < 2:
            # We need at least one positive and one negative sample for ROC AUC
            # to be defined: reject the sample
            continue
        score = roc_auc_score(all_labels[indices], all_values[indices])
        bootstrapped_scores.append(score)
#         print("Bootstrap #{} ROC area: {:0.3f}".format(i + 1, score))
    return bootstrapped_scores

def clopper_pearson(x, n, alpha=0.05):
    """Estimate the confidence interval for a sampled Bernoulli random
    variable.
    `x` is the number of successes and `n` is the number trials (x <=
    n). `alpha` is the confidence level (i.e., the true probability is
    inside the confidence interval with probability 1-alpha). The
    function returns a `(low, high)` pair of numbers indicating the
    interval on the probability.
    """
    b = scipy.stats.beta.ppf
    lo = b(alpha / 2, x, n - x + 1)
    hi = b(1 - alpha / 2, x + 1, n - x)
    return 0.0 if math.isnan(lo) else lo, 1.0 if math.isnan(hi) else hi

def print_result(all_labels, all_values, all_predicts):
    all_auc = bootstrap_auc(all_labels, all_values)
    auc_95_ci = scipy.stats.norm.interval(0.95, np.mean(all_auc), np.std(all_auc))
    print("AUC 95% CI:")
    print("(%.4f, %.4f)" % auc_95_ci)
    TP = ((all_predicts == 1) & (all_labels == 1)).sum()
    TN = ((all_predicts == 0) & (all_labels == 0)).sum()
    FN = ((all_predicts == 0) & (all_labels == 1)).sum()
    FP = ((all_predicts == 1) & (all_labels == 0)).sum()
    sensi_95 = []
    lo, hi = clopper_pearson(TP, TP + FN)
    sensi_95.append(lo)
    sensi_95.append(hi)
    print('Sensitivity 95% confidence interval: ({:.2f}, {:.2f})'.format(lo, hi))
    speci_95 = []
    lo, hi = clopper_pearson(TN, FP + TN)
    speci_95.append(lo)
    speci_95.append(hi)
    print('Specificity 95% confidence interval: ({:.2f}, {:.2f})'.format(lo, hi))
    return auc_95_ci, sensi_95, speci_95

def plot_roc_curve(fpr, tpr):
    roc_auc = auc(fpr, tpr)  ###计算auc的值

    plt.figure()
    lw = 2
    fig = plt.figure(figsize=(10, 10))
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.3f)' % roc_auc)  ###假正率为横坐标，真正率为纵坐标做曲线
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    return fig

class BalancedBatchSampler(BatchSampler):
    def __init__(self, labels, n_classes, n_samplers, batchsize):
        # super(BalancedBatchSampler, self).__init__()
        self.labels = labels
        self.labels_set = list(set(self.labels))
        self.labels_to_indices = {
            label: np.where(self.labels == label)[0] for label in self.labels_set
        }
        for i in self.labels_set:
            np.random.shuffle(self.labels_to_indices[i])

        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samplers
        #self.batch_size = self.n_classes * self.n_samples
        self.batch_size = batchsize
        self.n_dataset = len(self.labels)#总patch数

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < self.n_dataset:
            classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
            indices = []
            for class_ in classes:
                if class_ == classes[-1]:
                    temp = self.labels_to_indices[class_][
                        self.used_label_indices_count[class_]: self.used_label_indices_count[class_] + self.batch_size - (len(classes)-1) * self.n_samples]
                    indices.extend(temp)
                    self.used_label_indices_count[class_] += self.batch_size - (len(classes)-1) * self.n_samples
                else:
                    temp = self.labels_to_indices[class_][
                           self.used_label_indices_count[class_]: self.used_label_indices_count[class_] + self.n_samples]
                    indices.extend(temp)
                    self.used_label_indices_count[class_] += self.n_samples

                if self.used_label_indices_count[class_] + max(self.batch_size - (len(classes)-1) * self.n_samples, self.n_samples) > len(self.labels_to_indices[class_]):
                    np.random.shuffle(self.labels_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
            # print(indices)
            yield indices
            self.count += self.batch_size

    def __len__(self):
        return self.n_dataset // self.batch_size

'''def one_hot(values):
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)
    #print(integer_encoded)
    # binary encode
    onehot_encoder = OneHotEncoder(sparse=False, categories=[list(range(0,n_classes))])
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    #print(onehot_encoded)
    # invert first example
    inverted = label_encoder.inverse_transform([argmax(onehot_encoded[0, :])])
    return onehot_encoded'''

def one_hot(values):
    onehot_a = np.zeros((values.size, n_classes))
    onehot_a[np.arange(values.size), values] = 1
    return onehot_a
    

def multi_auc(onehot_labels, all_values):

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    print(onehot_labels.shape, all_values.shape)
    for i in range(3):
        fpr[i], tpr[i], _ = roc_curve(onehot_labels[:, i], all_values[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    # Finally average it and compute AUC
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    return roc_auc, fpr, tpr

def eer_threshold(all_label, all_value):
    threshold_dic = {}
    for cls in range(n_classes):
        all_values = all_value[:, cls]
        all_labels = all_label[:, cls]
        fpr, tpr, threshold = roc_curve(all_labels, all_values, pos_label=1)
        min_val = 999
        min_i = 0
        for i in range(len(fpr)):
            val = abs(fpr[i] + tpr[i] - 1)
            if val < min_val:
                min_val = val
                min_i = i
        print(threshold[min_i], fpr[min_i], tpr[min_i])
        threshold_dic[cls] = threshold[min_i]
    return threshold_dic


def compute_mAP(y_true, y_pred):
    AP = []
    for i in range(y_true.shape[1]): 
         AP.append(average_precision_score(y_true[:, i], y_pred[:, i]))
    return np.mean(AP)       

