'''
Amit Kumar
Roll No: 22CSM1R02
MTech-CSE, Sem01
DSF Assignment 1
'''
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate # for k-fold-cross-val
from tabulate import tabulate # for table printing
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, jaccard_score

# https://stackoverflow.com/a/30267328
class MultiColumnLabelEncoder:
    def __init__(self,columns = None):
        self.columns = columns # array of column names to encode

    def fit(self,X,y=None):
        return self # not relevant here

    def transform(self,X):
        '''
        Transforms columns of X specified in self.columns using
        LabelEncoder(). If no columns specified, transforms all
        columns in X.
        '''
        output = X.copy()
        if self.columns is not None:
            for col in self.columns:
                output[col] = LabelEncoder().fit_transform(output[col])
        else:
            for colname,col in output.iteritems():
                output[colname] = LabelEncoder().fit_transform(col)
        return output

    def fit_transform(self,X,y=None):
        return self.fit(X,y).transform(X)

def load_data(filename):
    filename = filename.strip()
    try:
        data = pd.read_csv(filename)
    except:
        print('Unable to load data!')

    print('Data successfully loaded from...', filename)
    return data

def clean_data(data):
    '''
    Takes in a pandas DataFrame and does the following on the datagrame
    - removes any null values, by removing the training example
    - convert categorical values to numeric values using LabelEncoder
    - standarize the data
    '''
    if data.isnull().values.any():
        print('Data contains null values! Dropping all training examples with null values!')
        data = data.dropna()

    # converting categorical values using LabelEncoder
    numerics = data.select_dtypes(include=np.number).columns.to_list()
    all_cols = data.columns.to_list()
    non_numeric = [x for x in all_cols if x not in numerics]

    # convert the categorical data to numerical data
    data = MultiColumnLabelEncoder(columns=non_numeric).fit_transform(data)
    y = data['y']
    del data['y']

    # standarizing the dataset using Standard Scaler
    scaler = StandardScaler()
    columns = data.columns
    data = pd.DataFrame(scaler.fit_transform(data), columns=columns)
    data['y'] = y

    return data

def get_x_y(data):
    y = data['y']
    del data['y']
    x = data.copy()
    return (x, y)


# K-Fold Cross-Validation
def cross_validation(model, _X, _y, _cv=5):
    '''Function to perform 5 Folds Cross-Validation
    Parameters
    ----------
    model: Python Class, default=None
          This is the machine learning algorithm to be used for training.
    _X: array
       This is the matrix of features.
    _y: array
       This is the target variable.
    _cv: int, default=5
      Determines the number of folds for cross-validation.
    Returns
    -------
    The function returns a dictionary containing the metrics 'accuracy', 'precision',
    'recall', 'f1' for both training set and validation set
    '''
    # https://stackoverflow.com/a/54718282
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.make_scorer.html
    _scoring = {
        'accuracy': make_scorer(accuracy_score), 
        'precision': make_scorer(precision_score, average='macro'), 
        'recall': make_scorer(recall_score, average='macro'), 
        'f1': make_scorer(f1_score, average='macro'), 
        'jaccard': make_scorer(jaccard_score, average='macro')
    }
    results = cross_validate(
        estimator=model,
        X=_X,
        y=_y,
        cv=_cv,
        scoring=_scoring,
        return_train_score=True
    )

    return {
        "Test Accuracy scores": results['test_accuracy'],
        "Mean Test Accuracy": results['test_accuracy'].mean()*100,
        "Test Precision scores": results['test_precision'],
        "Mean Test Precision": results['test_precision'].mean(),
        "Test Recall scores": results['test_recall'],
        "Mean Test Recall": results['test_recall'].mean(),
        "Test F1 scores": results['test_f1'],
        "Mean Test F1 Score": results['test_f1'].mean(),
        "Test jaccard scores": results['test_jaccard'],
        "Mean Test jaccard Score": results['test_jaccard'].mean(),
    }


# TODO: incomplete
def tabulate_data(one, two, three, four, five, name):
    table = [
        ["Dataset 1",one[0],two[0],three[0],four[0],five[0]],
        ["Dataset 2",one[1],two[1],three[1],four[1],five[1]], 
        ["Dataset 3",one[2],two[2],three[2],four[2],five[2]],
        ["Dataset 4",one[3],two[3],three[3],four[3],five[3]],
        ["Dataset 4",one[4],two[4],three[4],four[4],five[4]]
    ]
    headers=[
        name,
        "Logistic Regression", 
        "Naive Bayes Classifier", 
        "LDA",
        "k-NN", 
        "Decision Tree Classifier"
    ]
    t = tabulate(table, headers=headers, tablefmt="tsv")
    return t


def find_outputs(output):
    dict = {}
    dict['Mean Test Accuracy'] = output['Mean Test Accuracy']
    dict['Mean Test Precision'] = output['Mean Test Precision']
    dict['Mean Test Recall'] = output['Mean Test Recall']
    dict['Mean Test F1 Score'] = output['Mean Test F1 Score']
    dict['Mean Test jaccard Score'] = output['Mean Test jaccard Score']
    return dict


def collectMetric(name, data):
    l = []
    for i in data:
        l.append(i[name])
    return l


# https://stackoverflow.com/a/30333948
def writeToFile(a, p, r, f, j):
    with open('output.txt', 'w') as outputfile:
        outputfile.write(a)
        outputfile.write("\n\n")
        outputfile.write(p)
        outputfile.write("\n\n")
        outputfile.write(r)
        outputfile.write("\n\n")
        outputfile.write(f)
        outputfile.write("\n\n")
        outputfile.write(j)
    outputfile.close()


# https://stackoverflow.com/a/52733551
def writeToCSV(a, p, r, f, j):
    with open("output.csv","w") as outputfile:
        outputfile.write(a)
        outputfile.write("\n\n")
        outputfile.write(p)
        outputfile.write("\n\n")
        outputfile.write(r)
        outputfile.write("\n\n")
        outputfile.write(f)
        outputfile.write("\n\n")
        outputfile.write(j)
    outputfile.close()


def printOutputToFile(lr_final, nb_final, lda_final, knn_final, dt_final):
    lr_accuracy = collectMetric('Mean Test Accuracy', lr_final)
    lr_precision = collectMetric('Mean Test Precision', lr_final)
    lr_recall = collectMetric('Mean Test Recall', lr_final)
    lr_f1 = collectMetric('Mean Test F1 Score', lr_final)
    lr_jaccard = collectMetric('Mean Test jaccard Score', lr_final)

    nb_accuracy = collectMetric('Mean Test Accuracy', nb_final)
    nb_precision = collectMetric('Mean Test Precision', nb_final)
    nb_recall = collectMetric('Mean Test Recall', nb_final)
    nb_f1 = collectMetric('Mean Test F1 Score', nb_final)
    nb_jaccard = collectMetric('Mean Test jaccard Score', nb_final)

    lda_accuracy = collectMetric('Mean Test Accuracy', lda_final)
    lda_precision = collectMetric('Mean Test Precision', lda_final)
    lda_recall = collectMetric('Mean Test Recall', lda_final)
    lda_f1 = collectMetric('Mean Test F1 Score', lda_final)
    lda_jaccard = collectMetric('Mean Test jaccard Score', lda_final)

    knn_accuracy = collectMetric('Mean Test Accuracy', knn_final)
    knn_precision = collectMetric('Mean Test Precision', knn_final)
    knn_recall = collectMetric('Mean Test Recall', knn_final)
    knn_f1 = collectMetric('Mean Test F1 Score', knn_final)
    knn_jaccard = collectMetric('Mean Test jaccard Score', knn_final)

    dt_accuracy = collectMetric('Mean Test Accuracy', dt_final)
    dt_precision = collectMetric('Mean Test Precision', dt_final)
    dt_recall = collectMetric('Mean Test Recall', dt_final)
    dt_f1 = collectMetric('Mean Test F1 Score', dt_final)
    dt_jaccard = collectMetric('Mean Test jaccard Score', dt_final)

    accuracy_table = tabulate_data(lr_accuracy, nb_accuracy, lda_accuracy, knn_accuracy, dt_accuracy, "Accuracy")
    precision_table = tabulate_data(lr_precision, nb_precision, lda_precision, knn_precision, dt_precision, "Precision")
    recall_table = tabulate_data(lr_recall, nb_recall, lda_recall, knn_recall, dt_recall, "Recall")
    f1_table = tabulate_data(lr_f1, nb_f1, lda_f1, knn_f1, dt_f1, "F1 Score")
    jaccard_table = tabulate_data(lr_jaccard, nb_jaccard, lda_jaccard, knn_jaccard, dt_jaccard, "Jaccard Score")

    writeToCSV(
        accuracy_table, 
        precision_table, 
        recall_table, 
        f1_table, 
        jaccard_table
    )

    return (accuracy_score, precision_table, recall_table, f1_table, jaccard_table)
    