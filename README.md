# DSF-Assignment-01
Applying 5 Classification-Algorithms on 5 Datasets using k-fold-cross-validation(**k=5**) and comparing results of 5 Metrics

## Algorithms Used
1. Logistic Regression
2. Naive Bayes Classifier
3. LDA
4. k-NN
5. Decision Tree Classifier

## Datasets Used
1. https://www.kaggle.com/datasets/vivovinco/league-of-legends-champion-stats
2. https://www.kaggle.com/datasets/islombekdavronov/creditscoring-data
3. https://www.kaggle.com/datasets/uciml/iris
4. https://www.kaggle.com/datasets/whenamancodes/hr-employee-attrition
5. https://www.kaggle.com/datasets/rashikrahmanpritom/heart-attack-analysis-prediction-dataset

## Installing and running(virtualenv method)
- Download this repository, unzip it, and `cd` into this repository
- Make sure you have virtualenv installed. If not use  `pip install virtualenv`
- Create a new virtualenvironment with `python3 -m pip install virtualenv venv`
- Activate the virtualenv using
    - `sourve venv/local/bin/activate`, for linux, macOS
    - `venv\Scripts\activate`, for windows
- install the required packages using `pip install requirements.txt`
- Run the program using `python main.py`
- **Output will be displayed in the console and written to output.csv file!**


## My Output
```
Accuracy      Logistic Regression    Naive Bayes Classifier      LDA     k-NN    Decision Tree Classifier
----------  ---------------------  ------------------------  -------  -------  --------------------------
Dataset 1                 60.2683                   88.8992  87.197   66.6512                     99.5652
Dataset 2                 86.9388                   78.9796  86.6667  84.966                      77.9592
Dataset 3                 99.3261                   98.9263  97.5443  99.2233                     99.8287
Dataset 4                 90.6667                   95.3333  98       95.3333                     96
Dataset 4                 83.4863                   80.5082  83.1585  81.847                      75.5574

Precision      Logistic Regression    Naive Bayes Classifier       LDA      k-NN    Decision Tree Classifier
-----------  ---------------------  ------------------------  --------  --------  --------------------------
Dataset 1                 0.631164                  0.905203  0.900128  0.725561                    0.993333
Dataset 2                 0.78779                   0.658936  0.776924  0.812607                    0.609881
Dataset 3                 0.96511                   0.94925   0.883806  0.956965                    0.998424
Dataset 4                 0.912522                  0.958384  0.980606  0.956818                    0.96229
Dataset 4                 0.840127                  0.805765  0.844975  0.8203                      0.760299

Recall       Logistic Regression    Naive Bayes Classifier       LDA      k-NN    Decision Tree Classifier
---------  ---------------------  ------------------------  --------  --------  --------------------------
Dataset 1               0.592633                  0.899531  0.873369  0.640361                    0.995238
Dataset 2               0.661274                  0.707581  0.65793   0.54411                     0.621608
Dataset 3               0.991763                  0.985037  0.986661  0.993814                    0.989938
Dataset 4               0.906667                  0.953333  0.98      0.953333                    0.96
Dataset 4               0.829052                  0.801647  0.822511  0.81633                     0.752417

F1 Score      Logistic Regression    Naive Bayes Classifier       LDA      k-NN    Decision Tree Classifier
----------  ---------------------  ------------------------  --------  --------  --------------------------
Dataset 1                0.586982                  0.894779  0.87724   0.646736                    0.993732
Dataset 2                0.694744                  0.67261   0.689929  0.54124                     0.614567
Dataset 3                0.977829                  0.965091  0.926973  0.974507                    0.993915
Dataset 4                0.905681                  0.953047  0.979983  0.953014                    0.9599
Dataset 4                0.83084                   0.801838  0.825396  0.816076                    0.750532

Jaccard Score      Logistic Regression    Naive Bayes Classifier       LDA      k-NN    Decision Tree Classifier
---------------  ---------------------  ------------------------  --------  --------  --------------------------
Dataset 1                     0.441647                  0.823216  0.802066  0.506368                    0.988571
Dataset 2                     0.582981                  0.541514  0.578304  0.469065                    0.493141
Dataset 3                     0.95772                   0.934287  0.870467  0.951151                    0.988362
Dataset 4                     0.843782                  0.914141  0.96303   0.917063                    0.92697
Dataset 4                     0.71393                   0.673945  0.705317  0.692126 
```


