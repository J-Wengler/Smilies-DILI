from sklearn.linear_model import LogisticRegression
import sklearn
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import *
import numpy as np

dataPD = pd.read_csv('data.csv')
dili = dataPD['DILIConcern']

mol2PD = pd.read_csv('mol2_features.csv')

features = map(list, mol2PD.values)
featureList = []

for item in features:
    featureList.append(list(item))

#Convert the string DILI concerns to numbers
def diliStrToInt(dili, diliNums):
    for item in dili:
        if item == "Most-DILI-Concern":
            diliNums.append(2)
        elif item == "Less-DILI-Concern":
            diliNums.append(1)
        elif item == "No-DILI-Concern":
            diliNums.append(0)
        else:
            diliNums.append(0)
diliNums = []
diliStrToInt(dili, diliNums)


#Create X and y for the model

X = featureList
y = diliNums

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

rf_params = {

    'C' : np.arange(0.0, 100.0, 0.01),
    'fit_intercept' : [True, False],
    'solver' : ["newton-cg", "lbfgs", "liblinear", "sag", "saga"],
    'max_iter' : [x for x in range(100,1000)]

}


clf = LogisticRegression()

num_iter = 1000

ran_grid_search = sklearn.model_selection.RandomizedSearchCV(estimator = clf, param_distributions = rf_params,
 n_iter = num_iter, scoring = 'balanced_accuracy', n_jobs = -1, cv = 5)

ran_grid_search.fit(X_train,y_train)

y_pred = ran_grid_search.predict(X_test)

best_parameters = ran_grid_search.best_params_
best_estimator = ran_grid_search.best_estimator_

print(best_parameters)
print(best_estimator)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))