import sklearn
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import *




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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3) # 70% training and 30% test

#Tuning the hyperparameters
rf_params = {

    'n_estimators' : [x for x in range(10,200)],
    'min_samples_split' : [x for x in range(2,100)],
    "max_depth" : [x for x in range(1,100)],
    "min_samples_leaf" : [x for x in range(2,100)],
    "max_features" : ["auto", "sqrt", "log2"],
    "bootstrap" : [True, False]

}



num_iter = 10

clf=RandomForestClassifier()


ran_grid_search = sklearn.model_selection.RandomizedSearchCV(estimator = clf, param_distributions = rf_params,
 n_iter = num_iter, scoring = 'balanced_accuracy', n_jobs = -1, cv = 5)

ran_grid_search.fit(X_train,y_train)

y_pred = ran_grid_search.predict(X_test)

best_parameters = ran_grid_search.best_params_
best_estimator = ran_grid_search.best_estimator_

print(best_parameters)
print(best_estimator)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


print(np.unique(diliNums, return_counts=True))