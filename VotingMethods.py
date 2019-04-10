import pandas as pd
import pybel
import openbabel
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn import *


#Create a list of SMILES codes and of DILI concern
dataPD = pd.read_csv('data.csv')
SMILES = dataPD['SMILES']
dili = dataPD['DILIConcern']
smilesList = SMILES.tolist()

# Convert the string DILI concerns to numbers
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

mols = [pybel.readstring("smi", x) for x in smilesList]

# Generate molecular fingerprints from SMILES codes
fps = [x.calcfp() for x in mols]
fingerPrints = []
for i,fingerprint in enumerate(fps):
    fingerPrints.append(fps[i].fp)

# Convert the 'fingerprint' object to a list
fpAsList = [list(fp) for fp in fingerPrints]

# Create X and y for the model
X = fpAsList
y = diliNums

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)  # 70% training and 30% test

# Random Forest Model (averages 50%-60%)
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
clf=RandomForestClassifier(n_estimators=100)



# Logistic Regression Model (Averages 40%-50%)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()


# Naïve Bayes Model (Averages 30%-45%)
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()



# Stochastic Gradient Descent (Averages 30%-60%)
from sklearn.linear_model import SGDClassifier
sgd = SGDClassifier(loss = 'modified_huber', shuffle = True, random_state = 101)



# K-Nearest Neighbor (Averages 50%-65%)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 15)


estimators_to_use = [('rf', clf), ('lr', lr), ('gnb', nb), ('sgd', sgd), ('knn', knn)]

eclf1 = VotingClassifier(estimators = estimators_to_use, voting = 'hard')

vote_estimator = eclf1.fit(X_train, y_train)

y_pred = vote_estimator.predict(X_test)


print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


