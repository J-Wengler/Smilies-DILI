import pandas as pd
import pybel
import openbabel
from sklearn.model_selection import train_test_split
from sklearn import metrics

#Create a list of SMILES codes and of DILI concern
dataPD = pd.read_csv('data.csv')
SMILES = dataPD['SMILES']
dili = dataPD['DILIConcern']
smilesList = SMILES.tolist()

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

mols = [pybel.readstring("smi", x) for x in smilesList]

#Generate molecular fingerprints from SMILES codes
fps = [x.calcfp() for x in mols]
fingerPrints = []
for i,fingerprint in enumerate(fps):
    fingerPrints.append(fps[i].fp)

#Convert the 'fingerprint' object to a list
fpAsList = [list(fp) for fp in fingerPrints]

#combine features
# for i,mol in enumerate(featureList):
#     featureList[i] += fpAsList[i]

#Create X and y for the model

X = featureList
y = diliNums

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3) # 70% training and 30% test


RFnums = []
LRnums = []
NBnums = []
SGnums = []
KNnums = []


for i in range(10):
    #Random Forest Model (averages 50%-60%)
    from sklearn.ensemble import RandomForestClassifier
    clf=RandomForestClassifier(n_estimators=100)
    clf.fit(X_train,y_train)
    y_pred=clf.predict(X_test)
    RFnums.append(metrics.accuracy_score(y_test, y_pred))
    print(y_pred)

    #Logistic Regression Model (Averages 40%-50%)
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    LRnums.append(metrics.accuracy_score(y_test, y_pred))
    print(y_pred)

    #Naïve Bayes Model (Averages 30%-45%)
    from sklearn.naive_bayes import GaussianNB
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    y_pred = nb.predict(X_test)
    NBnums.append(metrics.accuracy_score(y_test, y_pred))
    print(y_pred)

    #Stochastic Gradient Descent (Averages 30%-60%)
    from sklearn.linear_model import SGDClassifier
    sgd = SGDClassifier(loss = 'modified_huber', shuffle = True, random_state = 101)
    sgd.fit(X_train, y_train)
    y_pred = sgd.predict(X_test)
    SGnums.append(metrics.accuracy_score(y_test, y_pred))
    print(y_pred)

    #K-Nearest Neighbor (Averages 50%-65%)
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors = 15)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    KNnums.append(metrics.accuracy_score(y_test, y_pred))
    print(y_pred)



print("Random Forest Average: " + str(sum(RFnums) / len(RFnums)))
print("Logistic Regression Average: " + str(sum(LRnums) / len(LRnums)))
print("Naïve Bayes Average: " + str(sum(NBnums) / len(NBnums)))
print("Stochastic Gradient Average: " + str(sum(SGnums) / len(SGnums)))
print("K-Nearest Neighbor: " + str(sum(KNnums) / len(KNnums)))
