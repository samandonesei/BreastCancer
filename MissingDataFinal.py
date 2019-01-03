import random

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from fancyimpute import KNN
import pandas
import numpy as np
from sklearn import metrics

def generate_random(coeficient_random,classif_column,fromList=False, atribut=None):
    set_date=pandas.read_csv("data.csv",header=0)
    set_date.drop("id",axis=1,inplace=True)
    set_date.drop("Unnamed: 32",axis=1,inplace=True)
    #set_date=set_date.copy()
    if fromList==False:
        if atribut == None:
            for i in range(len(set_date.columns)):
                if set_date.columns[i]!=classif_column:
                    for j in range(len(set_date[set_date.columns[i]])):
                        chance = random.randint(0, coeficient_random)
                        if chance==1:
                            set_date[set_date.columns[i]][j]=np.NaN
        else:
            print(len(atribut))
            if len(atribut)==1:
                for i in range(len(set_date.columns)):
                    if set_date.columns[i]==atribut[0] and set_date.columns[i]!=classif_column:
                        for j in range(len(set_date[set_date.columns[i]])):
                            chance=random.randint(0,coeficient_random)
                            if chance==1:
                                set_date[set_date.columns[i]][j]=np.NaN
                    '''elif set_date.columns[i]==classif_column:
                        print('Values will not be eliminated on the classification column!')
                        return set_date'''
            else:
                print('You provided a list of columns although fromList is False!')
                return set_date
    else:
        for i in range(len(set_date.columns)):
            if set_date.columns[i] in atribut:
                for j in range(len(set_date[set_date.columns[i]])):
                    chance = random.randint(0, coeficient_random)
                    if chance == 1:
                        set_date[set_date.columns[i]][j] = np.NaN
    return set_date
'''
def generate_random_WRelationship(coeficient_random,listColumns):
    set_date=pandas.read_csv("data.csv",header=0)
    set_date.drop("id",axis=1,inplace=True)
    set_date.drop("Unnamed: 32",axis=1,inplace=True)
    if len(listColumns)<=1:
        print('Cannot generate with relationship with only one attribute!')
        return set_date
    elif len(listColumns)==2:
    elif len(listColumns)==3:
    elif len(listColumns)==4:
'''


def printDetails_dataset(set_date):
    for i in range(len(set_date.columns)):
        print('Total missing values for '+str(set_date.columns[i])+' are '+str(set_date[set_date.columns[i]].isnull().sum()))

#def computeInitialPredictions(set_date):


def computeMissingValues_EliminationRow(set_date):
    #print(set_date.shape)
    #set_date_elim= copy.deepcopy(set_date)
    #set_date_elim=set_date_elim.dropna(inplace=True)
    #print(set_date_elim.head)
    #print(set_date_elim.shape)
    set_date_new=set_date.dropna()
    #print(set_date)
    #print(set_date.shape)
    return set_date_new

def computeMissingValues_EliminationColumn(set_date):
    #print(set_date.shape)
    #set_date_elim= copy.deepcopy(set_date)
    #set_date_elim=set_date_elim.dropna(inplace=True)
    #print(set_date_elim.head)
    #print(set_date_elim.shape)
    set_date_new=set_date.dropna(axis='columns')
    #print(set_date_new.apply(lambda x: x.count(), axis=1))
    #print(set_date)
    #print(set_date.shape)
    return set_date_new

def computeMissingValues_Mode(set_date):
    #print(set_date.shape)
    #print(set_date.apply(lambda x: x.count(), axis=1))
    set_date_new=set_date.fillna(set_date.mode().ix[0])
    #print(set_date_new.apply(lambda x: x.count(), axis=1))
    #print(set_date.shape)
    return set_date_new

def computeMissingValues_Mean(set_date):
    #print(set_date.shape)
    #print(set_date.apply(lambda x: x.count(), axis=1))
    set_date_new=set_date.fillna(set_date.mean())
    #print(set_date_new.apply(lambda x: x.count(), axis=1))
    #print(set_date.shape)
    return set_date_new

def computeMissingValues_Median(set_date):
    #print(set_date.shape)
    #print(set_date.apply(lambda x: x.count(), axis=1))
    set_date_new=set_date.fillna(set_date.median())
    #print(set_date_new.apply(lambda x: x.count(), axis=1))
    #print(set_date.shape)
    return set_date_new

def computeMissingValues_KNN(set_date):
    set_date_input=set_date[set_date.columns[1:len(set_date.columns)]]
    set_date_list=list(set_date_input)
    set_date_new=pandas.DataFrame(KNN(k=5).fit_transform(set_date_input))
    set_date_new.columns=set_date_list
    set_date_new.insert(0,'diagnosis',set_date.diagnosis)
    return set_date_new
    #print(set_date_new.head)

def computeMissingValue_RandomForest1VL(set_date):
    algoritm= RandomForestRegressor()
    #print(set_date.info())
    set_date_new=set_date.drop('diagnosis',axis=1)
    #print(set_date_new.info())
    #print(set_date_new[set_date_new['area_mean'].notnull()])
    antrenament_in=set_date_new[set_date_new['area_mean'].notnull()].drop('area_mean',axis=1)
    antrenament_out=set_date_new[set_date_new['area_mean'].notnull()]['area_mean']
    test_in=set_date_new[set_date_new['area_mean'].isnull()].drop('area_mean',axis=1)
    #print(test_in)

    algoritm.fit(antrenament_in,antrenament_out)
    valoriPrezise=algoritm.predict(test_in)
    set_date_new.area_mean[set_date_new.area_mean.isnull()]=valoriPrezise
    set_date_new.insert(0,'diagnosis',set_date.diagnosis)

    return set_date_new
    #print(set_date_new.info())

def computeMissingValue_RandomForest3VL(set_date):
    algoritm= RandomForestRegressor()
    #print(set_date.info())
    set_date_new=set_date.drop('diagnosis',axis=1)
    #print(set_date_new.info())
    set_date_new=set_date_new.drop('perimeter_worst',axis=1)
    set_date_new=set_date_new.drop('area_worst',axis=1)
    #print(set_date_new.info())
    #print(set_date_new[set_date_new['area_mean'].notnull()])
    antrenament_in=set_date_new[set_date_new['area_mean'].notnull()].drop('area_mean',axis=1)
    antrenament_out=set_date_new[set_date_new['area_mean'].notnull()]['area_mean']
    test_in=set_date_new[set_date_new['area_mean'].isnull()].drop('area_mean',axis=1)
    #print(test_in)

    algoritm.fit(antrenament_in,antrenament_out)
    valoriPrezise=algoritm.predict(test_in)
    set_date.area_mean[set_date.area_mean.isnull()]=valoriPrezise
    #print(set_date_new.info())

    set_date_new=set_date_new.drop('area_mean',axis=1)
    set_date_new.insert(0,'perimeter_worst',set_date.perimeter_worst)
    antrenament_in=set_date_new[set_date_new['perimeter_worst'].notnull()].drop('perimeter_worst',axis=1)
    antrenament_out=set_date_new[set_date_new['perimeter_worst'].notnull()]['perimeter_worst']
    test_in=set_date_new[set_date_new['perimeter_worst'].isnull()].drop('perimeter_worst',axis=1)
    #print(test_in)

    algoritm.fit(antrenament_in,antrenament_out)
    valoriPrezise=algoritm.predict(test_in)
    set_date.perimeter_worst[set_date.perimeter_worst.isnull()]=valoriPrezise
    #print(set_date_new.info())

    set_date_new=set_date_new.drop('perimeter_worst',axis=1)
    set_date_new.insert(0,'area_worst',set_date.area_worst)
    antrenament_in=set_date_new[set_date_new['area_worst'].notnull()].drop('area_worst',axis=1)
    antrenament_out=set_date_new[set_date_new['area_worst'].notnull()]['area_worst']
    test_in=set_date_new[set_date_new['area_worst'].isnull()].drop('area_worst',axis=1)
    #print(test_in)

    algoritm.fit(antrenament_in,antrenament_out)
    valoriPrezise=algoritm.predict(test_in)
    set_date_new.area_worst[set_date_new.area_worst.isnull()]=valoriPrezise

    set_date_new.insert(0,'perimeter_worst',set_date.perimeter_worst)
    set_date_new.insert(0,'area_mean',set_date.area_mean)
    set_date_new.insert(0,'diagnosis',set_date.diagnosis)

    #print(set_date_new.info())

    return set_date_new

def computeAccuracy(set_date,algorithm='KNN'):
    if algorithm=='KNN':
        algoritm = KNeighborsClassifier(n_neighbors=13)
    elif algorithm=='LogisticRegression':
        algoritm = LogisticRegression()
    elif algorithm=='RandomForest':
        algoritm = RandomForestClassifier()
    else:
        algoritm = GaussianNB()
    '''else:
        algoritm=DecisionTreeClassifier()'''
    antrenament, test = train_test_split(set_date, test_size=0.3, shuffle=False)
    antrenament_input = antrenament[antrenament.columns[1:len(set_date.columns)]]
    antrenament_output=antrenament.diagnosis

    test_input = test[test.columns[1:len(set_date.columns)]]
    test_output=test.diagnosis
    #test_input.drop("diagnosis", axis=1, inplace=True)
    algoritm.fit(antrenament_input,antrenament_output)
    #print(test_input)
    predictionInitial=algoritm.predict(test_input)
    return(metrics.accuracy_score(predictionInitial, test_output))


def accuracyComparisons_totallyRandomDataset():
    initialDataset=pandas.read_csv("data.csv",header=0)
    initialDataset.drop("id",axis=1,inplace=True)
    initialDataset.drop("Unnamed: 32",axis=1,inplace=True)

    randomDataset=generate_random(10,'diagnosis')
    rowEliminatedDataset=computeMissingValues_EliminationRow(randomDataset)
    meanComputedDataset=computeMissingValues_Mean(randomDataset)
    modeComputedDataset=computeMissingValues_Mode(randomDataset)
    medianComputedDataset=computeMissingValues_Median(randomDataset)
    knnComputedDataset=computeMissingValues_KNN(randomDataset)


    accInitial=computeAccuracy(initialDataset)
    accRowElim=computeAccuracy(rowEliminatedDataset)
    accMean=computeAccuracy(meanComputedDataset)
    accMode=computeAccuracy(modeComputedDataset)
    accMedian=computeAccuracy(medianComputedDataset)
    accKnn=computeAccuracy(knnComputedDataset)

    print('Acurracy on the initial dataset:'+str(accInitial)+'%')
    print('Acurracy on the dataset with eliminated rows:' + str(accRowElim) + '%')
    print('Acurracy on the dataset with mean compute:' + str(accMean) + '%')
    print('Acurracy on the dataset with mode compute:' + str(accMode) + '%')
    print('Acurracy on the dataset with median compute:' + str(accMedian) + '%')
    print('Acurracy on the dataset with KNN compute:' + str(accKnn) + '%')

def accuracyComparisons_oneRandomColumn(columnName,algoritm):
    initialDataset=pandas.read_csv("data.csv",header=0)
    initialDataset.drop("id",axis=1,inplace=True)
    initialDataset.drop("Unnamed: 32",axis=1,inplace=True)

    randomDataset=generate_random(5,'diagnosis',False,columnName)
    print(randomDataset.info())
    rowEliminatedDataset=computeMissingValues_EliminationRow(randomDataset)
    columnEliminatedDataset=computeMissingValues_EliminationColumn(randomDataset)
    meanComputedDataset=computeMissingValues_Mean(randomDataset)
    modeComputedDataset=computeMissingValues_Mode(randomDataset)
    medianComputedDataset=computeMissingValues_Median(randomDataset)
    knnComputedDataset=computeMissingValues_KNN(randomDataset)
    randfComputedDataset= computeMissingValue_RandomForest1VL(randomDataset) #area_mean

    accInitial = computeAccuracy(initialDataset,algoritm)
    accRowElim = computeAccuracy(rowEliminatedDataset,algoritm)
    accColElim=computeAccuracy(columnEliminatedDataset,algoritm)
    accMean = computeAccuracy(meanComputedDataset,algoritm)
    accMode = computeAccuracy(modeComputedDataset,algoritm)
    accMedian = computeAccuracy(medianComputedDataset,algoritm)
    accKnn = computeAccuracy(knnComputedDataset,algoritm)
    accRandF= computeAccuracy(randfComputedDataset,algoritm)

    print('Acurracy on the initial dataset:' + str(accInitial) + '%')
    print('Acurracy on the dataset with eliminated rows:' + str(accRowElim) + '%')
    print('Acurracy on the dataset with eliminated columns:' + str(accColElim) + '%')
    print('Acurracy on the dataset with mean compute:' + str(accMean) + '%')
    print('Acurracy on the dataset with mode compute:' + str(accMode) + '%')
    print('Acurracy on the dataset with median compute:' + str(accMedian) + '%')
    print('Acurracy on the dataset with KNN compute:' + str(accKnn) + '%')
    print('Accuracy on the dataset with RF compute' + str(accRandF) + '%')

def accuracyComparisons_threeRandomColumns(columnName,algoritm):
    initialDataset=pandas.read_csv("data.csv",header=0)
    initialDataset.drop("id",axis=1,inplace=True)
    initialDataset.drop("Unnamed: 32",axis=1,inplace=True)

    randomDataset=generate_random(5,'diagnosis',True,columnName)
    #print(randomDataset.info())
    rowEliminatedDataset=computeMissingValues_EliminationRow(randomDataset)
    columnEliminatedDataset=computeMissingValues_EliminationColumn(randomDataset)
    meanComputedDataset=computeMissingValues_Mean(randomDataset)
    modeComputedDataset=computeMissingValues_Mode(randomDataset)
    medianComputedDataset=computeMissingValues_Median(randomDataset)
    knnComputedDataset=computeMissingValues_KNN(randomDataset)
    randfComputedDataset= computeMissingValue_RandomForest3VL(randomDataset) #area_mean, area_worst, perimeter_worst

    accInitial = computeAccuracy(initialDataset,algoritm)
    accRowElim = computeAccuracy(rowEliminatedDataset,algoritm)
    accColElim=computeAccuracy(columnEliminatedDataset,algoritm)
    accMean = computeAccuracy(meanComputedDataset,algoritm)
    accMode = computeAccuracy(modeComputedDataset,algoritm)
    accMedian = computeAccuracy(medianComputedDataset,algoritm)
    accKnn = computeAccuracy(knnComputedDataset,algoritm)
    accRandF= computeAccuracy(randfComputedDataset,algoritm)

    print('Acurracy on the initial dataset:' + str(accInitial) + '%')
    print('Acurracy on the dataset with eliminated rows:' + str(accRowElim) + '%')
    print('Acurracy on the dataset with eliminated columns:' + str(accColElim) + '%')
    print('Acurracy on the dataset with mean compute:' + str(accMean) + '%')
    print('Acurracy on the dataset with mode compute:' + str(accMode) + '%')
    print('Acurracy on the dataset with median compute:' + str(accMedian) + '%')
    print('Acurracy on the dataset with KNN compute:' + str(accKnn) + '%')
    print('Accuracy on the dataset with RF compute' + str(accRandF) + '%')


def randomForest_featureImportance():
    initialDataset=pandas.read_csv("data.csv",header=0)
    initialDataset.drop("id",axis=1,inplace=True)
    initialDataset.drop("Unnamed: 32",axis=1,inplace=True)

    algoritm = RandomForestClassifier()
    antrenament, test = train_test_split(initialDataset, test_size=0.3, shuffle=False)
    antrenament_input = antrenament[antrenament.columns[1:len(initialDataset.columns)]]
    antrenament_output=antrenament.diagnosis

    test_input = test[test.columns[1:len(initialDataset.columns)]]
    test_output=test.diagnosis
    #test_input.drop("diagnosis", axis=1, inplace=True)
    algoritm.fit(antrenament_input,antrenament_output)
    #print(test_input)
    predictionInitial=algoritm.predict(test_input)
    print(metrics.accuracy_score(predictionInitial, test_output))
    atribute=pandas.Series(algoritm.feature_importances_, index=initialDataset.columns[1:len(initialDataset.columns)]).sort_values(ascending=True)
    print(atribute)


set_date_n=pandas.read_csv("data.csv",header=0)
set_date=generate_random(5,'diagnosis',True,['area_mean','perimeter_worst','area_worst'])
accuracyComparisons_threeRandomColumns(['perimeter_worst','area_worst','area_mean'],'KNN')
