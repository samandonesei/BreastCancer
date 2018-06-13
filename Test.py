from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import VarianceThreshold
from mlxtend.feature_selection import SequentialFeatureSelector
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import ExtraTreesClassifier
import pandas
import numpy as np
from sklearn import metrics

def KNN(iteratii):
    set_date=pandas.read_csv("data.csv",header=0)
    set_date.drop("id",axis=1,inplace=True)
    set_date.drop("Unnamed: 32",axis=1,inplace=True)
    first_set=list(set_date.columns[1:11])
    coloane = ['texture_mean','perimeter_mean','smoothness_mean','compactness_mean','symmetry_mean']
    medie=0
    for i in range(iteratii):
        antrenament, test = train_test_split(set_date, test_size = 0.3)
        antrenament_input = antrenament[coloane]
        antrenament_output=antrenament.diagnosis
        test_input= test[coloane]
        test_output =test.diagnosis
        algoritm=KNeighborsClassifier(n_neighbors=15)
        algoritm.fit(antrenament_input,antrenament_output)
        prediction=algoritm.predict(test_input)
        medie+=metrics.accuracy_score(prediction,test_output)
    print(medie/iteratii)
def RandomForest(iteratii):
    set_date=pandas.read_csv("data.csv",header=0)
    set_date.drop("id",axis=1,inplace=True)
    set_date.drop("Unnamed: 32",axis=1,inplace=True)
    first_set=list(set_date.columns[1:11])
    second_set=list(set_date.columns[11:20])
    last_set=list(set_date.columns[21:31])
    coloane=['texture_mean','perimeter_mean','smoothness_mean','compactness_mean','symmetry_mean']
    medie=0
    for i in range(iteratii):
        antrenament,test=train_test_split(set_date,test_size=0.3)
        #antrenament_input=antrenament[first_set]
        #antrenament_input=antrenament[second_set]
        antrenament_input=antrenament[last_set]
        #antrenament_input=antrenament[coloane]
        antrenament_output=antrenament.diagnosis
        #test_input=test[first_set]
        #test_input=test[second_set]
        test_input=test[last_set]
        #test_input=test[coloane]
        test_output=test.diagnosis
        algoritm=RandomForestClassifier(n_estimators=100)
        algoritm.fit(antrenament_input,antrenament_output)
        prediction=algoritm.predict(test_input)
        medie+=metrics.accuracy_score(prediction,test_output)
    print(medie/iteratii)
def LRegression(iteratii):
    set_date=pandas.read_csv("data.csv",header=0)
    set_date.drop("id",axis=1,inplace=True)
    set_date.drop("Unnamed: 32",axis=1,inplace=True)
    first_set=list(set_date.columns[1:11])
    second_set=list(set_date.columns[11:20])
    last_set=list(set_date.columns[21:31])
    coloane=['texture_mean','perimeter_mean','smoothness_mean','compactness_mean','symmetry_mean']
    medie=0
    for i in range(iteratii):
        antrenament,test=train_test_split(set_date,test_size=0.3)
        antrenament_input=antrenament[first_set]
        #antrenament_input=antrenament[second_set]
        #antrenament_input=antrenament[last_set]
        #antrenament_input=antrenament[coloane]
        antrenament_output=antrenament.diagnosis
        test_input=test[first_set]
        #test_input=test[second_set]
        #test_input=test[last_set]
        #test_input=test[coloane]
        test_output=test.diagnosis
        algoritm=LogisticRegression()
        algoritm.fit(antrenament_input,antrenament_output)
        prediction=algoritm.predict(test_input)
        medie+=metrics.accuracy_score(prediction,test_output)
    print(medie/iteratii)

def featureSelection(iteratii,optiune,number=20):#0 fara select,1 cu selectie chi2, 2 cu selectie f_regression
    # 0 fara select,1 cu selectie chi2, 2 cu selectie f_regression, 3 cu last_set, 4 cu mutual_info_classif
    set_date=pandas.read_csv("data.csv",header=0)
    #set_date.drop("id",axis=1,inplace=True)
    #set_date.drop("Unnamed: 32", axis=1,inplace=True)
    output=set_date.diagnosis
    deEliminat=['id','diagnosis','Unnamed: 32']
    input=set_date.drop(deEliminat,axis=1)
    algoritm = RandomForestClassifier()
    total=0
    if optiune==0:
        for i in range(iteratii):
            antrenament_input, test_input, antrenament_output, test_output = train_test_split(input, output, test_size=0.3)
            algoritm.fit(antrenament_input, antrenament_output)
            prediction = algoritm.predict(test_input)
            total=total+metrics.accuracy_score(prediction, test_output)
    elif optiune!=3:
        for i in range(iteratii):
            antrenament_input, test_input, antrenament_output, test_output = train_test_split(input, output, test_size=0.3)
            if optiune==1:
                atributeSelectate = SelectKBest(chi2, k=number).fit(antrenament_input, antrenament_output)
            if optiune==2:
                atributeSelectate = SelectKBest(f_classif, k=number).fit(antrenament_input, antrenament_output)
            if optiune==4:
                atributeSelectate= SelectKBest(mutual_info_classif,k=number).fit(antrenament_input,antrenament_output)
            antrenament_input2=atributeSelectate.transform(antrenament_input)
            test_input2=atributeSelectate.transform(test_input)
            algoritm.fit(antrenament_input2,antrenament_output)
            prediction=algoritm.predict(test_input2)
            total=total+metrics.accuracy_score(prediction,test_output)
    elif optiune==3:
        set_date = pandas.read_csv("data.csv", header=0)
        set_date.drop("id", axis=1, inplace=True)
        set_date.drop("Unnamed: 32", axis=1, inplace=True)
        first_set = list(set_date.columns[1:11])
        second_set = list(set_date.columns[11:20])
        last_set = list(set_date.columns[21:31])
        for i in range(iteratii):
            antrenament, test = train_test_split(set_date, test_size=0.3)
            antrenament_input=antrenament[first_set]
            #antrenament_input=antrenament[second_set]
            #antrenament_input = antrenament[last_set]
            #antrenament_input=antrenament[coloane]
            antrenament_output = antrenament.diagnosis
            test_input=test[first_set]
            #test_input=test[second_set]
            #test_input = test[last_set]
            #test_input=test[coloane]
            test_output = test.diagnosis
            algoritm.fit(antrenament_input, antrenament_output)
            prediction = algoritm.predict(test_input)
            total += metrics.accuracy_score(prediction, test_output)
    print(total/iteratii)


#selectie atribute fara iteratii, comparatie directa
def featureSelectionNS(number):#0 fara select,1 cu selectie chi2, 2 cu selectie f_regression
    # 0 fara select,1 cu selectie chi2, 2 cu selectie f_regression, 3 cu last_set, 4 cu mutual_info_classif
    set_date=pandas.read_csv("data.csv",header=0)
    #set_date.drop("id",axis=1,inplace=True)
    #set_date.drop("Unnamed: 32", axis=1,inplace=True)
    output=set_date.diagnosis
    deEliminat=['id','diagnosis','Unnamed: 32']
    #deEliminat = ['diagnosis', 'Unnamed: 32']
    input=set_date.drop(deEliminat,axis=1)
    algoritm = RandomForestClassifier()
    antrenament_input, test_input, antrenament_output, test_output = train_test_split(input, output, test_size=0.3,shuffle=False)
    algoritm.fit(antrenament_input, antrenament_output)
    prediction = algoritm.predict(test_input)
    print("Fara nicio selectie: "+str(metrics.accuracy_score(prediction, test_output)))
    algoritm=RandomForestClassifier()
    atributeSelectate1 = SelectKBest(chi2, k=number).fit(antrenament_input, antrenament_output)
    atributeSelectate2 = SelectKBest(f_classif, k=number).fit(antrenament_input, antrenament_output)
    atributeSelectate3 = SelectKBest(mutual_info_classif, k=number).fit(antrenament_input, antrenament_output)
    atributeSelectate4 = VarianceThreshold().fit(antrenament_input,antrenament_output)
    antrenament_input2=atributeSelectate1.transform(antrenament_input)
    test_input2=atributeSelectate1.transform(test_input)
    algoritm.fit(antrenament_input2,antrenament_output)
    prediction=algoritm.predict(test_input2)
    print("Cu selectie chi2 si k="+str(number)+" :"+str(metrics.accuracy_score(prediction, test_output)))
    antrenament_input2=atributeSelectate2.transform(antrenament_input)
    test_input2=atributeSelectate2.transform(test_input)
    algoritm= RandomForestClassifier()
    algoritm.fit(antrenament_input2,antrenament_output)
    prediction=algoritm.predict(test_input2)
    print("Cu selectie f_classif si k="+str(number)+" :"+str(metrics.accuracy_score(prediction, test_output)))
    antrenament_input2=atributeSelectate3.transform(antrenament_input)
    test_input2=atributeSelectate3.transform(test_input)
    algoritm= RandomForestClassifier()
    algoritm.fit(antrenament_input2,antrenament_output)
    prediction=algoritm.predict(test_input2)
    print("Cu selectie mutual_info_classif si k="+str(number)+" :"+str(metrics.accuracy_score(prediction, test_output)))
    antrenament_input2=atributeSelectate4.transform(antrenament_input)
    test_input2=atributeSelectate4.transform(test_input)
    algoritm= RandomForestClassifier()
    algoritm.fit(antrenament_input2,antrenament_output)
    prediction=algoritm.predict(test_input2)
    print("Cu selectie variance_threshold si k="+str(number)+" :"+str(metrics.accuracy_score(prediction, test_output)))
    atributeSelectate5 = RFE(RandomForestClassifier(),n_features_to_select=number).fit(antrenament_input,antrenament_output)
    antrenament_input2=atributeSelectate5.transform(antrenament_input)
    test_input2=atributeSelectate5.transform(test_input)
    algoritm= RandomForestClassifier()
    algoritm.fit(antrenament_input2,antrenament_output)
    prediction=algoritm.predict(test_input2)
    print("Cu selectie RFE(recursive feature elimination) si k="+str(number)+" :"+str(metrics.accuracy_score(prediction, test_output)))
    model = ExtraTreesClassifier().fit(antrenament_input,antrenament_output)
    importante=model.feature_importances_
    #print(importante)
    atributeSelectate6= SelectFromModel(model,prefit=True)
    antrenament_input2=atributeSelectate6.transform(antrenament_input)
    test_input2=atributeSelectate6.transform(test_input)
    algoritm= RandomForestClassifier()
    algoritm.fit(antrenament_input2,antrenament_output)
    prediction=algoritm.predict(test_input2)
    print("Cu TreeClassifier :"+str(metrics.accuracy_score(prediction,test_output)))

#incercare pentru forward selection (alte metode de wrapping inafara de RFE)
def featureSecvential():
    set_date = pandas.read_csv("data.csv", header=0)
    # set_date.drop("id",axis=1,inplace=True)
    # set_date.drop("Unnamed: 32", axis=1,inplace=True)
    output = set_date.diagnosis
    deEliminat = ['id', 'diagnosis', 'Unnamed: 32']
    # deEliminat = ['diagnosis', 'Unnamed: 32']
    input = set_date.drop(deEliminat, axis=1)
    algoritm = RandomForestClassifier()
    antrenament_input, test_input, antrenament_output, test_output = train_test_split(input, output, test_size=0.3,shuffle=False)
    forward=SequentialFeatureSelector(algoritm,k_features=16,scoring='accuracy')
    forward.fit(antrenament_input,antrenament_output)

#featureSecvential()


######## Selectarea atributelor si rularea pe RandomForest ########

#featureSelection(500,4,23)
#featureSelection(500,....) #cu shuffle
#0.9515-0.9525 fara selectie
#0.9524-0.9534 cu selectie chi2 si k=23
#0.9525-0.9541 cu select f_classif si k=20
#0.9535-0.9548 cu last_set
#0.8701-0.8714 cu second_set
#0.9318-0.9329 cu first_set

#featureSelectionNS(23)
#featureSelectionNS(...) #toate same time
#20 0.9649 nicio selectie
#   0.9707 chi2
#   0.9824 f_classif
#   0.9649 mutual_info_classif

#16 0.947 nicio selectie          | 0.9707 | 0.9641
#   0.9766 cu chi2                | 0.9824 | 0.9824
#   0.9883 cu f_classif           | 0.9590 | 0.9766
#   0.9707 cu mutual_info_classif | 0.9824 | 0.9649

#23 0.959 nicio selectie          | 0.9649 | 0.9766
#   0.9883 cu chi2                | 0.9707 | 0.9824
#   0.9883 cu f_classif           | 0.9707 | 0.9649
#   0.9766 cu mutual_info_classif | 0.9473 | 0.9707

#16 0.959 nicio selectie          | 0.9766 | 0.9532
#   0.9707 cu chi2                | 0.9649 | 0.9649
#   0.9532 cu f_classif           | 0.9824 | 0.9824
#   0.9649 cu mutual_info_classif | 0.959  | 0.9649
#   0.9941 cu variance_threshold  | 0.9649 | 0.9707
#   0.9707 cu RFE                 | 0.959  | 0.9707

#11 0.9649 nicio selectie         | 0.9532 | 0.9649
#   0.9532 cu chi2                | 0.9473 | 0.9415
#   0.9649 cu f_classif           | 0.9707 | 0.9649
#   0.9649 cu mutual_info_classif | 0.959  | 0.9532
#   0.9707 cu variance_threshold  | 0.9707 | 0.9766
#   0.9766 cu RFE                 | 0.9707 | 0.9649
#   0.959  cu TreeClassifier      | 0.959  | 0.9649

#23 0.9649 nicio selectie         | 0.959  | 0.9649
#   0.9824 cu chi2                | 0.9707 | 0.9649
#   0.959  cu f_classif           | 0.959  | 0.9766
#   0.9766 cu mutual_info_classif | 0.9707 | 0.9532
#   0.9532 cu variance_threshold  | 0.9824 | 0.9766
#   0.9356 cu RFE                 | 0.9707 | 0.9649
#   0.9473 cu TreeClassifier      | 0.9766 | 0.9298

###### Fara selectia atributelor ########


#KNN(100)
#0.874619883041 cu 1
#0.893976608187 cu 3
#0.892105263158 cu 5
#0.893801169591 cu 7
#0.892631578947 cu 9
#0.89350877193 cu 11
#0.900116959064 cu 13
#0.902923976608 cu 15
#0.903391812865 cu 17
#0.902222222222 cu 19


#RandomForest(100)
#0.9222105263158
#0.937 first set
#0.885 second set
#0.959 last set

#LRegression(100)
#0.891988304094
#0.903 first set
#0.885 second set
#0.952 last set

def test():
    set1=pandas.read_csv("data.csv")
    set2=pandas.read_csv("breast_cancer_data_1.csv")
    set3=pandas.read_csv("breast-cancer.csv")
    print(set1.head())
    print(set2.head())
    print(set3.head())
    #print(set2.columns)
#test()

