
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import VarianceThreshold
from mlxtend.feature_selection import SequentialFeatureSelector
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import RFECV
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.ensemble import ExtraTreesClassifier
from yellowbrick.regressor import AlphaSelection
import pandas
import seaborn
import matplotlib.pyplot as plt
from sklearn import metrics

def dataAnalysis(set_date):

    a,b=plt.subplots(figsize=(20, 20))
    seaborn.heatmap(set_date.corr(), annot=True, linewidths=2,linecolor='Black', fmt='.2f',ax=b,cbar= False,cmap="Blues")

    '''
    data = pandas.concat([set_date.diagnosis, set_date.perimeter_worst], axis=1)
    data = pandas.melt(data, id_vars="diagnosis",var_name="features",value_name='value')
    sns.violinplot(x='features',y='value',hue='diagnosis',palette='Set2',split=True,data=data)

    plt.show()

    data = pandas.concat([set_date.diagnosis, set_date.area_worst], axis=1)
    data = pandas.melt(data, id_vars="diagnosis",var_name="features",value_name='value')
    sns.violinplot(x='features',y='value',hue='diagnosis',palette='Set2',split=True,data=data)

    plt.show()

    data = pandas.concat([set_date.diagnosis, set_date.radius_worst], axis=1)
    data = pandas.melt(data, id_vars="diagnosis",var_name="features",value_name='value')
    sns.violinplot(x='features',y='value',hue='diagnosis',palette='Set2',split=True,data=data)
'''
    plt.show()

set_date=pandas.read_csv("data.csv",header=0)
set_date.drop("id", axis=1,inplace=True)
'''
set_date.drop("area_mean",axis=1,inplace=True)
set_date.drop("radius_mean",axis=1,inplace=True)
set_date.drop("perimeter_se",axis=1,inplace=True)
set_date.drop("radius_se",axis=1,inplace=True)
set_date.drop("area_worst",axis=1,inplace=True)
set_date.drop("radius_worst",axis=1,inplace=True)
set_date.drop("concave points_mean",axis=1,inplace=True)
set_date.drop("texture_worst",axis=1,inplace=True)
set_date.drop("perimeter_worst",axis=1,inplace=True)
'''

#set_date.drop("diagnosis", axis=1, inplace=True)
set_date.drop("Unnamed: 32", axis=1, inplace=True)
#dataAnalysis(set_date)
algoritm=AdaBoostClassifier()
antrenament, test = train_test_split(set_date, test_size=0.3, shuffle=False)
antrenament_input = antrenament[antrenament.columns[1:len(set_date.columns)]]
antrenament_output = antrenament.diagnosis
test_input = test[test.columns[1:len(set_date.columns)]]
test_output = test.diagnosis
algoritm.fit(antrenament_input, antrenament_output)
prediction=algoritm.predict(test_input)
print(metrics.accuracy_score(prediction, test_output))


def recursiveFeatureElim_CV(set_date):

    algoritm= RFECV(estimator=AdaBoostClassifier(),cv=10,scoring='accuracy')
    antrenament, test = train_test_split(set_date, test_size=0.3, shuffle=False)
    antrenament_input = antrenament[antrenament.columns[1:len(set_date.columns)]]
    antrenament_output=antrenament.diagnosis
    algoritm.fit(antrenament_input,antrenament_output)

    print('No of features to select: ', algoritm.n_features_)
    return algoritm.n_features_


def featureSecvential(set_date, optimum_number): #NOT WORKING YET

    antrenament, test= train_test_split(set_date,test_size=0.3,shuffle=False)
    antrenament_input = antrenament[antrenament.columns[1:len(set_date.columns)]]
    antrenament_output= antrenament.diagnosis



    algoritm=SequentialFeatureSelector(estimator=AdaBoostClassifier(),k_features=optimum_number,scoring='accuracy')
    atributeSelectate=algoritm.fit(antrenament_input,antrenament_output)

    test_input = test[test.columns[1:len(set_date.columns)]]
    test_output=test.diagnosis

    algoritm= AdaBoostClassifier()

    antrenament_input2=atributeSelectate.transform(antrenament_input)
    test_input2=atributeSelectate.transform(test_input)
    algoritm.fit(antrenament_input2,antrenament_output)
    prediction=algoritm.predict(test_input2)

    #atribute = atributeSelectate.get_support(indices=True)

    #print('Attributes selected by chi2: '+str(atributeSelectate.subsets_))


    print('Accuracy after FSelection:' + str(metrics.accuracy_score(prediction, test_output)) + '%')
    return(metrics.accuracy_score(prediction,test_output))



def chi2_Select(set_date,optimum_number):

    antrenament, test= train_test_split(set_date,test_size=0.3,shuffle=False)
    antrenament_input = antrenament[antrenament.columns[1:len(set_date.columns)]]
    antrenament_output= antrenament.diagnosis
    atributeSelectate = SelectKBest(chi2, k=optimum_number).fit(antrenament_input, antrenament_output)

    test_input = test[test.columns[1:len(set_date.columns)]]
    test_output=test.diagnosis

    algoritm= AdaBoostClassifier()

    antrenament_input2=atributeSelectate.transform(antrenament_input)
    test_input2=atributeSelectate.transform(test_input)
    algoritm.fit(antrenament_input2,antrenament_output)
    prediction=algoritm.predict(test_input2)

    atribute = atributeSelectate.get_support(indices=True)

    #print('Attributes selected by chi2: '+str(antrenament_input.columns[atribute]))

    print('Accuracy after chi2 selection:'+str(metrics.accuracy_score(prediction, test_output))+'%')
    return(metrics.accuracy_score(prediction, test_output))

def fclassif_Select(set_date,optimum_number):

    antrenament, test= train_test_split(set_date,test_size=0.3,shuffle=False)
    antrenament_input = antrenament[antrenament.columns[1:len(set_date.columns)]]
    antrenament_output= antrenament.diagnosis
    atributeSelectate=SelectKBest(f_classif, k=optimum_number).fit(antrenament_input, antrenament_output)
    test_input = test[test.columns[1:len(set_date.columns)]]
    test_output=test.diagnosis

    algoritm= AdaBoostClassifier()

    antrenament_input2=atributeSelectate.transform(antrenament_input)
    test_input2=atributeSelectate.transform(test_input)
    algoritm.fit(antrenament_input2,antrenament_output)
    prediction=algoritm.predict(test_input2)

    print(metrics.accuracy_score(prediction, test_output))
    return (metrics.accuracy_score(prediction, test_output))

def mutualinfoclassif_Select(set_date,optimum_number):

    antrenament, test= train_test_split(set_date,test_size=0.3,shuffle=False)
    antrenament_input = antrenament[antrenament.columns[1:len(set_date.columns)]]
    antrenament_output= antrenament.diagnosis
    atributeSelectate=SelectKBest(mutual_info_classif, k=optimum_number).fit(antrenament_input, antrenament_output)
    test_input = test[test.columns[1:len(set_date.columns)]]
    test_output=test.diagnosis

    algoritm= AdaBoostClassifier()

    antrenament_input2=atributeSelectate.transform(antrenament_input)
    test_input2=atributeSelectate.transform(test_input)
    algoritm.fit(antrenament_input2,antrenament_output)
    prediction=algoritm.predict(test_input2)

    print(metrics.accuracy_score(prediction, test_output))
    return (metrics.accuracy_score(prediction, test_output))

def variancethreshold_Select(set_date):

    antrenament, test= train_test_split(set_date,test_size=0.3,shuffle=False)
    antrenament_input = antrenament[antrenament.columns[1:len(set_date.columns)]]
    antrenament_output= antrenament.diagnosis
    atributeSelectate=VarianceThreshold().fit(antrenament_input,antrenament_output)
    test_input = test[test.columns[1:len(set_date.columns)]]
    test_output=test.diagnosis

    algoritm= AdaBoostClassifier()

    antrenament_input2=atributeSelectate.transform(antrenament_input)
    test_input2=atributeSelectate.transform(test_input)
    algoritm.fit(antrenament_input2,antrenament_output)
    prediction=algoritm.predict(test_input2)

    print(metrics.accuracy_score(prediction, test_output))

def recursivefeatureelim_Select(set_date,optimum_number):

    antrenament, test= train_test_split(set_date,test_size=0.3,shuffle=False)
    antrenament_input = antrenament[antrenament.columns[1:len(set_date.columns)]]
    antrenament_output= antrenament.diagnosis
    atributeSelectate= RFE(AdaBoostClassifier(),n_features_to_select=optimum_number).fit(antrenament_input,antrenament_output)
    test_input = test[test.columns[1:len(set_date.columns)]]
    test_output=test.diagnosis

    algoritm= AdaBoostClassifier()

    antrenament_input2=atributeSelectate.transform(antrenament_input)
    test_input2=atributeSelectate.transform(test_input)
    algoritm.fit(antrenament_input2,antrenament_output)
    prediction=algoritm.predict(test_input2)

    atribute = atributeSelectate.get_support(indices=True)

    #print('Attributes selected by RFE: '+str(antrenament_input.columns[atribute]))

    print('Accuracy after RFE selection:' + str(metrics.accuracy_score(prediction, test_output)) + '%')
    return (metrics.accuracy_score(prediction, test_output))

def extratreeclassif_Select(set_date):

    antrenament, test= train_test_split(set_date,test_size=0.3,shuffle=False)
    antrenament_input = antrenament[antrenament.columns[1:len(set_date.columns)]]
    antrenament_output= antrenament.diagnosis
    model= ExtraTreesClassifier().fit(antrenament_input, antrenament_output)
    atributeSelectate= SelectFromModel(model, prefit=True)
    test_input = test[test.columns[1:len(set_date.columns)]]
    test_output=test.diagnosis

    algoritm= AdaBoostClassifier()

    antrenament_input2=atributeSelectate.transform(antrenament_input)
    test_input2=atributeSelectate.transform(test_input)
    algoritm.fit(antrenament_input2,antrenament_output)
    prediction=algoritm.predict(test_input2)

    print(metrics.accuracy_score(prediction, test_output))



def lassoCV_visualiseAlpha(set_date):
    alphas = [0, 0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 2]
    model=LassoCV(alphas=alphas)
    set_date.replace(['M','B'],[0,1],inplace=True)
    print(set_date.head())
    antrenament, test= train_test_split(set_date,test_size=0.3,shuffle=False)
    antrenament_input = antrenament[antrenament.columns[1:len(set_date.columns)]]
    antrenament_output= antrenament.diagnosis
    model=LassoCV(alphas=alphas)
    plot=AlphaSelection(model)
    plot.fit(antrenament_input,antrenament_output)
    plot.poof()

def lasso_Select(set_date,alpha):

    set_date.replace(['M','B'],[0,1],inplace=True)
    #print(set_date.head())
    antrenament, test= train_test_split(set_date,test_size=0.3,shuffle=False)
    antrenament_input = antrenament[antrenament.columns[1:len(set_date.columns)]]
    antrenament_output= antrenament.diagnosis

    model =Lasso(alpha=alpha,normalize=True).fit(antrenament_input,antrenament_output)

    atributeSelectate= SelectFromModel(model, prefit=True)
    test_input = test[test.columns[1:len(set_date.columns)]]
    test_output=test.diagnosis

    algoritm= AdaBoostClassifier()

    antrenament_input2=atributeSelectate.transform(antrenament_input)
    test_input2=atributeSelectate.transform(test_input)
    algoritm.fit(antrenament_input2,antrenament_output)
    prediction=algoritm.predict(test_input2)

    atribute = atributeSelectate.get_support(indices=True)

    print('Attributes selected by LASSO: '+str(antrenament_input.columns[atribute]))

    print('Accuracy after LASSO selection:' + str(metrics.accuracy_score(prediction, test_output)) + '%')

def selectionComparisons():
    set_date=pandas.read_csv("data.csv",header=0)
    set_date=set_date.drop("id",axis=1)
    set_date=set_date.drop("Unnamed: 32", axis=1)

    #lassoCV_visualiseAlpha(set_date)

    antrenament, test = train_test_split(set_date, test_size=0.3, shuffle=False)
    antrenament_input = antrenament[antrenament.columns[1:len(set_date.columns)]]
    antrenament_output = antrenament.diagnosis

    # atributeSelectate = SelectKBest(chi2, k=optimum_number).fit(antrenament_input, antrenament_output)

    test_input = test[test.columns[1:len(set_date.columns)]]
    test_output = test.diagnosis
    algoritm = AdaBoostClassifier()
    algoritm.fit(antrenament_input, antrenament_output)
    prediction = algoritm.predict(test_input)
    print('Initial accuracy:' + str(metrics.accuracy_score(prediction, test_output)) + '%')

    #optimum_number= int(recursiveFeatureElim_CV(set_date))
    #chi2_Select(set_date,optimum_number)
    #fclassif_Select(set_date,optimum_number)
    #mutualinfoclassif_Select(set_date,optimum_number)
    #variancethreshold_Select(set_date)
    #recursivefeatureelim_Select(set_date,optimum_number)
    #extratreeclassif_Select(set_date)
    #lasso_Select(set_date,0.0001)
    #featureSecvential(set_date,optimum_number)


#selectionComparisons()

def exhaustiveSearch(method):
    set_date=pandas.read_csv("data.csv",header=0)
    set_date=set_date.drop("id",axis=1)
    set_date=set_date.drop("Unnamed: 32", axis=1)

    x=[]
    y=[]

    if method==1:
        for i in range(1,len(set_date.columns)):
            print(str(i)+':')
            value=chi2_Select(set_date, i)
            y.append(value)
            x.append(i)
        plt.plot(x, y)
        plt.title('chi2')
    elif method==2:
        for i in range(1,len(set_date.columns)):
            print(str(i)+':')
            value=mutualinfoclassif_Select(set_date, i)
            y.append(value)
            x.append(i)
        plt.plot(x, y)
        plt.title('mutual_info_classif')
    elif method==3:
        for i in range(1,len(set_date.columns)):
            print(str(i)+':')
            value=fclassif_Select(set_date, i)
            y.append(value)
            x.append(i)
        plt.plot(x, y)
        plt.title('f_classif')
    elif method==4:
        for i in range(1,len(set_date.columns)):
            print(str(i)+':')
            value=recursivefeatureelim_Select(set_date, i)
            y.append(value)
            x.append(i)
        plt.plot(x, y)
        plt.title('RFE')
    elif method==5:
        for i in range(1,len(set_date.columns)):
            print(str(i)+':')
            value=featureSecvential(set_date, i)
            y.append(value)
            x.append(i)
        plt.plot(x, y)
        plt.title('FS')

    print(x)
    print(y)

    plt.show()

#exhaustiveSearch(5)
