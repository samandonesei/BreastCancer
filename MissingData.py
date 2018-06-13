from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas
import numpy as np
from sklearn import metrics
import random
import sklearn
import csv

def KNN_normal():
    set_date=pandas.read_csv("data.csv",header=0)
    set_date.drop("id",axis=1,inplace=True)
    set_date.drop("Unnamed: 32",axis=1,inplace=True)
    first_set=list(set_date.columns[1:11])
    coloane = ['texture_mean','perimeter_mean','smoothness_mean','compactness_mean','symmetry_mean']
    antrenament, test = train_test_split(set_date, test_size=0.3, shuffle=False)
    print(antrenament.shape)
    antrenament_input = antrenament[coloane]
    antrenament_output = antrenament.diagnosis
    test_input = test[coloane]
    test_output = test.diagnosis
    algoritm = KNeighborsClassifier(n_neighbors=15)
    algoritm.fit(antrenament_input, antrenament_output)
    prediction = algoritm.predict(test_input)
    print(metrics.accuracy_score(prediction, test_output))
def LRegression_normal():
    set_date=pandas.read_csv("data.csv",header=0)
    set_date.drop("id",axis=1,inplace=True)
    set_date.drop("Unnamed: 32",axis=1,inplace=True)
    '''
    first_set=list(set_date.columns[1:11])
    second_set=list(set_date.columns[11:20])
    last_set=list(set_date.columns[21:31])
    '''
    #coloane=['texture_mean','perimeter_mean','smoothness_mean','compactness_mean','symmetry_mean']
    antrenament,test=train_test_split(set_date,test_size=0.3)
    #antrenament_input=antrenament[first_set]
    #antrenament_input=antrenament[second_set]
    #antrenament_input=antrenament[last_set]
    #antrenament_input=antrenament[coloane]
    antrenament_output=antrenament.diagnosis
    antrenament.drop("diagnosis",axis=1,inplace=True)
    antrenament_input=antrenament
    #test_input=test[first_set]
    #test_input=test[second_set]
    #test_input=test[last_set]
    #test_input=test[coloane]
    test_output=test.diagnosis
    test.drop("diagnosis",axis=1,inplace=True)
    test_input=test
    algoritm=LogisticRegression()
    algoritm.fit(antrenament_input,antrenament_output)
    prediction=algoritm.predict(test_input)
    print(metrics.accuracy_score(prediction,test_output))
def generate_random(coeficient_random):
    set_date=pandas.read_csv("data.csv",header=0)
    set_date.drop("id",axis=1,inplace=True)
    set_date.drop("Unnamed: 32",axis=1,inplace=True)
    first_set=list(set_date.columns[1:11])
    coloane_test=['texture_mean','perimeter_mean','smoothness_mean','compactness_mean','symmetry_mean']
    coloane = ['texture_mean','perimeter_mean','smoothness_mean','compactness_mean','symmetry_mean','diagnosis']
    antrenament, test = train_test_split(set_date, test_size = 0.3 , shuffle= False)
    antrenament = antrenament[coloane]
    #antrenament_output= antrenament.diagnosis
    test_input = test[coloane_test]
    test_output= test.diagnosis
    #print(antrenament_input)
    #print(antrenament.perimeter_mean)
    #print(antrenament_input[coloane[1]][3])
    #print(antrenament_input.perimeter_mean[3])
    for i in range(5):
        length=len(antrenament[coloane[i]])
        for j in range(length):
            chance=random.randint(0,coeficient_random)
            if chance==1:
                antrenament[coloane[i]][j]=np.NaN
    print(antrenament.isnull().sum())
    '''algoritm=KNeighborsClassifier(n_neighbors=15)
    algoritm.fit(antrenament_input,antrenament_output)
    prediction=algoritm.predict(test_input)
    medie+=metrics.accuracy_score(prediction,test_output)'''
    return antrenament,test_input,test_output


def tratare_eliminare_KNN():
    antrenament,test_input, test_output=generate_random(2)
    #print(antrenament_input.head(20))
    algoritm=KNeighborsClassifier(n_neighbors=15)
    print(antrenament.shape)
    antrenament.dropna(inplace=True)
    print(antrenament.shape)
    coloane = ['texture_mean', 'perimeter_mean', 'smoothness_mean', 'compactness_mean', 'symmetry_mean']
    antrenament_input=antrenament[coloane]
    print(antrenament_input.shape)
    antrenament_output=antrenament.diagnosis
    print(antrenament_output.shape)
    print(test_input.shape)
    print(test_output.shape)
    algoritm.fit(antrenament_input,antrenament_output)
    #print(test_input)
    prediction=algoritm.predict(test_input)
    print(metrics.accuracy_score(prediction, test_output))
def tratare_eliminare_LRegression(antrenament,test_input,test_output):
    #antrenament,test_input,test_output=generate_random(5)
    print(antrenament.shape)
    algoritm = LogisticRegression()
    antrenament.dropna(inplace=True)
    print(antrenament.shape)
    #coloane = ['texture_mean', 'perimeter_mean', 'smoothness_mean', 'compactness_mean', 'symmetry_mean']
    #antrenament_input=antrenament[coloane]
    #print(antrenament_input.shape)
    #print(antrenament_input.shape)
    antrenament_output=antrenament.diagnosis
    antrenament.drop("diagnosis",axis=1,inplace=True)
    antrenament_input=antrenament
    algoritm.fit(antrenament_input, antrenament_output)
    prediction = algoritm.predict(test_input)
    print(metrics.accuracy_score(prediction, test_output))
def tratare_medie_LRegression(antrenament,test_input,test_output):
    #antrenament,test_input,test_output=generate_random(5)
    print(antrenament.shape)
    algoritm = LogisticRegression()
    antrenament.fillna(antrenament.mean(),inplace=True)
    print(antrenament.shape)
    #coloane = ['texture_mean', 'perimeter_mean', 'smoothness_mean', 'compactness_mean', 'symmetry_mean']
    #antrenament_input=antrenament[coloane]
    #print(antrenament_input.shape)
    #print(antrenament_input.shape)
    antrenament_output=antrenament.diagnosis
    antrenament.drop("diagnosis",axis=1,inplace=True)
    antrenament_input=antrenament
    algoritm.fit(antrenament_input, antrenament_output)
    prediction = algoritm.predict(test_input)
    print(metrics.accuracy_score(prediction, test_output))
def tratare_mediana_LRegression(antrenament,test_input,test_output):
    #antrenament,test_input,test_output=generate_random(5)
    print(antrenament.shape)
    algoritm = LogisticRegression()
    antrenament.fillna(antrenament.median(),inplace=True)
    print(antrenament.shape)
    #coloane = ['texture_mean', 'perimeter_mean', 'smoothness_mean', 'compactness_mean', 'symmetry_mean']
    #antrenament_input=antrenament[coloane]
    #print(antrenament_input.shape)
    #print(antrenament_input.shape)
    antrenament_output=antrenament.diagnosis
    antrenament.drop("diagnosis",axis=1,inplace=True)
    antrenament_input=antrenament
    algoritm.fit(antrenament_input, antrenament_output)
    prediction = algoritm.predict(test_input)
    print(metrics.accuracy_score(prediction, test_output))
def tratare_mode_LRegression(antrenament,test_input,test_output):
    print(antrenament.shape)
    algoritm = LogisticRegression()
    antrenament.fillna(antrenament.mode().ix[0],inplace=True)
    print(antrenament.shape)
    #coloane = ['texture_mean', 'perimeter_mean', 'smoothness_mean', 'compactness_mean', 'symmetry_mean']
    #antrenament_input=antrenament[coloane]
    #print(antrenament_input.shape)
    #print(antrenament_input.shape)
    antrenament_output=antrenament.diagnosis
    antrenament.drop("diagnosis",axis=1,inplace=True)
    antrenament_input=antrenament
    algoritm.fit(antrenament_input, antrenament_output)
    prediction = algoritm.predict(test_input)
    print(metrics.accuracy_score(prediction, test_output))
def tratare_KNNfill_LRegression(antrenament,test_input,test_output):
    print(antrenament.shape)
    algoritm=LogisticRegression()

LRegression_normal()
antrenament,test_input,test_output=generate_random(5)
antrenament2=antrenament.copy()
antrenament3=antrenament.copy()
antrenament4=antrenament.copy()
#antrenament5=antrenament.copy()
test_input2=test_input.copy()
test_input3=test_input.copy()
test_input4=test_input.copy()
#test_input5=test_input.copy()
test_output2=test_output.copy()
test_output3=test_output.copy()
test_output4=test_output.copy()
#test_output5=test_output.copy()
tratare_eliminare_LRegression(antrenament,test_input,test_output)
tratare_medie_LRegression(antrenament2,test_input2,test_output2)
tratare_mediana_LRegression(antrenament3,test_input3,test_output3)
tratare_mode_LRegression(antrenament4,test_input4,test_output4)

##### Outputuri #####

# ~0.94-0.96           |        |        |        |        |        |        |
# 0.9181 cu eliminare  | 0.9356 | 0.9181 | 0.8713 | 0.9239 | 0.9181 | 0.8771 | 0.8888
# 0.9181 cu medie      | 0.9064 | 0.9239 | 0.9064 | 0.9064 | 0.9181 | 0.9122 | 0.9122
# 0.9005 cu mediana    | 0.8888 | 0.9122 | 0.8947 | 0.9005 | 0.9005 | 0.9064 | 0.9064
# 0.8245 cu mode       | 0.8187 | 0.8596 | 0.8128 | 0.8421 | 0.8654 | 0.8421 | 0.8304


'''
set_date=pandas.read_csv("data.csv",header=0)
set_date.drop("id", axis=1, inplace=True)
set_date.drop("Unnamed: 32", axis=1, inplace=True)
first_set = list(set_date.columns[1:11])
coloane_test = ['texture_mean', 'perimeter_mean', 'smoothness_mean', 'compactness_mean', 'symmetry_mean']
coloane = ['texture_mean', 'perimeter_mean', 'smoothness_mean', 'compactness_mean', 'symmetry_mean', 'diagnosis']
date=set_date[coloane]
for i in range(5):
    length=len(date[coloane[i]])
    for j in range(length):
        chance=random.randint(0,5)
        if chance==1:
            date[coloane[i]][j]=np.NaN
print(date)
'''
#writer=csv.writer(open("date.csv",'w'))
#writer.writerows(date)
#tratare_mode_LRegression(antrenament4,test_input4,test_output4)
#KNN_normal()
#tratare_eliminare_KNN()