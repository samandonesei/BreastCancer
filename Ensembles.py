
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
import pandas
from sklearn import metrics


def randomForest(iteratii):
    set_date = pandas.read_csv("data.csv", header=0)
    set_date.drop("id", axis=1, inplace=True)
    set_date.drop("Unnamed: 32", axis=1, inplace=True)
    medie = 0
    for i in range(iteratii):
        antrenament, test = train_test_split(set_date, test_size=0.25)
        antrenament_input = antrenament[antrenament.columns[1:len(set_date.columns)]]
        antrenament_output = antrenament.diagnosis
        test_input = test[test.columns[1:len(set_date.columns)]]
        test_output = test.diagnosis
        algoritm = RandomForestClassifier()
        algoritm.fit(antrenament_input, antrenament_output)
        prediction = algoritm.predict(test_input)
        medie += metrics.accuracy_score(prediction, test_output)
        print(i)
    print(medie / iteratii)

def gridSearch_RF():
    set_date = pandas.read_csv("data.csv", header=0)
    set_date.drop("id", axis=1, inplace=True)
    set_date.drop("Unnamed: 32", axis=1, inplace=True)
    antrenament, test = train_test_split(set_date, test_size=0.25)
    antrenament_input = antrenament[antrenament.columns[1:len(set_date.columns)]]
    antrenament_output = antrenament.diagnosis
    test_input = test[test.columns[1:len(set_date.columns)]]
    test_output = test.diagnosis
    algoritm = RandomForestClassifier()
    parameters = {
        'n_estimators': [100, 200, 500, 1000],
        'max_features': ['sqrt', 'log2'],
        'max_depth': [None, 3, 4, 5, 6, 7, 8, 9, 10],
        'criterion': ['gini', 'entropy']
    }
    RF_GS = GridSearchCV(estimator=algoritm, param_grid=parameters, cv=10)
    RF_GS.fit(antrenament_input, antrenament_output)

    print(RF_GS.best_params_)

def maxVote_ensemble():
    set_date = pandas.read_csv("data.csv", header=0)
    set_date.drop("id", axis=1, inplace=True)
    set_date.drop("Unnamed: 32", axis=1, inplace=True)
    antrenament, test = train_test_split(set_date, test_size=0.25)
    antrenament_input = antrenament[antrenament.columns[1:len(set_date.columns)]]
    antrenament_output = antrenament.diagnosis
    test_input = test[test.columns[1:len(set_date.columns)]]
    test_output = test.diagnosis
    clasificator_1=GaussianNB()
    clasificator_2=KNeighborsClassifier()
    clasificator_3=DecisionTreeClassifier()
    clasificator_4=RandomForestClassifier(n_estimators=500)
    clasificator_5=AdaBoostClassifier()
    maxVote = VotingClassifier(estimators=[('nb', clasificator_1), ('knn', clasificator_2), ('dt', clasificator_3)], voting='hard')
    maxVote.fit(antrenament_input,antrenament_output)
    prediction = maxVote.predict(test_input)
    accuracy_0=metrics.accuracy_score(prediction, test_output)
    print(metrics.accuracy_score(prediction, test_output))

    clasificator_1.fit(antrenament_input,antrenament_output)
    prediction = clasificator_1.predict(test_input)
    accuracy_1=metrics.accuracy_score(prediction, test_output)
    print('Accuracy for clasificator_1:'+str(accuracy_1))

    clasificator_2.fit(antrenament_input,antrenament_output)
    prediction = clasificator_2.predict(test_input)
    accuracy_2=metrics.accuracy_score(prediction, test_output)
    print('Accuracy for clasificator_2:'+str(accuracy_2))

    clasificator_3.fit(antrenament_input,antrenament_output)
    prediction = clasificator_3.predict(test_input)
    accuracy_3=metrics.accuracy_score(prediction, test_output)
    print('Accuracy for clasificator_3:'+str(accuracy_3))

    return accuracy_0,accuracy_1,accuracy_2,accuracy_3

def moreaccurate_maxVote(iterations):

    mean_acc_1=0
    mean_acc_2=0
    mean_acc_3=0
    mean_acc_4=0

    for i in range(iterations):
        value_1, value_2, value_3, value_4= maxVote_ensemble()
        mean_acc_1+=value_1
        mean_acc_2+=value_2
        mean_acc_3+=value_3
        mean_acc_4+=value_4

    print('Average results:')
    print('MV:'+str(mean_acc_1/iterations)+'%')
    print('C1:' + str(mean_acc_2/iterations) + '%')
    print('C2:' + str(mean_acc_3/iterations) + '%')
    print('C3:' + str(mean_acc_4/iterations) + '%')





def maxVote_ensemble2():
    set_date = pandas.read_csv("data.csv", header=0)
    set_date.drop("id", axis=1, inplace=True)
    set_date.drop("Unnamed: 32", axis=1, inplace=True)
    antrenament, test = train_test_split(set_date, test_size=0.25)
    antrenament_input = antrenament[antrenament.columns[1:len(set_date.columns)]]
    antrenament_output = antrenament.diagnosis
    test_input = test[test.columns[1:len(set_date.columns)]]
    test_output = test.diagnosis
    clasificator_1=GaussianNB()
    clasificator_2=KNeighborsClassifier()
    clasificator_3=DecisionTreeClassifier()
    clasificator_4=RandomForestClassifier(n_estimators=500)
    clasificator_5=AdaBoostClassifier()
    maxVote = VotingClassifier(estimators=[('nb', clasificator_1), ('knn', clasificator_2), ('dt', clasificator_3), ('rf', clasificator_4), ('ab',clasificator_5)], voting='hard')
    maxVote.fit(antrenament_input,antrenament_output)
    prediction = maxVote.predict(test_input)
    accuracy_0=metrics.accuracy_score(prediction, test_output)
    print(metrics.accuracy_score(prediction, test_output))

    clasificator_1.fit(antrenament_input,antrenament_output)
    prediction = clasificator_1.predict(test_input)
    accuracy_1=metrics.accuracy_score(prediction, test_output)
    print('Accuracy for clasificator_1:'+str(accuracy_1))

    clasificator_2.fit(antrenament_input,antrenament_output)
    prediction = clasificator_2.predict(test_input)
    accuracy_2=metrics.accuracy_score(prediction, test_output)
    print('Accuracy for clasificator_2:'+str(accuracy_2))

    clasificator_3.fit(antrenament_input,antrenament_output)
    prediction = clasificator_3.predict(test_input)
    accuracy_3=metrics.accuracy_score(prediction, test_output)
    print('Accuracy for clasificator_3:'+str(accuracy_3))

    clasificator_4.fit(antrenament_input, antrenament_output)
    prediction = clasificator_4.predict(test_input)
    accuracy_4 = metrics.accuracy_score(prediction, test_output)
    print('Accuracy for clasificator_4:' + str(accuracy_4))

    clasificator_5.fit(antrenament_input, antrenament_output)
    prediction = clasificator_5.predict(test_input)
    accuracy_5 = metrics.accuracy_score(prediction, test_output)
    print('Accuracy for clasificator_5:' + str(accuracy_5))

    return accuracy_0,accuracy_1,accuracy_2,accuracy_3, accuracy_4, accuracy_5

def moreaccurate_maxVote2(iterations):

    mean_acc_0=0
    mean_acc_1=0
    mean_acc_2=0
    mean_acc_3=0
    mean_acc_4=0
    mean_acc_5=0

    for i in range(iterations):
        value_0, value_1, value_2, value_3, value_4, value_5= maxVote_ensemble2()
        mean_acc_0+=value_0
        mean_acc_1+=value_1
        mean_acc_2+=value_2
        mean_acc_3+=value_3
        mean_acc_4+=value_4
        mean_acc_5+=value_5

    print('Average results:')
    print('MV:'+str(mean_acc_0/iterations)+'%')
    print('C1:' + str(mean_acc_1/iterations) + '%')
    print('C2:' + str(mean_acc_2/iterations) + '%')
    print('C3:' + str(mean_acc_3/iterations) + '%')
    print('C4:' + str(mean_acc_4 / iterations) + '%')
    print('C5:' + str(mean_acc_5 / iterations) + '%')





#gridSearch_RF()

#randomForest()

#maxVote_ensemble()

#moreaccurate_maxVote(500)

#moreaccurate_maxVote2(100)

maxVote_ensemble2()