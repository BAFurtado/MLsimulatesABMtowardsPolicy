
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import normalize
from sklearn.svm import SVC


def normalize_trial(x, xt):
    n = []
    for each in [x, xt]:
        n.append(normalize(each.as_matrix()))
    return n[0], n[1]


def run_classifiers(x, x_test, y, y_test):

    models = ['Tree', 'MPL', 'SVC', 'Voting']
    # models = ['Tree']

    m1 = RandomForestClassifier(n_estimators=10000, criterion='gini', bootstrap=True, max_depth=15)
    m3 = SVC(C=1, kernel='poly', degree=3, probability=True)
    m4 = MLPClassifier(solver='lbfgs', early_stopping=True, activation='tanh', max_iter=2000)
    voting = VotingClassifier(estimators=[('rf', m1), ('svc', m3), ('neural', m4)], voting='soft')

    # Fitting models
    cls = [m1, m3, m4, voting]
    # cls = [m1]
    for each in cls:
        each.fit(x, y)

    models = dict(zip(models, cls))

    # Calculating accuracies and printing
    for key in models.keys():
        print('Score {}: {:.4f}.'.format(key, models[key].score(x_test, y_test)))
        # Examining confusion matrix
        yhat = models[key].predict(x_test)
        cm = confusion_matrix(y_test, yhat)
        with open(f'output/confusion_matrix_{key}.txt', 'w') as handler:
            msg = 'Confusion Matrix {}:\n {}.'.format(key, cm)
            handler.write(msg)
            print('Confusion Matrix {}:\n {}.'.format(key, cm))
    # Returns a dictionary of models' names and the model itself
    return models


def predict(model, data):
    return model.predict(data)


def fit(model, a, b):
    model.fit(a, b)
