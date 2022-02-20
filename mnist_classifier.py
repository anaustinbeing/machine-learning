import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB
from sklearn.datasets import fetch_openml
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix


mnist = fetch_openml('mnist_784', version=1)
print('Keys: ', mnist.keys())

X_austin = mnist['data'].to_numpy()
y_austin = mnist['target'].to_numpy()

print('Type of data: ', type(X_austin))
print('Type of target: ', type(y_austin))

print('Shape of data: ', X_austin.shape)
print('Shape of target: ', y_austin.shape)

some_digit1 = X_austin[7]
some_digit2 = X_austin[5]
some_digit3 = X_austin[0]

some_digit1_image = some_digit1.reshape(28, 28)
some_digit2_image = some_digit2.reshape(28, 28)
some_digit3_image = some_digit3.reshape(28, 28)

plt.imshow(some_digit1_image, cmap=mpl.cm.binary)
plt.title('some_digit1 - X[7]')
plt.show()

plt.imshow(some_digit2_image, cmap=mpl.cm.binary)
plt.title('some_digit2 - X[5]')
plt.show()

plt.imshow(some_digit3_image, cmap=mpl.cm.binary)
plt.title('some_digit3 - X[0]')
plt.show()

# Preprocessing

y_austin = np.uint8(y_austin)

y_austin_new = np.where((0 <= y_austin) & (y_austin <= 3), 0, y_austin)
y_austin_new = np.where((4 <= y_austin_new) & (y_austin_new <= 6), 1, y_austin_new)
y_austin_new = np.where((7 <= y_austin_new) & (y_austin_new <= 9), 9, y_austin_new)

print('Frequency of target 0: ', (y_austin_new == 0).sum())
print('Frequency of target 1: ', (y_austin_new == 1).sum())
print('Frequency of target 9: ', (y_austin_new == 9).sum())

# splitting dataset for training and testing
x_train, x_test, y_train, y_test = X_austin[:60000], X_austin[60000:], y_austin_new[:60000], y_austin_new[60000:]

NB_clf_austin = MultinomialNB().fit(x_train, y_train)
some_digit1_predict = NB_clf_austin.predict(some_digit1.reshape(1, -1))
print('Prediction for some_digit1: ', some_digit1_predict)
some_digit2_predict = NB_clf_austin.predict(some_digit2.reshape(1, -1))
print('Prediction for some_digit2: ', some_digit2_predict)
some_digit3_predict = NB_clf_austin.predict(some_digit3.reshape(1, -1))
print('Prediction for some_digit3: ', some_digit3_predict)



def perform_cross_validation(model_classifier, X, Y):
    print("\n****** Performing cross validation ******")
    cross_score = cross_val_score(model_classifier, X, Y, cv=3, scoring='accuracy')
    score_min = cross_score.min()
    score_max = cross_score.max()
    score_mean = cross_score.mean()
    print("MIN: ", score_min, "MEAN: ", score_mean, "MAX: ", score_max);
    print('Cross val score: ', cross_score)

perform_cross_validation(NB_clf_austin, x_train, y_train)

y_test_predicts = NB_clf_austin.predict(x_test)
print('Accuracy score on test data: ', accuracy_score(y_test, y_test_predicts))
# or you can use: NB_clf_austin.score(x_test, y_test)

# to view the classes: NB_clf_austin.classes_

print('Accuracy matrix: \n', confusion_matrix(y_test, y_test_predicts))

########### Logistic Regression

from sklearn.linear_model import LogisticRegression

'''
    Using solver='lbfgs'
'''
print('\n########## Using solver="lbfgs" ###########\n')
LR_clf_austin = LogisticRegression(multi_class='multinomial', max_iter=1000, tol=0.1, solver='lbfgs')
LR_clf_austin.fit(X_austin, y_austin_new)

LR_clf_austin.predict_proba(X_austin)#(X_austin[0].reshape(1, -1))

some_digit1_predict = LR_clf_austin.predict(some_digit1.reshape(1, -1))
print('Prediction for some_digit1: ', some_digit1_predict)
some_digit2_predict = LR_clf_austin.predict(some_digit2.reshape(1, -1))
print('Prediction for some_digit2: ', some_digit2_predict)
some_digit3_predict = LR_clf_austin.predict(some_digit3.reshape(1, -1))
print('Prediction for some_digit3: ', some_digit3_predict)

perform_cross_validation(LR_clf_austin, x_train, y_train)

y_test_predicts = LR_clf_austin.predict(x_test)
print('Accuracy score on test data: ', LR_clf_austin.score(x_test, y_test))
print('Precision: ', precision_score(y_test, y_test_predicts, average=None))
print('Recall: ', recall_score(y_test, y_test_predicts, average=None))

'''
    Using solver='saga'
'''

print('\n########## Using solver="saga" ###########\n')
LR_clf_austin = LogisticRegression(multi_class='multinomial', max_iter=1000, tol=0.1, solver='saga')
LR_clf_austin.fit(X_austin, y_austin_new)

LR_clf_austin.predict_proba(X_austin)#(X_austin[0].reshape(1, -1))

some_digit1_predict = LR_clf_austin.predict(some_digit1.reshape(1, -1))
print('Prediction for some_digit1: ', some_digit1_predict)
some_digit2_predict = LR_clf_austin.predict(some_digit2.reshape(1, -1))
print('Prediction for some_digit2: ', some_digit2_predict)
some_digit3_predict = LR_clf_austin.predict(some_digit3.reshape(1, -1))
print('Prediction for some_digit3: ', some_digit3_predict)

perform_cross_validation(LR_clf_austin, x_train, y_train)

y_test_predicts = LR_clf_austin.predict(x_test)
print('Accuracy score on test data: ', LR_clf_austin.score(x_test, y_test))
print('Precision: ', precision_score(y_test, y_test_predicts, average=None))
print('Recall: ', recall_score(y_test, y_test_predicts, average=None))