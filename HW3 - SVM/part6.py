import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.model_selection import cross_val_score

# loading in the data
train = np.genfromtxt('mnist_train.txt', delimiter= ',')
test = np.genfromtxt('mnist_test.txt', delimiter = ',')

# part a) normalizing the data
train_norm = 2 * train / 255 - 1
X_train = train_norm[:,1:]
y_train = train[:,0]


test_norm = 2 * test / 255 - 1
X_test = test_norm[:,1:]
y_test = test[:,0]

# part b) exploring non-linear kernels
C = 1
kernel = 'rbf'
gamma = 1/len(X_train[0])

# fitting the model
def model(X, y, X_test, y_test, C, kernel, gamma):
	model = svm.SVC(C=C, kernel=kernel, gamma=gamma)
	model.fit(X,y)

	support_vectors = model.support_vectors_
	n_support = model.n_support_

	score = model.score(X_test, y_test)

	err_score = cross_val_score(model, X, y, cv=10)

	print("Accuracy: %0.2f (+/- %0.2f)" % (err_score.mean(), err_score.std() * 2))

	return err_score

# model(X_train, y_train, X_test, y_test, C, kernel, gamma)

# trying it with different C values

print('gamma = 2 gamma')
model(X_train, y_train, X_test, y_test, C, kernel, 2 * gamma)
print('gamma = 3 gamma')
model(X_train, y_train, X_test, y_test, C, kernel, 3 * gamma)
print('gamma = 4 gamma')
model(X_train, y_train, X_test, y_test, C, kernel, 4 * gamma)
print('gamma = 0.5 gamma')
model(X_train, y_train, X_test, y_test, C, kernel, 0.5 * gamma)
print('gamma = 0.25 gamma')
model(X_train, y_train, X_test, y_test, C, kernel, 0.25 * gamma)

# trying it with different gamma values

print('C = 0.5')
model(X_train, y_train, X_test, y_test, 0.5, kernel, gamma)
print('C = 0.8')
model(X_train, y_train, X_test, y_test, 0.8, kernel, gamma)
print('C = 1.1')
model(X_train, y_train, X_test, y_test, 1.1, kernel, gamma)
print('C = 1.5')
model(X_train, y_train, X_test, y_test, 1.5, kernel, gamma)
print('C = 2.0')
model(X_train, y_train, X_test, y_test, 2.0, kernel, gamma)



