import numpy as np

x0 = np.array([1,1,1,1])
x1 = np.array([1600,2400,1416,3000])
x2 = np.array([3,3,2,4])
y = np.array([330, 369, 232, 540])
m = len(y)
alf = 0.1

print(x1.mean(), x2.mean(), y.mean())
print(x1.std(), x2.std(), y.std())

x1_norm = ( x1 - x1.mean() ) / x1.std()
x2_norm = ( x2 - x2.mean() ) / x2.std()
y_norm = ( y - y.mean() ) / y.std()

X = np.array([x0, x1_norm, x2_norm, y_norm]).T

print(X)

w = np.zeros(shape = (3,1))

def gradient_descent():

	for i in range(4):
		for j in range(3):
			w[j] -= (alf) * 1/m * np.dot(X[i,3],X[i,j])
			if j == 1:
				print(np.dot(X[i,3],X[i,j]))

	print(w)

gradient_descent()
