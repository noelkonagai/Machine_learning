import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from sklearn import svm


# for plotting, p indicates the coordinates of positive points, n that of negative points
def plot_part1():

	xp = [1,1]
	yp = [2,2]
	zp = [2,0]

	xn = [0,0]
	yn = [1,0]
	zn = [0,1]


	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')

	n = 100

	ax.scatter(xp, yp, zp, s=48, c='r', marker='.', depthshade=False)
	ax.scatter(xn, yn, zn, s=48, c='b', marker='.', depthshade=False)

	ax.set_xlabel('X')
	ax.set_ylabel('Y')
	ax.set_zlabel('Z')
	ax.set_xlim([0,2])

	plt.show()

# data has shape (6,4)
data = np.array([[1,1,1],[1,2,2],[1,2,0],[-1,0,0],[-1,1,0],[-1,0,1]])

y = data[:,0]
X = data[:,1:]

print(X.shape)
print(y.shape)

model = svm.SVC()
model.fit(X,y)

n = len(y) #rows
m = len(X[0]) #columns

# w = model.coef_
support = model.support_vectors_
n_support = model.n_support_

alphas = np.abs(model.dual_coef_).T

L = len(alphas)
w = np.zeros(len(X[0]))

for i in range(L):
	w += alphas[i][0] * y[i] * X[i]

b = support[0]

print('w',w)
print('b',b)
print(support)



