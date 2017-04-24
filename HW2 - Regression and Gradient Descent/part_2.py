import numpy as np
import matplotlib.pyplot as plt

#loading data into a matrix
data = np.loadtxt('housing.txt', delimiter=',')

# columns 0, 1 are features x1, x2; column 2 is the housing price y
X = data[:,0:2]
y = data[:, 2]

# in housing.txt m = 47, n = 2
m = y.size 
n = X.shape[1]

#redefining X with a column of 1's in the front, as entries of x0 are all ones
a = np.ones(shape=(1,m))
X = np.vstack((a, X.T)).T

#variables for the gradient descent
itr = 20
alpha = 0.1

J_vals = []
J_vals_stochastic = []

def mean_stddev(X):
	#function for part 2A-b

	#column 1
	mean_1 = np.mean(X[:, 1])
	stddev_1 = np.std(X[:, 1])

	#column 2
	mean_2 = np.mean(X[:, 2])
	stddev_2 = np.std(X[:, 2])

	#price, y value
	mean_y = np.mean(y)
	stddev_y = np.std(y)

	return mean_1, stddev_1, mean_2, stddev_2, mean_y, stddev_y

def normalize(X, y, mean_1, stddev_1, mean_2, stddev_2, mean_y, stddev_y):
	#function for part 2A-c
	X_norm = X
	y_norm = y

	#normalize column 1
	X_norm[:, 1] = (X_norm[:, 1] - mean_1) / stddev_1

	#normalize column 2
	X_norm[:, 2] = (X_norm[:, 2] - mean_2) / stddev_2

	#normalize y
	y_norm = (y_norm - mean_y) / stddev_y

	header = 'mean_1:' + str(mean_1) + ', stddev_1: ' + str(stddev_1) + 'mean_2:' + str(mean_2) + ', stddev_2: ' + str(stddev_2) + 'mean_y:' + str(mean_y) + ', stddev_y: ' + str(stddev_y) 
	np.savetxt('normalized_X.txt', X_norm, delimiter=',', newline='\n', header= header)
	np.savetxt('normalized_y.txt', y_norm, delimiter=',', newline='\n', header= header)

	return X_norm, y_norm

def sqr_err(X, y, w):
	#function for 2B-a; this finds the square error for given row of x, and y

	f = np.dot(X,w)
	sqr_err = (f - y) ** 2
	s = 1/(2 * m) * sqr_err.sum()

	return s

def sqr_err_sgd(X_sgd, y_sgd, w_sgd):
	#function for 2B-a; this finds the square error for given row of x, and y

	f_sgd = np.dot(X_sgd,w_sgd)
	y_sgd.shape = (m,1)
	sqr_err_sgd = (f_sgd - y_sgd) ** 2
	s_sgd = 1/(2 * m) * sqr_err_sgd.sum()

	return s_sgd

def update_w(X, y, itr, alpha):

	decrease = []

	y.shape = (m,1)
	w_gd = np.zeros(shape=(n+1,1))

	for i in range(itr):

		f = np.dot(X,w_gd)

		for j in range(n+1):

			X_col = X[:,j].T
			X_col.shape = (1,m)

			w_gd[j] = w_gd[j] - alpha * 1/m * np.dot(X[:,j].T, (f-y))

		sqr_gd = sqr_err(X, y, w_gd)
		decrease.append(sqr_gd)
		
		# if decrease[i] - decrease[i-1] < 0:
		# 	print('iteration', i, 'gd decreasing. J(w) =', decrease[i])
		# else:
		# 	print('iteration',  i, 'gd not decreasing. J(w) =', decrease[i])


	J_vals.append([decrease[itr-1]])

	return w_gd, J_vals

mean_1, stddev_1, mean_2, stddev_2, mean_y, stddev_y = mean_stddev(X)
X_norm, y_norm = normalize(X, y, mean_1, stddev_1, mean_2, stddev_2, mean_y, stddev_y)

def predict(x1, x2):
	#2C finding the predicted price with values for x1 and x2
	w, J_vals = update_w(X_norm, y, 80, 0.3)

	x0 = 1
	x1_norm = (x1 - mean_1) / stddev_1
	x2_norm = (x2 - mean_2) / stddev_2

	x_array = np.array([x0, x1_norm, x2_norm])

	y_prediction = np.dot(x_array,w)
	
	return y_prediction

# y_prediction = predict(1650, 3)
# print(y_prediction)

def stochastic(X, y, itr, alpha):
	#2D finding the stochastic gradient descent

	#because we randomly shuffle values, we create an augmented X that contains y values as well
	X_full = np.vstack((X.T, y.T)).T

	decrease_sgd = []

	w_sgd = np.zeros(shape=(n+1,1))

	for k in range(itr):

		np.random.shuffle(X_full)

		y_temp = X_full[:,3]
		y_temp.shape = (m, 1)
		temp2 = X_full[0:1,1]

		f = np.dot(X_full[:,0:3],w_sgd)

		temp = (f - y_temp)[0:1]

		for i in range(m):
			for j in range(n+1):
				w_sgd[j] += - (alpha * 1/m * np.dot((f - y_temp)[i],X_full[i,j]))

		sqr_sgd = sqr_err_sgd(X_full[:,0:3], X_full[:,3], w_sgd)
		decrease_sgd.append(sqr_sgd)
		
		if decrease_sgd[k] - decrease_sgd[k-1] < 0:
			print('iteration ', k, 'sgd decreasing. J(w) =', decrease_sgd[k])
		else:
			print('iteration ', k, 'sgd not decreasing. J(w) =', decrease_sgd[k])

	J_vals_stochastic.append([decrease_sgd[itr-1]])

	return w_sgd, J_vals_stochastic

#Q2D, comparison of vanilla and stochastic gradient decents
# w_stochastic, J_vals_stochastic = stochastic(X_norm, y, 3, 0.1)
# w_vanilla, J_vals_vanilla = update_w(X_norm, y, 80, 0.1)

# print(J_vals_stochastic, w_stochastic)
# print(J_vals_vanilla)

def J_vs_itr():
	#2B-b plotting J(w) against iterations, holding alpha at a constant 0.1
	iterations = [10, 20, 30, 40, 50, 60, 70, 80]

	for i in iterations:
		w, J_vals = update_w(X_norm, y_norm, i, 0.05)

	plt.plot(iterations, J_vals)
	plt.ylabel('Values of J(w)')
	plt.xlabel('Iterations')
	plt.show()

def J_vs_alpha():
	#2B-d plotting J(w) against alpha values, holding itr at a constant

	alpha_vals = [0.05, 0.1, 0.3]
	# alpha_vals = [0.005, 0.007, 0.01, 0.02, 0.05, 0.1, 0.3]

	# for i in alpha_vals:
	# 	w, J_vals = stochastic(X_norm, y_norm, 10, i)

	# plt.plot(alpha_vals, J_vals)
	# plt.ylabel('Values of J(w)')
	# plt.xlabel('Values of alpha')
	# plt.show()

	for i in alpha_vals:
		w, J_vals = update_w(X_norm, y_norm, 80, i)
		print(J_vals)

	plt.plot(alpha_vals, J_vals)
	plt.ylabel('Values of J(w)')
	plt.xlabel('Values of alpha')
	plt.show()

# J_vs_alpha()
J_vs_itr()
