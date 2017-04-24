import numpy as np
import math
import random

def load_and_process():
	raw_data = np.loadtxt('ps5_data.csv', delimiter=',')
	raw_labels = np.loadtxt('ps5_data-labels.csv', delimiter='\n')
	theta1 = np.loadtxt('ps5_theta1.csv', delimiter=',')
	theta2 = np.loadtxt('ps5_theta2.csv', delimiter=',')

	rows = len(raw_data[:,0:1])
	cols = len(raw_data[0])

	ones = np.ones((rows,1))

	data = np.zeros((rows,cols + 1))
	data[:,0:1] = ones
	data[:,1:] = raw_data
	labels = raw_labels - np.ones((5000,))

	np.savetxt('data.csv', data, delimiter=',')
	np.savetxt('labels.csv', labels, delimiter=',')

	return data, labels, theta1, theta2

def load_files():
	theta1 = np.loadtxt('ps5_theta1.csv', delimiter=',')
	theta2 = np.loadtxt('ps5_theta2.csv', delimiter=',')
	data = np.loadtxt('data.csv', delimiter=',')
	labels = np.loadtxt('labels.csv', delimiter=',')

	return data, labels, theta1, theta2

def activation_value(data, weight, unit_num):
		
	z = np.dot(data, weight[unit_num])
	g = (1 / (1 + math.exp(-z)))

	return g

def hidden_layer(data, theta1, layers, inputs):

	a = [1]

	for i in range(layers):
		a.append(activation_value(data, theta1, i))
	
	a = np.asarray(a)

	return a

def output_layer(theta1, theta2, inputs, layers, outputs):

	prediction = []
	out = []
	a2 = []

	for i in range(inputs):
		a = hidden_layer(data[i], theta1, layers, inputs)
		a2.append(a)

		h = []

		for i in range(outputs):
			g = activation_value(a, theta2, i)
			h.append(g)

		prediction.append(h.index(max(h)))
		out.append(h)

	out = np.asarray(out)
	a2 = np.asarray(a2)

	prediction = np.asarray(prediction)

	return prediction, out, a2

def error_rate(labels, prediction, inputs):

	mistakes = 0

	for i in range(inputs):
		if(int(labels[i]) != int(prediction[i])):
			mistakes += 1

	error = mistakes / inputs * 100

	return error

def label_transform(labels, inputs, outputs):
	#the function transforms the labels ranging from 0-9 into arrays of 10, where the ith index has 1 to indicate which number it is
	y = np.zeros((inputs, outputs))

	for i in range(inputs):
		index = int(labels[i])
		y[i][index] = 1

	return y

def mle_single(h, y, outputs):
	
	cost = 0

	for i in range(outputs):
		cost -= (y[i] * math.log(h[i]) + (1 - y[i]) * math.log(1 - h[i]))

	return cost

def mle_full(h, y, inputs, outputs):

	cost = []

	for i in range(inputs):
		cost.append(mle_single(h[i], y[i], outputs))

	cost = np.asarray(cost)

	j = 1/inputs * sum(cost)

	return j

def derivatives(h, y, a1, a2, tehta1, theta2):

	d3 = h - y

	for j in range(len(d3)):

		d2 = np.dot(theta2.T,d3[j]) * a2 * (1 - a2)

		# for i in range(len(d2)):
		# 	d1 = np.dot(theta1.T,d2[i][1:]) * a1 * (1 - a1)

	# np.savetxt('d1.csv', d1, delimiter=',')
	np.savetxt('d2.csv', d2, delimiter=',')
	np.savetxt('d3.csv', d3, delimiter=',')

	print("Done.")

	return d2, d3

def stats_derivatives(a1, a2):

	d2 = np.loadtxt('d2.csv', delimiter=',')

	derivatives = []

	''' #this is the partial derivatives for all the 5000 datapoints

	for k in range(5000):
		for j in range(len(a1[0])):
			for i in range(len(d2[0])):
				derivatives.append(a1[k][j] * d2[k][i])

	'''

	for j in range(len(a1[0])):
		for i in range(len(d2[0])):
			derivatives.append(a1[0][j] * d2[0][i])

	abs_der = np.asarray(np.absolute(derivatives))

	mean = np.mean(abs_der)
	std = np.std(abs_der)
	med = np.median(abs_der)

	print("mean: ", mean)
	print("standard deviation: ", std)
	print("median: ", med)

	return mean, std, med

def generate_theta(theta1, theta2):

	print(theta1[0][0])

	for i in range(len(theta1)):
		for j in range(len(theta1[0])):
			theta1[i][j] = random.randrange(9, 11) / 10 * theta1[i][j]

	for i in range(len(theta2)):
		for j in range(len(theta2[0])):
			theta2[i][j] = random.randrange(9, 11) / 10 * theta2[i][j]

	print(theta1[0][0])

	return theta1, theta2

data, labels, theta1, theta2 = load_files()
prediction, h, a2 = output_layer(theta1, theta2, 5000, 25, 10)
error = error_rate(labels, prediction, 5000)
y = label_transform(labels, 5000, 10)
cost = mle_full(h, y, 5000, 10)
# d2, d3 = derivatives(h, y, data, a2, theta1, theta2)
# mean, std, med = stats_derivatives(data, a2)
new_theta1, new_theta2 = generate_theta(theta1, theta2)

# running with the new thetas
prediction2, h2, a22 = output_layer(new_theta1, new_theta2, 5000, 25, 10)
error2 = error_rate(labels, prediction, 5000)
cost2 = mle_full(h2, y, 5000, 10)
d2, d3 = derivatives(h, y, data, a2, new_theta1, new_theta2)
mean, std, med = stats_derivatives(data, a2)

# print('cost', cost)
# print('error percentage', error)
print('cost with new thetas', cost2)
print('error percentage with new thetas', error2)


