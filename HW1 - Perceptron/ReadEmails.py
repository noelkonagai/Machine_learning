import timeit, json
import numpy as np

features = []

stop = 0

f_train = open('spam_train.txt', 'r')
f_test = open('spam_test.txt', 'r')
f_words = open('words_dict.txt', 'r')
f_frequency = open('words_frequency.txt', 'w')
f_words_5000 = open('words_dict_5000.txt', 'w')
f_frequency_5000 = open('words_frequency_5000.txt', 'w')

train_sample = 4000
validation_sample = 5000
X = 60

start = timeit.default_timer()

def words(data,X):
	global e, words_frequent

	# words = {}
	words_frequent = {}

	words = json.load(f_words)

	e = [ line.rstrip().split(" ") for line in data ]

	# toggle commenting out to create new words dictionary
	# for i in range(train_sample):
	# 	for word in e[i]:
	# 		words.setdefault(word, 0)

	# 	for word in words:
	# 		if word in e[i]:
	# 			words[word] += 1

	# words.pop('1', None)
	# words.pop('-1', None)

	k = 0
	for key in words:
		if words[key] >= X:
			words_frequent[key] = k
			k += 1

	data.close()

	# json.dump(words, f_words)
	# json.dump(words_frequent, f_frequency)

	print('number of features', len(words_frequent))

	return words_frequent

def feature_vector(email):
	global features
	
	x = []

	for key in words_frequent:
		if key in email:
			x.append(1)
		else:
			x.append(0)

	features.append(x)

	return features

def perceptron_train(data):
	global w

	itr = 0
	w = np.zeros(len(words_frequent))

	while True:
		itr += 1
		k = 0

		for i in range(train_sample):
			x = x_vectors[i]
			y = int(e[i][0])

			if 1 in x:
				if y * np.dot(x,w) <= 0:
					w = w + (y * x)
					k += 1
			else:
				k = 0

		
		if itr > 1 and k == 0:
			print('itr: ', itr, ' k: ', k)
			return w

		print('itr: ', itr, ' k: ', k)

def perceptron_error(w, data):

	k = 0
	c = validation_sample - train_sample

	for i in range(train_sample,validation_sample):
		x = x_vectors[i]
		y = int(e[i][0])

		if y * np.dot(x,w) <= 0:
			k += 1

	error = k/c

	print('error: ', error)

	return error

words(f_train, X)

for i in range(validation_sample):
	feature_vector(e[i])

x_vectors = np.array(features)

perceptron_train(e)
perceptron_error(w, e)

largest_12 = np.argsort(w)[-12:]
smallest_12 = np.argsort(w)[:12]


print('the most positively weighed words are')
for i in range(12):
	print(str(i+1) + 'th', list(words_frequent)[largest_12[i]])

print('the most negatively weighed words are')
for i in range(12):
	print(str(i+1) + 'th', list(words_frequent)[smallest_12[i]])

stop = timeit.default_timer()

print('seconds taken', stop - start)

f_train.close()
f_test.close()
f_words.close()
f_frequency.close()

# emails trip you over, 3208, 1108, make sure w changes