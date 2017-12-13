# Text Generation using basic RNN via Tensorflow / Keras
# Andrew Bossie

import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

file = "Datasets/bible.txt"
inputText = open(file).read()

#unique chars
chars = sorted(list(set(inputText)))
char_to_int = dict((c,i) for i,c in enumerate(chars))
int_to_char = dict((i,c) for i,c in enumerate(chars))
n_chars = len(inputText)
n_unique = len(chars)
print(chars)
print(chars[13])

#hyperparams
sequence_length = 50
n_neurons = 200
epochs = 200
batch_size = 25
num_batches = n_unique/sequence_length
eta = 1e-4

#string to int

print("Formatting text...")
dataX = []
dataY = []
for i in range(0, n_chars - sequence_length, 1):
	seq_in = inputText[i:i+sequence_length]
	seq_out = inputText[i + sequence_length]
	dataX.append([char_to_int[char] for char in seq_in])
	dataY.append(char_to_int[seq_out])
patterns = len(dataX)
print("Success")

#reshape [total length, time steps, fetaures]
formatted_text = np.reshape(dataX, (patterns, sequence_length, 1))
y = np_utils.to_categorical(dataY)
# formatted_text = formatted_text / float(n_unique)

define model
X = tf.placeholder(tf.float32, [None, sequence_length, 1])
y = tf.placeholder(tf.float32, [None, sequence_length, 1])

print("Building model...")
RNN_Cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons, activation=tf.nn.relu)
outputs, states = tf.nn.dynamic_rnn(RNN_Cell, X, dtype=tf.float32)

loss = tf.reduce_mean(tf.square(outputs - y))
optimizer = tf.train.AdamOptimizer(learning_rate=eta)
training_op = optimizer.minimize(loss)
print("Success")

init = tf.global_variables_initializer()

train
with tf.Session() as sess:

	init.run()

	p=0

	print("Training Data...")
	for epoch in range(epochs):
		X_batch = formatted_text[p:p+sequence_length]
		y_batch = formatted_text[p+1:p+sequence_length + 1]
		sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
		# print(X_batch)
		if epoch % 100 == 0:
			mse = loss.eval(feed_dict={X: X_batch, y: y_batch})
			print(epoch, "\tMSE: ", mse)
		p += sequence_length

	print("Success")

#generate
sample_length = 20
start_index = np.random.randint(0, len(dataX) - 1)
pattern = dataX[start_index]
print("\"", ''.join([int_to_char[value] for value in pattern]), "\"")

for i in range(sample_length):
	X_new = np.reshape(pattern, (1, len(pattern), 1))
	# X_new = X_new/float(n_unique)
	y_pred = sess.run(outputs, feed_dict={X:X_new})
	index = np.argmax(y_pred)
	result = int_to_char[index]
	seq_in = [int_to_char[value] for value in pattern]
	print(result)
	pattern.append(index)
	pattern = pattern[1:len(pattern)]


