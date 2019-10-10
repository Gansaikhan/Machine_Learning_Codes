# @Author Gansaikhan Shur
# Written for Python 3.6

import tensorflow as tf
import tflearn
import numpy as np
import nltk
import random
import json
from nltk.stem.lancaster import LancasterStemmer

with open('intents.json') as f:
    data = json.load(f)

words = []
labels = []
docsx = []
docsy = []
training = []
output = []
stemmer = LancasterStemmer()


for message in data['Messages']:
    for pattern in message['patterns']:
        # Getting each word one by one
        words.extend(nltk.word_tokenize(pattern))
        docsy.append(message['tag'])
        wrds = nltk.word_tokenize(pattern)
        words.extend(wrds)
        docsx.append(wrds)

    if message['tag'] not in labels:
        labels.append(message['tag'])

# Removes strings and tries to get the root of the word
words = [stemmer.stem(word.lower())
         for word in words if word not in ("?" or "!")]
# No Duplicate Words
words = sorted(list(set(words)))
labels = sorted(labels)
output_emp = [0 for _ in range(len(labels))]

for idx, doc in enumerate(docsx):
    bag = []

    strings = [stemmer.stem(s) for s in doc]

    for word in words:
        if word in strings:
            bag.append(1)
        else:
            bag.append(0)

    output_r = output_emp[:]
    # One Hot
    output_r[labels.index(docsy[idx])] = 1

    training.append(bag)
    output.append(output_r)

training = np.array(training)
output = np.array(output)

# *Reset Graph
tf.reset_default_graph()

epochs = 300
batch_s = 8


network = tflearn.input_data(shape=[None, len(training[0])])
# Hidden layers with 10 Neurons
network = tflearn.fully_connected(network, 10)
network = tflearn.fully_connected(network, 10)
network = tflearn.fully_connected(
    network, len(output[0]), activation='softmax')
network = tflearn.regression(network)
model = tflearn.DNN(network)
model.fit(training, output, n_epoch=epochs,
          batch_size=batch_s, show_metric=True)
model.save("chatbotmodel.tflearn")
