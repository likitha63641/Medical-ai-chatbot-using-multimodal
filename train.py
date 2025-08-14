# Import necessary libraries
import random
import numpy as np
import json
import nltk
import pickle
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.optimizers import SGD
from keras.models import Sequential
from keras.layers import Dense, Dropout

# Initialize the lemmatizer and download necessary NLTK data
lemmatizer = WordNetLemmatizer()
nltk.download('omw-1.4')
nltk.download("punkt")
nltk.download("wordnet")

# Initialize variables
words = []
classes = []
documents = []
ignore_words = ["?", "!"]
data_file = open("intents.json").read()
intents = json.loads(data_file)

# Tokenize and process patterns in intents
for intent in intents["intents"]:
    for pattern in intent["patterns"]:
        # Tokenize each word in the sentence
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        # Add the tokenized words to the documents list with the corresponding intent
        documents.append((w, intent["tag"]))

        # Add the intent (tag) to the class list if not already present
        if intent["tag"] not in classes:
            classes.append(intent["tag"])

# Lemmatize and lower each word and remove duplicates
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))

# Sort classes
classes = sorted(list(set(classes)))

# Print data summary
print(f"{len(documents)} documents")
print(f"{len(classes)} classes: {classes}")
print(f"{len(words)} unique lemmatized words: {words}")

# Save the words and classes lists using pickle
pickle.dump(words, open("words.pkl", "wb"))
pickle.dump(classes, open("classes.pkl", "wb"))

# Initialize training data
training = []
output_empty = [0] * len(classes)

# Create the bag of words and output row for each document
for doc in documents:
    # Initialize the bag of words with 0s for each word in the vocabulary
    bag = [0] * len(words)

    # Get the list of tokenized words for the current pattern
    pattern_words = doc[0]

    # Lemmatize each word in the pattern
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]

    # Create the bag of words: 1 if the word exists in the pattern, otherwise 0
    for w in words:
        if w in pattern_words:
            bag[words.index(w)] = 1

    # Create an output row: 0 for all intents, except 1 for the current intent
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    # Add the bag of words and corresponding output row to the training data
    training.append([bag, output_row])

# Shuffle the training data and convert it to a numpy array
random.shuffle(training)
training = np.array(training, dtype=object)

# Split the training data into input (train_x) and output (train_y)
train_x = np.array([i[0] for i in training], dtype=np.float32)
train_y = np.array([i[1] for i in training], dtype=np.float32)

print("Training data created")

# Create the neural network model
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation="softmax"))

# Print the model summary
model.summary()

# Compile the model using Stochastic Gradient Descent (SGD) with Nesterov
sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])

# Train the model
hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)

# Save the trained model
model.save("chatbot_model.h5")

print("Model created and saved")
