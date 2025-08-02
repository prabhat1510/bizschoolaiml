#This code is a simple example of image recognition using the MNIST dataset with TensorFlow and Keras.
#It demonstrates loading, preprocessing data, building a neural network model, training it, evaluating performance, and making predictions.
#It predicst the digit in the first, tenth, and seventh images of the test set.
#It also includes comments explaining each step in the process.
#Step 1: Importing Libraries
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
import numpy as np
'''
TensorFlow: Popular deep learning library.
MNIST: Dataset of 28x28 grayscale images of digits 0-9
Sequential model: A stack of layers where each layer has one input tensor and one output tensor.
Flatten/Dense: Layers used in building fully connected neural networks.
to_categorical: Converts integer labels to one-hot encoded vectors.
NumPy: Used for numerical operations (like reshaping).

'''
#Step 2: Load & Preprocess the Data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)
'''
Load MNIST: Returns training and test data.
Normalization: Pixel values scaled to [0, 1] for better training stability.
One-hot encoding: Converts digit labels (0-9) into 10-dimensional vectors.
'''
#Step 3: Build the Model
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])
'''
Flatten: Converts 2D image to 1D vector (28×28 → 784).
Dense(128): First hidden layer with 128 neurons and ReLU activation.
Dense(64): Second hidden layer.
Dense(10): Output layer with 10 neurons (each representing a digit), and Softmax activation to produce probabilities.
'''
#Step 4: Compile the Model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
'''
Adam optimizer: Efficient gradient-based optimizer.
Categorical crossentropy: Loss function used for multi-class classification.
Accuracy: Metric to monitor during training and evaluation.
'''
#Step 5: Train the Model
model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))
#Trains the model for 5 epochs and evaluates on test data at each epoch.

#Step 6: Evaluate the Model
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_accuracy:.4f}")

#Returns the loss and accuracy on the unseen test set.

#Step 7: Make Predictions
pred = model.predict(np.expand_dims(x_test[0], axis=0))
print(f'Predicted label: {np.argmax(pred)}')

'''
np.expand_dims: Adds a batch dimension (1 sample).
model.predict: Gets output probabilities.
np.argmax: Returns the index of the highest probability—i.e., predicted digit.
Repeated for samples 10 and 7 as well.
'''

#model.save('mnist_digit_model.h5')
# print("Model saved successfully.")
#Saves the trained model to disk (commented out here).