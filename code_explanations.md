# Code Explanations: Labs 1, 2, and 3

This document provides a line-by-line explanation of the code in each of the three Jupyter Notebooks. It explains what each block does, the purpose of each function, and why specific settings are used.

---

## 🧠 Lab 1: Simulate a Perceptron using NumPy
**File**: [ANN_lab1.ipynb](./ANN_lab1.ipynb)

This script builds a single Perceptron from scratch using only raw Python and **NumPy**. It uses a sharp step activation function and learns the weights dynamically through training.

### Full Code Code Block
```python
import numpy as np

# Step Activation Function
def step_function(x):
    if x >= 0:
        return 1
    return 0

# Perceptron Class
class Perceptron:
    def __init__(self, input_size, learning_rate=0.1):
        self.weights = np.zeros(input_size)
        self.bias = 0
        self.learning_rate = learning_rate

    # Predict output
    def predict(self, x):
        total = np.dot(x, self.weights) + self.bias
        return step_function(total)

    # Train the perceptron
    def train(self, X, y, epochs=10):
        for _ in range(epochs):
            for inputs, target in zip(X, y):
                prediction = self.predict(inputs)
                error = target - prediction

                # Update weights and bias
                self.weights += self.learning_rate * error * inputs
                self.bias += self.learning_rate * error

# Training data for AND Gate
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y = np.array([0, 0, 0, 1])

# Create and train perceptron
perceptron = Perceptron(input_size=2)
perceptron.train(X, y, epochs=10)

# Display predictions
print("Predictions:")
for sample in X:
    print("Input:", sample, "Output:", perceptron.predict(sample))
```

### Detailed Code Walkthrough

1. **`import numpy as np`**
   - Imports the NumPy library, which provides support for multi-dimensional arrays and fast mathematical operations.
2. **`def step_function(x):`**
   - Defines the activation function. It acts as a hard ON/OFF switch. If the net input score `x` is greater than or equal to `0`, it returns `1` (ON). If it is negative, it returns `0` (OFF).
3. **`class Perceptron:`**
   - Defines the class structure for our single-cell network.
4. **`def __init__(self, input_size, learning_rate=0.1):`**
   - The constructor initializes the Perceptron's state:
     - `self.weights = np.zeros(input_size)`: Starts weights at zero for all inputs (e.g., `[0.0, 0.0]`).
     - `self.bias = 0`: Starts the bias threshold at 0.
     - `self.learning_rate = learning_rate`: Sets how fast the model adjusts its weights (set to `0.1` by default).
5. **`def predict(self, x):`**
   - Calculates the dot product of the input array `x` and the weights (`x * weights`), adds the bias, and sends the total score to the `step_function` to get a `0` or `1` output.
6. **`def train(self, X, y, epochs=10):`**
   - This is the learning loop:
     - Runs through the whole dataset `epochs` times (set to 10 rounds).
     - Loops over each input sample and its target label using `zip(X, y)`.
     - Calculates the prediction error: `error = target - prediction`.
     - If `error` is not `0` (i.e. prediction was wrong), it updates the weights and bias:
       - `self.weights += self.learning_rate * error * inputs`
       - `self.bias += self.learning_rate * error`
7. **`X = np.array([...])` and `y = np.array([0, 0, 0, 1])`**
   - Defines the inputs and targets for the **logical AND gate**. The target `y` is only `1` when both inputs are `1`.
8. **`perceptron = Perceptron(input_size=2)`**
   - Creates a Perceptron instance with 2 input features (representing $x_1$ and $x_2$).
9. **`perceptron.train(X, y, epochs=10)`**
   - Runs the training loop for 10 rounds to adjust the weights and bias.
10. **`print(...)` and inference loop**
    - Feeds each of the four AND inputs back into the trained model and prints the final outputs to verify that the Perceptron has successfully learned the AND function.

---

## 🔢 Lab 2: Single Neuron Forward Pass (Sigmoid)
**File**: [ANN_lab2.ipynb](./ANN_lab2.ipynb)

This script demonstrates **forward propagation** manually through a single neuron. Instead of training or updating weights, we hardcode the weights and bias to trace how a neuron computes outputs using a smooth **Sigmoid** activation function.

### Full Code Code Block
```python
import numpy as np

# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Input values
X = np.array([[0,0],
              [0,1],
              [1,0],
              [1,1]])

# Initial weights and bias
weights = np.array([0.5, 0.5])
bias = -0.7

print("Single Neuron Model Output:\n")

for x in X:
    # Weighted sum
    z = np.dot(x, weights) + bias

    # Apply sigmoid activation
    output = sigmoid(z)

    # Convert to binary class
    prediction = 1 if output >= 0.5 else 0

    print(f"Input: {x}, Weighted Sum = {z:.2f}, Output = {output:.4f}, Class = {prediction}")
```

### Detailed Code Walkthrough

1. **`import numpy as np`**
   - Imports NumPy for array structures and vectorized mathematical functions.
2. **`def sigmoid(x):`**
   - Implements the sigmoid activation function: $\sigma(x) = \frac{1}{1 + e^{-x}}$. 
   - `np.exp(-x)` computes $e^{-x}$ for the input value. The sigmoid squashes any score into a continuous range between `0` and `1`, representing confidence/probability.
3. **`X = np.array([...])`**
   - Sets up the four combinations of inputs for the AND gate dataset.
4. **`weights = np.array([0.5, 0.5])` and `bias = -0.7`**
   - Manually defines fixed, preset settings. These weights and bias are chosen specifically so that the single neuron can solve the AND gate without any training.
5. **`for x in X:`**
   - Starts a loop to feed each of the four input samples through the neuron.
6. **`z = np.dot(x, weights) + bias`**
   - Computes the weighted sum of inputs and weights, then adds the bias:
     - For `[0, 0]`, $z = (0 \times 0.5) + (0 \times 0.5) - 0.7 = -0.7$.
     - For `[1, 1]`, $z = (1 \times 0.5) + (1 \times 0.5) - 0.7 = 0.3$.
7. **`output = sigmoid(z)`**
   - Passes the score `z` to the sigmoid function to get a decimal output (probability). For example, `sigmoid(0.3)` is approximately `0.5744`.
8. **`prediction = 1 if output >= 0.5 else 0`**
   - Applies thresholding: if the probability is `0.5` or higher, the class prediction is `1` (YES). If it is less than `0.5`, the prediction is `0` (NO).
9. **`print(f"...")`**
   - Prints the formatted inputs, weighted sum, sigmoid probability, and final class prediction.

---

## ⚡ Lab 3: Feedforward Neural Network (TensorFlow/Keras)
**File**: [ANN_lab3.ipynb](./ANN_lab3.ipynb)

This script builds, compiles, and trains a multi-layer neural network (a hidden layer of 4 neurons + 1 output neuron) using **TensorFlow** and **Keras** to solve the logical AND gate.

### Full Code Code Block
```python
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense

# AND Gate Dataset
X = np.array([[0,0],
              [0,1],
              [1,0],
              [1,1]], dtype=np.float32)

y = np.array([[0],
              [0],
              [0],
              [1]], dtype=np.float32)

# Feedforward Neural Network
model = Sequential([
    Input(shape=(2,)),
    Dense(4, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train the model
model.fit(X, y, epochs=500, verbose=0)

# Predictions
print("Predictions:")
predictions = model.predict(X)

for i in range(len(X)):
    print(f"Input: {X[i]} => Predicted: {predictions[i][0]:.4f} => Class: {int(predictions[i][0] >= 0.5)}")
```

### Detailed Code Walkthrough

1. **Imports (`numpy`, `tensorflow`, `Sequential`, `Input`, `Dense`)**
   - `numpy`: For dataset structure.
   - `tensorflow`: The core backend deep learning framework.
   - `Sequential`: A Keras model class that lets us stack layers sequentially (one after another).
   - `Input`, `Dense`: Keras layers. `Input` specifies the input size, and `Dense` defines a fully connected layer (where every neuron connects to all inputs from the previous layer).
2. **`X` and `y` Arrays with `dtype=np.float32`**
   - Defines inputs and targets for the AND gate.
   - We specify `float32` instead of the default 64-bit float because deep learning frameworks like TensorFlow run faster and consume less memory with 32-bit floats.
3. **`model = Sequential([...])`**
   - Constructs the network architecture:
     - `Input(shape=(2,))`: Tells the model that each input sample has 2 features.
     - `Dense(4, activation='relu')`: The hidden layer with 4 neurons. It uses **ReLU** (Rectified Linear Unit) activation, which filters out negative scores (turning them to 0) and passes positive scores through.
     - `Dense(1, activation='sigmoid')`: The output layer with 1 neuron. It uses **Sigmoid** activation to return a probability value between `0` and `1`.
4. **`model.compile(...)`**
   - Configures the model before training:
     - `optimizer='adam'`: The Adam optimizer, which updates weights dynamically using running momentum.
     - `loss='binary_crossentropy'`: The loss function for binary (yes/no) classification, which penalizes incorrect confident predictions.
     - `metrics=['accuracy']`: Tells Keras to log classification accuracy during training.
5. **`model.fit(X, y, epochs=500, verbose=0)`**
   - Trains the model by running the dataset through it 500 times (`epochs`).
   - `verbose=0`: Suppresses print logs for each epoch so the console output remains clean.
6. **`predictions = model.predict(X)`**
   - Feeds the input data `X` back into the trained model to perform inference. This returns raw float probability values.
7. **Inference Loop & Thresholding**
   - Loops over predictions and converts the float probability to a binary class: `int(predictions[i][0] >= 0.5)`. Prints the result for each input sample.
