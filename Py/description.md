# Digit Classification with a Neural Network (from scratch, using NumPy)

We'll be doing digit classification using the **MNIST** dataset.

MNIST is a large library of 28x28, low-resolution grayscale images of handwritten digits (0–9). We will build a simple two-layer neural network **from scratch** (using only NumPy — no deep learning frameworks) that classifies these images and outputs which digit is present in an input.

---

## 1. The Data

- The MNIST training set contains `m = 60,000` images.
- Each image is `28 x 28 = 784` pixels.
- Each pixel has an intensity value ranging from `0` (black) to `255` (white).
- Each image also has a **label**: the true digit (0–9) it represents.

When loaded from `mnist_train.csv`, each row is one sample: the first column is the label, and the remaining 784 columns are the pixel values. So the raw data matrix has shape:

```
data : (m, 785)        # m = 60,000, 785 = 1 label + 784 pixels
```

### 1.1 Reshaping the data

We transpose the data so that **each column is one training example** instead of each row. This is a common convention in neural network math, since it lets us write forward propagation as compact matrix multiplications.

```
data.T : (785, m)
```

After transposing and splitting off the label row:

- `Y` — shape `(m,)`  → the true labels (one integer per example)
- `X` — shape `(784, m)` → the pixel values, one **column** per example

We also **normalize** the pixel values from the range `[0, 255]` to `[0, 1]` by dividing by 255:

$$
X := \frac{X}{255}
$$

This keeps input values small, which helps the network train faster and more stably.

The dataset is split into:
- `X_dev`, `Y_dev` — a small held-out set (1,000 samples) used to check performance
- `X_train`, `Y_train` — the remaining ~59,000 samples used for training

---

## 2. The Network Architecture

We build a neural network with **2 layers** (one hidden layer + one output layer):

| Layer                 | Neurons | Description                                  |
|-----------------------|---------|-----------------------------------------------|
| Input layer            | 784     | one neuron per pixel                          |
| Hidden layer (Layer 1) | 10      | learns intermediate patterns/features         |
| Output layer (Layer 2) | 10      | one neuron per digit class (0–9)              |

### 2.1 Parameters

The connections between layers are represented by **weight matrices** and **bias vectors**:

| Parameter | Shape     | Connects                          |
|-----------|-----------|------------------------------------|
| `W1`      | (10, 784) | input layer → hidden layer         |
| `b1`      | (10, 1)   | bias for hidden layer              |
| `W2`      | (10, 10)  | hidden layer → output layer        |
| `b2`      | (10, 1)   | bias for output layer              |

All parameters are randomly initialized with small values in the range `[-0.5, 0.5]`:

$$
W^{[l]}, b^{[l]} \sim \text{Uniform}(-0.5,\ 0.5)
$$

Random initialization breaks symmetry — if all weights started equal, every neuron would learn the exact same thing.

---

## 3. Training: Three Steps

Training a neural network is an iterative loop of three steps, repeated many times (**iterations**):

1. **Forward propagation** — compute a prediction from the input
2. **Backward propagation** — compute how wrong the prediction was, and how much each parameter contributed to that error
3. **Parameter update** — nudge each parameter slightly to reduce the error

---

### 3.1 Forward Propagation

We push the input image(s) through the network and compute the output layer by layer.

**Layer 0 (input):**

$$
A^{[0]} = X
$$

**Layer 1 (hidden layer):**

$$
Z^{[1]} = W^{[1]} A^{[0]} + b^{[1]}
$$

$$
A^{[1]} = g(Z^{[1]}) = \text{ReLU}(Z^{[1]})
$$

**Layer 2 (output layer):**

$$
Z^{[2]} = W^{[2]} A^{[1]} + b^{[2]}
$$

$$
A^{[2]} = \text{softmax}(Z^{[2]})
$$

Here `Z` is a *linear* combination of the previous layer's activations (a weighted sum plus bias), and the activation function `g` introduces **non-linearity**, which is what allows the network to model complex, non-linear relationships in the data. Without a non-linear activation function, stacking multiple layers would collapse into a single linear transformation, no matter how many layers we added.

#### ReLU (Rectified Linear Unit)

Used for the hidden layer. It simply zeroes out negative values and passes positive values through unchanged:

$$
\text{ReLU}(z) = \max(0, z)
$$

Its derivative (needed later for backpropagation) is a step function:

$$
\text{ReLU}'(z) =
\begin{cases}
1 & \text{if } z > 0 \\
0 & \text{if } z \leq 0
\end{cases}
$$

Other common activation functions include `tanh` and `sigmoid`, but ReLU is popular because it's computationally cheap and helps avoid the *vanishing gradient* problem.

#### Softmax

Used for the output layer. It converts the 10 raw output scores (**logits**) into a probability distribution over the 10 digit classes — each value is between 0 and 1, and all 10 values sum to 1:

$$
\text{softmax}(z)_i = \frac{e^{z_i}}{\sum_{j=1}^{10} e^{z_j}}
$$

The predicted digit is simply the class with the highest probability:

$$
\hat{y} = \underset{i}{\text{argmax}}\ A^{[2]}_i
$$

---

### 3.2 Backward Propagation

Once we have a prediction, we need to measure how wrong it was and figure out how much each parameter (`W1`, `b1`, `W2`, `b2`) contributed to that error. This is done using the **chain rule** of calculus, propagating the error backward from the output layer to the input layer — hence "backpropagation."

#### One-hot encoding the labels

Since the network outputs a probability for each of the 10 classes, we need the ground-truth label in the same format. We **one-hot encode** the label: a vector of 10 entries, all zero except a `1` in the position of the correct digit.

For example, if the true label is `3`:

$$
Y_{\text{one-hot}} = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
$$

#### Gradients

Using the one-hot encoded labels, we compute the error at the output layer, then propagate it backward layer by layer:

**Output layer error:**

$$
dZ^{[2]} = A^{[2]} - Y_{\text{one-hot}}
$$

This elegant expression is the derivative of the *softmax + cross-entropy loss* combination, and it directly measures how far off each output probability is from the true (one-hot) label.

**Gradients for Layer 2 parameters:**

$$
dW^{[2]} = \frac{1}{m} \, dZ^{[2]} (A^{[1]})^{T}
$$

$$
db^{[2]} = \frac{1}{m} \sum dZ^{[2]}
$$

**Propagate the error back to Layer 1:**

$$
dZ^{[1]} = (W^{[2]})^{T} dZ^{[2]} \odot \text{ReLU}'(Z^{[1]})
$$

Here `⊙` denotes element-wise multiplication. Multiplying by `ReLU'(Z^[1])` ensures we only propagate error through neurons that were "active" (i.e., where `Z^[1] > 0`) during forward propagation.

**Gradients for Layer 1 parameters:**

$$
dW^{[1]} = \frac{1}{m} \, dZ^{[1]} (A^{[0]})^{T}
$$

$$
db^{[1]} = \frac{1}{m} \sum dZ^{[1]}
$$

In all of the above, `m` is the number of training examples in the batch, and dividing by `m` gives us the *average* gradient across all examples, rather than the sum.

---

### 3.3 Updating the Parameters (Gradient Descent)

Now that we know each parameter's gradient (i.e., how much it contributed to the error), we update it in the direction that **reduces** the error. This is called **gradient descent**:

$$
W^{[1]} := W^{[1]} - \alpha \, dW^{[1]}
$$

$$
b^{[1]} := b^{[1]} - \alpha \, db^{[1]}
$$

$$
W^{[2]} := W^{[2]} - \alpha \, dW^{[2]}
$$

$$
b^{[2]} := b^{[2]} - \alpha \, db^{[2]}
$$

Here `α` (**alpha**) is the **learning rate**, a hyperparameter that controls how big a step we take with each update:

- Too large → training becomes unstable and may never converge
- Too small → training converges very slowly

Unlike `W` and `b`, `α` is **not learned** by the model — it is chosen manually (in our case, `α = 0.01`), typically through experimentation.

---

## 4. The Training Loop

Steps 3.1–3.3 (forward propagation → backward propagation → parameter update) are repeated for a fixed number of **iterations** (in our case, `10,000`). Each iteration processes the *entire* training set at once (this is called **batch gradient descent**).

Every 100 iterations, we check the model's current accuracy on the training set:

$$
\text{Accuracy} = \frac{\text{number of correct predictions}}{\text{total number of predictions}}
$$

As training progresses, accuracy climbs steadily — starting around 8% (roughly random guessing, since there are 10 classes) and rising to about 88% by the 10,000th iteration.

---

## 5. Evaluating the Model

After training on `X_train`/`Y_train`, we evaluate the model on the held-out **dev set** (`X_dev`, `Y_dev`), which the network never saw during training. This tells us how well the model **generalizes** to new, unseen data rather than just memorizing the training examples.

Prediction on new data uses only forward propagation (no backprop needed):

$$
\hat{Y} = \underset{i}{\text{argmax}}\ \text{softmax}\Big(W^{[2]}\, \text{ReLU}(W^{[1]}X + b^{[1]}) + b^{[2]}\Big)
$$

With this simple 2-layer network trained for 10,000 iterations at `α = 0.01`, we achieve roughly **89% accuracy** on the dev set — a solid result for such a minimal architecture built entirely from first principles.

We can also visually inspect individual predictions by plotting a single test image alongside the model's predicted label and the true label, which is a useful sanity check beyond the aggregate accuracy number.

---

## 6. Summary

| Concept | Purpose |
|---|---|
| Normalization (`X/255`) | Keeps input values small and training stable |
| ReLU | Non-linearity in the hidden layer |
| Softmax | Converts output logits into class probabilities |
| One-hot encoding | Represents categorical labels in a form comparable to softmax output |
| Backpropagation | Computes how each parameter contributed to the error |
| Gradient descent | Updates parameters to reduce error |
| Learning rate (`α`) | Controls the size of each parameter update |
| Iterations | Number of times we repeat the train loop |
