We'll be doing digit classification using the MNIST dataset.
MNIST is a large library of 28x28, low resolution grayscale images of of handwritten digits
We will be building a neural network that classifies these images and outputs what digit is present in an input.

The MATH

1. Starting with 28x28 pixel (784 pixels), m training images
2. Each pixel has a pixel value ranging from 0 (black) to 255 (white)
3. We can represent the training data as a matrix M of size m x 784, where each row will be a sample
4. We transpose M, thus, turning each row (which is an example) into a column
5. We take an image, do some processing and then predict which of the 10 classes does the input belongs to
6. We will build a neural network with 2 layers
    i. the input layer has 784 neurons, where each pixel maps to a neuron
   ii. the first layer will be a hidden layer with 10 neurons
  iii. the output (second) layer will have 10 neurons, each corresponding to one class
7. The training consists of three parts:
    i. forward propagation: we push the input image through the network and compute what the output is.
    A[0] = x
    z[1] = w[1]A[0] + b[1]
    A[1] = g(z[1]) = ReLU (z[1])
    a. Here z is a non-linear activation function which allows the neural network to model data better. We can use different activation functions, but the most commonly used ones are ReLU, tanh and sigmoid.
    b. For the second layer, we perform the same operation with. a different activation function, called the softmax function, which converts the logits corresponding to each class into a suitable probability bounded between 0 and 1.
    ii. backward propagation: in order to learn the weights and biases that will give us the correct class for each input, we use our current prediction to understand how far was our prediction from the actual class, which will act as an error. We then use this error and see how much did each weight/bias (parameters from now on) contributed to the wrong prediction and use the error to correct the parameters.
    a. we one-hot encode the ground truth, i.e., our output vector will have 10 entries, with 0 everywhere and a 1 in the correct class
    iii. updating the parameters using the errors (gradients of error wrt each parameter) and a learning rate alpha which controls our training speed/convergence.
    a. alpha here is a hyperparameter, which is not trained by the model, we set it manually using some heuristics
    iv. and we keep doing it until we get a good accuracy