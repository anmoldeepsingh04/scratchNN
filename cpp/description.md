We perform a classification task for MNIST and FashionMNIST by coding all the required functionalities in C++, as even the DL libraries used in Python, are basically written in C++ for faster and efficient execution. 
We will only be using the standard libraries in C++.
We want to implement a pytorch inspired DL framework

1. We implement the tensor class, which is the building block of all our NN
- tensor is an object that can store some numbers and can be used to store the weights of our NN. It has addition, multiplication defined which allows us to create new tensors
- we can use any data structure to store data and then geenralize it to be a tensor, but we want random read and write functionalit and the size of each data structure, in our CPP version of tensor, we would like to use the vector data strucutre.
- we will be using float as the data type as it uses 32 bits which is the default datatype being used by pytorch.
- we want to use a method in various files throughout the program, so we need to define a header file for each method which will have the definition and then a cpp file which will have the implementation of the method