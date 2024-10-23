# Autoencoder-Pytorch-MNIST
The autoencoder features a simple linear architecture of 784→128→64→128→784. The encoder compresses the input image (flattened from a 28x28 pixel grid to an input dimension of 784) into a lower-dimensional representation through two fully connected layers, using ReLU activation functions to introduce non-linearity. The decoder then reconstructs the original input image from this compressed representation, employing a similar set of fully connected layers and ending with a Sigmoid activation to produce pixel values in the range [0, 1].

Training employs the Adam optimizer and minimizes reconstruction error with Mean Squared Error (MSELoss). The prepare_data function pairs digits (0-1, 2-3, etc.) for training, utilizing the datasets.MNIST library.


The training time ranges from 60 to 400 seconds, depending on the variables such as learning rate (lr), epochs, and whether ReLU activations are enabled.
![image](https://github.com/user-attachments/assets/8a23823e-d3a1-4812-b716-950238a615ae)

Below we can see the results with the reconstructions of the next digit (n+1).

![image](https://github.com/user-attachments/assets/67b8bb48-4d4d-4358-86f5-db2f8432e556)
