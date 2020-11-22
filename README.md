# Convolutional-Neural-Networks

CNN implemention for classifying image patches. Basic steps necessary for building a simple CNN that can classify images with acceptable accuracy. Dataset contains examples for training, validation and testing. It consists of RGB images of three categories: persons, riders and cars which were down-scaled to at most 100 × 100 pixels for memory reasons.
Dataset Link: https://omnomnom.vision.rwth-aachen.de/data/cityscapesExtracted.zip


# input_cs.py :
-The dataset contains extraction artifacts e.g. small parts of cars or far away pedestrians. In order to remove images which do not contain any reasonable content, complete input_cs and filter out these artifacts by thresholding the size of the content to at least 900 pixels.
  - Take a look at the Dataset.filter function of TensorFlow and use it with the given input_cs._filter_size function
  -  need to resize the images before passing them to the CNN (Resize the image 64 by 64 pixels using TensorFlow’s tf.resize_images function)


- Further preprocessing such as normalization will be necessary before using the CNN. In my case, a normalization of the channel values from [0 ... 255] to [0 ... 1] is sufficient and implemented in input_cs (Take a look at Dataset.map and rescale the images)


# model.py :
- The main building blocks of CNNs are convolution and pooling layers, respectively. A small CNN with 3 layers of each type will be sufficient where each
convolutional layer is succeeded by a ReLU activation and a max-pooling layer. The convolutional layers have a receptive field of 5 by 5 pixels. The layers may
comprise 24, 32, and 50 filters which are applied with a stride size of 1 also pool over blocks of 3 by 3 with a stride size of 2 for all pooling layers. 

- Implemention of the proposed architecture in model.build_model.
  - For implementing the convolution in low level TensorFlow it needs to use TensorFlow’s tf.get_variable for adding weight variables for the convolutions and biases to the computational graph.

  - Using these variables, apply the convolution with tf.nn.conv2d and add the biases with tf.nn.bias_add afterwards.

  - Finally, it needs tf.nn.relu and tf.nn.max_pool for non-linearity application and pooling, respectively.

- In order to map the representation after the last convolutional layer to a class label,the implemention of three fully connected layers of size 100, 50 and number of categories are done in model.build_model.
  - It's not a good idea to implement a softmax on top of the last fully connected layer since we will use a stable internal implementation later on.

- For training purposes, it needs to specify a proper loss on top of a softmax which will be appropriate to predict the final class. In our case, a suitable loss function is the cross entropy loss which has implemented in the model.loss function.
  - It's possible to use tf.nn.sparse_softmax_cross_entropy_with_logits, a stable implementation of softmax and cross entropy error.
  - The function should return the per-sample loss for book-keeping reasons and the batch loss, respectively

- Implement the training operation in model.get_train_op_for_loss


# train.py 
Complete the basic training loop in train.train
- Build and train the network
- One possibility to regularize the model is early stopping based on the validation set error. After training for a certain number of epochs, the error on a validation set is measured. One may save the best model or stop training if the error increases to much. Although not crucial for this simple task, you should implement a validation step which is used to save the best model. Completed the code in train.train which computes the mean cross entropy error and the accuracy on the validation set.


# main_cityscape.py :
Finally, use the model to predict class labels for the images in the test set. Implemention of main_cityscape.run_test which should print the test accuracy. Furthermore, compute the accuracy for each call separately and print the mean of these accuracies. Also possible to compute the confusion matrix i.e. a matrix where entry (prediction, label) corresponds to how often the model predicted prediction for an instance of label label
  - Possible to use tf.argmax for predicting the output if its necessary to do the prediction in TensorFlow.
