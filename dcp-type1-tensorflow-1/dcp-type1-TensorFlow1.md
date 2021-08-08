![DevIncept](https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse4.mm.bing.net%2Fth%3Fid%3DOIP.Aa1Ojw7Azjcpr51sEtQSJgAAAA%26pid%3DApi&f=1)
1. **Tensorflow - *Yugal Agarwal***
   * About Tensorflow
   * Installation (Import)
   * Introduction to Tensors
2. **Machine Learning - *Vaibhav Gupta***
   * Linear Regression
   * Classification
   * Hidden Markov Models
3. **Deep Learning - *Sai Dileep Kumar Mukkamala***
   * What is Keras
   * Neural Networks
   * Convolutional neural network (CNN)
   * CNN Code Implementation

# Tensorflow
## **AboutTenserFlow**
TensorFlow is a free and open-source software library for machine learning. It can be used across a range of tasks but has a particular focus on training and inference of deep neural networks. Tensorflow is a symbolic math library based on dataflow and differentiable programming.
[![Teserflow](https://i.ytimg.com/vi/yjprpOoH5c8/maxresdefault.jpg)](https://www.tensorflow.org)
### **Why Tenserflow**
TensorFlow is an end-to-end open source platform for machine learning. It has a comprehensive, flexible ecosystem of tools, libraries and community resources that lets researchers push the state-of-the-art in ML and developers easily build and deploy ML powered applications.
## **Installation**
In this section we will understand how to install Tenserflow in your system.
#### **System Requirements**
* Python 3.6 - 3.9
* pip 19.0 or later
* Ubuntu 16.04 or later (64-bit)
* macOS 10.12.6 (Sierra) or later (64-bit)
* Windows 7 or later (64-bit) 
### **1. Install Python environment on your system**
Requires Python 3.6 - 3.9, pip and venv >= 19.0

If these are already installed, skip to the next step.  
Otherwise, install [Python](https://www.python.org/), the [pip package manager](https://pip.pypa.io/en/stable/installation/) and [venv](https://docs.python.org/3/library/venv.html).
### **2. Create a virtual environment**
Virtual Environment are used to isolate package installation from the system.  
### **3. Install the TenserFlow pip package**
Choose one of the following TensorFlow packages to install from [PyPI](https://pypi.org/project/tensorflow/) :  
* **tenserflow** - Latest stable release with CPU and GPU support (Ubuntu and Windows) .  
* **tf - nighlty** - Preview build (unstable) . Ubuntu and Windows include GPU support .  
* **tenserflow==1.5** - The final version of TenserFlow 1.x.

##  **Introduction To Tensors**

By programming perspective **Tensors** are multi-dimensional arrays with a uniform type (called a **dtype**).  
![](https://media.geeksforgeeks.org/wp-content/uploads/two-d.png)  
The above image is a simple 2-dimensional array.
When we go in more fundamental mathematics perspective **tensor** is a generalization of scalar vector and matrix. 
![](https://miro.medium.com/max/891/0*jGB1CGQ9HdeUwlgB)  
For eg. a vector is a one-dimensional tensor, and matrix is a two-dimensional matrix.

If you're familiar with **NumPy**, tensors are (kind of) like **np.arrays**.

All tensors are immutable like Python numbers and strings: you can never update the contents of a tensor, only create a new one.

# Machine Leanring
## Linear Regression
Linear regression is one of the most basic forms of machine learning and is used to predict numeric values. In this tutorial we will use a linear model to predict the survival rate of passangers from the titanic dataset.
## Classification
Now that we've covered linear regression it is time to talk about classification. Where regression was used to predict a numeric value, classification is used to seperate data points into classes of different labels. In this example we will use a TensorFlow estimator to classify flowers.

Since we've touched on how estimators work earlier, I'll go a bit quicker through this example.

Documntation - https://www.tensorflow.org/tutorials/estimator/premade

## Hidden Markov Models

"The Hidden Markov Model is a finite set of states, each of which is associated with a (generally multidimensional) probability distribution. Transitions among the states are governed by a set of probabilities called transition probabilities." (http://jedlik.phy.bme.hu/~gerjanos/HMM/node4.html)

A hidden markov model works with probabilities to predict future events or states. In this section we will learn how to create a hidden markov model that can predict the weather.

Documentation -  https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/HiddenMarkovModel
###Data
Let's start by discussing the type of data we use when we work with a hidden markov model. 

In the previous sections we worked with large datasets of 100's of different entries. For a markov model we are only interested in probability distributions that have to do with states. 

We can find these probabilities from large datasets or may already have these values. We'll run through an example in a second that should clear some things up, but let's discuss the components of a markov model.

**States:** In each markov model we have a finite set of states. These states could be something like "warm" and "cold" or "high" and "low" or even "red", "green" and "blue". These states are "hidden" within the model, which means we do not direcly observe them.

**Observations:** Each state has a particular outcome or observation associated with it based on a probability distribution. An example of this is the following: *On a hot day Tim has a 80% chance of being happy and a 20% chance of being sad.*

**Transitions:** Each state will have a probability defining the likelyhood of transitioning to a different state. An example is the following: *a cold day has a 30% chance of being followed by a hot day and a 70% chance of being follwed by another cold day.*

To create a hidden markov model we need.
- States
- Observation Distribution
- Transition Distribution

For our purpose we will assume we already have this information available as we attempt to predict the weather on a given day.

# Deep Learning
## What is Keras?
   
>* To Understand about that , First we need to know about Deep Learning.
>
>* Deep learning is a machine learning technique that teaches computers to do what comes naturally to humans: learn by example. Deep learning is a key technology behind driverless cars, enabling them to recognize a stop sign, or to distinguish a pedestrian from a lamppost. It is the key to voice control in consumer devices like phones, tablets, TVs, and hands-free speakers. Deep learning is getting lots of attention lately and for good reason. It’s achieving results that were not possible before.
>
>* In deep learning, a computer model learns to perform classification tasks directly from images, text, or sound. Deep learning models can achieve state-of-the-art accuracy, sometimes exceeding human-level performance. Models are trained by using a large set of labeled data and neural network architectures that contain many layers.    
![Alt text](https://databricks.com/wp-content/uploads/2019/04/logo-keras.png)
### Keras Deep Learning library with TensorFlow
>Keras is a minimalist, highly modular neural networks library, written in Python and capable of running on top of either TensorFlow or Theano. 
>
>It was developed with a focus on enabling fast experimentation. Being able to go from idea to result with the least possible delay is key to doing good research.
ref: https://keras.io/
#### Why Keras?

>Keras is an API designed for human beings, not machines. Keras follows best practices for reducing cognitive load: it offers consistent & simple APIs, it minimizes the number of user actions required for common use cases, and it provides clear and actionable feedback upon user error.
    >
>This makes Keras easy to learn and easy to use. As a Keras user, you are more productive, allowing you to try more ideas than your competition, faster -- which in turn helps you win machine learning competitions.
    >
>This ease of use does not come at the cost of reduced flexibility: because Keras integrates with lower-level deep learning languages (in particular TensorFlow), it enables you to implement anything you could have built in the base language. In particular, as tf.keras, the Keras API integrates seamlessly with your TensorFlow workflows.

#### Keras Working Pipeline
![Alt text](https://blog.keras.io/img/keras-tensorflow-logo.jpg)
### MODEL Definition
There are two types of models available in Keras: the Sequential model and the Model class used with functional API.

#### Sequential Model
The simplest model is defined in the Sequential class which is a linear stack of Layers. You can create a Sequential model and define all of the layers in the constructor, for example:

      from keras.models import Sequential
      model = Sequential(...)

A more useful idiom is to create a Sequential model and add your layers in the order of the computation you wish to perform, for example:       
       
       from keras.models import Sequential
       model = Sequential()
       model.add(...)
       model.add(...)
       model.add(...)
#### Functional API
The Keras functional API provides a more flexible way for defining models.

It specifically allows you to define multiple input or output models as well as models that share layers. More than that, it allows you to define ad hoc acyclic network graphs.

Models are defined by creating instances of layers and connecting them directly to each other in pairs, then defining a Model that specifies the layers to act as the input and output to the model,For Example:

                         inputs = Input(shape=(3,))
                         x = Dense(50, activation='relu')(inputs)
                         output = Dense(1, activation = 'sigmoid')(x)
                         n_net = Model(inputs, output)
                         n_net.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
                         n_net.fit(x=dat_train, y=y_classifier_train, epochs=10,
                         verbose=1, validation_data=(dat_test, y_classifier_test))
### Model Compilation
> Before training a model, you need to configure the learning process, which is done via the compile method. It receives three arguments:
>
>* An optimizer. This could be the string identifier of an existing optimizer (such as rmsprop or adagrad), or an instance of the Optimizer class. See: optimizers.
>* A loss function. This is the objective that the model will try to minimize. It can be the string identifier of an existing loss function (such as categorical_crossentropy or mse), or it can be an objective function. See: losses.
>* A list of metrics. For any classification problem you will want to set this to metrics=['accuracy']. A metric could be the string identifier of an existing metric or a custom metric function.

#### For a multi-class classification problem
      model.compile(optimizer='rmsprop',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])

### For a binary classification problem
      model.compile(optimizer='rmsprop',
                    loss='binary_crossentropy',
                    metrics=['accuracy'])

### For a mean squared error regression problem
      model.compile(optimizer='rmsprop',
                    loss='mse')

### For custom metrics
      import keras.backend as K

      def mean_pred(y_true, y_pred):
          return K.mean(y_pred)

      model.compile(optimizer='rmsprop',
                    loss='binary_crossentropy',
                    metrics=['accuracy', mean_pred])
### Applications
Keras Applications are deep learning models that are made available alongside pre-trained weights. These models can be used for prediction, feature extraction, and fine-tuning.

Weights are downloaded automatically when instantiating a model. They are stored at ~/.keras/models/.
## Neural Networks

![NN](https://i2.wp.com/vinodsblog.com/wp-content/uploads/2019/01/recurrent-neural-networks-4259348843-1549021778666.png?resize=1300%2C650&ssl=1)
Neural networks are artificial systems that were inspired by biological neural networks. These systems learn to perform tasks by being exposed to various datasets and examples without any task-specific rules. The idea is that the system generates identifying characteristics from the data they have been passed without being programmed with a pre-programmed understanding of these datasets.
Neural networks are based on computational models for threshold logic. Threshold logic is a combination of algorithms and mathematics. Neural networks are based either on the study of the brain or on the application of neural networks to artificial intelligence. The work has led to improvements in finite automata theory.
Components of a typical neural network involve neurons, connections, weights, biases, propagation function, and a learning rule. Neurons will receive an input p_j(t) from predecessor neurons that have an activation a_j(t), threshold $\theta$_j, an activation function f, and an output function f_{out}. Connections consist of connections, weights and biases which rules how neuron $i$ transfers output to neuron $j$. Propagation computes the input and outputs the output and sums the predecessor neurons function with the weight. The learning rule modifies the weights and thresholds of the variables in the network.
**Supervised vs Unsupervised Learning:**

* Neural networks learn via supervised learning; 

* Supervised machine learning involves an input variable x and output variable y. The algorithm learns from a training dataset. With each correct answers, algorithms iteratively make predictions on the data. The learning stops when the algorithm reaches an acceptable level of performance.
Unsupervised machine learning has input data X and no corresponding output variables. The goal is to model the underlying structure of the data for understanding more about the data. The keywords for supervised machine learning are classification and regression. For unsupervised machine learning, the keywords are clustering and association.
**Evolution of Neural Networks:**

Hebbian learning deals with neural plasticity. Hebbian learning is unsupervised and deals with long term potentiation. Hebbian learning deals with pattern recognition and exclusive-or circuits; deals with if-then rules.

Back propagation solved the exclusive-or issue that Hebbian learning could not handle. This also allowed for multi-layer networks to be feasible and efficient. If an error was found, the error was solved at each layer by modifying the weights at each node. This led to the development of support vector machines, linear classifiers, and max-pooling. The vanishing gradient problem affects feedforward networks that use back propagation and recurrent neural network. This is known as deep-learning.

Hardware-based designs are used for biophysical simulation and neurotrophic computing. They have large scale component analysis and convolution creates new class of neural computing with analog. This also solved back-propagation for many-layered feedforward neural networks.

Convolutional networks are used for alternating between convolutional layers and max-pooling layers with connected layers (fully or sparsely connected) with a final classification layer. The learning is done without unsupervised pre-training. Each filter is equivalent to a weights vector that has to be trained. The shift variance has to be guaranteed to dealing with small and large neural networks. This is being resolved in Development Networks.
**Types of Neural Networks:**

There are seven types of neural networks that can be used.

>* The first is a multilayer perceptron which has three or more layers and uses a nonlinear activation function.
>
>* The second is the convolutional neural network that uses a variation of the multilayer perceptrons.
>
>* The third is the recursive neural network that uses weights to make structured predictions.
>
>* The fourth is a recurrent neural network that makes connections between the neurons in a directed cycle. The long short-term memory neural network uses the recurrent neural network architecture and does not use activation function.
>
>* The final two are sequence to sequence modules which uses two recurrent networks and shallow neural networks which produces a vector space from an amount of text. These neural networks are applications of the basic neural network demonstrated below.
## Convolutional neural network(CNN)
* As we discussed about various types of Neural Networks.
* CNN is one of those.
**Convolutional neural network (ConvNets or CNNs) is one of the main categories to do images recognition, images classifications. Objects detections, recognition faces etc.,**
![cnn](https://i1.wp.com/www.michaelchimenti.com/wp-content/uploads/2017/11/Deep-Neural-Network-What-is-Deep-Learning-Edureka.png)
***CNN image classifications takes an input image, process it and classify it under certain categories (Eg., Dog, Cat, Tiger, Lion). Computers sees an input image as array of pixels and it depends on the image resolution. Based on the image resolution, it will see h x w x d( h = Height, w = Width, d = Dimension ). Eg., An image of 6 x 6 x 3 array of matrix of RGB (3 refers to RGB values) and an image of 4 x 4 x 1 array of matrix of grayscale image.***
**Technically, deep learning CNN models to train and test, each input image will pass it through a series of convolution layers with filters (Kernals), Pooling, fully connected layers (FC) and apply Softmax function to classify an object with probabilistic values between 0 and 1. The below figure is a complete flow of CNN to process an input image and classifies the objects based on values.**
![](https://miro.medium.com/max/1400/1*XbuW8WuRrAY5pC4t-9DZAQ.jpeg)
**Convolution Layer**

* Convolution is the first layer to extract features from an input image. Convolution preserves the relationship between pixels by learning image features using small squares of input data. It is a mathematical operation that takes two inputs such as image matrix and a filter or kernel.
![](https://miro.medium.com/max/576/1*kYSsNpy0b3fIonQya66VSQ.png)
* Consider a 5 x 5 whose image pixel values are 0, 1 and filter matrix 3 x 3 as shown in below
![](https://miro.medium.com/max/516/1*4yv0yIH0nVhSOv3AkLUIiw.png)
* Then the convolution of 5 x 5 image matrix multiplies with 3 x 3 filter matrix which is called “Feature Map” as output shown in below.
![](https://miro.medium.com/max/335/1*MrGSULUtkXc0Ou07QouV8A.gif)
* Convolution of an image with different filters can perform operations such as edge detection, blur and sharpen by applying filters. The below example shows various convolution image after applying different types of filters (Kernels).

**Strides**
* Stride is the number of pixels shifts over the input matrix. When the stride is 1 then we move the filters to 1 pixel at a time. When the stride is 2 then we move the filters to 2 pixels at a time and so on. The below figure shows convolution would work with a stride of 2.
![](https://miro.medium.com/max/869/1*nGHLq1hx0gt02OK4l8WmRg.png)

**Padding**

Sometimes filter does not fit perfectly fit the input image. We have two options:
* Pad the picture with zeros (zero-padding) so that it fits
* Drop the part of the image where the filter did not fit. This is called valid padding which keeps only valid part of the image.

**Non Linearity (ReLU)**

* ReLU stands for Rectified Linear Unit for a non-linear operation. The output is **ƒ(x) = max(0,x)**.

* Why ReLU is important : ReLU’s purpose is to introduce non-linearity in our ConvNet. Since, the real world data would want our ConvNet to learn would be non-negative linear values.
![](https://miro.medium.com/max/561/1*gcvuKm3nUePXwUOLXfLIMQ.png)
* There are other non linear functions such as tanh or sigmoid that can also be used instead of ReLU. Most of the data scientists use ReLU since performance wise ReLU is better than the other two.

**Pooling Layer**

Pooling layers section would reduce the number of parameters when the images are too large. Spatial pooling also called subsampling or downsampling which reduces the dimensionality of each map but retains important information. Spatial pooling can be of different types:
* Max Pooling
* Average Pooling
* Sum Pooling

Max pooling takes the largest element from the rectified feature map. Taking the largest element could also take the average pooling. Sum of all elements in the feature map call as sum pooling.
![](https://miro.medium.com/max/753/1*SmiydxM5lbTjoKWYPiuzWQ.png)

**Fully Connected Layer**

The layer we call as FC layer, we flattened our matrix into vector and feed it into a fully connected layer like a neural network.
![](https://miro.medium.com/max/693/1*Mw6LKUG8AWQhG73H1caT8w.png)

In the above diagram, the feature map matrix will be converted as vector (x1, x2, x3, …). With the fully connected layers, we combined these features together to create a model. Finally, we have an activation function such as softmax or sigmoid to classify the outputs as cat, dog, car, truck etc.,
![](https://miro.medium.com/max/875/1*4GLv7_4BbKXnpc6BRb0Aew.png)