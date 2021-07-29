<div align="center">
  <img src="https://i.ytimg.com/vi/yjprpOoH5c8/maxresdefault.jpg">
</div>

[![Python](https://img.shields.io/pypi/pyversions/tensorflow.svg?style=plastic)](https://badge.fury.io/py/tensorflow)
[![PyPI](https://badge.fury.io/py/tensorflow.svg)](https://badge.fury.io/py/tensorflow)

[TensorFlow](https://www.tensorflow.org/) is open source artificial intelligence library, using data flow graphs to build models. 
It allows developers to develop and train ML models so that they can create large-scale neural networks with many layers.
It is mainly used for Classification, Perception, Understanding, Discovering, Prediction and Creation of the models.

TensorFlow was originally developed by researchers and engineers working on the
Google Brain team within Google's Machine Intelligence Research organization to
conduct machine learning and deep neural networks research.


<h2> Interesting Facts about Tensorflow </h2>

It’s a powerful machine learning framework. The most famous deep learning library in the world is Google's TensorFlow.
Google product uses machine learning in all of its products to improve the search engine, translation, image captioning or recommendations.

TensorFlow not only provides stable [Python](https://www.tensorflow.org/api_docs/python)
but also prividing [C++](https://www.tensorflow.org/api_docs/cc) APIs, as well as
non-guaranteed backward compatible API for other languages.

## Installation

To download the source code, clone this repository:
```
git clone https://github.com/anubhav201241/Tensorflow.git
```

To run them, you also need the latest version of TensorFlow. To install it:
```
pip install tensorflow
```

or (with GPU support):
```
pip install tensorflow_gpu
```

For more details about TensorFlow installation, you can check [TensorFlow Installation Guide](https://www.tensorflow.org/install/)


<h2> TensorFlow Architecture </h2>
<div align="center">
  <img src="https://cdn.educba.com/academy/wp-content/uploads/2019/11/What-is-TensorFlow-Architecture.png">
</div>

TensorFlow Serving is a flexible, high-performance serving system for machine learning models, designed for production environments. 
TensorFlow Serving makes it easy to deploy new algorithms and experiments, while keeping 
the same server architecture and APIs. TensorFlow Serving provides out of the box integration with TensorFlow models, 
but can be easily extended to serve other types of models.

Tensorflow architecture works in three parts:
<ul>
  <li>Preprocessing the data</li>
  <li>Build the model</li>
  <li>Train and estimate the model</li>
</ul>
<div align="center">
  <img src="https://i.ytimg.com/vi/7qUHca-GKHU/maxresdefault.jpg">
</div>

## TensorFlow Servable

These are the central uncompleted units in TensorFlow serving. Servables are the objects that the clients use to perform the computation.
The size of a servable is flexible. A single servable may consist of anything from a lookup table to a unique model in a tuple of interface models. 
Servable should be of any type and interface, which enabling flexibility and future improvements such as Streaming results, Asynchronous modes of operation
and Experimental APIs

## Servable Versions
TensorFlow server can handle one or more versions of the servables, over the lifetime of any single server instance. 
It opens the door for new algorithm configurations, weights, and other data can be loaded over time. 
They also can enable more than one version of a servable to be charged at a time. 
They also allow more than one version of a servable to be loaded concurrently, supporting roll-out and experimentation gradually.

## Servable Streams
A sequence of versions of any servable sorted by increasing version of numbers.

## TensorFlow Models
A serving represents a model in one or more servables. A machine-learned model includes one or more algorithm and lookup the embedding tables. 
A servable can also serve like a fraction of a model for example, an example, a large lookup table be served as many instances.

## TensorFlow Loaders
Loaders manage a servable's life cycle. The loader API enables common infrastructure which is independent of the specific learning algorithm, 
data, or product use-cases involved.

## Sources in TensorFlow Architecture
In simple terms, sources are modules that find and provide servable. Each reference provides zero or more servable streams at a time. 
For each servable stream, a source supplies only one loader instance for every servable.
Each source also provides zero or more servable streams. For each servable stream, a source supplies only one loader instance and makes available to be loaded.

## TensorFlow Managers

TensorFlow managers handle the full lifecycle of a Servables, including Loading Servables, Serving Servables, Unloading Servables
Manager observes to sources and tracks all versions. The Manager tries to fulfill causes, but it can refuse to load an Aspired version.
Managers may also postpone an "unload."

## TensorFlow Core

This manages the below aspects of servables:
<ul>
  <li>Lifecycle</li>
  <li>Metrics</li>
  <li>TensorFlow serving core satisfaction servables and loaders as the opaque objects.</li>
</ul>  
<div align="center">
  <img src="https://static.javatpoint.com/tutorial/tensorflow/images/tensorflow-architecture.png">
</div>
<br><br>

# Basics of Tensorflow

## Tensor
Tensors are multi-dimensional arrays with a uniform type. They are kind of np.arrays but you didn't update the contents if a tensor,only create a new one.
For example, a 4-D array of floating point numbers representing a mini-batch of images with dimensions[batch,height,width,channel]. 

## Constant 
It is use to create a contant tensor from tensor-like object. It can be created using tf.constant() function.
There are 4 paramenters of a TensoFlow constant:
* value: A constant value (or a list) of output dtype.
* dtype: A type of data type (int float boolean)
* shape: Optional dimensions of resulting Tensor.
* name: Optional name for the Tensor.

## Variables
It allows to add new trainable parameters to the graph(structure of neural network). It can be created using tf.Variable() function.
The Variable() constructor requires an initial value for the variable, which can be a Tensor of any type and shape. This initial value defines the type and shape of the variable. After construction, the type and shape of the variable are fixed. It means that shape and type will never change. The value can be changed using one of the assign methods.

## Placeholder
It allows to feed the data to a TensorFlow model from outside a model. It allows us to create our operation and build our computation graph, without needing the data.
It can be created using tf.placeholder() function.

## Session 
In order to run any meaningful operation on the graph,you need a Session. It initiates a TensorFlow Graph object in which tensors are processed through operations.

# Neural Networks Using TensorFlow
  
## Introducton:  
Neural Networks are a subset of Machine Learning, that uses interconnected layers of nodes, called neurons, to mimmick the working of the biological neurons. The idea of Neural Networks was inspired by the human brain and it forms the basis of Deep Learning. The basic architecture of a Neural Network consists of an input layer, one or more hidden layers, and an output layer.   
  
![image.png](https://1.cms.s81c.com/sites/default/files/2021-01-06/ICLH_Diagram_Batch_01_03-DeepNeuralNetwork-WHITEBG.png)
  
## Building Neural Networks in TensorFlow:  
To build and train Neural Networks with Tensorflow we use the [**keras module**](https://www.tensorflow.org/api_docs/python/tf/keras), which is an implementation of the high-level Keras API of TensorFlow. Keras is a deep learning API written in Python, running on top of the machine learning platform TensorFlow. It was developed with a focus on enabling fast experimentation. It is simple to use, flexible and powerful. Keras provides methods to prepare the data for processing, build and traain the model as well as methods to evaluate and fine tune the model parameters. 
  
![image.png](https://i.pinimg.com/originals/f3/ff/48/f3ff4855a71201b102f92a733fd5a875.png)
  
### Defining the Architecture
A simple Neural Network can be built using the [Sequential class](https://www.tensorflow.org/guide/keras/sequential_model). A Sequential model is appropriate for a plain stack of layers where each layer has exactly one input tensor and one output tensor. Keras provides a wide range of layers that can be added to the model in it's [layers module](https://www.tensorflow.org/api_docs/python/tf/keras/layers). Layers can be added using the add() function or they can directly be defined in the object definition. The model is built using the compile() function.
  
### Training the model
The fit() method is used to train the model by passing the training data as arguments. Keras also keeps checkpoints during training and callbacks can be used to stop the training when specified requirements are met. Keras also keeps logs of the training process thus making it poassible to understand exactly what happens during the training process.  
  
### Evaluting the model
The performance of the model can be evaluated using the evaluate() method by passing the test data and labels as arguments. The metrics on which it has to be evaluated can be specified during the compilation. To get the predictions of the model the predict() method is used. After evaluation the model can be refined by tuning the hyperparameters. 

### Saving and Loading the model
The trained model can be saved using the save() method. Keras allows the model weights to be saved in several formats such as the HDF5 format and the SavedModel format. The saved model can then be loaded for use in other applications or it can be deplyed in a web service or an Edge device. 
  
Keras and TensorFlow provides an approachable, highly-productive interface for solving machine learning problems, with a focus on modern deep learning. Thus, TensorFlow is the go-to platform for researchers and engineers working with Neural Networks. 

<br><br>

# Convolutional Neural Networks With Tensorflow

### Learning Objectives
1.   create CNN network from scratch.
2.   understanding what is convolution.
3.   Train your model and visualize the prediction.

#### MNIST Dataset
We will train cnn network on dataset called [MNIST](http://yann.lecun.com/exdb/mnist/). It contains 60,000 images of handwritten digits, from 0 to 9, like these:


![dataset image](https://miro.medium.com/max/700/1*LyRlX__08q40UJohhJG9Ow.png)

# Convolution Neural Network

### Convolution layer
You will usually hear about 2D Convolution while dealing with convolutional neural networks for images. It is a simple mathematical operation in which we slide a matrix or kernel of weights over 2D data and perform element-wise multiplication with the data that falls under the kernel. Finally, we sum up the multiplication result to produce one output of that operation.


![dataset image](https://miro.medium.com/max/1320/1*LT0l-KXw5FXIkcGVl-KXlQ.gif)
Input shape : (1, 9, 9) — Output Shape : (1, 7, 7) — K : (3, 3) — P : (0, 0) — S : (1, 1) — D : (1, 1) — G : 1

### Max Pooling layer
Max pooling is a pooling operation that selects the maximum element from the region of the feature map covered by the filter. Thus, the output after max-pooling layer would be a feature map containing the most prominent features of the previous feature map.
![dataset image](https://media.geeksforgeeks.org/wp-content/uploads/20190721025744/Screenshot-2019-07-21-at-2.57.13-AM.png)


### model summary
![dataset image](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F1723677%2F664483930a8dae8d6bdde8521d743b22%2Fimg2.png?generation=1602506057159013&alt=media)

Conv2D layers are convolutions. Each filter (64 in the first  and 128 in the second convolution layers) transforms a part of the image (3*3 for the first two Conv2D layers. The transformation is applied on the whole image.

MaxPool2D is a downsampling filter. It reduces a 2x2 matrix of the image to a single pixel with the maximum value of the 2x2 matrix. The filter aims to conserve the main features of the image while reducing the size.

Dropout is a regularization layer. In our model, 33% of the nodes in the layer are randomly ignores, allowing the network to learn different features. This prevents overfitting.

relu is the rectifier, and it is used to find nonlinearity in the data. It works by returning the input value if the input value >= 0. If the input is negative, it returns 0.

Flatten converts the tensors into a 1D vector.

The Dense layers are an artificial neural network (ANN). The last layer returns the probability that an image is in each class (one for each digit).

As this model aims to categorize the images, we will use a categorical_crossentropy loss function.

### conclusion
Neural Network with convolutional layer has a great impact on accuracy. our model got 99% accuracy with just two convolution layer. Convolutional neural networks (CNNs) have accomplished astonishing achievements across a variety of domains, including medical research, and an increasing interest has emerged in radiology.



