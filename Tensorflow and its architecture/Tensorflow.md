<div align="center">
  <img src="https://i.ytimg.com/vi/yjprpOoH5c8/maxresdefault.jpg">
</div>

[![Python](https://img.shields.io/pypi/pyversions/tensorflow.svg?style=plastic)](https://badge.fury.io/py/tensorflow)
[![PyPI](https://badge.fury.io/py/tensorflow.svg)](https://badge.fury.io/py/tensorflow)

[TensorFlow](https://www.tensorflow.org/) is an open-source end-to-end framework for building Machine Learning apps. It’s a symbolic math toolkit that performs a variety of tasks including deep neural network training and inference using dataflow and differentiable programming. 
It allows developers to develop and train ML models so that they can create large-scale neural networks with many layers.

It is mainly used for Classification, Perception, Understanding, Discovering, Prediction and Creation of the models.
It enables programmers to construct machine learning applications by utilizing a variety of tools, frameworks, and community resources.

Google’s TensorFlow is now the most well-known deep learning package on the planet. Machine learning is used by Google in all of its products to enhance search, translation, picture captioning, and recommendations.
For instance, Google users may benefit from AI-assisted search that is both faster and more refined. When a user puts a keyword into Google’s search field, the search engine makes a suggestion for the following word.

Google aims to employ machine learning to make the most of their huge datasets in order to provide the best possible experience for its consumers. Machine learning is used by three different organizations. Researchers, Programmers, and Data Scientists.They may all cooperate and increase their efficiency by using the same toolbox.

Tensor Flow was developed to scale since Google has more than just data; they also have the world’s most powerful computer. TensorFlow is a machine learning and deep neural network research library created by the Google Brain Team. It was designed to work on multiple CPUs or GPUs, as well as mobile operating systems in some circumstances, and it includes wrappers in Python, C++, and Java.

<h2> Interesting Facts about Tensorflow </h2>

It’s a powerful machine learning framework. The most famous deep learning library in the world is Google's TensorFlow.
Google product uses machine learning in all of its products to improve the search engine, translation, image captioning or recommendations.

TensorFlow not only provides stable [Python](https://www.tensorflow.org/api_docs/python)
but also prividing [C++](https://www.tensorflow.org/api_docs/cc) APIs, as well as
non-guaranteed backward compatible API for other languages.

## Tensorflow’s Structure

* Preprocessing the data, building the model, and finally training and estimating the model are the three elements of the Tensorflow structure.
* Tensorflow gets its name from the fact that it takes input in the form of a multi-dimensional array, commonly known as tensors. You can create a flowchart of the operations you’d want to run on that input. The input enters at one end, travels through this system of various processes, and emerges as output at the other.
* The tensor goes in, runs through a set of operations, and then comes out the other side, which is why it’s called TensorFlow.

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
and Experimental APIs In other words, after the model is deployed, the client sends a request to the server and some calculations are performed on the server. Now, the object that the client uses to perform calculations, which is deployed on the server, is known as a Servable. The size and granularity of the servable are flexible.

## Servable Versions
TensorFlow server can handle one or more versions of the servables, over the lifetime of any single server instance. 
It opens the door for new algorithm configurations, weights, and other data can be loaded over time. 
They also can enable more than one version of a servable to be charged at a time. 
They also allow more than one version of a servable to be loaded concurrently, supporting roll-out and experimentation gradually.

## Servable Streams
A sequence of versions of any servable sorted by increasing version of numbers.

## Tensorflow Serving

TensorFlow Serving is a flexible, high-performance serving system for machine learning models, designed for production environments. TensorFlow Serving makes it easy to deploy new algorithms and experiments, while keeping the same server architecture and APIs. In this we do tensorflow model serving which is the process of exposing of a trained model so that it can be accessed by an endpoint. Endpoint here can mean a direct user or other software. Tensorflow Serving can handle multiple versions of the servable and allows easy swaps between different versions of the Servables.

## Steps involved in Tensorflow Serving

1. Create your model and train and evaluate it using tensorflow libraries.
2. Save your model.
3. Examine your saved model.
4. Serve your model with TensorFlow Serving. Add TensorFlow Serving distribution URI as a package source: Install TensorFlow Serving
5. Make a request to your model in TensorFlow Serving. Make REST requests.
6. We'll ask our server to give us the latest version of our servable by not specifying a particular version.
7. Get final output from the server.

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


