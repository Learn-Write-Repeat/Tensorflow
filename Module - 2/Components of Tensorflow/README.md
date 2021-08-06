# Installation of  TensorFlow 

```
pip install tensorflow 
pip install tensorflow-gpu [for gpu version]
```  
# Inrtroduction to TensorFlow
In TensorFlow, computation is described using data flow graphs. Each node of the graph represents an instance of a mathematical operation (like addition, division, or multiplication) and each edge is a multi-dimensional data set (tensor) on which the operations are performed.

  ![tensorflowGraph](https://uploads.toptal.io/blog/image/124829/toptal-blog-image-1511963286232-e03ddeaf3e7b01619064bdb4ceb2f4b0.png)

- **Tensor** : A tensor is a generalization of vectors and matrices to potentially higher dimensions. Internally, TensorFlow represents tensors as n-dimensional arrays of base datatypes. 
- **Nodes** : In TensorFlow, each node represents the instantion of an operation. Each operation has >= inputs and >= 0 outputs.
  
- **Operation** : This represents an abstract computation, such as adding or multiplying matrices. An operation manages tensors. It can just be polymorphic: the same operation can manipulate different tensor element types. For example, the addition of two int32 tensors, the addition of two float tensors, and so on.
- **Session** : When the client program has to establish communication with the TensorFlow runtime system, a session must be created. As soon as the session is created for a client, an initial graph is created and is empty. It has two fundamental methods:
    
    - **session.extend** : In a computation, the user can extend the execution graph, requesting to add more operations (nodes) and edges (data).

    - **session.run** : Using TensorFlow, sessions are created with some graphs, and these full graphs are executed to get some outputs, or sometimes, subgraphs are executed thousands/millions of times using run invocations. Basically, the method runs the execution graph to provide outputs.

![dataFlowGraph](https://static.packt-cdn.com/products/9781786468574/graphics/image_01_006.jpg)
A Tensor has following properties:

- a data type (int32 , float64 , string etc...)
- a shape

### Importing TensorFlow:
> ``` import tensorflow as tf ```

### Tensor Types:
- **Constant** : Constants are used as constants. They create a node that takes value and it does not change. We can create constansts by
``` tf.constant()  ``` it takes five arguments:

```
tf.constant(value, dtype=None, shape=None, name='Const', verify_shape=False)
```
### Example:
```
# create graph
a = tf.constant(2)
b = tf.constant(3)
c = a + b
# launch the graph in a session
with tf.Session() as sess:
    print(sess.run(c))
```
- **VARIABLE** : A TensorFlow [variable](https://www.tensorflow.org/guide/variable#create_a_variable) is the recommended way to represent shared, persistent state your program manipulates.
``` 
w = tf.Variable(<initial-value>, name=<optional-name>
```
- **Placeholder** : A placeholder is simply a variable that we will assign data to at a later date. It allows us to create our operations and build our computation graph, without needing the data. In TensorFlowterminology, we then feed data into the graph through these placeholders.

 ### Example :
 ``` 
 import tensorflow as tf

x = tf.placeholder("float", [None, 3])
y = x * 2

with tf.Session() as session:
    x_data = [[1, 2, 3],
              [4, 5, 6],]
    result = session.run(y, feed_dict={x: x_data})
    print(result)
    
``` 

## Parameters of Tensor :
***
* **Rank** : Identifies the number of dimensions of the tensor. A rank is known as the order or n-dimensions of a tensor, where for example rank 1 tensor is a vector or rank 2 tensor is matrix.
We use ```tf.rank()``` to see the rank.
```
tensor_rank2= tf.Variable([["hello", "hi"], ["thank", "you"]], tf.string)#rank2 tensor

#To see the rank of a tensor
print(tf.rank(tensor_rank2)) #tf.rank()used to determine the rank

```
  
* **Shape** :  The shape of a tensor is the number of rows and columns it has. We use ``` .shape ``` to seterminw the shape.
```
tensor_rank2= tf.Variable([["hello", "hi"], ["thank", "you"]], tf.string)#rank2 tensor
print(tensor_rank2.shape)
```
* **Type** : The data type assigned to tensor elements.

![tensor](https://cdn.guru99.com/images/1/080418_1250_WhatisaTens3.png)

## Tensor operations :
|TensorFlow operator        | Description |
| ----------- | ----------- |
| ``` tf.add(a,b) ```       | a+b    |
| ``` tf.substract(a, b)```      | a-b      |
|``` tf.multiply(a, b) ```| a*b|
|``` tf.div(a, b) ```| a/b|
|``` tf.pow(a, b)```|a^b |
|```tf.sqrt(a)```| sqrt(a)|
|``` tf.exp(a)```|e^x|
|```tf.maximum(a,b)```|max(a,b)|
|```tf.log(a)```|log(a)|
|```tf.cos(a)```|cos(a)|
|``` tf.sin(a)```|sin(a)|
 
# Linear regression :

Linear regression is one of the most basic forms of machine learning and is used to predict numeric values.

In this tutorial we will use a linear model to predict the survival rate of passangers from the titanic dataset.

### Setup:
1. install sklearn:
``` pip install  sklearn ```

2. 
``` 
    import os
    import sys

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from IPython.display import clear_output
    from six.moves import urllib
    import tensorflow.compat.v2.feature_column as fc

    import tensorflow as tf
```
### Loading Dataset:

TrainingDataset Link : <https://storage.googleapis.com/tf-datasets/titanic/train.csv>
TestingDataset Link : <https://storage.googleapis.com/tf-datasets/titanic/eval.csv>

The **training data** is what we feed to the model so that it can develop and learn. It is usually a much larger size than the testing data.

The **testing data** is what we use to evaulate the model and see how well it is performing. We must use a seperate set of data that the model has not been trained on to evaluate it.
```
# Load dataset.
dftrain = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv')
dfeval = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv')
y_train = dftrain.pop('survived')
y_eval = dfeval.pop('survived')
```
The ```pd.read_csv()``` method will return to us a new pandas dataframe. You can think of a dataframe like a table. In fact, we can actually have a look at the table representation.

We've decided to pop the "survived" column from our dataset and store it in a new variable. This column simply tells us if the person survived our not.

To look at the data we'll use the ```.head()``` method from pandas. This will show us the first 5 items in our dataframe.
``` 
dftrain.head()
```
![head](https://github.com/archismansaha/100DaysOfCode/blob/main/head.PNG?raw=true)

For more statistical view we can use ``` .describe()``` method :
``` 
dftrain.describe() 
```
![describe](https://github.com/archismansaha/100DaysOfCode/blob/main/describe.PNG?raw=true)

Lets look at the shape: 

``` 
dftrain.shape

Output:(627,9)
```
Let's have a look at our survival information.
```
y_train.head()
```
![y_train](https://github.com/archismansaha/100DaysOfCode/blob/main/y_train.PNG?raw=true)

Let's visualize the informations for better understanding:]
```
dftrain.age.hist(bins=10)
```
![age](https://github.com/archismansaha/100DaysOfCode/blob/main/age.PNG?raw=true)
```
dftrain.sex.value_counts().plot(kind='barh')
```
![sex](https://github.com/archismansaha/100DaysOfCode/blob/main/sex.PNG?raw=true)
```
dftrain['class'].value_counts().plot(kind='barh')
```
![class](https://github.com/archismansaha/100DaysOfCode/blob/main/class.PNG?raw=true)

After analyzing this information, we should notice the following:
- Most passengers are in their 20's or 30's 
- Most passengers are male
- Most passengers are in "Third" class
- Females have a much higher chance of survival

### Feature Columns: 
***
In our dataset we have two different kinds of information: **Categorical and Numeric**

Our **categorical data** is anything that is not numeric! For example, the sex column does not use numbers, it uses the words "male" and "female".

Before we continue and create/train a model we must convet our categorical data into numeric data. We can do this by encoding each category with an integer (ex. male = 1, female = 2).
## Feature Engineering to Model :
***
Estimators use a system called feature columns to describe how the model should interpret each of the raw input features. An Estimator expects a vector of numeric inputs, and feature columns describe how the model should convert each feature.

Selecting and crafting the right set of feature columns is key to learning an effective model. A feature column can be either one of the raw inputs in the original features dict (a base feature column), or any new columns created using transformations defined over one or multiple base columns (a derived feature columns).

The linear estimator uses both numeric and categorical features. Feature columns work with all TensorFlow estimators and their purpose is to define the features used for modeling. Additionally, they provide some feature engineering capabilities like one-hot-encoding, normalization, and bucketization.
```
#converting catagorical data to numeric data

Catagorical=['sex', 'n_siblings_spouses', 'parch', 'class', 'deck', 'embark_town', 'alone'] #catagorical coloumn

Numeric=['age', 'fare'] #numeric coloumn
feature_columns=[]

for feature_name in Catagorical:
  vocabulary = dftrain[feature_name].unique()  # getting list of all unique values from given feature column
  feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))

for feature_name in Numeric:
  feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))

print(feature_columns)
```
![numeric](https://github.com/archismansaha/100DaysOfCode/blob/main/numeric.PNG?raw=true)


### The Training Process
****
So, we are almost done preparing our dataset and I feel as though it's a good time to explain how our model is trained. Specifically, how input data is fed to our model. 

For this specific model data is going to be streamed into it in small batches of 32. This means we will not feed the entire dataset to our model at once, but simply small batches of entries. We will feed these batches to our model multiple times according to the number of **epochs**. 

An **epoch** is simply one stream of our entire dataset. The number of epochs we define is the amount of times our model will see the entire dataset. We use multiple epochs in hope that after seeing the same data multiple times the model will better determine how to estimate it.

Ex. if we have 10 ephocs, our model will see the same dataset 10 times. 

Since we need to feed our data in batches and multiple times, we need to create something called an **input function**. The input function simply defines how our dataset will be converted into batches at each epoch.

``` 
def make_input_fn(data_df, label_df, num_epochs=10, shuffle=True, batch_size=32):
  def input_function():  # this inner function will be returned
    ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))  # creating a tf.data.Dataset object with data and its label
    if shuffle:
      ds = ds.shuffle(1000)  # randomizing the order of data
    ds = ds.batch(batch_size).repeat(num_epochs)  # split dataset into batches of 32 and repeat process for number of epochs
    return ds  # return a batch of the dataset
  return input_function  # return a function object for use

train_input_fn = make_input_fn(dftrain, y_train)  
eval_input_fn = make_input_fn(dfeval, y_eval, num_epochs=1, shuffle=False)
```
### Creating Model :
***
```
linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns) # passing feature coloumns to create a linear estimator
```
### Training the model :
***
```
linear_est.train(train_input_fn) #training model
result = linear_est.evaluate(eval_input_fn) #testing model

clear_output() 
print(result)
```
![result]( https://github.com/archismansaha/100DaysOfCode/blob/main/TRAIN.PNG?raw=true)

We can use the ```.predict()``` method to get survival probabilities from the model. This method will return a list of dicts that store a predicition for each of the entries in our testing data set. 

![prediction](https://github.com/archismansaha/100DaysOfCode/blob/main/predicted.PNG?raw=true)

********************************

### SOURCE: 
<https://www.toptal.com/machine-learning/tensorflow-machine-learning-tutorial>
<https://leonardoaraujosantos.gitbook.io/artificial-inteligence/appendix/tensorflow>
<https://www.guru99.com/tensor-tensorflow.html#7>
<https://www.tensorflow.org/>
<https://tensorflow.rstudio.com/guide/tensorflow/tensors/>









