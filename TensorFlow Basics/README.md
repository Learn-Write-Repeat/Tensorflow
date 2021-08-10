<h1 align="center">
    Session
</h1>


A session allows to execute graphs or part of graphs. It allocates resources (on one or more machines) for that and holds the actual values of intermediate results and variables.
A Session object encapsulates the environment in which Operation objects are executed, and Tensor objects are evaluated
A session may own resources, such as tf.Variable, tf.queue.QueueBase, and tf.compat.v1.ReaderBase. It is important to release these resources when they are
no longer required. 
<br></br>
![Session](https://docs.graphcore.ai/projects/tensorflow1-user-guide/en/latest/_images/Session_Graph.png)

A "TensorFlow Session", as shown above, is an environment for running a graph. The session is in charge of allocating the operations to GPU(s) and/or CPU(s), including remote machines. Let’s see how you use it:


```python
with tf.Session(graph=graph) as sess:
  sess.run(initialize)
  sess.run(assign)
  print(sess.run(variable))
# Output: 13
```

The code has created the tensor.The next step is to evaluate the tensor in a session.

The code creates a session instance, `sess`, using `tf.Session`. The `sess.run()` function then evaluates the tensor and returns the results.

### Session's feed_dict


```python
x = tf.placeholder(tf.string)

with tf.Session() as sess:
    output = sess.run(x, feed_dict={x: 'Hello World'})
    print(output)
```

    Hello World


Use the feed_dict parameter in `tf.session.run()` to set the placeholder tensor. The above example shows the tensor `x` being set to the string `"Hello, world"`. It's also possible to set more than one tensor using `feed_dict` as shown below:


```python
x = tf.placeholder(tf.string)
y = tf.placeholder(tf.int32)
z = tf.placeholder(tf.float32)

with tf.Session() as sess:
    output_x = sess.run(x, feed_dict={x: 'Test String', y: 123, z: 45.67})
    output_y = sess.run(y, feed_dict={x: 'Test String', y: 123, z:45.67})
    print(output_x)
    print(output_y)
```

    Test String
    123


**Note**: If the data passed to the `feed_dict` doesn’t match the tensor type and can’t be cast into the tensor type, you’ll get the error `“ValueError: invalid literal for...”`.

# Tensorflow Operator

---

## Introduction to Operator
Tensorflow prodives certain predefined operators that can be performed on tensors. Tensorflow operator helps us to perform basic operation on tensor easily.
___
There are sevral types of operations that can be done on tensorflow. Some of them are mentioned below:
* Arithmetic operations : add, subtract, multiply,...
* Matrix operations : matmul, matrix_inverse, ...
* Control flow operations : tuple, group, ...
* Logical operations : logical_and, logical_not, ...
* Comparision operations: equal, less, not_equal, ...
* Debugging operations : is_finite, is_inf,..

Example of each type of operations by operators is shown in the code file.

### Conclusion
Tensorflow operation is very useful in especially when we are working with tensor and matrix. [Click Here](https://www.tensorflow.org/lite/guide/op_select_allowlist) to visit list of all the operator in tensorflow.

# Tensorflow Variables

A TensorFlow variable is the recommended way to represent shared, persistent state your program manipulates. This guide covers how to create, update, and manage instances of tf.Variable in TensorFlow.

Variables are created and tracked via the tf.Variable class. A tf.Variable represents a tensor whose value can be changed by running ops on it. Specific ops allow you to read and modify the values of this tensor. Higher level libraries like tf.keras use tf.Variable to store model parameters.

# Creation

We can create tf.Variableobjects with the tf.Variable() function. The tf.Variable() function accepts different data types as parameter such as integers, floats, strings, lists, and tf.Constant objects.

# Variable Value

Every Variable must specify an initial_value. Otherwise, TensorFlow raises an error and says that Value Error: initial_value must be specified. Therefore, make sure that you pass on an initial_valueargument when creating Variable objects. To be able to view a Variable’s values, we can use the .value() function as well as the .numpy() function. 

# Variable Name 

Name is a Variable attribute which helps developers to track the updates on a particular variable. You can pass a name argument while creating the Variable object. If you don’t specify a name, TensorFlow assigns a default name,like Variable:0.

# Variable Dtype

Each Variable must have a uniform data type that it stores. Since there is a single type of data stored for every Variable, you can also view this type with the .dtype attribute.

# Shape, Rank, and Size

The shape property shows the size of each dimension in the form of a list. We can view the shape of the Variable object with the .shape attribute. Then, we can view the number of dimensions that a Variable object has with the tf.size() function. Finally, Size corresponds to the total number of elements a Variable has. We need to use the tf.size() function to count the number of elements in a Variable. 


