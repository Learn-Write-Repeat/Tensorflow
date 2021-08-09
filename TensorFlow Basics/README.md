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
