# TensorFlow Basics:

- ### **Tensors**

    A tensor is a higher-dimensional generalisation of vectors and matrices. Tensors are internally represented by TensorFlow as n-dimensional arrays of base datatypes. Tensors are a fundamental aspect of TensorFlow, which should come as no surprise. They are the primary objects that are passed around and moved around throughout the programme. Each tensor represents a partially defined computation that will yield a value in the end. TensorFlow programmes operate by constructing a graph of Tensor objects that describes how tensors are related. Running different parts of the graph allows for the generation of results.

    Each tensor is associated with a data type and a shape. Float32, int32, string, and other data types are available., while shape represents a data dimension. Tensors, like vectors and matrices, can be subjected to operations such as addition, subtraction, dot product, cross product, and so on.

    - Creating Tensors

        Simply define the tensor's value and the datatype, and you're ready to go! It's worth noting that we usually deal with tensors of numeric data; string tensors are uncommon.

        ```python
        string = tf.Variable("DevIncept", tf.string) 
        number = tf.Variable(2021, tf.int16)
        floating = tf.Variable(20.21, tf.float64)
        ```

    - Rank of a Tensor

        Degree is another word for rank; these terms simply refer to the number of dimensions in the tensor. We've just created a *tensor of rank 0*, also known as a scalar. We'll now make some tensors with higher degrees/ranks.

        ```python
        // Source Code

        Rank1 = tf.Variable(["DevIncept"], tf.string) 
        Rank2 = tf.Variable([["Open", "Source"], ["Contribution", "2021"]], tf.string)

        tf.rank(Rank2) // Determine rank of a tensor
        ```

        ```python
        //Output

        <tf.Tensor: shape=(), dtype=int32, numpy=1>
        ```

        A tensor's rank is directly related to the deepest level of nested lists. As the deepest level of nesting is 1, you can see that ```["DevIncept"]``` is a rank 1 tensor in the first example. Whereas ```[["Open", "Source"], ["Contribution", "2021"]]``` is a rank 2 tensor in the second example because the deepest level of nesting is 2.

    - Shape of a Tensor

        A tensor's shape is simply the number of elements in each dimension. TensorFlow will attempt to determine a tensor's shape, but it may be unknown at times.

        ```python
        //Source Code

        Rank1.shape
        ```

        ```python
        //Output

        TensorShape([1])
        ```

    - Changing Shape

        A tensor's number of elements is the product of the sizes of all its shapes. There are frequently many shapes with the same number of elements, making the ability to change the shape of a tensor convenient.

        ```python
        //Source Code

        tensor1 = tf.ones([1,2,3])  # A shape [1,2,3] tensor is created by calling tf.ones().
        tensor2 = tf.reshape(tensor1, [2,3,1])  # [2,3,1] reshape existing data to shape.
        tensor3 = tf.reshape(tensor2, [3, -1])  # -1 tells the tensor to calculate the size of the dimension in that place.
                                                # Tensor [3,3] will be reshaped by this.
        # There MUST be an equal number of elements in the newly-reshaped Tensor.

        print(tensor1, tensor2, tensor3)
        ```

        ```python
        //Output

        tf.Tensor(
        [[[1. 1. 1.]
          [1. 1. 1.]]], shape=(1, 2, 3), dtype=float32)
        tf.Tensor(
        [[[1.]
          [1.]
          [1.]]

         [[1.]
          [1.]
          [1.]]], shape=(2, 3, 1), dtype=float32)
        tf.Tensor(
        [[1. 1.]
         [1. 1.]
         [1. 1.]], shape=(3, 2), dtype=float32)
        ```

    - Tensor Types

        There are various types of tensors. Variable, Constant, Placeholder, and SparseTensor are the most commonly used. All of these tensors are immutable when executed with Variable, which means their value cannot change during execution. For the time being, let us consider Variable tensor is used when we want to potentially change the value of our tensor.

- ### **Sessions**

    A session allows you to run graphs or parts of graphs. It does so by allocating resources (on one or more machines) and storing the actual values of intermediate results and variables. A Session object represents the environment in which Operation objects and Tensor objects are evaluated. A session may be the owner of resources like tf.Variable, tf.queue.QueueBase, and tf.compat.v1.ReaderBase. It is critical to decommission these resources when they are no longer needed.

    ```python
    with tf.Session(graph=graph) as sess:
      sess.run(initialize)
      sess.run(assign)
      print(sess.run(variable))
    ```

    The tensor was created by the code. The tensor will then be evaluated in a session. Using tf.Session, the code creates a session instance, sess. After that, the sess.run() function evaluates the tensor and returns the results.

    ```python
    # Source Code
    x = tf.placeholder(tf.string)
    y = tf.placeholder(tf.int32)
    with tf.Session() as sess:
    		output1 = sess.run(x, feed_dict={x: 'DevIncept'})
        output2 = sess.run(y, feed_dict={x: 'DevIncept', y: 123})
        output3 = sess.run(y, feed_dict={y: 'DevIncept', x: 123})
        print(output1)
        print(output2)
        print(output3)

    # Output
    DevIncept
    DevIncept
    123
    ```

    To set the placeholder tensor, use the feed dict parameter in tf.session.run(). In the preceding example, the tensor x is set to the string "DevIncept." It is also possible to use feed dict to set more than one tensor, as shown below: Note: If the data passed to feed dict does not match the tensor type and cannot be cast into the tensor type, the error “ValueError: invalid literal for...” will be returned.

- ### **Operators**

    Tensorflow generates predefined operators that can be applied to tensors. The Tensorflow operator makes it simple to perform basic tensor operations. Some of them are as follows:

    - Arithmetic operations include add, subtract, multiply, and divide.

    ```python
    # Source Code
    A1  = tf.constant([1,6,2,8])
    A2  = tf.constant([9,4,3,0])
    ans = tf.add(A1, A2)
    print(ans)

    # Output
    Tensor("Add:0", shape=(4,), dtype=int32)
    ```

    - Matrix operations include tf.diag, tf.transpose, tf.matrix_inverse, tf.norm, etc.

    ```python
    # Source Code
    A1  = tf.constant([1,6,2,8])
    trans = tf.transpose(A1)
    print(trans)

    # Output
    Tensor("transpose:0", shape=(4,), dtype=int32)
    ```

    - Control flow operations.

    Operations in Tensorflow allow you to control the execution of operations and add conditional dependencies to them, examples of these classes include the following: tf.tuple, tf.group, tf.cond, etc.

    ```python
    # Source Code
    A1  = tf.constant([1,6,2,8])
    iden = tf.identity(A1)
    print(iden)

    # Output
    Tensor("Identity:0", shape=(4,), dtype=int32)
    ```

    - Logical operations include logical and, logical not, and so on.

    ```python
    # Source Code
    x  = tf.constant([False,False,True,True])
    y  = tf.constant([False,True,True,False])
    logic = tf.math.logical_or(x,y)
    print(logic)

    # Output
    Tensor("LogicalOr:0", shape=(4,), dtype=bool)
    ```

    - Comparison operations include equal, less, not equal, and so on.

    ```python
    # Source Code
    A1  = tf.constant([1,6,2,8])
    A2  = tf.constant([9,4,3,0])
    eq = tf.equal(A1, A2)
    print(eq)

    # Output
    Tensor("Equal:0", shape=(4,), dtype=bool)
    ```

    - Debugging operations include: is finite, is inf, and so on. Used to validate values.

    ```python
    # Source Code
    x = tf.constant([5.0, 4.8, 6.8, np.inf, np.nan])
    fin = tf.math.is_finite(x) 
    print(fin)

    # Output
    Tensor("IsFinite:0", shape=(5,), dtype=bool)
    ```

- ### **Variables**

    You should use TensorFlow variables to represent shared and persistent state that your programme manipulates.

    - Creating Variables

        ```python
        # Source Code
        tf.Variable(1)
        tf.Variable("DevIncept") #using String
        tf.Variable(['2','0','2','1']) #using Array
        tf.constant([1,2,3,4]) #Constant
        tf.Variable([[1,2],[3,4]], shape = (2,2), dtype="int32") # Different Shape
        tf.reshape([1,2,3,4], (2,2))

        # Output
        <tf.Variable 'Variable:0' shape=() dtype=int32>
        <tf.Variable 'Variable_3:0' shape=() dtype=string>
        <tf.Variable 'Variable_2:0' shape=(4,) dtype=string>
        <tf.Tensor 'Const_5:0' shape=(4,) dtype=int32>
        <tf.Variable 'Variable_5:0' shape=(2, 2) dtype=int32>
        <tf.Tensor 'Reshape:0' shape=(2, 2) dtype=int32>
        ```

    - Convert Variable into a Tensor

        ```python
        # Source Code
        a = tf.Variable([1,2,3,4])
        tf.convert_to_tensor(a)

        # Output
        <tf.Tensor 'ReadVariableOp:0' shape=(4,) dtype=int32>
        ```

    - Variable Name and Value

        ```python
        # Source Code
        a = tf.Variable([[1.0, 2.0],[1.0, 2.0]])
        print(a.name)
        print(a.value())

        # Output
        Variable_10:0
        Tensor("ReadVariableOp_1:0", shape=(2, 2), dtype=float32)
        ```

    - Shape of a Variable

        ```python
        # Source Code
        a = tf.Variable([[1.0, 2.0],[1.0, 2.0]])
        print(a.shape)

        # Output
        (2, 2)
        ```