
# TensorFlow - Mathematical Foundations

***Mathematics*** is considered as the ***heart of any machine learning algorithm***. It is with the help of core concepts of Mathematics, a solution for specific machine learning algorithm is defined. Therefore it is important to understand mathematics before creating the basic application in TensorFlow.

## Vector

An array of numbers, which is either continuous or discrete, is defined as a vector. Machine learning algorithms deal with fixed length vectors for better output generation.

Machine learning algorithms deal with multidimensional data so vectors play a crucial role.

![](https://www.tutorialspoint.com/tensorflow/images/vector.jpg)

The pictorial representation of vector model is as shown below −

![](https://www.tutorialspoint.com/tensorflow/images/vector_model.jpg)


## Scalar
Scalar can be defined as one-dimensional vector. Scalars are those, which include only magnitude and no direction. With scalars, we are only concerned with the magnitude.

Examples of scalar include weight and height parameters of children.

## Matrix

Matrix can be defined as multi-dimensional arrays, which are arranged in the format of rows and columns. The size of matrix is defined by row length and column length. Following figure shows the representation of any specified matrix.


![](https://www.tutorialspoint.com/tensorflow/images/multi_dimensional_arrays.jpg)


Consider the matrix with “m” rows and “n” columns as mentioned above, the matrix representation will be specified as “m*n matrix” which defined the length of matrix as well.



## Mathematical Computations
In this section, we will learn about the different Mathematical Computations in TensorFlow.

### 1. Addition of matrices
Addition of two or more matrices is possible if the matrices are of the same dimension. The addition implies addition of each element as per the given position.

Consider the following example to understand how addition of matrices works −
> **Example:**

        [ 1 2 ]        [ 5 6 ]            [ 1+5 2+6 ]            [ 6   8  ]
    A = [ 3 4 ],   B = [ 7 8 ],   A + B = [ 3+7 4+8 ],   A + B = [ 10  12 ]
		

### 2. Subtraction of matrices
The subtraction of matrices operates in similar fashion like the addition of two matrices. The user can subtract two matrices provided the dimensions are equal.
> **Example:** 

        [ 1 2 ]        [ 5 6 ]            [ 1-5 2-6 ]            [-4 -4 ]
    A = [ 3 4 ],   B = [ 7 8 ],   A - B = [ 3-7 4-8 ],   A - B = [-4 -4 ]

### 3. Multiplication of matrices
For two matrices **A m*n** and **B p*q** to be multipliable, **n should be equal to p**. The resulting matrix is −

**C m*q**

> **Example:**

        [ 1 2 ]        [ 5 6 ]
    A = [ 3 4 ],   B = [ 7 8 ]

    C11 = [ 1 2 ][5]    C12 = [ 7 8 ][6]   
                 [7],                [8], 
		
    C21 = [ 3 4 ][5]    C22 = [ 3 4 ][6]
                 [7],                [8], 
								 
    C11=> 19,  C12=>22
    C21=> 43,  C22=>50

        [ C11 C12 ]
    C = [ C21 C22 ]
		
### 4.Transpose of matrix
The transpose of a matrix A, m*n is generally represented by AT (transpose) n*m and is obtained by transposing the column vectors as row vectors.

> **Example:**

        [ 1 2 ]          [ 1 3 ]
    A = [ 3 4 ],   A^T = [ 2 4 ]

### 5. Dot product of vectors
Any vector of dimension n can be represented as a matrix v = R^n*1.

         [ v11 ]         [ v21 ]
    v1 = [ v12 ]    v2 = [ v22 ]
         [  .  ]         [  .  ]
         [  .  ]         [  .  ]
         [ v1n ]         [ v2n ]
The dot product of two vectors is the sum of the product of corresponding components − Components along the same dimension and can be expressed as

    v1.v2=(v1^T)v2=(v2^T)v1=v11v21+v12v22+⋅⋅⋅⋅+v1nv2n=∑k=1nv1kv2k
The example of dot product of vectors is mentioned below −

> **Example:**

         [ 1 ]         [  3 ]
    v1 = [ 2 ]    v2 = [  5 ]
         [ 3 ]         [ -1 ]
	 
    v1.v2=(v1^T)v2=1×3+2×5−3×1=>10
