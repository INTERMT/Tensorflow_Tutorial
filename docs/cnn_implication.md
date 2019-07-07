#### 稀疏交互

在生物学家休博尔和维瑟尔早期关于猫视觉皮层的研究中发现，视觉皮层中存在一些细胞对输入空间也就是图像中的子区域非常敏感，我们称为感受野。在神经网络中，稀疏交互就是下一层节点只和上一层中的部分节点进行连接的操作。稀疏交互可以显著的降低神经网络中参数的数量。

左边是全连接方式，隐藏节点都需要所有的输入；右边是稀疏交互，隐藏层节点只接受一个区域内的节点输入。

![./images/5-5.png](http://www.tensorflownews.com/wp-content/uploads/2018/03/5-15.png)

##### 稀疏交互的实现
<!-- -->

以 MNIST 数据集为例，来实现稀疏交互，并输出结果对应的图片。

MNIST 原始图片：

![./images/Figure_1.png](http://www.tensorflownews.com/wp-content/uploads/2018/03/Figure_1.png)

为了进行局部连接，有两个重要的参数需要选择：

1.局部区域的大小

局部区域的大小，首先以 5 * 5 的局部区域为例：

2.局部特征的抽取次数

针对局部区域可以进行多次特征抽取，可以选择局部特征抽取的次数，首先以抽取 5 次为例。

3.步长

在确定局部区域大小之后，可以平滑的每次移动一个像素，也可以间隔 N 个像素进行移动。
如图：

![./images/5-10.jpg](http://www.tensorflownews.com/wp-content/uploads/2018/03/5-10.jpg)

也可以使用不同的特征提取器对同一片区域，进行多次特征提取，如图所示：

<!-- > 引用图片

https://devblogs.nvidia.com/parallelforall/image-segmentation-using-digits-5/

Left: An example input volume in red and an example volume of neurons in the first Convolutional layer. Each neuron in the convolutional layer is connected only to a local region in the input volume spatially, but to the full depth (i.e. all color channels). Note, there are multiple neurons (5 in this example) along the depth, all looking at the same region in the input. Right: The neurons still compute a dot product of their weights with the input followed by a non-linearity, but their connectivity is now restricted to be local spatially. Source: Stanford CS231 course.

-->

![./images/5-7.png](http://www.tensorflownews.com/wp-content/uploads/2018/03/5-7.png)

4.边缘填充

在进行一次局部连接的过程中，如果不进行边缘填充，图像的维度将会发生变化，如图所示：

4 * 4 的图像，进行了 3 * 3 的局部连接，维度发生了变化。

![./images/5-11.png](http://www.tensorflownews.com/wp-content/uploads/2018/03/5-11.png)

对于边缘的两种处理方法：

<!-- 引用图片 https://leonardoaraujosantos.gitbooks.io/artificial-inteligence/content/convolutional_neural_networks.html 边缘不填充： ![./images/no_padding_no_strides.gif](http://www.tensorflownews.com/wp-content/uploads/2018/03/no_padding_no_strides.gif) 边缘填充： ![./images/same_padding_no_strides.gif](http://www.tensorflownews.com/wp-content/uploads/2018/03/same_padding_no_strides.gif) -->

![./images/5-8.png](http://www.tensorflownews.com/wp-content/uploads/2018/03/5-8.png)
#### 权值共享

##### 权值共享的由来

降低网络中参数的个数，还有一个方法就是共享参数。每一组参数都可以认为是一个特征提取器，即使图像有一定的偏移，还是可以将相应的特征用同一组参数提取出来。

#### TensorFlow 实现局部连接和权值共享

如下图所示：
为了演示局部连接和权值共享在特征提取方面的作用，接下来将使用在稀疏交互中一种很常用的一组权值，它的作用是边缘提取。

就是这个矩阵组成的权值：

```
[[-1,-1,-1],
[-1,8,-1],
[-1,-1,-1]]
```

使用 TensorFlow 的 convolution 函数对 MNIST 数据集做卷积操作。因为这部分代码涉及到维度相关的操作比较多，在稍后卷积网络部分会有详细说明。这段代码单独实现了卷积功能：

```
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt

#引入 MNIST 数据集
mnist = input_data.read_data_sets("/tmp/data/", one_hot=False)

#选取训练集中的第 1 个图像的矩阵
mnist_one=mnist.train.images[0]

plt.subplot(121)
plt.imshow(mnist_one.reshape((28,28)), cmap=plt.cm.gray)

#输出图片的维度，结果是：(784,)
print(mnist_one.shape)

#因为原始的数据是长度是 784 向量，需要转换成 1*28*28*1 的矩阵。
mnist_one_image=mnist_one.reshape((1,28,28,1))

#输出矩阵的维度
print(mnist_one_image.shape)

#滤波器参数
filter_array=np.asarray([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])

#滤波器维度
print(filter_array.shape)

#调整滤波器维度
filter_tensor=filter_array.reshape((3,3,1,1))

#卷机操作
conv_image_tensor=tf.nn.convolution(mnist_one_image,filter=tf.to_float(filter_tensor),padding="SAME")

#返回的张量维度
print(conv_image_tensor.shape)

#调整为二维图片
conv_image=tf.reshape(conv_image_tensor,[28,28])

with tf.Session() as sess:

#获得张量的值
conv_image=sess.run(conv_image)

plt.subplot(122)

#使用 matplotlib 输出为图片
plt.imshow(conv_image, cmap=plt.cm.gray)

plt.show()
```

![./images/5-16.png](http://www.tensorflownews.com/wp-content/uploads/2018/03/5-16.png)

### 池化介绍

#### 池化

除了之前的两种方式，在数据量很大，类比现实生活中事情纷繁复杂的时候，我们总是想抓住重点，在图像中，可以在一个区域选取一个重要的点。

一般是选择值最大的点，作为这一个区域的代表：

如图所示：

![./images/5-12.png](http://www.tensorflownews.com/wp-content/uploads/2018/03/5-12.png)
这个池化选取的是 2 * 2 的区域，留下值最大点，步长为 2。原来 4 * 4 的图片矩阵池化之后变成了 2 * 2 的图片矩阵。

### 手写数字识别

接下来将会以 MNIST 数据集为例，使用卷积层和池化层，实现一个卷积神经网络来进行手写数字识别，并输出卷积和池化效果。

#### 数据准备

* MNIST 数据集下载

MNIST 数据集可以从 THE MNIST DATABASE of handwritten digits 的网站直接下载。
网址：http://yann.lecun.com/exdb/mnist/

train-images-idx3-ubyte.gz: 训练集图片
train-labels-idx1-ubyte.gz: 训练集列标
t10k-images-idx3-ubyte.gz: 测试集图片
t10k-labels-idx1-ubyte.gz: 测试集列标

TensorFlow 有加载 MNIST 数据库相关的模块，可以在程序运行时直接加载。

代码如下：

```
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as pyplot

#引入 MNIST 数据集
mnist = input_data.read_data_sets("/tmp/data/", one_hot=False)

#选取训练集中的第 1 个图像的矩阵
mnist_one=mnist.train.images[0]

#输出图片的维度，结果是：(784,)
print(mnist_one.shape)

#因为原始的数据是长度是 784 向量，需要转换成 28*28 的矩阵。
mnist_one_image=mnist_one.reshape((28,28))

#输出矩阵的维度
print(mnist_one_image.shape)

#使用 matplotlib 输出为图片
pyplot.imshow(mnist_one_image)

pyplot.show()
```

代码的输出依次是：
1.单个手写数字图片的维度：
(784,)

2.转化为二维矩阵之后的打印结果：
(28, 28)

3.使用 matplotlib 输出为图片

![./images/5-13.png](http://www.tensorflownews.com/wp-content/uploads/2018/03/5-13.png)