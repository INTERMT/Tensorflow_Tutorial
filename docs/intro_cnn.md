## 卷积神经网络

### 图像识别问题和数据集

> 计算机视觉中有哪些问题？典型问题：经典数据集。

在 2012 年的 ILSVRC 比赛中 Hinton 的学生 Alex Krizhevsky 使用深度卷积神经网络模型 AlexNet 以显著的优势赢得了比赛，top-5 的错误率降低至了 16.4% ，相比第二名的成绩 26.2% 错误率有了巨大的提升。AlexNet 再一次吸引了广大研究人员对于卷积神经网络的兴趣，激发了卷积神经网络在研究和工业中更为广泛的应用。现在基于卷积神经网络计算机视觉还广泛的应用于医学图像处理，人脸识别，自动驾驶等领域。越来越多的人开始了解卷积神经网络相关的技术，并且希望学习和掌握相关技术。因为卷积神经网络需要大量的标记数据集，有一些经典的数据集可以用来学习，同时解决一些常见的计算机视觉问题。

* 卷积神经网络的具体应用，经典数据集。

比如最常用的 mnist 手写数字数据集，这个数据集有 60000个训练样本，10000个测试样本；cfair 10 数据集包含 60000 个 32x32 像素 的彩色图片，它们分别属于 10 个类别，每一个类别有 6000 个图片，其中 50000 个作为训练集，10000个作为测试集。

* 卷积神经网络在这些应用上取得的成果。

针对 mnist 手写数字数据集，现在已经达到了 99% 以上的识别率，在稍后的学习中，也会实现一个准确率达到 99% 以上的模型。

### 卷积神经网络简介

> 卷积神经网络是什么，以及卷积神经网络将如何解决计算机视觉的相关问题。

<!-- 非常好的 Motivation Motivation Convolutional Neural Networks (CNN) are biologically-inspired variants of MLPs. From Hubel and Wiesel’s early work on the cat’s visual cortex [Hubel68], we know the visual cortex contains a complex arrangement of cells. These cells are sensitive to small sub-regions of the visual field, called a receptive field. The sub-regions are tiled to cover the entire visual field. These cells act as local filters over the input space and are well-suited to exploit the strong spatially local correlation present in natural images. Additionally, two basic cell types have been identified: Simple cells respond maximally to specific edge-like patterns within their receptive field. Complex cells have larger receptive fields and are locally invariant to the exact position of the pattern. The animal visual cortex being the most powerful visual processing system in existence, it seems natural to emulate its behavior. Hence, many neurally-inspired models can be found in the literature. To name a few: the NeoCognitron [Fukushima], HMAX [Serre07] and LeNet-5 [LeCun98], which will be the focus of this tutorial. -->

图像数据集的特点，对于神经网络的设计提出了一些新的挑战。

#### 维度比较高

因为图像的维度普遍比较高，例如 MNIST 数据集，每一个图片是 28 * 28 的图片。

如果直接用神经网络，假设采用2个 1000个神经元的隐藏层加 1 个10个神经元的隐藏层，最后使用 softmax 分类层，输出 10 个数字对应的概率。

参数的数量有：

786 * 1000 * 1000 * 10

如果是更大一点的图片，网络的规模还会进一步快速的增长。为了应对这种问题，
Yann LeCun 在贝尔实验室做研究员的时候提出了卷积网络技术，并展示如何使用它来大幅度提高手写识别能力。接下来将介绍卷积和池化以及卷积神经网络。

### 卷积介绍

我们尝试用一个简单的神经网络，来探讨如何解决这个问题。假设有4个输入节点和4个隐藏层节点的神经网络，如图所示：

<img class="wp-image-1324 aligncenter" src="http://www.tensorflownews.com/wp-content/uploads/2018/03/5-1.png" alt="" width="312" height="243" />
<p style="text-align: center;">图1  全连接神经网络</p>
&nbsp;

&nbsp;

每一个输入节点都要和隐藏层的 4 个节点连接，每一个连接需要一个权重参数 w：

<img class="wp-image-1326 aligncenter" src="http://www.tensorflownews.com/wp-content/uploads/2018/03/5-3.png" alt="" width="344" height="263" />
<p style="text-align: center;">图2 一个输入节点向下一层传播</p>
一共有 4 个输入节点，，所以一共需要 4*4=16个参数。

相应的每一个隐藏层节点，都会接收所有输入层节点：

<img class="wp-image-1328 aligncenter" src="http://www.tensorflownews.com/wp-content/uploads/2018/03/5-4-1.png" alt="" width="363" height="279" />
<p style="text-align: center;">图3 每个隐藏层节点接收所有输入层节点输入</p>
这是一个简化版的模型，例如手写数据集 MNIST 28 * 28 的图片，输入节点有 784 个，假如也只要一个隐藏层有 784 个节点，那么参数的个数都会是：784 * 784=614656，很明显参数的个数随着输入维度指数级增长。

因为神经网络中的参数过多，会造成训练中的困难，所以降低神经网络中参数的规模，是图像处理问题中的一个重要问题。

有两个思路可以进行尝试：

1.隐藏层的节点并不需要连接所有输入层节点，而只需要连接部分输入层。

如图所示：

&nbsp;

<img class="wp-image-1338 aligncenter" src="http://www.tensorflownews.com/wp-content/uploads/2018/03/140C7C92945B23BD7C7DFFA5AED16818.png" alt="" width="264" height="214" />
<p style="text-align: center;">图4 改为局部连接之后的网络结构</p>
每个隐藏层节点，只接受两个输入层节点的输入，那么，这个网络只需要 3 * 2 =6个连接。使用局部连接之后，单个输出层节点虽然没有连接到所有的隐藏层节点，但是隐藏层汇总之后所有的输出节点的值都对网络有影响。

2.局部连接的权重参数，如果可以共享，那么网络中参数规模又会明显的下降。如果把局部连接的权重参数当做是一个特征提取器的话，可以尝试将这个特征提取器用在其他的地方。

那么这个网络最后只需要 2 个参数，就可以完成输入层节点和隐藏层节点的连接。

这两个思路就是卷积神经网络中的稀疏交互和权值共享，下一篇文章将会详细讲解以及使用 TensorFlow 实现。