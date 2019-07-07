<p style="text-align: right;"><strong>作者：AI小昕</strong></p>
本系列教程将手把手带您从零开始学习Tensorflow，并最终通过Tensorflow实现一些经典的项目。欢迎您持续关注我们的教程，关注更多机器学习、深度学习相关的优质博文。

Tensorflow是由谷歌大脑团队于2015年11月开发的第二代开源的机器学习系统。Tensorflow支持python、C++、java、GO等多种编程语言，以及CNN、RNN和GAN等深度学习算法。Tensorflow除可以在Windows、Linux、MacOS等操作系统运行外，还支持Android和iOS移动平台的运行、以及适用于多个CPU/GPU组成的分布式系统中。

Tensorflow是目前最火的深度学习框架，广泛应用于自然语言处理、语音识别、图像处理等多个领域。不仅深受全球深度学习爱好者的广泛欢迎，Google、eBay、Uber、OPenAI等众多科技公司的研发团队也都在使用它。

相较于其它的深度学习框架，如：Caffe、Torch、Keras、MXnet、Theano等，Tensorflow的主要优势有以下几点：高度的灵活性、支持python语言开发、可视化效果好、功能更加强大、运行效率高、强大的社区。

本节将从Tensorflow的安装配置、Tensorflow的核心——计算图模型开始讲起，带大家走进Tensorflow的世界。好了，随小编一起进入正文吧。

<strong>1.Tensorflow</strong><strong>安装与配置</strong>

目前，Windows、Linux和MacOS均已支持Tensorflow。文章将以Windows系统的安装为例。

在安装Tensorflow前，我们要先安装Anaconda，因为它集成了很多Python的第三方库及其依赖项，方便我们在编程中直接调用。

Anaconda下载地址为：<a href="https://www.anaconda.com/download/。">https://www.anaconda.com/download/。</a>（分为python3.6版本和python2.7版本，本书使用的是python3.6版本。）

下载好安装包后，一步步执行安装过程，直到出现如图1-1所示的界面，完成Anaconda的安装：

<img class="alignnone size-full wp-image-1510 aligncenter" src="http://www.tensorflownews.com/wp-content/uploads/2018/03/图片56.png" alt="" width="552" height="378" />
<p style="text-align: center;">图1-1 Anaconda安装成功截图</p>
安装好Anaconda后，我们便可以打开命令提示符，输入pip install Tensorflow完成Tensorflow的安装。

之后我们进入python可执行界面，输入import tensorflow as tf来检验Tensorflow是否安装成功。如果没有报任何错，可以正常执行，则说明Tensorflow已经安装成功。

Jupyter Notebook是一款非常好用的交互式开发工具，不仅支持40多种编程语言，还可以实时运行代码、共享文档、数据可视化、支持markdown等，适用于机器学习、统计建模数据处理、特征提取等多个领域。尤其在Kaggle、天池等数据科学竞赛中，快捷、实时、方便的优点深受用户欢迎。本书后边的章节中，均将以Jupyter Notebook作为开发环境，运行Tensorflow程序。

<strong>2.</strong><strong>计算图模型</strong>

Tensorflow是一种计算图模型，即用图的形式来表示运算过程的一种模型。Tensorflow程序一般分为图的构建和图的执行两个阶段。图的构建阶段也称为图的定义阶段，该过程会在图模型中定义所需的运算，每次运算的的结果以及原始的输入数据都可称为一个节点（operation ，缩写为op）。我们通过以下程序来说明图的构建过程：

程序1：

<img class="alignnone size-full wp-image-1511 aligncenter" src="http://www.tensorflownews.com/wp-content/uploads/2018/03/图片57.png" alt="" width="623" height="224" />

程序1定义了图的构建过程，“import tensorflow as tf”，是在python中导入tensorflow模块,并另起名为“tf”；接着定义了两个常量op，m1和m2，均为1*2的矩阵；最后将m1和m2的值作为输入创建一个矩阵加法op，并输出最后的结果result。

我们分析最终的输出结果可知，其并没有输出矩阵相加的结果，而是输出了一个包含三个属性的Tensor(Tensor的概念我们会在下一节中详细讲解，这里就不再赘述)。

以上过程便是图模型的构建阶段：只在图中定义所需要的运算，而没有去执行运算。我们可以用图1-1来表示：

<img class="alignnone size-full wp-image-1509 aligncenter" src="http://www.tensorflownews.com/wp-content/uploads/2018/03/图片55.png" alt="" width="664" height="269" />
<p style="text-align: center;">图1-2 图的构建阶段</p>
第二个阶段为图的执行阶段，也就是在会话（session）中执行图模型中定义好的运算。

我们通过程序2来解释图的执行阶段：

程序2：

<img class="alignnone size-full wp-image-1515 aligncenter" src="http://www.tensorflownews.com/wp-content/uploads/2018/03/图片61.png" alt="" width="621" height="163" />

程序2描述了图的执行过程，首先通过“tf.session()”启动默认图模型，再调用run()方法启动、运行图模型，传入上述参数result，执行矩阵的加法，并打印出相加的结果，最后在任务完成时，要记得调用close()方法，关闭会话。

除了上述的session写法外，我们更建议大家，把session写成如程序3所示“with”代码块的形式，这样就无需显示的调用close释放资源，而是自动地关闭会话。

程序3：

<img class="alignnone size-full wp-image-1512 aligncenter" src="http://www.tensorflownews.com/wp-content/uploads/2018/03/图片58.png" alt="" width="623" height="103" />

此外，我们还可以利用CPU或GPU等计算资源分布式执行图的运算过程。一般我们无需显示的指定计算资源，Tensorflow可以自动地进行识别，如果检测到我们的GPU环境，会优先的利用GPU环境执行我们的程序。但如果我们的计算机中有多于一个可用的GPU，这就需要我们手动的指派GPU去执行特定的op。如下程序4所示，Tensorflow中使用with...device语句来指定GPU或CPU资源执行操作。

程序4：

<img class="alignnone size-full wp-image-1513 aligncenter" src="http://www.tensorflownews.com/wp-content/uploads/2018/03/图片59.png" alt="" width="623" height="164" />

上述程序中的“tf.device(“/gpu:2”)”是指定了第二个GPU资源来运行下面的op。依次类推，我们还可以通过“/gpu:3”、“/gpu:4”、“/gpu:5”...来指定第N个GPU执行操作。

关于GPU的具体使用方法，我们会在下面的章节结合案例的形式具体描述。

Tensorflow中还提供了默认会话的机制，如程序5所示，我们通过调用函数as_default()生成默认会话。

程序5：

<img class="alignnone size-full wp-image-1514 aligncenter" src="http://www.tensorflownews.com/wp-content/uploads/2018/03/图片60.png" alt="" width="620" height="289" />

我们可以看到程序5和程序2有相同的输出结果。我们在启动默认会话后，可以通过调用eval()函数，直接输出变量的内容。

有时，我们需要在Jupyter或IPython等python交互式环境开发。Tensorflow为了满足用户的这一需求，提供了一种专门针对交互式环境开发的方法InteractiveSession(),具体用法如程序6所示：

程序6：

<img class="size-full wp-image-1516 aligncenter" src="http://www.tensorflownews.com/wp-content/uploads/2018/03/图片62.png" alt="" width="621" height="256" />

程序6就是交互式环境中经常会使用的InteractiveSession()方法，其创建sess对象后，可以直接输出运算结果。

综上所述，我们介绍了Tensorflow的核心概念——计算图模型，以及定义图模型和运行图模型的几种方式。接下来，我们思考一个问题，为什么Tensorflow要使用图模型？图模型有什么优势呢？

首先，图模型的最大好处是节约系统开销，提高资源的利用率，可以更加高效的进行运算。因为我们在图的执行阶段，只需要运行我们需要的op,这样就大大的提高了资源的利用率；其次，这种结构有利于我们提取中间某些节点的结果，方便以后利用中间的节点去进行其它运算；还有就是这种结构对分布式运算更加友好，运算的过程可以分配给多个CPU或是GPU同时进行，提高运算效率；最后，因为图模型把运算分解成了很多个子环节，所以这种结构也让我们的求导变得更加方便。

好了，相信读到这里，大家对Tensorflow这一高深莫测的技术有了基本的了解，在接下来的内容中我们将持续为您讲解Tensorflow的变量、常量，以及如何使用Tensorflow去运行深度学习的项目等。欢迎大家关注我们的网站。

&nbsp;