<h3>前边的章节介绍了什么是Tensorflow，本节将带大家真正走进Tensorflow的世界，学习Tensorflow一些基本的操作及使用方法。同时也欢迎大家关注我们的网站和系列教程：<a href="http://www.tensorflownews.com/">http://www.tensorflownews.com</a><a href="http://www.tensorflownews.com/">/</a>，学习更多的机器学习、深度学习的知识！</h3>
Tensorflow是一种计算图模型，即用图的形式来表示运算过程的一种模型。Tensorflow程序一般分为图的构建和图的执行两个阶段。图的构建阶段也称为图的定义阶段，该过程会在图模型中定义所需的运算，每次运算的的结果以及原始的输入数据都可称为一个节点（operation ，缩写为op）。我们通过以下程序来说明图的构建过程：

程序2-1：

<img class="alignnone size-full wp-image-1620 aligncenter" src="http://www.tensorflownews.com/wp-content/uploads/2018/03/图片57-1.png" alt="" width="623" height="224" />

程序2-1定义了图的构建过程，“import tensorflow as tf”，是在python中导入tensorflow模块,并另起名为“tf”；接着定义了两个常量op，m1和m2，均为1*2的矩阵；最后将m1和m2的值作为输入创建一个矩阵加法op，并输出最后的结果result。

我们分析最终的输出结果可知，其并没有输出矩阵相加的结果，而是输出了一个包含三个属性的Tensor(Tensor的概念我们会在下一节中详细讲解，这里就不再赘述)。

以上过程便是图模型的构建阶段：只在图中定义所需要的运算，而没有去执行运算。我们可以用图2-1来表示：

<img class="alignnone size-full wp-image-1617 aligncenter" src="http://www.tensorflownews.com/wp-content/uploads/2018/03/图片9.png" alt="" width="664" height="269" />
<p style="text-align: center;">图2-1 图的构建阶段</p>
第二个阶段为图的执行阶段，也就是在会话（session）中执行图模型中定义好的运算。

我们通过程序2-2来解释图的执行阶段：

程序2-2：

<img class="alignnone size-full wp-image-1623 aligncenter" src="http://www.tensorflownews.com/wp-content/uploads/2018/03/图片61-1.png" alt="" width="621" height="163" />

&nbsp;

程序2-2描述了图的执行过程，首先通过“tf.session()”启动默认图模型，再调用run()方法启动、运行图模型，传入上述参数result，执行矩阵的加法，并打印出相加的结果，最后在任务完成时，要记得调用close()方法，关闭会话。

除了上述的session写法外，我们更建议大家，把session写成如程序2-4所示“with”代码块的形式，这样就无需显示的调用close释放资源，而是自动地关闭会话。

程序2-3：

<img class="alignnone size-full wp-image-1621 aligncenter" src="http://www.tensorflownews.com/wp-content/uploads/2018/03/图片58-1.png" alt="" width="623" height="103" />

此外，我们还可以利用CPU或GPU等计算资源分布式执行图的运算过程。一般我们无需显示的指定计算资源，Tensorflow可以自动地进行识别，如果检测到我们的GPU环境，会优先的利用GPU环境执行我们的程序。但如果我们的计算机中有多于一个可用的GPU，这就需要我们手动的指派GPU去执行特定的op。如下程序2-4所示，Tensorflow中使用with...device语句来指定GPU或CPU资源执行操作。

程序2-4：

<img class="alignnone size-full wp-image-1622 aligncenter" src="http://www.tensorflownews.com/wp-content/uploads/2018/03/图片59-1.png" alt="" width="623" height="164" />

&nbsp;

上述程序中的“tf.device(“/gpu:2”)”是指定了第二个GPU资源来运行下面的op。依次类推，我们还可以通过“/gpu:3”、“/gpu:4”、“/gpu:5”...来指定第N个GPU执行操作。

关于GPU的具体使用方法，我们会在下面的章节结合案例的形式具体描述。

Tensorflow中还提供了默认会话的机制，如程序2-5所示，我们通过调用函数as_default()生成默认会话。

程序2-5：

<img class="alignnone size-full wp-image-1624 aligncenter" src="http://www.tensorflownews.com/wp-content/uploads/2018/03/图片60-1.png" alt="" width="620" height="289" />

我们可以看到程序2-5和程序2-2有相同的输出结果。我们在启动默认会话后，可以通过调用eval()函数，直接输出变量的内容。

有时，我们需要在Jupyter或IPython等python交互式环境开发。Tensorflow为了满足用户的这一需求，提供了一种专门针对交互式环境开发的方法InteractiveSession(),具体用法如程序2-6所示：

程序2-6：

<img class="size-full wp-image-1625 aligncenter" src="http://www.tensorflownews.com/wp-content/uploads/2018/03/图片62-1.png" alt="" width="621" height="256" />

程序2-6就是交互式环境中经常会使用的InteractiveSession()方法，其创建sess对象后，可以直接输出运算结果。

综上所述，我们介绍了Tensorflow的核心概念——计算图模型，以及定义图模型和运行图模型的几种方式。接下来，我们思考一个问题，为什么Tensorflow要使用图模型？图模型有什么优势呢？

首先，图模型的最大好处是节约系统开销，提高资源的利用率，可以更加高效的进行运算。因为我们在图的执行阶段，只需要运行我们需要的op,这样就大大的提高了资源的利用率；其次，这种结构有利于我们提取中间某些节点的结果，方便以后利用中间的节点去进行其它运算；还有就是这种结构对分布式运算更加友好，运算的过程可以分配给多个CPU或是GPU同时进行，提高运算效率；最后，因为图模型把运算分解成了很多个子环节，所以这种结构也让我们的求导变得更加方便。

<strong><b>2.3.2 Tensor介绍</b></strong>

Tensor（张量）是Tensorflow中最重要的数据结构，用来表示Tensorflow程序中的所有数据。Tensor本是广泛应用在物理、数学领域中的一个物理量。那么在Tensorflow中该如何理解Tensor的概念呢？

实际上，我们可以把Tensor理解成N维矩阵（N维数组）。其中零维张量表示的是一个标量，也就是一个数；一维张量表示的是一个向量，也可以看作是一个一维数组；二维张量表示的是一个矩阵；同理，N维张量也就是N维矩阵。

在计算图模型中，操作间所传递的数据都可以看做是Tensor。那Tensor的结构到底是怎样的呢？我们可以通过程序2-7更深入的了解一下Tensor。

程序2-7：

<img class="alignnone size-full wp-image-1627 aligncenter" src="http://www.tensorflownews.com/wp-content/uploads/2018/03/图片11.png" alt="" width="570" height="254" />

程序2-7的输出结果表明：构建图的运算过程输出的结果是一个Tensor，且其主要由三个属性构成：Name、Shape和Type。Name代表的是张量的名字，也是张量的唯一标识符，我们可以在每个op上添加name属性来对节点进行命名，Name的值表示的是该张量来自于第几个输出结果（编号从0开始），上例中的“mul_3:0”说明是第一个结果的输出。Shape代表的是张量的维度，上例中shape的输出结果(1,1)说明该张量result是一个二维数组，且每个维度数组的长度是1。最后一个属性表示的是张量的类型，每个张量都会有唯一的类型，常见的张量类型如图2-2所示。

<img class="alignnone size-full wp-image-1618 aligncenter" src="http://www.tensorflownews.com/wp-content/uploads/2018/03/图片10.png" alt="" width="691" height="492" />
<p style="text-align: center;">图2-2 常用的张量类型</p>
我们需要注意的是要保证参与运算的张量类型相一致，否则会出现类型不匹配的错误。如程序2-8所示，当参与运算的张量类型不同时，Tensorflow会报类型不匹配的错误：

程序2-8：

<img class="alignnone size-full wp-image-1628 aligncenter" src="http://www.tensorflownews.com/wp-content/uploads/2018/03/图片12.png" alt="" width="570" height="214" />

正如程序的报错所示：m1是int32的数据类型，而m2是float32的数据类型，两者的数据类型不匹配，所以发生了错误。所以我们在实际编程时，一定注意参与运算的张量数据类型要相同。

<strong><b>2.3.3 常量、变量及占位符</b></strong>

Tensorflow中对常量的初始化，不管是对数值、向量还是对矩阵的初始化，都是通过调用constant()函数实现的。因为constant()函数在Tensorflow中的使用非常频繁，经常被用于构建图模型中常量的定义，所以接下来，我们通过程序2-9了解一下constant()的相关属性：
程序2-9：

<img class="alignnone size-full wp-image-1629 aligncenter" src="http://www.tensorflownews.com/wp-content/uploads/2018/03/图片13.png" alt="" width="569" height="166" />

如程序2-9所示，函数constant有五个参数，分别为value，name，dtype，shape和verify_shape。其中value为必选参数，其它均为可选参数。Value为常量的具体值，可以是一个数字，一维向量或是多维矩阵。Name是常量的名字，用于区别其它常量。Dtype是常量的类型，具体类型可参见图2-2。Shape是指常量的维度，我们可以自行定义常量的维度。

verify_shape是验证shape是否正确，默认值为关闭状态(False)。也就是说当该参数true状态时，就会检测我们所写的参数shape是否与value的真实shape一致，若不一致就会报TypeError错误。如：上例中的实际shape为(2,0)，若我们将参数中的shape属性改为(2,1)，程序就会报如下错误：

TypeError: Expected Tensor's shape: (2, 1), got (2,).

Tensorflow还提供了一些常见常量的初始化，如：tf.zeros、tf.ones、tf.fill、tf.linspace、tf.range等，均可以快速初始化一些常量。例如：我们想要快速初始化N维全0的矩阵，我们可以利用tf.zeros进行初始化，如程序2-10所示：

程序2-10：

<img class="alignnone size-full wp-image-1630 aligncenter" src="http://www.tensorflownews.com/wp-content/uploads/2018/03/图片14.png" alt="" width="571" height="347" />

程序2-10向我们展示了tf.zeros和tf.zeros_like的用法。其它常见常量的具体初始化用法可以参考Tensorflow官方手册：<a href="https://www.tensorflow.org/api_guides/python/constant_op"><u>https://www.tensorflow.org/api_guides/python/constant_op</u></a>。

此外，Tensorflow还可以生成一些随机的张量，方便快速初始化一些随机值。如：tf.random_normal()、tf.truncated_normal()、tf.random_uniform()、tf.random_shuffle()等。如程序2-11所示，我们以tf.random_normal()为例，来看一下随机张量的具体用法：

程序2-11：

<img class="alignnone size-full wp-image-1631 aligncenter" src="http://www.tensorflownews.com/wp-content/uploads/2018/03/图片15.png" alt="" width="565" height="249" />

随机张量random_normal()有shape、mean、stddev、dtype、seed、name六个属性。 shape是指张量的形状，如上述程序是生成一个2行3列的tensor；mean是指正态分布的均值；stddev是指正太分布的标准差；dtype是指生成tensor的数据类型；seed是分发创建的一个随机种子；而name是给生成的随机张量命名。

Tensorflow中的其它随机张量的具体使用方法和属性介绍，可以参见Tensorflow官方手册：<a href="https://www.tensorflow.org/api_guides/python/constant_op"><u>https://www.tensorflow.org/api_guides/python/constant_op</u></a>。这里将不在一一赘述。

除了常量constant()，变量variable()也是在Tensorflow中经常会被用到的函数。变量的作用是保存和更新参数。执行图模型时，一定要对变量进行初始化，经过初始化后的变量才能拿来使用。变量的使用包括创建、初始化、保存、加载等操作。首先，我们通过程序2-12了解一下变量是如何被创建的：

程序2-12：

<img class="alignnone size-full wp-image-1632 aligncenter" src="http://www.tensorflownews.com/wp-content/uploads/2018/03/图片16-1.png" alt="" width="574" height="195" />

程序2-12展示了创建变量的多种方式。我们可以把函数variable()理解为构造函数，构造函数的使用需要初始值，而这个初始值是一个任何形状、类型的Tensor。也就是说，我们

既可以通过创建数字变量、一维向量、二维矩阵初始化Tensor，也可以使用常量或是随机常量初始化Tensor，来完成变量的创建。

当我们完成了变量的创建，接下来，我们要对变量进行初始化。变量在使用前一定要进行初始化，且变量的初始化必须在模型的其它操作运行之前完成。通常，变量的初始化有三种方式，如程序2-13所示：

程序2-13：

<img class="alignnone size-full wp-image-1633 aligncenter" src="http://www.tensorflownews.com/wp-content/uploads/2018/03/图片17-1.png" alt="" width="570" height="382" />

程序2-13说明了初始化变量的三种方式：初始化全部变量、初始化变量的子集以及初始化单个变量。首先，global_variables_initializer()方法是不管全局有多少个变量，全部进行初始化，是最简单也是最常用的一种方式；variables_initializer()是初始化变量的子集，相比于全部初始化化的方式更加节约内存；Variable()是初始化单个变量，函数的参数便是要初始化的变量内容。通过上述的三种方式，我们便可以实现变量的初始化，放心的使用变量了。

我们经常在训练模型后，希望保存训练的结果，以便下次再使用或是方便日后查看，这时就用到了Tensorflow变量的保存。变量的保存是通过tf.train.Saver()方法创建一个Saver管理器，来保存计算图模型中的所有变量。具体代码如程序2-14所示：

程序2-14：

<img class="alignnone size-full wp-image-1634 aligncenter" src="http://www.tensorflownews.com/wp-content/uploads/2018/03/图片18-1.png" alt="" width="572" height="380" />

我们要注意，我们的存储文件save.ckpt是一个二进制文件，Saver存储器提供了向该二进制文件保存变量和恢复变量的方法。保存变量的方法就是程序中的save()方法，保存的内容是从变量名到tensor值的映射关系。完成该存储操作后，会在对应目录下生成如图2-3所示的文件：

<img class="alignnone size-full wp-image-1635 aligncenter" src="http://www.tensorflownews.com/wp-content/uploads/2018/03/图片19-1.png" alt="" width="628" height="130" />
<p style="text-align: center;">图2-3 保存变量生成的相应文件</p>
Saver提供了一个内置的计数器自动为checkpoint文件编号。这就支持训练模型在任意步骤多次保存。此外，还可以通过global_step参数自行对保存文件进行编号，例如：global_step=2，则保存变量的文件夹为model.ckpt-2。

那如何才能恢复变量呢？首先，我们要知道一定要用和保存变量相同的Saver对象来恢复变量。其次，不需要事先对变量进行初始化。具体代码如程序2-15所示：
程序2-15：

<img class="size-full wp-image-1636 aligncenter" src="http://www.tensorflownews.com/wp-content/uploads/2018/03/图片20-1.png" alt="" width="575" height="259" />

本程序示例中，我们要注意：变量的获取是通过restore()方法，该方法有两个参数，分别是session和获取变量文件的位置。我们还可以通过latest_checkpoint()方法，获取到该目录下最近一次保存的模型。

以上就是对变量创建、初始化、保存、加载等操作的介绍。此外，还有一些与变量相关的重要函数，如：eval()等。

认识了常量和变量，Tensorflow中还有一个非常重要的常用函数——placeholder。placeholder是一个数据初始化的容器，它与变量最大的不同在于placeholder定义的是一个模板，这样我们就可以session运行阶段，利用feed_dict的字典结构给placeholder填充具体的内容，而无需每次都提前定义好变量的值，大大提高了代码的利用率。Placeholder的具体用法如程序2-16所示：

程序序2-16：

<img class="alignnone size-full wp-image-1637 aligncenter" src="http://www.tensorflownews.com/wp-content/uploads/2018/03/图片21-1.png" alt="" width="573" height="194" />

&nbsp;

程序2-16演示了placeholder占位符的使用过程。Placeholder()方法有dtype，shape和name三个参数构成。dtype是必填参数，代表传入value的数据类型；shape是选填参数，代表传入value的维度；name也是选填参数，代表传入value的名字。我们可以把这三个参数看作为形参，在使用时传入具体的常量值。这也是placeholder不同于常量的地方，它不可以直接拿来使用，而是需要用户传递常数值。

最后，Tensorflow中还有一个重要的概念——fetch。Fetch的含义是指可以在一个会话中同时运行多个op。这就方便我们在实际的建模过程中，输出一些中间的op，取回多个tensor。Fetch的具体用法如程序2-17所示：

程序2-17：

<img class="alignnone size-full wp-image-1638 aligncenter" src="http://www.tensorflownews.com/wp-content/uploads/2018/03/图片22-1.png" alt="" width="570" height="319" />

程序2-17展示了fetch的用法，即我们利用session的run()方法同时取回多个tensor值，方便我们查看运行过程中每一步op的输出结果。

小结：本节旨在让大家学会Tensorflow的基础知识，为后边实战的章节打下基础。主要讲了Tensorflow的核心——计算图模型，如何定义图模型和计算图模型；还介绍了Tensor的概念，以及Tensorflow中的常量、变量、占位符、feed等知识点。大家都掌握了吗？
<h3>最后，对深度学习感兴趣，热爱Tensorflow的小伙伴，欢迎关注我们的网站！<a href="http://www.tensorflownews.com/">http://www.tensorflownews.com</a></h3>