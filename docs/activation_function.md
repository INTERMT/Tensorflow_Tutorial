<h2>欢迎关注我们的网站：<a href="http://www.tensorflownews.com">http://www.tensorflownews.com</a><a href="http://www.tensorflownews.com">/</a>，学习更多的机器学习、深度学习的知识！</h2>
之前我们介绍了Sigmoid函数能够将输入的数据转换到0和1之间，其实Sigmoid函数本质上是一种常用的激活函数，是神经元最重要的组成部分。那什么是激活函数呢？激活函数有什么作用呢？常见的激活函数都有哪些？以及如何选择合适的激活函数？本节我们将重点对上述问题进行讨论。

线性模型在处理非线性问题时往往手足无措，这时我们需要引入激活函数来解决线性不可分问题。激活函数（Activation function），又名激励函数，往往存在于神经网络的输入层和输出层之间，作用是给神经网络中增加一些非线性因素，使得神经网络能够解决更加复杂的问题，同时也增强了神经网络的表达能力和学习能力。

常用的激活函数有Sigmoid函数、双曲正切激活函数（tanh）、修正线性单元（ReLU）等。接下来我们将一一学习。
<h3><a name="_Toc507703042"></a><strong><b>3</b></strong><strong><b>.4.1 </b></strong><strong><b> Sigmoid</b></strong><strong><b>函数</b></strong></h3>
Sigmoid函数是神经网络中最常用到的激活函数，数学表达式为：

<img class="size-full wp-image-758 alignnone" src="http://www.buluo360.com/wp-content/uploads/2018/03/图片4-1.png" alt="" width="90" height="51" />

函数图像如下图3-8所示。

<img class="size-full wp-image-757 aligncenter" src="http://www.buluo360.com/wp-content/uploads/2018/03/图片3-1.png" alt="" width="405" height="222" />
<p style="text-align: center;">图3-8  Logistics Sigmoid函数图像</p>
由函数图像可知，Sigmoid函数是单调增函数，输出范围在[0,1]之间，且越是负向增大，越接近于0，逼近速度越来越慢；越是正向增大，越接近于1，逼近速度也是越来越慢；因为 Sigmoid函数求导比较容易，可解释性也很强，所以在历史上被广泛的使用。

与此同时，Sigmoid函数也有两个很大的缺点：首先是Sigmoid函数会造成梯度消失问题，从图像中我们也可以得知，当输入特别大或是特别小时，神经元的梯度几乎接近于0，这就导致神经网络不收敛，模型的参数不会更新，训练过程将变得非常困难。另一方面，Sigmoid函数的输出不是以0为均值的，导致传入下一层神经网络的输入是非0的。这就导致一个后果：若Sigmoid函数的输出全部为正数，那么传入下一层神经网络的值永远大于0，这时参数无论怎么更新梯度都为正。正是基于上述的缺点，Sigmoid函数近年来的使用频率也在渐渐减弱。
<h3><a name="_Toc507703043"></a><strong><b>3</b></strong><strong><b>.4.2 </b></strong><strong><b>双曲正切激活函数（Tanh）</b></strong></h3>
Tanh函数又名双曲正切激活函数，是Sigmoid函数的变形，其数学表达式为：

<img class="alignnone size-full wp-image-759" src="http://www.buluo360.com/wp-content/uploads/2018/03/图片5-1.png" alt="" width="134" height="51" />

函数图像如图3-9所示：

<img class="size-full wp-image-760 aligncenter" src="http://www.buluo360.com/wp-content/uploads/2018/03/图片6-1.png" alt="" width="439" height="280" />
<p style="text-align: center;">图3-9  tanh函数图像</p>
由上图可知，tanh激活函数与Sigmoid函数不同的是，函数的输出范围在[-1,1]之间，且Tanh函数的输出是以为0均值的，这就一定程度上解决了上述Sigmoid函数的第二个缺点，所以其在实际应用中的效果要好于Sigmoid函数。但当输入特别大或是特别小时，仍然存在梯度消失的问题。
<h3><a name="_Toc507703041"></a><strong><b>3</b></strong><strong><b>.4.3 </b></strong><strong><b>修正线性单元ReLU</b></strong></h3>
ReLU激活函数又名修正线性单元，是目前深层神经网络中越来越受欢迎的一种激活函数，其数学表达式为：f(x) = max(0,x)，函数图像如下图所示：

<img class="alignnone size-full wp-image-761 aligncenter" src="http://www.buluo360.com/wp-content/uploads/2018/03/图片7.png" alt="" width="442" height="314" />
<p style="text-align: center;">图3-10  ReLU函数图像</p>
从ReLU的函数图像我们可以发现，函数原点左侧的部分，输出值为0，斜率为0；函数原点右侧是斜率为1的直线，且输出值就是输入值。相比于上述的Sigmoid和tanh两种激活函数，ReLU激活函数完美的解决了梯度消失的问题，因为它的线性的、非饱和的。此外，它的计算也更加简单，只需要设置一个特定的阈值就可以计算激活值，这样极大的提高了运算的速度。所以近年来，ReLU激活函数的应用越来越广泛。

但是ReLU激活函数也有一些缺陷：训练的时候不适合大梯度的输入数据，因为在参数更新之后，ReLU的神经元不会再任何数据节点被激活，这就会导致梯度永远为0。比如：输入的数据小于0时，梯度就会为0，这就导致了负的梯度被置0，而且今后也可能不会被任何数据所激活，也就是说ReLU的神经元“坏死”了。

所以针对ReLU函数上述的缺点，又出现了带泄露的ReLU（Leaky ReLU）和带参数的ReLU（Parametric ReLU）。
<h3><a name="_Toc507703044"></a><strong><b>3</b></strong><strong><b>.4.4 </b></strong><strong><b>其它激活函数</b></strong></h3>
Leaky ReLU是ReLU激活函数的变形，主要是针对ReLU函数中神经元容易坏死的缺陷，将原点左侧小于0的部分，赋予一个很小的斜率。其效果通常要好于ReLU激活函数，但实践中使用的频率并没有那么高。数据公式为：f(x) = max(0, x) + γmin(0, x)。通常，γ是一个很小的常数，如：0.01。

Parametric ReLU是ReLU激活函数的另一种变形，和Leaky ReLU函数一样是非饱和函数，解决了坏死难题。不同之处在于其在函数中引入一个可学习的参数，往往不同的神经元有不同的参数，所以第i个神经元的数学表达式为：f(x) = max(0, x) + γi min(0, x)。当γi 取0时，便可以视为ReLU函数，取很小的常数时，可视为Leaky ReLU函数。相对而言，Parametric ReLU激活函数的使用频率也不是很高。

上述两种ReLU激活函数的变形Leaky ReLU、Parametric ReLU函数图如图3-10所示：

<img class="alignnone size-full wp-image-762 aligncenter" src="http://www.buluo360.com/wp-content/uploads/2018/03/图片8.png" alt="" width="426" height="352" />
<p style="text-align: center;">图3-11 Leaky ReLU/Parametric ReLU函数图像</p>
对深度学习感兴趣，热爱Tensorflow的小伙伴，欢迎关注我们的网站！<a href="http://www.tensorflownews.com">http://www.tensorflownews.com</a>
<h2></h2>