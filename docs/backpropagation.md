反向传播算法（Backpropagation Algorithm，简称BP算法）是深度学习的重要思想基础，对于初学者来说也是必须要掌握的基础知识！本文希望以一个清晰的脉络和详细的说明，来让读者彻底明白BP算法的原理和计算过程。

全文分为上下两篇，上篇主要介绍BP算法的原理（即公式的推导），介绍完原理之后，我们会将一些具体的数据带入一个简单的三层神经网络中，去完整的体验一遍BP算法的计算过程；下篇是一个项目实战，我们将带着读者一起亲手实现一个BP神经网络（不适用任何第三方的深度学习框架）来解决一个具体的问题。

读者在学习的过程中，有任何的疑问，欢迎加入我们的交流群（扫描文章最后的二维码即可加入），和大家一起讨论！

<strong>1.BP</strong><strong>算法的推导</strong>

<img class="alignnone size-full wp-image-1410 aligncenter" src="http://www.tensorflownews.com/wp-content/uploads/2018/03/图片16.png" alt="" width="857" height="477" />
<p style="text-align: center;">图1 一个简单的三层神经网络</p>
图1所示是一个简单的三层（两个隐藏层，一个输出层）神经网络结构，假设我们使用这个神经网络来解决二分类问题，我们给这个网络一个输入样本 ，通过前向运算得到输出 。输出值 的值域为 ，例如 的值越接近0，代表该样本是“0”类的可能性越大，反之是“1”类的可能性大。

<strong>1.1</strong><strong>前向传播的计算</strong>

为了便于理解后续的内容，我们需要先搞清楚前向传播的计算过程，以图1所示的内容为例：

输入的样本为：

<img class="alignnone size-full wp-image-1422 aligncenter" src="http://www.tensorflownews.com/wp-content/uploads/2018/03/图片28.png" alt="" width="246" height="57" />

第一层网络的参数为：

<img class="alignnone size-full wp-image-1423 aligncenter" src="http://www.tensorflownews.com/wp-content/uploads/2018/03/图片29.png" alt="" width="433" height="94" />

第二层网络的参数为：

<img class="alignnone size-full wp-image-1424 aligncenter" src="http://www.tensorflownews.com/wp-content/uploads/2018/03/图片30.png" alt="" width="438" height="66" />

第三层网络的参数为：

<img class="alignnone size-full wp-image-1425 aligncenter" src="http://www.tensorflownews.com/wp-content/uploads/2018/03/图片31.png" alt="" width="364" height="50" />

<strong>1.1.1</strong><strong>第一层隐藏层的计算</strong>

<img class="alignnone size-full wp-image-1411 aligncenter" src="http://www.tensorflownews.com/wp-content/uploads/2018/03/图片17.png" alt="" width="756" height="411" />
<p style="text-align: center;">图2 计算第一层隐藏层</p>
<img class="alignnone  wp-image-1480" src="http://www.tensorflownews.com/wp-content/uploads/2018/03/25.png" alt="" width="840" height="319" />

<img class="alignnone wp-image-1451" src="http://www.tensorflownews.com/wp-content/uploads/2018/03/3-1.png" alt="" width="817" height="123" />

<strong>1.1.2</strong><strong>第二层隐藏层的计算</strong>

<img class="size-full wp-image-1412 aligncenter" src="http://www.tensorflownews.com/wp-content/uploads/2018/03/图片18.png" alt="" width="755" height="413" />
<p style="text-align: center;">图3 计算第二层隐藏层</p>
<img class="alignnone wp-image-1452" src="http://www.tensorflownews.com/wp-content/uploads/2018/03/4.png" alt="" width="1060" height="384" />

<strong>1.1.3</strong><strong>输出层的计算</strong>

<img class="alignnone size-full wp-image-1413 aligncenter" src="http://www.tensorflownews.com/wp-content/uploads/2018/03/图片19.png" alt="" width="753" height="416" />
<p style="text-align: center;">图4 计算输出层</p>
<img class="alignnone size-full wp-image-1453" src="http://www.tensorflownews.com/wp-content/uploads/2018/03/5-5.png" alt="" width="923" height="43" />

<img class="alignnone size-full wp-image-1432 aligncenter" src="http://www.tensorflownews.com/wp-content/uploads/2018/03/图片38.png" alt="" width="317" height="43" />

即：

<img class="alignnone size-full wp-image-1433 aligncenter" src="http://www.tensorflownews.com/wp-content/uploads/2018/03/图片39.png" alt="" width="326" height="39" />

<img class="alignnone size-full wp-image-1454" src="http://www.tensorflownews.com/wp-content/uploads/2018/03/6-1.png" alt="" width="981" height="95" />

<img class="alignnone size-full wp-image-1455" src="http://www.tensorflownews.com/wp-content/uploads/2018/03/7.png" alt="" width="998" height="232" />

<img class="alignnone size-full wp-image-1414 aligncenter" src="http://www.tensorflownews.com/wp-content/uploads/2018/03/图片20.png" alt="" width="300" height="232" />

<img class="alignnone size-full wp-image-1456" src="http://www.tensorflownews.com/wp-content/uploads/2018/03/8.png" alt="" width="1080" height="358" />

<img class="alignnone size-full wp-image-1458" src="http://www.tensorflownews.com/wp-content/uploads/2018/03/9.png" alt="" width="982" height="461" />

<img class="alignnone size-full wp-image-1461" src="http://www.tensorflownews.com/wp-content/uploads/2018/03/11.png" alt="" width="981" height="453" />

<img class="alignnone size-full wp-image-1464" src="http://www.tensorflownews.com/wp-content/uploads/2018/03/12.png" alt="" width="982" height="304" />

<img class="alignnone size-full wp-image-1465" src="http://www.tensorflownews.com/wp-content/uploads/2018/03/13.png" alt="" width="973" height="177" />

<strong>单纯的公式推导看起来有些枯燥，下面我们将实际的数据带入图1所示的神经网络中，完整的计算一遍。</strong>

<strong>2.</strong><strong>图解BP算法</strong>

<img class="alignnone size-full wp-image-1415 aligncenter" src="http://www.tensorflownews.com/wp-content/uploads/2018/03/图片21.png" alt="" width="857" height="477" />
<p style="text-align: center;">图5 图解BP算法</p>
我们依然使用如图5所示的简单的神经网络，其中所有参数的初始值如下：

输入的样本为（假设其真实类标为“1”）：

<img class="alignnone size-full wp-image-1466 aligncenter" src="http://www.tensorflownews.com/wp-content/uploads/2018/03/14.png" alt="" width="244" height="58" />

第一层网络的参数为：

<img class="alignnone size-full wp-image-1469 aligncenter" src="http://www.tensorflownews.com/wp-content/uploads/2018/03/17.png" alt="" width="681" height="104" />

第二层网络的参数为：

<img class="alignnone size-full wp-image-1468 aligncenter" src="http://www.tensorflownews.com/wp-content/uploads/2018/03/16.png" alt="" width="703" height="84" />

第三层网络的参数为：

<img class="alignnone size-full wp-image-1470 aligncenter" src="http://www.tensorflownews.com/wp-content/uploads/2018/03/18.png" alt="" width="537" height="67" />

<img class="alignnone size-full wp-image-1472" src="http://www.tensorflownews.com/wp-content/uploads/2018/03/19-1.png" alt="" width="764" height="162" />

<strong>2.1</strong><strong>前向传播</strong>

我们首先初始化神经网络的参数，计算第一层神经元：

<img class="alignnone size-full wp-image-1416 aligncenter" src="http://www.tensorflownews.com/wp-content/uploads/2018/03/图片22.png" alt="" width="736" height="412" />

<img class="alignnone size-full wp-image-1448 aligncenter" src="http://www.tensorflownews.com/wp-content/uploads/2018/03/图片54.png" alt="" width="378" height="163" />

<img class="alignnone size-full wp-image-1473" src="http://www.tensorflownews.com/wp-content/uploads/2018/03/20.png" alt="" width="785" height="80" />

<img class="alignnone size-full wp-image-1417 aligncenter" src="http://www.tensorflownews.com/wp-content/uploads/2018/03/图片23.png" alt="" width="583" height="325" />

<img class="alignnone wp-image-1444 aligncenter" src="http://www.tensorflownews.com/wp-content/uploads/2018/03/图片50.png" alt="" width="624" height="357" />

<img class="alignnone size-full wp-image-1474" src="http://www.tensorflownews.com/wp-content/uploads/2018/03/21.png" alt="" width="737" height="37" />

<img class="alignnone size-full wp-image-1418 aligncenter" src="http://www.tensorflownews.com/wp-content/uploads/2018/03/图片24.png" alt="" width="736" height="411" />

<img class="alignnone wp-image-1475" src="http://www.tensorflownews.com/wp-content/uploads/2018/03/22.png" alt="" width="966" height="266" />

<img class="alignnone size-full wp-image-1419 aligncenter" src="http://www.tensorflownews.com/wp-content/uploads/2018/03/图片25.png" alt="" width="734" height="412" />

<img class="alignnone wp-image-1476" src="http://www.tensorflownews.com/wp-content/uploads/2018/03/23.png" alt="" width="1026" height="278" />

<img class="alignnone size-full wp-image-1420 aligncenter" src="http://www.tensorflownews.com/wp-content/uploads/2018/03/图片26.png" alt="" width="744" height="414" />

<img class="alignnone size-full wp-image-1442 aligncenter" src="http://www.tensorflownews.com/wp-content/uploads/2018/03/图片48.png" alt="" width="646" height="164" />

<strong>2.2</strong><strong>误差反向传播</strong>

<img class="alignnone size-full wp-image-1421 aligncenter" src="http://www.tensorflownews.com/wp-content/uploads/2018/03/图片27.png" alt="" width="737" height="483" />

&nbsp;

&nbsp;

<img class="alignnone size-full wp-image-1478" src="http://www.tensorflownews.com/wp-content/uploads/2018/03/24.png" alt="" width="802" height="75" />

<img class="alignnone size-full wp-image-1438 aligncenter" src="http://www.tensorflownews.com/wp-content/uploads/2018/03/图片44.png" alt="" width="588" height="128" />

接着计算第二层隐藏层的误差项，根据误差项的计算公式有：

<img class="alignnone size-full wp-image-1437 aligncenter" src="http://www.tensorflownews.com/wp-content/uploads/2018/03/图片43.png" alt="" width="585" height="205" /> 最后是计算第一层隐藏层的误差项：

<img class="alignnone size-full wp-image-1436 aligncenter" src="http://www.tensorflownews.com/wp-content/uploads/2018/03/图片42.png" alt="" width="604" height="337" />

&nbsp;

<strong>2.3</strong><strong>更新参数</strong>

上一小节中我们已经计算出了每一层的误差项，现在我们要利用每一层的误差项和梯度来更新每一层的参数，权重W和偏置b的更新公式如下：

<img class="alignnone size-full wp-image-1443 aligncenter" src="http://www.tensorflownews.com/wp-content/uploads/2018/03/图片49.png" alt="" width="374" height="87" />

通常权重W的更新会加上一个正则化项来避免过拟合，这里为了简化计算，我们省去了正则化项。上式中的 是学习率，我们设其值为0.1。参数更新的计算相对简单，每一层的计算方式都相同，因此本文仅演示第一层隐藏层的参数更新：

<img class="alignnone size-full wp-image-1435 aligncenter" src="http://www.tensorflownews.com/wp-content/uploads/2018/03/图片41.png" alt="" width="664" height="582" />

<strong>3.</strong><strong>小结</strong>

至此，我们已经完整介绍了BP算法的原理，并使用具体的数值做了计算。在下篇中，我们将带着读者一起亲手实现一个BP神经网络（不适用任何第三方的深度学习框架），敬请期待！有任何疑问，欢迎加入我们一起交流！