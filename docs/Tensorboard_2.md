<p style="text-align: right;">间提壶华小厨</p>

<h2><a name="_Toc512706904"></a>1  Tensorflow监控指标可视化</h2>
除了GRAPHS栏目外，tensorboard还有IMAGES、AUDIO、SCALARS、HISTOGRAMS、DISTRIBUTIONS、FROJECTOR、TEXT、PR CURVES、PROFILE九个栏目，本小节将详细介绍这些子栏目各自的特点和用法。
<h3><a name="_Toc512706905"></a>1.1 IMAGES</h3>
图像仪表盘，可以显示通过tf.summary.image()函数来保存的png图片文件。
<code>
1.	# 指定图片的数据源为输入数据x，展示的相对位置为[-1,28,28,1]
2.	image_shape=tf.reshape(x, [-1, 28, 28,1])
3.	# 将input命名空间下的图片放到summary中，一次展示10张
4.	tf.summary.image('input', image_shape, 10)
</code>

如上面代码，将输入数据中的png图片放到summary中，准备后面写入日志文件。运行程序，生成日志文件，然后在tensorboard的IMAGES栏目下就会出现如下图一所示的内容（实验用的是mnist数据集）。仪表盘设置为每行对应不同的标签，每列对应一个运行。图像仪表盘仅支持png图片格式，可以使用它将自定义生成的可视化图像（例如matplotlib散点图）嵌入到tensorboard中。该仪表盘始终显示每个标签的最新图像。

<img class="alignnone size-full wp-image-2274" src="http://www.tensorflownews.com/wp-content/uploads/2018/05/图片1-2.png" alt="" width="1755" height="809" />
<p style="text-align: center;"><em>图一 tensorboard中的IMAGES栏目内容展开界面</em></p>

<h3><a name="_Toc512706906"></a>1.2 AUDIO</h3>
音频仪表盘，可嵌入音频的小部件，用于播放通过tf.summary.audio()函数保存的音频。

一个音频summary要存成  的二维字符张量。其中，k为summary中记录的音频被剪辑的次数，每排张量是一对[encoded_audio, label]，其中，encoded_audio 是在summary中指定其编码的二进制字符串，label是一个描述音频片段的UTF-8编码的字符串。

仪表盘设置为每行对应不同的标签，每列对应一个运行。该仪表盘始终嵌入每个标签的最新音频。
<h3><a name="_Toc512706907"></a>1.3 SCALARS</h3>
Tensorboard 的标量仪表盘，统计tensorflow中的标量（如：学习率、模型的总损失）随着迭代轮数的变化情况。如下图二所示，SCALARS栏目显示通过函数tf.summary.scalar()记录的数据的变化趋势。如下所示代码可添加到程序中，用于记录学习率的变化情况。
<code>
1.	# 在learning_rate附近添加，用于记录learning_rate
2.	tf.summary.scalar('learning_rate', learning_rate)
</code>
Scalars栏目能进行的交互操作有：
点击每个图表左下角的蓝色小图标将展开图表
拖动图表上的矩形区域将放大
双击图表将缩小
鼠标悬停在图表上会产生十字线，数据值记录在左侧的运行选择器中。

<img class="alignnone size-full wp-image-2275" src="http://www.tensorflownews.com/wp-content/uploads/2018/05/图片2-2.png" alt="" width="1770" height="855" />
<p style="text-align: center;"><em>图二 tensorboard中的SCALARS栏目内容展开界面</em></p>
此外，读者可通过在仪表盘左侧的输入框中，编写正则表达式来创建新文件夹，从而组织标签。

&nbsp;
<h3><a name="_Toc512706909"></a>1.4 HISTOGRAMS</h3>
Tensorboard的张量仪表盘，统计tensorflow中的张量随着迭代轮数的变化情况。它用于展示通过tf.summary.histogram记录的数据的变化趋势。如下代码所示：
<code>
tf.summary.histogram(weights, 'weights')
</code>

上述代码将神经网络中某一层的权重weight加入到日志文件中，运行程序生成日志后，启动tensorboard就可以在HISTOGRAMS栏目下看到对应的展开图像，如下图三所示。每个图表显示数据的时间“切片”，其中每个切片是给定步骤处张量的直方图。它依据的是最古老的时间步原理，当前最近的时间步在最前面。通过将直方图模式从“偏移”更改为“叠加”，如果是透视图就将其旋转，以便每个直方图切片都呈现为一条相互重叠的线。

<img class="alignnone size-full wp-image-2276" src="http://www.tensorflownews.com/wp-content/uploads/2018/05/图片3-1.png" alt="" width="1268" height="535" />
<p style="text-align: center;"><em>图三 tensorboard中的HISTOGRAMS栏目内容展开界面</em></p>

<h3><a name="_Toc512706908"></a>1.5 DISTRIBUTIONS</h3>
Tensorboard的张量仪表盘，相较于HISTOGRAMS，用另一种直方图展示从tf.summary.histogram()函数记录的数据的规律。它显示了一些分发的高级统计信息。

如下图四所示，图表上的每条线表示数据分布的百分位数，例如，底线显示最小值随时间的变化趋势，中间的线显示中值变化的方式。从上至下看时，各行具有以下含义：[最大值，93％，84％，69％，50％，31％，16％，7％，最小值]。这些百分位数也可以看作标准偏差的正态分布：[最大值，μ+1.5σ，μ+σ，μ+0.5σ，μ，μ-0.5σ，μ-σ，μ-1.5σ，最小值]，使得从内侧读到外侧的着色区域分别具有宽度[σ，2σ，3σ]。

<img class="alignnone size-full wp-image-2277 aligncenter" src="http://www.tensorflownews.com/wp-content/uploads/2018/05/图片4-1.png" alt="" width="1741" height="820" />

&nbsp;
<p style="text-align: center;"><em>图四 tensorboard中的DISTRIBUTIONS栏目内容展开界面</em></p>

<h3><a name="_Toc512706910"></a>1.6 PROJECTOR</h3>
嵌入式投影仪表盘，全称Embedding Projector，是一个交互式的可视化工具，通过数据可视化来分析高维数据。例如，读者可在模型运行过程中，将高维向量输入，通过embedding projector投影到3D空间，即可查看该高维向量的形式，并执行相关的校验操作。Embedding projector的建立主要分为以下几个步骤：

1）建立embedding tensor
<code>
#1. 建立 embeddings
embedding_var = tf.Variable(batch_xs, name="mnist_embedding")
summary_writer = tf.summary.FileWriter(LOG_DIR)
</code>

2）建立embedding projector 并配置
<code>
1.	config = projector.ProjectorConfig()
2.	embedding = config.embeddings.add()
3.	embedding.tensor_name = embedding_var.name
4.	embedding.metadata_path = path_for_mnist_metadata   #'metadata.tsv'
5.	embedding.sprite.image_path = path_for_mnist_sprites  #'mnistdigits.png'
6.	embedding.sprite.single_image_dim.extend([28,28])
7.	projector.visualize_embeddings(summary_writer, config)
</code>

3）将高维变量保存到日志目录下的checkpoint文件中
<code>
1.	sess = tf.InteractiveSession()
2.	sess.run(tf.global_variables_initializer())
3.	saver = tf.train.Saver()
4.	saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"), 1)
</code>

4）将metadata与embedding联系起来，将 vector 转换为 images，反转灰度，创建并保存 sprite image
<code>
1.	to_visualise = batch_xs
2.	to_visualise = vector_to_matrix_mnist(to_visualise)
3.	to_visualise = invert_grayscale(to_visualise)
4.	sprite_image = create_sprite_image(to_visualise)
5.	plt.imsave(path_for_mnist_sprites,sprite_image,cmap='gray')
</code>

5）运行程序，生成日志文件，启动服务，tensorboard中的PROJECTOR栏将展示投影后的数据的动态图，如下图五所示。

<img class="alignnone size-full wp-image-2278" src="http://www.tensorflownews.com/wp-content/uploads/2018/05/图片5-1.png" alt="" width="1256" height="648" />
<p style="text-align: center;"><em>图五 tensorboard中的PROJECTOR栏目内容展开界面</em></p>
Embedding Projector从模型运行过程中保存的checkpoint文件中读取数据，默认使用主成分分析法（PCA）将高维数据投影到3D空间中，也可以设置选择另外一种投影方法，T-SNE。除此之外，也可以使用其他元数据进行配置，如词汇文件或sprite图片。
<h3><a name="_Toc512706911"></a>1.7 TEXT</h3>
文本仪表盘，显示通过tf.summary.text()函数保存的文本片段，包括超链接、列表和表格在内的Markdown功能均支持。
<h3><a name="_Toc512706912"></a>1.8 PR CURVES</h3>
PR CURVES仪表盘显示的是随时间变化的PR曲线，其中precision为横坐标，recall为纵坐标。如下代码创建了一个用于记录PR曲线的summary。
<code>
1.	# labels为输入的y， predition为预测的y值
2.	# num_thresholds为多分类的类别数量
3.	tensorboard.summary.pr_curve(name='foo',
4.	                     predictions=predictions,
5.	                     labels=labels,
6.	                     num_thresholds=11)
</code><img class="alignnone size-full wp-image-2279" src="http://www.tensorflownews.com/wp-content/uploads/2018/05/图片6-1.png" alt="" width="1217" height="641" />

<em>图六 tensorboard中的PR CURVES栏目内容展开界面</em>

<em>(</em><em>图片来自tensorboard官方的github项目，链接为：</em>

<em>https://github.com/tensorflow/tensorboard/blob/master/tensorboard/plugins/pr_curve/images/pr_curves_intro.png)</em>

上图六为tensorboard上PR CURVES栏目在有内容时的首页，没有内容时就隐藏在INACTIVE栏目下。

训练模型时，经常需要在查准率和查全率之间权衡，PR曲线能够帮助我们找到这个权衡点。每条曲线都对应一个二分类问题，所以，针对多分类问题，每一个类都会生成一条对应的PR曲线。
<h3><a name="_Toc512706913"></a>1.9 PROFILE</h3>
Tensorboard的配置文件仪表盘，该仪表盘上包含了一套TPU工具，可以帮助我们了解，调试，优化tensorflow程序，使其在TPU上更好的运行。

但并不是所有人都可以使用该仪表盘，只有在Google Cloud TPU上有访问权限的人才能使用配置文件仪表盘上的工具。而且，该仪表盘与其他仪表盘一样，都需要在模型运行时捕获相关变量的跟踪信息，存入日志，方可用于展示。

在PROFILE仪表盘的首页上，显示的是程序在TPU上运行的工作负载性能，它主要分为五个部分：Performance Summary、Step-time Graph、Top 10 Tensorflow operations executed on TPU、Run Environment和Recommendation for Next Step。如下图七所示：

<img class="alignnone size-full wp-image-2280 aligncenter" src="http://www.tensorflownews.com/wp-content/uploads/2018/05/图片7-1.png" alt="" width="1483" height="845" />

<em>图七 tensorboard中的PROFILE栏目内容展开界面</em>

<em>（图片来自tensorboard的github项目</em>

<em><a href="https://github.com/tensorflow/tensorboard/blob/master/tensorboard/plugins/profile/docs/overview-page.png">https://github.com/tensorflow/tensorboard/blob/master/tensorboard/plugins/profile/docs/overview-page.png</a></em><em>）</em>

其中，Performance Summary包括以下四项：

1）所有采样步骤的平均步长时间

2）主机空闲时间百分比

3）TPU空闲时间百分比

4）TPU矩阵单元的利用率

Run Environment（运行环境）包括以下五方面：

1）使用的主机数量

2）使用的TPU类型

3）TPU内核的数量

4）训练批次的大小（batch size）

5）作业信息（构建命令和运行命令）

&nbsp;
<h2><a name="_Toc512706914"></a>2. 总结</h2>
本系列主要介绍了tensorflow中一个非常重要的工具——tensorboard。Tensorboard是一个可视化工具，它能够以直方图、折线图等形式展示程序运行过程中各标量、张量随迭代轮数的变化趋势，它也可以显示高维度的向量、文本、图片和音频等形式的输入数据，用于对输入数据的校验。Tensorflow函数与tensorboard栏目的对应关系如表1所示。

Tensorboard的可视化功能对于tensorflow程序的训练非常重要，使用tensorboard进行调参主要分为以下几步：

1）校验输入数据

如果输入数据的格式是图片、音频、文本的话，可以校验一下格式是否正确。如果是处理好的低维向量的话，就不需要通过tensorboard校验。

2）查看graph结构

查看各个节点之间的数据流关系是否正确，再查看各个节点所消耗的时间和空间，分析程序优化的瓶颈。

3）查看各变量的变化趋势

在SCALAR、HISTOGRAMS、DISTRIBUTIONS等栏目下查看accuracy、weights、biases等变量的变化趋势，分析模型的性能

4）修改code

根据3）和4）的分析结果，优化代码。

5）选择最优模型

6）用Embedding Projector进一步查看error出处

Tensorboard虽然只是tensorflow的一个附加工具，但熟练掌握tensorboard的使用，对每一个需要对tensorflow程序调优的人都非常重要，它可以显著提高调参工作的效率，帮助我们更快速地找到最优模型。

&nbsp;
<p style="text-align: center;">表1 tensorflow函数与tensorboard栏目的对照表</p>

<table class=" aligncenter" width="553">
<tbody>
<tr>
<td width="184">Tensorboard栏目</td>
<td width="184">tensorflow日志生成函数</td>
<td width="184">内容</td>
</tr>
<tr>
<td width="184">GRAPHS</td>
<td width="184">默认保存</td>
<td width="184">显示tensorflow计算图</td>
</tr>
<tr>
<td width="184">SCALARS</td>
<td width="184">tf.summary.scalar</td>
<td width="184">显示tensorflow中的张量随迭代轮数的变化趋势</td>
</tr>
<tr>
<td width="184">DISTRIBUTIONS</td>
<td width="184">tf.summary.histogram</td>
<td width="184">显示tensorflow中张量的直方图</td>
</tr>
<tr>
<td width="184">HISTOGRAMS</td>
<td width="184">tf.summary.histogram</td>
<td width="184">显示tensorflow中张量的直方图（以另一种方式）</td>
</tr>
<tr>
<td width="184">IMAGES</td>
<td width="184">tf.summary.image</td>
<td width="184">显示tensorflow中使用的图片</td>
</tr>
<tr>
<td width="184">AUDIO</td>
<td width="184">tf.summary.audio</td>
<td width="184">显示tensorflow中使用的音频</td>
</tr>
<tr>
<td width="184">TEXT</td>
<td width="184">tf.summary.text</td>
<td width="184">显示tensor flow中使用的文本</td>
</tr>
<tr>
<td width="184">PROJECTOR</td>
<td width="184"></td>
<td width="184">通过读取checkpoint文件可视化高维数据</td>
</tr>
</tbody>
</table>
<p style="text-align: center;">
</p>
<p style="text-align: center;"></p>
<p style="text-align: center;"></p>
<p style="text-align: center;"></p>
<p style="text-align: center;"></p>