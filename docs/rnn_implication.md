【前言】：在前面的内容里，我们已经学习了循环神经网络的基本结构和运算过程，这一小节里，我们将用TensorFlow实现简单的RNN，并且用来解决时序数据的预测问题，看一看RNN究竟能达到什么样的效果，具体又是如何实现的。

在这个演示项目里，我们使用随机生成的方式生成一个数据集（由0和1组成的二进制序列），然后人为的增加一些数据间的关系。最后我们把这个数据集放进RNN里，让RNN去学习其中的关系，实现二进制序列的预测<sup>1</sup>。数据生成的方式如下：

循环生成规模为五十万的数据集，每次产生的数据为0或1的概率均为0.5。如果连续生成了两个1（或两个0）的话，则下一个数据强制为0（或1）。

&nbsp;

1.我们首先导入需要的Python模块：
<code>
1	#!/usr/bin/python
2	# -*- coding: UTF-8 -*-
3	import numpy as np
4	import tensorflow as tf
5	import matplotlib.pyplot as plt
6	from tensorflow.contrib import rnn</code>

2. 定义一个Data类，用来产生数据：
<code>
1	class Data:
2	    def __init__(self, data_size, num_batch, batch_size, time_step):
3	        self.data_size = data_size      # 数据集的大小
4	        self.batch_size = batch_size    # 一个batch的大小
5	        self.num_batch = num_batch   # batch的数目（num_batch=data_size//batch_size）
6	        self.time_step = time_step     # RNN的时间步
7	        self.data_without_rel = []      # 保存随机生成的数据，数据间没有联系
8	        self.data_with_rel = []         # 保存有时序关系的数据
</code>
3. 在构造方法“__init__”中，我们初始化了数据集的大小“data_size”、一个batch的大小“batch_size”、一个epoch中的batch数目“num_batch”以及RNN的时间步“time_step”。接下来我们定义一个“generate_data”方法：
<code>
9	def generate_data(self):
10	    # 随机生成数据
11	    self.data_without_rel = np.array(np.random.choice(2, size=(self.data_size,)))
12
13	    for i in range(self.data_size):
14	        if self.data_without_rel[i-1] == 1 and self.data_without_rel[i-2] == 1:
15	            # 之前连续出现两个1，当前数据设为0
16	            self.data_with_rel.append(0)
17	            continue
18	        elif self.data_without_rel[i-1] == 0 and self.data_without_rel[i-2] == 0:
19	            # 之前连续出现两个0，当前数据设为1
20	            self.data_with_rel.append(1)
21	            continue
22	        # np.random.rand()产生的随机数范围：[0,1]
23	        else:
24	            if np.random.rand() &gt;= 0.5:
25	                self.data_with_rel.append(1)
26	            else:
27	                self.data_with_rel.append(0)
28	    return self.data_without_rel, self.data_with_rel
</code>

在第11行代码中，我们用了 “np.random.choice”函数生成的由0和1组成的长串数据。接下来我们用了一个for循环，在“data_without_rel”保存的数据的基础上重新生成了一组数据，并保存在“data_with_rel”数组中。为了使生成的数据间具有一定的序列关系，我们使用了前面介绍的很简单的数据生成方式：以“data_without_rel”中的数据为参照，如果出现了连续两个1（或0）则生成一个0（或1），其它情况则以相等概率随机生成0或1。

有了数据我们接下来要用RNN去学习这些数据，看看它能不能学习到我们产生这些数据时使用的策略，即数据间的联系。评判RNN是否学习到规律以及学习的效果如何的依据，是我们在第三章里介绍过的交叉熵损失函数。根据我们生成数据的规则，如果RNN没有学习到规则，那么它预测正确的概率就是0.5，否则它预测正确的概率为：（在“data_without_rel”中，连续出现的两个数字的组合为：00、01、10和11。00和11出现的总概率占0.5，在这种情况下，如果RNN学习到了规律，那么一定能预测出下一个数字，00对应1，11对应0。而如果出现的是01或10的话，RNN预测正确的概率就只有0.5，所以综合起来就是0.75）。

根据交叉熵损失函数，在没有学习到规律的时候，其交叉熵损失为：

loss = - (0.5 * np.log(0.5) + 0.5 * np.log(0.5)) = 0.6931471805599453

在学习到规律的时候，其交叉熵损失为：

Loss = -0.5*(0.5 * np.log(0.5) + np.log(0.5))=-0.25 * (1 * np.log(1) ) - 0.25 * (1 * np.log(1))=0.34657359027997264

&nbsp;

4.我们定义“generate_epochs”方法处理生成的数据：
<code>
29	def generate_epochs(self):
30	    # 生成数据
31	    self.generate_data()
32
33	    data_x = np.zeros([self.num_batch, self.batch_size], dtype=np.int32)
34	    data_y = np.zeros([self.num_batch, self.batch_size], dtype=np.int32)
35
36	    # 将数据划分成num_batch组
37	    for i in range(self.num_batch):
38	        data_x[i] = self.data_without_rel[self.batch_size * i:self.batch_size * (i + 1)]
39	        data_y[i] = self.data_with_rel[self.batch_size * i:self.batch_size * (i + 1)]
40	    # 将每个batch的数据按time_step进行切分
41	    epoch_size = self.batch_size // self.time_step
42
43	    # 返回最终的数据
44	    for i in range(epoch_size):
45	        x = data_x[:, self.time_step * i:self.time_step * (i + 1)]
46	        y = data_y[:, self.time_step * i:self.time_step * (i + 1)]
47	        yield (x, y)</code>

5.接下来实现RNN部分：
<code>
48	class Model:
49	    def __init__(self, data_size, batch_size, time_step, state_size):
50	        self.data_size = data_size
51	        self.batch_size = batch_size
52	        self.num_batch = self.data_size // self.batch_size
53	        self.time_step = time_step
54	        self.state_size = state_size
55
56	    # 输入数据的占位符
57	    self.x = tf.placeholder(tf.int32, [self.num_batch, self.time_step], name='input_placeholder')
58	    self.y = tf.placeholder(tf.int32, [self.num_batch, self.time_step], name='labels_placeholder')
59
60	    # 记忆单元的占位符
61	    self.init_state = tf.zeros([self.num_batch, self.state_size])
62	    # 将输入数据进行one-hot编码
63	    self.rnn_inputs = tf.one_hot(self.x, 2)
64
65	    # 隐藏层的权重矩阵和偏置项
66	    self.W = tf.get_variable('W', [self.state_size, 2])
67	    self.b = tf.get_variable('b', [2], initializer=tf.constant_initializer(0.0))
68
69	    # RNN隐藏层的输出
70	    self.rnn_outputs, self.final_state = self.model()
71
72	    # 计算输出层的输出
73	    logits = tf.reshape( tf.matmul(tf.reshape(self.rnn_outputs, [-1, self.state_size]), self.W) + self.b, [self.num_batch, self.time_step, 2])
74
75	    self.losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y, logits=logits)
76	    self.total_loss = tf.reduce_mean(self.losses)
77	    self.train_step = tf.train.AdagradOptimizer(0.1).minimize(self.total_loss)
</code>

6.定义RNN模型：
<code>
78	    def model(self):
79	        cell = rnn.BasicRNNCell(self.state_size)
80	        rnn_outputs, final_state = tf.nn.dynamic_rnn(cell, self.rnn_inputs,
initial_state=self.init_state)
81	        return rnn_outputs, final_state
</code>

这里我们使用了“dynamic_rnn”，因此每次会同时处理所有batch的第一组数据，总共处理的次数为：batch_size / time_step。
<code>
82	    def train(self):
83	        with tf.Session() as sess:
84	            sess.run(tf.global_variables_initializer())
85	            training_losses = []
86	            d = Data(self.data_size, self.num_batch, self.batch_size, self.time_step)
87	            training_loss = 0
88	            training_state = np.zeros((self.num_batch, self.state_size))
89	            for step, (X, Y) in enumerate(d.generate_epoch()):
90	                tr_losses, training_loss_, training_state, _ = \
sess.run([self.losses, self.total_loss, self.final_state, self.train_step],
feed_dict={self.x: X, self.y: Y, self.init_state: training_state})
91	                training_loss += training_loss_
92	                if step % 20 == 0 and step &gt; 0:
93	                    training_losses.append(training_loss/20)
94	                    training_loss = 0
95	        return training_losses</code>

7.到这里，我们已经实现了整个RNN模型，接下来初始化相关数据，看看RNN的学习效果如何：
<code>
96	if __name__ == '__main__':
97	    data_size = 500000
98	    batch_size = 2000
99	    time_step = 5
100	    state_size = 6
101
102	    m = Model(data_size, batch_size, time_step, state_size)
103	    training_losses = m.train()
104	    plt.plot(training_losses)
105	    plt.show()
</code>
定义数据集的大小为500000，每个batch的大小为2000，RNN的“时间步”设为5，隐藏层的神经元数目为6。将训练过程中的loss可视化，结果如下图中的左侧图像所示：

<img class="alignnone size-full wp-image-3935" src="http://www.tensorflownews.com/wp-content/uploads/2018/11/图片1（左边）.png" alt="" width="666" height="499" /><img class="alignnone size-full wp-image-3934" src="http://www.tensorflownews.com/wp-content/uploads/2018/11/图片1（右边）.png" alt="" width="674" height="505" />
<p style="text-align: center;"><em>图1 二进制序列数据训练的loss曲线</em></p>
从左侧loss曲线可以看到，loss最终稳定在了0.35左右，这与我们之前的计算结果一致，说明RNN学习到了序列数据中的规则。右侧的loss曲线是在调整了序列关系的时间间隔后（此时的time_step过小，导致RNN无法学习到序列数据的规则）的结果，此时loss稳定在0.69左右，与之前的计算也吻合。

&nbsp;

下一篇，我们将介绍几种常见的RNN循环神经网络结构以及部分代码示例。

&nbsp;

&nbsp;

&nbsp;