<p style="text-align: right;">fendouai</p>
<strong>目录：</strong>

<strong>介绍</strong>

<strong>记录设备状态</strong>

<strong>手动分配状态</strong>

<strong>允许GPU内存增长</strong>

<strong>在多GPU系统是使用单个GPU</strong>

<strong>使用多个 GPU</strong>

<strong> </strong>

<strong> </strong>

<strong>一、介绍</strong>

在一个典型的系统中，有多个计算设备。在 TensorFlow 中支持的设备类型包括 CPU 和 GPU。他们用字符串来表达，例如：

&nbsp;
<ul>
 	<li>"/cpu:0": 机器的 CPU</li>
 	<li>"/device:GPU:0": 机器的 GPU 如果你只有一个</li>
 	<li>"/device:GPU:1": 机器的第二个 GPU</li>
</ul>
&nbsp;

如果 TensorFlow 操作同时有 CPU 和 GPU 的实现，操作将会优先分配给 GPU 设备。例如，matmul 同时有 CPU 和 GPU 核心，在一个系统中同时有设备 cpu:0 和 gpu:0，gpu:0 将会被选择来执行 matmul。

&nbsp;

<strong>二、记录设备状态</strong>

&nbsp;

为了确定你的操作和张量分配给了哪一个设备，创建一个把 log_device_placement 的配置选项设置为 True 的会话即可。

&nbsp;

&nbsp;

# 创建一个计算图

a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')

b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')

c = tf.matmul(a, b)

# 创建一个 session，它的 log_device_placement 被设置为 True.

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

# 运行这个操作

print(sess.run(c))

你将会看到一下输出:

&nbsp;

Device mapping:

/job:localhost/replica:0/task:0/device:GPU:0 -&gt; device: 0, name: Tesla K40c, pci bus

id: 0000:05:00.0

b: /job:localhost/replica:0/task:0/device:GPU:0

a: /job:localhost/replica:0/task:0/device:GPU:0

MatMul: /job:localhost/replica:0/task:0/device:GPU:0

[[ 22.  28.]

[ 49.  64.]]

&nbsp;

<strong>三、手动分配设备</strong>

&nbsp;

如果你希望一个特定的操作运行在一个你选择的设备上，而不是自动选择的设备，你可以使用 tf.device 来创建一个设备环境，这样所有在这个环境的操作会有相同的设备分配选项。

&nbsp;

# 创建一个会话

with tf.device('/cpu:0'):

a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')

b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')

c = tf.matmul(a, b)

# 创建一个 session，它的 log_device_placement 被设置为 True

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

# 运行这个操作

print(sess.run(c))

&nbsp;

你将会看到 a 和 b 被分配给了 cpu:0。因为没有指定特定的设备来执行 matmul 操作，TensorFlow 将会根据操作和已有的设备来选择(在这个例子中是 gpu:0)，并且如果有需要会自动在设备之间复制张量。

&nbsp;

Device mapping:

/job:localhost/replica:0/task:0/device:GPU:0 -&gt; device: 0, name: Tesla K40c, pci bus

id: 0000:05:00.0

b: /job:localhost/replica:0/task:0/cpu:0

a: /job:localhost/replica:0/task:0/cpu:0

MatMul: /job:localhost/replica:0/task:0/device:GPU:0

[[ 22.  28.]

[ 49.  64.]]

&nbsp;

<strong>四、允许 GPU 内存增长</strong>

&nbsp;

默认情况下，TensorFlow 将几乎所有的 GPU的显存（受 CUDA_VISIBLE_DEVICES 影响）映射到进程。 通过减少内存碎片，可以更有效地使用设备上宝贵的GPU内存资源。

&nbsp;

在某些情况下，只需要分配可用内存的一个子集给进程，或者仅根据进程需要增加内存使用量。 TensorFlow 在 Session 上提供了两个 Config 选项来控制这个选项。

&nbsp;

第一个是 allow_growth 选项，它根据运行时的需要分配 GPU 内存：它开始分配很少的内存，并且随着 Sessions 运行并需要更多的 GPU 内存，我们根据 TensorFlow 进程需要继续扩展了GPU所需的内存区域。请注意，我们不释放内存，因为这会导致内存碎片变得更糟。要打开此选项，请通过以下方式在 ConfigProto 中设置选项：

&nbsp;

&nbsp;

config = tf.ConfigProto()

config.gpu_options.allow_growth = True

session = tf.Session(config=config, ...)

&nbsp;

&nbsp;

第二种方法是 per_process_gpu_memory_fraction 选项，它决定了每个可见GPU应该分配的总内存量的一部分。例如，可以通过以下方式告诉 TensorFlow 仅分配每个GPU的总内存的40％：

&nbsp;

config = tf.ConfigProto()

config.gpu_options.per_process_gpu_memory_fraction = 0.4

session = tf.Session(config=config, ...)

&nbsp;

&nbsp;

如果要真正限制 TensorFlow 进程可用的GPU内存量，这非常有用。

&nbsp;

<strong>五、在多GPU系统上使用单个GPU</strong>

&nbsp;

如果您的系统中有多个GPU，则默认情况下将选择具有最低ID的GPU。 如果您想在不同的GPU上运行，则需要明确指定首选项：

&nbsp;

# 创建一个计算图

with tf.device('/device:GPU:2'):

a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')

b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')

c = tf.matmul(a, b)

# 创建一个 log_device_placement 设置为True 的会话

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

# 运行这个操作

print(sess.run(c))

&nbsp;

&nbsp;

你会看到现在 a 和 b 被分配给 cpu:0。 由于未明确指定设备用于 MatMul 操作，因此 TensorFlow 运行时将根据操作和可用设备（本例中为 gpu:0）选择一个设备，并根据需要自动复制设备之间的张量。

&nbsp;

如果指定的设备不存在，将得到 InvalidArgumentError：

InvalidArgumentError: Invalid argument: Cannot assign a device to node 'b':

Could not satisfy explicit device specification '/device:GPU:2'

[[Node: b = Const[dtype=DT_FLOAT, value=Tensor&lt;type: float shape: [3,2]

values: 1 2 3...&gt;, _device="/device:GPU:2"]()]]

&nbsp;

如果希望 TensorFlow 在指定的设备不存在的情况下自动选择现有的受支持设备来运行操作，则可以在创建会话时在配置选项中将 allow_soft_placement 设置为 True。

&nbsp;

# 创建计算图

with tf.device('/device:GPU:2'):

a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')

b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')

c = tf.matmul(a, b)

# 创建一个 allow_soft_placement 和 log_device_placement 设置为 True 的会话

&nbsp;

sess = tf.Session(config=tf.ConfigProto(

allow_soft_placement=True, log_device_placement=True))

# 运行这个操作

print(sess.run(c))

&nbsp;

<strong>六、使用多个 GPU</strong>

&nbsp;

如果您想要在多个 GPU 上运行 TensorFlow ，则可以采用多塔式方式构建模型，其中每个塔都分配有不同的 GPU。 例如：

&nbsp;

&nbsp;

# 创建计算图

c = []

for d in ['/device:GPU:2', '/device:GPU:3']:

with tf.device(d):

a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3])

b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2])

c.append(tf.matmul(a, b))

with tf.device('/cpu:0'):

sum = tf.add_n(c)

# 创建一个 log_device_placement 设置为 True 的会话

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

# 运行这个操作

print(sess.run(sum))

&nbsp;

你将会看到以下的输出：

&nbsp;

Device mapping:

/job:localhost/replica:0/task:0/device:GPU:0 -&gt; device: 0, name: Tesla K20m, pci bus

id: 0000:02:00.0

/job:localhost/replica:0/task:0/device:GPU:1 -&gt; device: 1, name: Tesla K20m, pci bus

id: 0000:03:00.0

/job:localhost/replica:0/task:0/device:GPU:2 -&gt; device: 2, name: Tesla K20m, pci bus

id: 0000:83:00.0

/job:localhost/replica:0/task:0/device:GPU:3 -&gt; device: 3, name: Tesla K20m, pci bus

id: 0000:84:00.0

Const_3: /job:localhost/replica:0/task:0/device:GPU:3

Const_2: /job:localhost/replica:0/task:0/device:GPU:3

MatMul_1: /job:localhost/replica:0/task:0/device:GPU:3

Const_1: /job:localhost/replica:0/task:0/device:GPU:2

Const: /job:localhost/replica:0/task:0/device:GPU:2

MatMul: /job:localhost/replica:0/task:0/device:GPU:2

AddN: /job:localhost/replica:0/task:0/cpu:0

[[  44.   56.]

[  98.  128.]]

&nbsp;

&nbsp;

翻译自：

<a href="https://www.tensorflow.org/programmers_guide/using_gpu">https://www.tensorflow.org/programmers_guide/using_gpu</a>