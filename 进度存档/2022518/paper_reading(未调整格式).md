# Paper Reading -Attention is all you need

[toc]

## 1. Attention is all you need

### 1.abstract

弃用：Recurrence 结构 ， convolution 结构

只使用：Attention Mechanism

效果：简单，又快，又好

### 2.introduction

Recurrence 结构的问题 ：依赖序列，因此限制并行计算。目前的一些改进方法：分解技巧，条件计算。但不能从根本上改变序列计算的问题。

Attention 机制可以不需要考虑输入输出序列之间的距离(有些工作还将 Attention 连接到 recurrent network 中)

Transformer 只用了 Attention。

### 3.background

一些模型基于 Convolution network 提高并行计算的能力，但计算符号的数量会随着 input 或 output 的两个位置之间的距离而增长 ， 但 transformer 的运算符是 constant ，虽然这是通过降低 有效分辨率 来形成的(在多头中的平均处理)

Self-attention 被认为通过依赖一个序列不同的位置来计算 sequence 的 representation 。

### 3.1 Model Architecture

`encoder-decoder` 一般结构 ：encoder 对输入符号进行编码 ，decoder 一次输出 1 symbol， 输出会考虑先前输出的 symbol 。

<img src="https://github.com/H-shw/Transformer_etc./blob/master/%E8%BF%9B%E5%BA%A6%E5%AD%98%E6%A1%A3/2022518/pics/1.png" alt="png" style="zoom:50%;" />

`encoder` : $N=6$ , Self - Attention +  1个简单的 fully connected feed-forward network ， 中间用残差网络连接(为了保证残差网络的适用性，要求 dimension always = 512)

`decoder` : $N=6$ ,  Masked Multi-Head Attention 层应该是为了防止 序列号(position)更大的 参与之前的预测(保证了一个顺序)

### 3.2 Attention

An attention function can be described as mapping a **query** and a set of **key-value** pairs to an output, where the query, keys, values, and output are all vectors. 

The output is computed as `a weighted sum of the values`, where the weight assigned to each **value** is computed by a compatibility function of the `query with the corresponding key`.

Q : 去匹配 KEY

K : 被匹配 Q

V : 通过上面两者的交互，加权得到表示

Query，Key，Value的概念取自于信息检索系统，举个简单的搜索的例子来说。

当你在某电商平台搜索某件商品（年轻女士冬季穿的红色薄款羽绒服）时，你在搜索引擎上输入的内容是Query，然后搜索引擎根据Query为你匹配Key（例如商品的种类，颜色，描述等），然后根据Query和Key的相似度得到匹配的内容（Value)。





<img src="https://4143056590-files.gitbook.io/~/files/v0/b/gitbook-legacy-files/o/assets%2F-LpO5sn2FY1C9esHFJmo%2F-M1uVIrSPBnanwyeV0ps%2F-M1uVKsKWfvSHDX8l-ff%2Fencoder-introduction.jpg?generation=1583677010388256&alt=media" alt="img" style="zoom:50%;" />

这个绿色的框（Encoder #1）就是Encoder里的一个独立模块。下面绿色的输入的是两个单词的embedding。这个模块想要做的事情就是想**把**x_1*x*1**转换为另外一个向量**$r_1$，这两个向量的维度是一样的。然后就一层层往上传。

转化的过程分成几个步骤，第一个步骤就是Self-Attention，第二个步骤就是普通的全连接神经网络。但是注意，**Self-Attention框里是所有的输入向量共同参与了这个过程，也就是说，**$x_1$和$x_2$通过某种信息交换和杂糅，得到了中间变量$z_1$和$z_2$。而全连接神经网络是割裂开的，$z_1$和$z_2$各自独立通过全连接神经网络，得到了$r_1$和$r_2$。

$x_1$ 和 $x_2$ 互相不知道对方的信息，但因为在第一个步骤Self-Attention中发生了信息交换，所以$r_1$和$r_2$各自都有从得来的信$x_1$和$x_2$信息了。

如果我们用**直觉的方式来理解Self-Attention**，假设左边的句子就是输入$x_1,x_2,...,x_{14}$，然后通过Self-Attention映射为$z_1,z_2,...,z_{14}$，**为什么叫Self-Attention呢，就是一个句子内的单词，互相看其他单词对自己的影响力有多大**。比如单词`it`，它和句子内其他单词最相关的是哪个，如果颜色的深浅来表示影响力的强弱，那显然我们看到对`it`影响力最强的就是`The`和`Animal`这两个单词了。所以**Self-Attention就是说，句子内各单词的注意力，应该关注在该句子内其他单词中的哪些单词上**。

![img](https://4143056590-files.gitbook.io/~/files/v0/b/gitbook-legacy-files/o/assets%2F-LpO5sn2FY1C9esHFJmo%2F-M1uVIrSPBnanwyeV0ps%2F-M1uVKsRIyR8d4qMnFoq%2Fself-attention-intuiation.jpg?generation=1583677019227629&alt=media)

<img src="https://github.com/H-shw/Transformer_etc./blob/master/%E8%BF%9B%E5%BA%A6%E5%AD%98%E6%A1%A3/2022518/pics/4.png" alt="image" style="zoom:67%;" />



自注意力机制公式如下
$$
Attention(Q,K,V) = softmax(\frac{QK^{T}}{\sqrt{d_k}})V
$$
注：两个经常用的 attention 函数 ：点积(dot-product) 和 additive attention (使用真自由一层隐藏层的前馈神经网络)，两者复杂度相似，但点积更快  (GPU 对 矩阵乘法的优化)

$\frac{1}{\sqrt{d_k}}$ 的原因 ： 防止点积之后的结果太大，使得 `softmax` 的结果推到梯度比较小的位置。

#### 3.2.2 多头注意力

多头注意力机制可以用来学习不同的子空间，不同的位置上的表示，如果只有一个 `attention head` 将会平均这个情况。(本段缺乏解释性的验证)

一个对维度的考虑。For each of these we use $dk = dv = d_{model}/h = 64.$ Due to the reduced dimension of each head, the total computational cost is similar to that of single-head attention with full dimensionality.

#### 3.2.3 Applications of Attention in our Model

共有3处使用了这种多头注意力的机制

<img src="https://github.com/H-shw/Transformer_etc./blob/master/%E8%BF%9B%E5%BA%A6%E5%AD%98%E6%A1%A3/2022518/pics/5.png" style="zoom:67%;" />

* Query  来自前一层的 decoder , key value 来自本层。允许 decoder 与输入的 sequence 交互，模仿了经典的 encoder-decoder attention mechanism
* encoder 中包含，使得 encoder 中的每个位置可以和上一层encoder的每个位置交互
* decooder 中也是相同的。另外需要保证的一点是，通过使用 MASK 防止信息向左边流动。

### 3.3 position-wise Feed-Forward Networks

$$
FFN(x) = max(0,xW_{1}+b_{1})W_{2} + b_{2}
$$

两层的 FC ，中间接 $RELU$ 。两层的参数不共享。

实现上的维度：The dimensionality of input and output is dmodel = 512, and the inner-layer has dimensionality dff = 2048.

### 3.4 Embeddings and Softmax

We also use the usual learned linear transformation and softmax function to convert the decoder output `to predicted next-token probabilities`. 

In our model, we share ` the same weight matrix ` between the `two embedding layers` and the `pre-softmax linear transformation`. 

In the embedding layers, we `multiply those weights` by $\sqrt{ d_{model} }$. (应该是前面除过了，现在进行还原)

<img src="https://github.com/H-shw/Transformer_etc./blob/master/%E8%BF%9B%E5%BA%A6%E5%AD%98%E6%A1%A3/2022518/pics/6.png" style="zoom:50%;" />

### 3.5 Positional Encoding

捕捉相对位置关系。
$$
PE_{(pos,2i)}=sin(pos/10000^{2i/d_{model}})\\
PE_{(pos,2i+1)}=sin(pos/10000^{2i/d_{model}})\\
$$
主要是获得相对位置的表示(毕竟三角函数是周期函数)。 

除此之外还使用了 training 的方法获得位置表示，但效果与上面的相似，用 sin 还能获得更大的位置的表示。

10000 参数的疑问 ：猜测可能就是为了获得更大的位置的表示吧。

### 4 Why Self-Attention

transformer 本质 ：a sequence to another

三个需求：

1.每一层的复杂度

2.并行技术的数量以及 sequential operation 所需要的最小数量(即尽量每个减少序列计算的操作数量)

3.在长距离的依赖中的路径长度(path length)，越短越容易学习到长距离的依赖关系。

<img src="https://github.com/H-shw/Transformer_etc./blob/master/%E8%BF%9B%E5%BA%A6%E5%AD%98%E6%A1%A3/2022518/pics/7.png" style="zoom:67%;" />

对 Complexity 的理解：

* Self-Attention : 每个词($n$)都要考虑其他所有词($n$) , 所用的方法为直接相乘
* Recurrent : 1个词考虑其他所有词 (1层一个) 
* Convolution : 增加了核的因素

Sequential Operation:

* 只有 Recurrent 需要重复 $N$ 次，因为一次处理一个词

Maximum:

* Recurrent 需要重复 $N$ 次，因为一次处理一个词
* Attention 完全并行处理
* Convolution 要看核的大小，一次处理核大小的数据

**Regularization**:

drop out 

Label Smoothing

**Optimizer:**

Adam
$$
\beta_{1} = 0.9\, , \beta_{2}=0.98\,,\epsilon=10^{-9}\\
lrate = d_{model}^{-0.5} \cdot min(step\_num^{-0.5},step\_num \cdot warmup\_steps^{-1.5})
$$
相当于先线性增长，再衰减。We used $warmup\_steps = 4000$



### 补充：Result 部分调参的带来的一些启示

In Table 3 rows (B), we observe that `reducing the attention key size` $d_k$ `hurts model quality`. This suggests that determining compatibility is not easy and that a more sophisticated compatibility function than dot product may be beneficial. 

We further observe in rows (C) and (D) that, as expected, bigger models are better, and `dropout` is very helpful in avoiding over-fitting. 

In row (E) we replace our sinusoidal positional encoding with learned positional embeddings [8], and `observe nearly identical results to the base model`. 训练 和 cos/sin 的效果大致相似 

### 来自中文博客的一些补充资料

https://luweikxy.gitbook.io/machine-learning-notes/self-attention-and-transformer

> Attention和self-attention的区别
>
> 以Encoder-Decoder框架为例，输入Source和输出Target内容是不一样的，比如对于英-中机器翻译来说，Source是英文句子，Target是对应的翻译出的中文句子，Attention发生在Target的元素Query和Source中的所有元素之间。
>
> Self Attention，指的不是Target和Source之间的Attention机制，而是Source内部元素之间或者Target内部元素之间发生的Attention机制，也可以理解为Target=Source这种特殊情况下的Attention。
>
> 两者具体计算过程是一样的，只是计算对象发生了变化而已。



下图是有八个Attention，先看右图，这八个Attention用八种不同的颜色表示，从蓝色到灰色。然后我们可以看到一个单词，在这八个Attention上对句子里每个单词的权重，颜色越深，代表权重越大。我们只挑出橙色和绿色（即第二个和第三个色块），看它们分别是怎样的注意力。然后把橙色和绿色的色块拉长就得到了左边这个图。

![img](https://4143056590-files.gitbook.io/~/files/v0/b/gitbook-legacy-files/o/assets%2F-LpO5sn2FY1C9esHFJmo%2F-M1uVIrSPBnanwyeV0ps%2F-M1uVKsm7hF0Pgr2phbd%2Fmulti-headed-attention-5.jpg?generation=1583677012711592&alt=media)

我们现在看左边，先看橙色部分，单词`it`连接的权重最重的是`animal`，这是从某一个侧面来看，那从另一个侧面来看，看绿色部分，`it`最关注的是`tired`。橙色的注意力主要表明`it`是个什么东西，从东西的角度说明它是一种动物，而不是苹果或者香蕉。如果我们从状态这个层面来看，`it`这个动物现在是在怎么样的一个状态，它的状态是`tired`，而不是兴奋。所以**不同的Self-Attention Head是不同方面的理解**。









