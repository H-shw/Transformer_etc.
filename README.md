# 介绍与进度

本仓库用于存储学习进度。应当在完工的时候会包括 : 

* *transformer* 论文阅读
* *transformer* 复现
* *BERT* 论文阅读
* *Huggingface* 里的  *Transformers*
* *fairseq* 的知识等。

**欢迎 check 进度，欢迎告知可以改进的方面(比如,代码风格，阅读论文的方法等)。**



## 进度

这里会用很简短的语言描述今天做了什么(也可能包含不只是关于本次学习的内容)，不过如果今天摸鱼了也会写上来。。



### **2022.5.17**

获得指导和材料，晚上开始阅读 [*Attention is all you need*](http://papers.neurips.cc/paper/7181-attention-is-all-you-need.pdf)。

### **2022.5.18**

今天比较空闲，就多花了时间在这上面。

早上大概把论文看完了，然后阅读了一篇质量不错的博客辅助，加深理解(https://luweikxy.gitbook.io/machine-learning-notes/self-attention-and-transformer)。

一个非常初期的报告点击[这里](https://github.com/H-shw/Transformer_etc./blob/master/%E8%BF%9B%E5%BA%A6%E5%AD%98%E6%A1%A3/2022518/paper_reading(%E6%9C%AA%E8%B0%83%E6%95%B4%E6%A0%BC%E5%BC%8F).md)。*Github* 不支持 *LaTex* 数学符号的显示。。如 $\frac{1}{n}$

下午开始看代码，着手复现 *Transformer* 。

晚上感觉差不多把 *Transformer* 模型大概搭好了？应该说 *Transformer* 代码量并不大，不过感觉参考别人的思路多了些。。不过有些细节按照自己的想法写了，也许有 *bug* , 后面还需要检查。

明天课比较多，还打算周五组会要汇报的东西。

### **2022.5.19**

大概把 *Transformer* 整个训练框架搭好了 (其实也就只写了`train.py` , `test.py` ，一点点`utils.py` 内容 和 *config* 文件，还是把之前工作中的搬来改改的。。具体任务还没确定，因此 *dataloader* 还没写)

可能考虑找个简单的任务测试一下 *MyTransformer* 效果？

另外还写了一份计算机网络的实验报告，虽然是验证性实验不过内容相当多。

还看了一篇关于  关于 *VQA* 的论文 ，在计算构成前面准则的要件的时候，套用了 *Transformer* 的结构，把 *Self-Attention* 机制换成了它所需要的 *Attention* 结构。

### **2022.5.20**

抽空增加了 *optimizer* 的 *transformer* 定制版，和 *reference* 实现的 *transformer* 对照过一遍，改了改一些细节。

晚上组会开完看看有没有时间找一下任务。



### **2022.5.21**

找了一个[文本分类的任务](https://github.com/649453932/Chinese-Text-Classification-Pytorch)(因为目前找到的翻译任务感觉数据量很大)。目前已经调试运行成功(框架用人家的)，使用我自己的 *Transformer* 模型达到了和该任务作者使用的 *Transformer* 基本相似的效果(感觉这个作者的 *dropout* 很奇怪就没加)。

可能存在的不足是这里只用上了 *encoder* , 毕竟文本分类不需要 *decoder* 翻译为人类可以理解的内容。

另外，*code*文件夹下的 *transformer* 中相比之前的模型，增加了 共享权重 的机制。(分别共享了 *encoder* 和 *decoder* 的 embedding 权重，以及在 *encoder-decoder* 之后的全连接层的权重)

目前暂时认为复现工作大概就这样，之后还会再改改，还不算非常完全。

下一步看 [*BERT*](https://arxiv.org/pdf/1810.04805.pdf) 论文吧。







 

