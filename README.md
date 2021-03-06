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



### **2022.5.22**

今天还没开始看 BERT 。

早上迅速重读了 3 篇 *causal inference* 的论文，下午需要汇报。晚上在复习计网。明天除了上课，可能还是要花大量时间复习计网吧，毕竟要考试了。

值得一提的是，今晚还听了一位学长的科研经历与经验的分享。

首先我觉得讲的很好，表述很清晰，学长是个会说话的人。通过他对自己经历的描述，有一种一步接一步进取提升的感觉，从一个刚开始学 *python* 的萌新成长为了能够自己想 *idea* , 有着属于自己的一套对 *idea* 的评估、测试方法 的大佬。但应该在整个过程中付出了相当多的努力，可能也有失败的经历吧，但这就不足为外人道了。

**一些启示：**

* **论文阅读：**
  * 前沿的，未必是想要立即用上，把握领域的方向(如 *prompt* )   
  * 与当前工作相关的，找当前取得的成果的上存在的一些问题，以及当前工作的可能的解决方案。
* ***idea* 相关：**
  * 最好能够提出有 **启发性** 的 *idea* 或 问题
  * 获得方式：可以找前人工作中存在的不足
  * 对 *idea* 应当进行评估( 创新性与实验风险，如时间资源的消耗)，分段测试 *idea* 的效果(分析可能的效果，根据简单任务的实际表现判断是代码的问题还是 *idea* 的问题)
* **加强交流：**
  * 加强与指导者的交流，注意应当带着**有思考性的问题**去提问和交流
  * 充分利用碎片，较为随意的时间交流一些方向和目前的效果的问题

(希望以上部分是可以发出来的，非常感谢学长的分享)

以下是一些个人的想法。

从我个人的经历上看，感觉对学长提到的论文阅读里最好能阅读当前工作相关的内容，思考改进的方法以及对目前任务的应用尤为重要。

之前做的一个工作一开始属于一个空想的状态，和指导者讨论出的实现方法并没能取得什么效果。在比较大量地阅读工作相关的论文内容后，发现了本方向的发展趋势，再尝试着将其修改，适用到我的任务中来，相比于之前，显然是走上了一条更有依据基础的路。

另外一点我也很有感触的是与指导者的交流。在最开始的时候，作为萌新感觉提出一些大的想法，方向还可以，落到具体的实验上就不太OK，于是就常常带着问题和我的思考去找我的指导者交流。一开始还是有些拘束，害怕太过打扰人家，但是后来发现这种交流是必要的，没有交流，工作就无法推进下去，况且对方也愿意花时间和我交流(在此感谢)。从我个人的经历上来说，感到交流也确实是十分重要的，可以完善自己想法中存在的不足，遇到困难时候可以想办法推进，只要带着有思考去交流，我想对方也是愿意指导我的吧。



### **2022.5.23**

今晚晚点时候开始看BERT论文。

加加班，一会把 BERT 模型部分看完。

[模型部分](https://github.com/H-shw/Transformer_etc./blob/master/%E8%BF%9B%E5%BA%A6%E5%AD%98%E6%A1%A3/2022523/Reading_BERT.md)看完了，论文的总结暂时先写到这里，明天继续。



### **2022.5.24**

简单看了一下 BERT 在几个任务上的表现，感觉没什么特别值得记录之处，下一步就是看附录了。

今天主要复习计网去了。怎么计网知识这么杂。

 

### **2022.5.25**

BERT 论文的阅读基本完工了。[简单的总结链接](https://github.com/H-shw/Transformer_etc./blob/master/%E8%BF%9B%E5%BA%A6%E5%AD%98%E6%A1%A3/2022525/Reading_BERT.md)

要是有空的话感觉最多半天就能看完了。。

关于 huggingface 的 [transformers](https://github.com/huggingface/transformers) 以及 [fairseq](https://github.com/facebookresearch/fairseq) 打算先放放，链接先放这里。

最近先复习期末，刷刷leetcode , 补一补人工智能可能会问的问题。

### **2022.5.26**

抱歉，必须要准备期末考试了。今天起到5月29就先不更了。

### **2022.5.29**

今天考完计网了。明天打算更一下学习因果推理前门准则后门准则之类的内容。

### **2022.5.30**

现在还在整理笔记，应该明天上传。

### **2022.5.31**

已上传。之前看论文的时候不大理解的前面准则，后门准则已经更新到 [此处](https://github.com/H-shw/Transformer_etc./blob/master/notes/casual%20inference/%E5%9B%A0%E6%9E%9C%E6%8E%A8%E7%90%86%E5%AD%A6%E4%B9%A0%E7%AC%94%E8%AE%B0.md) 。下次组会应该要汇报这个了。打了这么多公式感觉好累。

### **2022.6.1**

儿童节快乐！今天看了中介，反事实的一部分，更新到 [此处](https://github.com/H-shw/Transformer_etc./blob/master/notes/casual%20inference/%E5%9B%A0%E6%9E%9C%E6%8E%A8%E7%90%86%E5%AD%A6%E4%B9%A0%E7%AC%94%E8%AE%B0.md) 。

看了一下之前上传的 `markdown` ，好像有的公式显示不出来，我用 *Typora* 写的，看来是 md 方言的问题。

### **2022.6.2**

因果部分差不多整理好了，对应的就是 *CAUSAL INFERENCE IN STATISTICS A PRIMER* 的内容了 [pdf版下载地址](https://zh.usa1lib.org/book/2664651/adcbf6) 的内容，但感觉了解的还是偏浅。

### **2022.6.4**

更新了一下昨天复习的一些AI基础知识内容(基本概念部分)。[链接](https://github.com/H-shw/Transformer_etc./blob/master/notes/AI%20note/AI%E5%86%85%E5%AE%B9%E5%A4%8D%E4%B9%A0.md)

### **2022.6.5**

更新了一些基础模型(一直到集成学习概述部分)。[链接](https://github.com/H-shw/Transformer_etc./blob/master/notes/AI%20note/AI%E5%86%85%E5%AE%B9%E5%A4%8D%E4%B9%A0.md)

### **2022.6.6**

集成学习的看了一部分。[链接](https://github.com/H-shw/Transformer_etc./blob/master/notes/AI%20note/AI%E5%86%85%E5%AE%B9%E5%A4%8D%E4%B9%A0.md)

### **2022.6.7**

今天主要在做期末大作业，打算明天复习深度学习部分。

### **2022.6.8**

简单复习了HMM,神经元模型,CNN,RNN,LSTM,GRU。[链接](https://github.com/H-shw/Transformer_etc./blob/master/notes/AI%20note/AI%E5%86%85%E5%AE%B9%E5%A4%8D%E4%B9%A0.md)

### **2022.6.9**

今天做数据库的大作业，主要是做一个DBMS的应用场景。

### **2022.6.10**

主要还是在做数据库大作业。但是组会前准备，复习因果推理的时候修正了之前笔记的一些错误。 [此处](https://github.com/H-shw/Transformer_etc./blob/master/notes/casual%20inference/%E5%9B%A0%E6%9E%9C%E6%8E%A8%E7%90%86%E5%AD%A6%E4%B9%A0%E7%AC%94%E8%AE%B0.md)

另外再分享一本书。[Causality- Models, Reasoning, and Inference](https://zh.sg1lib.club/book/2780725/2ec8f1)

然后还了解到现在因果推理往往是一种顶层设计，用来组合和应用由深度学习学习出的特征。另外一个常见的使用方法是用反事实的方法生成额外的数据做数据增强。

### **2022.6.12**

抱歉做大作业两天没更新了，明天继续开始学习。

### **2022.6.13**

明天！明天！明天继续学习。中午下午晚上都在装打印机的驱动和适配，一定不要买佳能打印机，特别是二手旧款的！

### **2022.6.14**

今天看了 [word2vec](https://zhuanlan.zhihu.com/p/114538417) ，总结暂时没写。主要时间在复习数据库原理备考，也许这是本科期间最后一次有意义的考试了。

