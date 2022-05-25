# BERT

## 背景

语言模型的预训练已经在 NLP 的多种任务上展现了任务，包括 自然语言推理，释义的句子任务 以及 Token 层面的任务如 命名实体识别 和 产出词汇层面合适粒度的输出。

运用预训练语言模型的下游任务主要有 $2$ 种策略 : 

* *feature-base* : 将预训练的表示当做额外的表示加入 *task_specific* 的框架中
* *fine-tuning approach* : 进入少量的任务特定的参数，通过简单地调整参数来适应任务。

现有的应用方法所存在的问题 ： 只考虑在**单向**运用语言模型来学习语言的表示。这种方式可能会限制预训练表示的效果，尤其是 *fine tuning approach* 。单向的方式限制了模型在预训练时采用的结构选择(如 OpenAI GPT)，只使用了从左到右的单向 *Transformer* ， 每个 *token* 都只能考虑之前出现的*token* 。 

*BERT* 通过使用 *masked language model* 来解决单向的问题，通过随机地 *mask* 一些 *token* ，通过上下文来预测被 *mask* 的词是什么。以此来融合前后的上下文，以此训练一个 *deep bidirectional Transformer*。

**贡献**：

* 双向的运用语言的表示。
* 显示 pre-trained 表示不需要大量工程的特定任务结构。
* SOTA for 11 task

## Related Work

***pre-trained* 的历程 ：** 

**Unsupervised Feature-based Approaches**

*embedding* 任务(*char* , *word*, *sentence*, 前人的工作使用了对 下一个句子打分的方法，以此从左到右生成新词汇作为前面句子的表示 或 *denoising auto encoder derived objectives*(?))

ELMo 通过双向地提取 *context-sensitive feature* ，*token* 将从左到右和从右到左的表示 *concat* 起来作为表示，将上下文的表示融入特定的任务框架中

**Unsupervised Fine-tuning Approaches**:

只需要较小的调整。

**Transfer Learning from Supervised Data**

如 CV 等领域也开始用一种预训练的方法。

## BERT

两个部分： *pre-training* & *fine-tuning*

在 *pre-training* 部分使用 *unlabeled data* 训练，然后用这些参数来初始化。

在 *fine-tuning* 阶段使用下有任务的 *labeled data* 。

BERT 的一个特性是一个联合的结构，可以用于多种任务。

BERT 结构图

<img src="https://github.com/H-shw/Transformer_etc./blob/master/%E8%BF%9B%E5%BA%A6%E5%AD%98%E6%A1%A3/2022523/pics/1.png" style="zoom:67%;" />

*[CLS] is a special symbol added in front of every input example, and [SEP] is a special separator token*



$BERT_{base}$ : 

* $L\,\,(number\,\,of\,\,Transformer\_block)$ : $12$
* $H\,\,(Hidden \, size)$ : $768$
* $A\, (self-attention\,heads)$ : $12$
* 总参数量 : $110M$

 

$BERT_{Large}$ : 

* $L\,\,(number\,\,of\,\,Transformer\_block)$ : $24$
* $H\,\,(Hidden \, size)$ : $1024$
* $A\, (self-attention\,heads)$ : $16$
* 总参数量 : $340M$



### 输入输出

输入会将一个 句子 或 句子*pair* 视作一个 *token sequence*。

*sentence* 句子只是一段连续文本，可能和一般意义上的句子不同。

*sequence* 指代 *token sequence* 可能是 1句 ， 或 2句。

The first token of every sequence is always a special classification token ([CLS]). 

这个 *token* 的表示用来聚合句子的表示，用于分类任务。

分句子方法：用 [SEP] , 并且加了一个学习的 embedding 给每个 token , 让他们知道他们是属于哪一个句子(一个 *pair* 中有两个句子)

input embedding as $E$ , [$CLS$] represented by $C \in R^{H}$ , 最终结果的$i^{th}$ token 表示为 $T_{i} \in R^{h}$。

对于 $1$ 个 *token* , 表示的方法是加上 token , segment(属于哪个句子的 embedding) 和 position embedding。



### Pre-training BERT

#### 1.Masked LM

在以往，往往只能单方向的进行训练，这是因为在双向的条件下，允许每个 *token* 间接地看到自己，因此在多层的上下文环境下，模型可以轻易地预测到目标词。

因此为了训练深层的双向表达， 随机 *mask* 一定比例的 *token* , 成为 *masked LM*。

最后的 *mask token* 对应的隐藏层向量放到 *softmax* 中预测 。 mask 比例 :$15\%$。

这将到导致 *pre-train* 的任务和 *fine-tune* 的任务有所不同，因此采用的方法是：挑出的 $15\%$ *mask token* 将有 $80\%$ 概率为 *[MASK]* , $10\%$ 的概率是随机的 *token* , $10\%$ 的概率是原来的词。

#### Next Sentence Prediction (NSP)

用于抓住两个句子之间的 *relationship* 。

选择一些句子进行训练，一个句子有 50% 的概率是真实的后一句， 50% 概率不是。简单但有效。

### Fine-tuning

一般的任务在解决一个 pair 的问题时，往往是先分开分别 *encode* , 再用 *bidirectional cross attention* 。

BERT 中将两个过程合二为一了。

*Fine-tuning* 的过程只需要简单的输入 *input & output* , 是端到端的。

在 *input* 中 ， *pair sentence A  & B* 被视作 :

* *sentence pairs in* 改写任务
* 假设前提 *pairs in* 推理任务
* *question passage pairs in QA*
* *a degenerate text-∅ pair in* 文本分类和句子排序任务

在 *output* 中，*token representation* 放入 *output layer* 用于 *token level* 的任务，比如 句子排序(*sentence tagging*) 或 *QA* , *[CLS]* 将会放到 *output layer* 中用作分类，比如推理和语义分析任务。

*fine-tuning*  任务比较简单，需要的时间和计算资源相比预训练过程都十分微小。

### 实验部分

实验的几个结论

* 双向的语言模型效果好于单向的
* 即使是数据量很小的任务，使用大规模的预训练仍然是有效的; 大规模的模型能够提高效果。
* 使用 *Feature-based Approach* (不更改其中的 *parament* , 后面接上 *BiLSTM* 直接分类) 一样能取得较好的效果。

### Appendix

#### A

通过 80% , 10% random , 10% origin 的方法， Transformer encoder 不知道需要预测或者被替换的词是什么，因此会去学输入的每一个次的上下文表示。

由于只用 $15\%$ 的量来预测，因此需要更多的预测次数。



 *Comparison of BERT, ELMo ,and OpenAI GPT*

![image-20220525201341783](C:\Users\14163\AppData\Roaming\Typora\typora-user-images\image-20220525201341783.png)

BERT , OPENAI GPT 是 Fine-tune 的

ELMo 是 feature-base 的

同时也能很明显的看出来， BERT 在每一层都是双向的。



另外还描述了一些调参得出的结论，如 *epoch* 数量， *mask,random,origin* 比率的影响。另外还得出了 *fine-tune* 的方法对于这个比例更具有鲁棒性。
