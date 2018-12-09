## Sorting Problem By Pointer Network

佘国榛

15307130224

In this report, we introduced a new encoder&decoder arch network to solve the sorting problem as the general purpose. The report can be seperated into several parts: 

1.The Attention mechanism and the seq2seq model

2.The architecture&usage of the pointer network

3.The implementation of the code

4.The performance analysis of the result work.

## Attention mechanism

If we talk about the attention mechanism, we must know what the RNN&seq2seq is.

In the NLP problem, we would meet the translation problem quite offten.  We usually use the seq2seq model to solve it, the following architecture is a standard seq2seq model contains both the encoder and the decoder.

![屏幕快照 2018-12-09 下午3.48.21](/Users/hashibami/Desktop/屏幕快照 2018-12-09 下午3.48.21.png) 



We have omitted the loss function design in the seq2seq, since it varies with different implementation. In the pointer network, we will represent a much more detailed information about the given network. In this implementation, we can see some shortcomings very easily. For example, once the sentence length expands, we can see that the yellow cell can not memory the enough information. 

So we brought out a new conecept **Attention**, in this implementation, we can fully utilize the encoder's output information, the following picture show the partial architecture of attention mechanism.



 