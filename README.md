# DrWhoAI
## Introduction
As a fan of **Dr Who** series and artificial intelligence lover my goal it is to create an artificial agent with the personality of Dr Who.

The idea of generating an equivalent artificial agent of any person through their interactions with the real world is not a new idea in literature. For example in the sci-fi serie Caprica a virtual agent was created based on all existing records about a person. That idea could be moved to a fiction characters, if we have enough information on how a person speaks and how it interacts with other characters, it could be generated an equivalent model to this character. The information, that we have in a book about character is usually small and complex to transfer to a models. Furthermore the television series and movies, full of dialogues among characters could generate primitive models of conversation. One of these early BOTs generated based on dialogs from movies can be seen in http://arxiv.org/pdf/1506.05869.pdf (July 2015).  This paper present a model generated through an algorithm based on deeplearning “seq2seq” getting meaningful answers. One of the problems is the training method, thousands of thousands of dialogues from diferents characters without any relation and context. My point is this, why not train with only one character dialogues. Dr. Who is one of the oldest television characters and the hero of many of us.
The aim would be to generate an equivalent to Doctor model, based on all the dialogues of his years of issue.


The main technologies to be used are Deeplearning and other AGI technologies

![alt tag](http://i0.wp.com/nerdgeekfeelings.com/wp-content/uploads/2014/12/doctor-who-all-doctors-fanart.jpg?resize=1024%2C576)

## Deep Learning Approach

The main 

### LSTM Model


Long Short Term Memory networks – usually just called “LSTMs” – are a special kind of RNN, capable of learning long-term dependencies. They were introduced by Hochreiter & Schmidhuber (1997), and were refined and popularized by many people in following work.1 They work tremendously well on a large variety of problems, and are now widely used.

LSTMs are explicitly designed to avoid the long-term dependency problem. Remembering information for long periods of time is practically their default behavior, not something they struggle to learn.  In this model, ordinary neurons, i.e. units which apply a sigmoidal activation to a linear combination of their inputs, are replaced by memory cells. Each memory cell is associated with an input gate, an output gate and an internal state that feeds into itself unperturbed across time steps.
Introduction of the LSTM model:


All recurrent neural networks have the form of a chain of repeating modules of neural network. In standard RNNs, this repeating module will have a very simple structure, such as a single tanh layer.
![enter image description here](http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-chain.png)


More information can be obtained in
*    [[pdf]](http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf) Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. Neural computation, 9(8), 1735-1780.

Addition of the forget gate to the LSTM model:

*   [[pdf]](http://www.mitpressjournals.org/doi/pdf/10.1162/089976600300015015) Gers, F. A., Schmidhuber, J., & Cummins, F. (2000). Learning to forget: Continual prediction with LSTM. Neural computation, 12(10), 2451-2471.

Graves LSTM paper:

*   [[pdf]](http://www.cs.toronto.edu/~graves/preprint.pdf) Graves, Alex. Supervised sequence labelling with recurrent neural networks. Vol. 385\. Springer, 2012.

A very good introduction

*   [[Understanding LSTM Networks]](http://colah.github.io/posts/2015-08-Understanding-LSTMs)

Other Introduction

* [[Demystifying LSTM neural networks]](http://blog.terminal.com/demistifying-long-short-term-memory-lstm-recurrent-neural-networks/)


### Seq2Seq Model

A basic sequence-to-sequence model, as introduced in Cho et al., 2014, consists of two recurrent neural networks (RNNs): an encoder that processes the input and a decoder that generates the output. This basic architecture is depicted below

![enter image description here](https://www.tensorflow.org/versions/master/images/basic_seq2seq.png)

The strength of this model lies in its simplicity and generality. We can use this model for machine translation, question/answering, and conversations without major changes in the architecture.

The main papers about this model:

* [ [pdf] Sequence to Sequence Learning with Neural Networks](http://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf)
* [ [pdf] Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation](http://arxiv.org/pdf/1406.1078.pdf)
* [ [pdf] Neural Machine Translation by Jointly Learning to Align and Translate](http://arxiv.org/pdf/1409.0473v6.pdf)
* [ [pdf] A Neural Conversational Model](http://arxiv.org/pdf/1506.05869v1.pdf)


### How to Execute
First download the TensorFlow library depending on your platform:

**Ubuntu/Linux 64-bit, CPU only:**
``` bash
$ sudo pip install --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.6.0-cp27-none-linux_x86_64.whl
```
**Ubuntu/Linux 64-bit, GPU enabled:**
``` bash
$ sudo pip install --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow-0.6.0-cp27-none-linux_x86_64.whl
```
**Mac OS X, CPU only:**
``` bash
$ sudo easy_install --upgrade six
$ sudo pip install --upgrade https://storage.googleapis.com/tensorflow/mac/tensorflow-0.6.0-py2-none-any.whl
```
First of all you should download the Dr Who trancriptions http://www.chakoteya.net/DoctorWho/. The Code will  
``` 
python -m DeepLearning.Utils.crawlerEpisodes
``` 
Then you should obtain the Dialogues of the Doctor, trying to get when the doctos is asked. We will obtain a CSV with the information structured.
``` 
python -m DeepLearning.Utils.getTrainingData
``` 
Gets train, dev split the CSV data generated, Builds vocabulary and Converts parallel data into ids
``` 
python -m DeepLearning.Utils.prepare_corpus
```

From the root directory:
Delete /cache to start anew.

### Train

Use the example BASH script to train the model. This saves the check points in the `--train_dir` directory.
If you run it again, the training process continues from the check point. To restart with fresh parameters,
simply delete/rename the check points.

```
./run.sh
```
### Evaluate

Use a BASH script to execute several sentences examples to evaluate the model generated.
```
./run_example.sh
```
### Interactive 

Use a BASH script to make a chatbot interactive with the Dr. Who
```
./run_decode.sh
```


### Problems encountered
The current main issues are the following:

* Not reasoning, is some thing like random answer
* Dificult to evaluate how good or bad is de model generated, not is enought to mantain one evaluation part of the corpus 

## Current example results
Person Question/comment | Doctor Answer
 --- | ---
Person: what is your name ? | Doctor: i don't know .
Person: are you the doctor ? | Doctor: yes , i know .
Person: What's that noise ? | Doctor: what noise ?
Person: (wearing the sunglasses) So what was it ? Your ghost. | Doctor:i can't be surprised.
Person:  Why did they only come out at night ? | Doctor: what 's that ?
Person: What will happen to them ? | Doctor: i 'll know.
Person: What do I do now? | Doctor: i 'm going to get back to the tardis .
Person: I do. You keep going. You have to. Take it from me, there is a whole world out there. A galaxy, a life. What would O'Donnell have wanted ? | Doctor: yes .
Person: What will UNIT do with the ghosts ? | Doctor: i don't know.
Person: Here's what I don't understand. You did change the future. You stopped the Fisher King from returning . | Doctor: well , i 'm not sure . i 'm delighted to hear . i 'm delighted to hear .
Person: And saying the chamber will open ? | Doctor: very well .
Person: Smart. | Doctor: i'm sorry .
Person: How do you mean ? | Doctor: cataclysm .


### Based on other project
 The main projects and code used are:
* [TensorFlow translate seq2seq example](https://www.tensorflow.org/versions/master/tutorials/seq2seq/index.html) 
* [Shakespeare translations using TensorFlow](https://github.com/tokestermw/tensorflow-shakespeare)

### Other Interesting articles

[[pdf]](http://arxiv.org/abs/1511.03729) This is an algorithm which better models language semantics in context (helps Google's model)

[[pdf]](http://arxiv.org/abs/1510.08565) This is a conversation model with attention with intention (more human like)

[[pdf]](http://arxiv.org/abs/1510.03055) This is showing how using MMI over seq2seq generates more relevant results for sentence generation.

[[pdf]](http://arxiv.org/abs/1511.06440) This improved supervised learning cost functions drastically.

[[pdf]](http://arxiv.org/abs/1512.08301) Another new type of network outperforming rnn's and lstm's.

Author

Rafael del Hoyo

License

Apache 2.0

Future Plans

- Im working to introduce Word2vec as embedded layer trying to improve the vocabulary of the Doctor
- Trying to genereste the proces in 2 steps based on the same ideas of [solving the Facebook AI Research bAbI tasks](https://docs.google.com/viewer?a=v&pid=forums&srcid=MTYzMDU0NjEwNDE5NjI1MzYxMjMBMTI5MzA0MzY1OTIyODc5MjE3MDQBeUtSY3UxSS1aYXNKATAuMQEBdjI)
