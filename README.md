# DrWhoAI
## Introduction
As a fan of **Dr Who** series and artificial intelligence lover my goal it is to create an artificial agent with the personality of Dr Who.

The idea of generating an equivalent artificial agent of any person through their interactions with the real world is not a new idea in literature. For example in the sci-fi serie Caprica a virtual agent was created based on all existing records about a person. That idea could be moved to a fiction characters, if we have enough information on how a person speaks and how it interacts with other characters, the fiction character could be generate an equivalent model to this character. The information, that we have in a book about character is usually small and complex to transfer to a models. Furthermore the television series and movies, full of dialogues among characters could generate primitive models of conversation. One of these early BOTs generated based on dialogs from movies can be seen in http://arxiv.org/pdf/1506.05869.pdf (July 2015).  This paper present a model generated through an algorithm based on deeplearning “seq2seq” getting meaningful answers. One of the problems is the training method, thousands of thousands of dialogues from diferents characters without any relation and context. My point is this, why not train with only one character dialogues. Dr. Who is one of the oldest television characters and the hero of many of us.
The aim would be to generate an equivalent to Doctor model, based on all the dialogues of his years of issue.


The main technologies to be used are Deeplearning and other AGI technologies

![alt tag](http://i0.wp.com/nerdgeekfeelings.com/wp-content/uploads/2014/12/doctor-who-all-doctors-fanart.jpg?resize=1024%2C576)

## Deep Learning Approach

The main 

### LSTM Model
### Seq2Seq Model

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


### Based on
 The main projects and code used are:
* [TensorFlow translate seq2seq example](https://www.tensorflow.org/versions/master/tutorials/seq2seq/index.html) 
* [Shakespeare translations using TensorFlow](https://github.com/tokestermw/tensorflow-shakespeare)

Author

Rafael del hoyo

License

Apache 2.0

Future Plans
