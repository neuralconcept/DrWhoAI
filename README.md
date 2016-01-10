# DrWhoAI
## Introduction
As a fan of **Dr Who** series and artificial intelligence loving my goal it is to create an artificial agent with the personality of Dr Who.

The main technologies to be used are Deeplearning and other AGI technologies

![alt tag](http://i0.wp.com/nerdgeekfeelings.com/wp-content/uploads/2014/12/doctor-who-all-doctors-fanart.jpg?resize=1024%2C576)

## Deep Learning Approach

The main 

### LSTM Model
### Seq2Seq Model

### How to Execute
First download the TensorFlow library depending on your platform:
```
pip install https://storage.googleapis.com/tensorflow/mac/tensorflow-0.5.0-py2-none-any.whl # for mac
pip install https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.5.0-cp27-none-linux_x86_64.whl # for ubuntu
```

1. Grabs parallel data.
2. Gets train, dev split.
3. Builds vocabulary
4. Converts parallel data into ids

From the root directory:

```
python -m Deeplearning.get_data
python -m tensorshake.prepare_corpus
```

Delete /cache to start anew.

## Train

Use the example BASH script to train the model. This saves the check points in the `--train_dir` directory.
If you run it again, the training process continues from the check point. To restart with fresh parameters,
simply delete/rename the check points.

```
./run.sh
```
Firs of all you should download the Dr Who trancriptions http://www.chakoteya.net/DoctorWho/. The Code will  
```bash

```

### Problems encountered
* Not reasoning, is some thing like random answare
* Dificult to evaluate how good or bad is de model generated, not is enought to mantain one evaluation part of the corpus 
* 
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
