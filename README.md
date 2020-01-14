# Question-Answering-System
  
  This system implements a Deep NLP pipeline which is capable of extracting the exact one word answer to any question entered in Natural Language by searching a scalable Corpora of over 30 Wikipedia Articles in the raw text format. The Question can be entered during runtime or a set of questions can be dumped in a text file. The System prints the fetched answers on the command line as well as in a well structured JSON file.
  The developed NLP pipeline exercises various concepts such as TF-IDF and Cosine similarity, Tokenization, Part of Speech Tagging, Lemmatizing, Dependency Parsing, Lesk Algorithm, Wordnet Interpretations, Named Entity Recognition and sentence overlapping and text matching.

## Getting Started

Firstly, clone the repository to your local system. Make sure you have a well functioning Computer with a decent RAM and sufficient storage. This System is cross platform functional and will work on any known Operating System.  

### Prerequisites

Before you begin make sure the follwing things are installed on your computer:
a. Python3
b. Data Science and scientific processing Libraries. It is not mandatory although recommended to install Anaconda for Python 3.7 as it pretty much contains all important data science libraries.  
c. The libraries needed for this system to work are: sklearn, spacy, glob, nltk, sys, numpy, wordnet, string, json, pprint.


## Deployment

Once the environment is set up and everthing is installed do the following:
a. Navigate to the cloned directory through the command line shell.
b. Execute the program named questionanswer.py by typing:
```
python questionanswer.py
```
c. The program will ask to input name of the file which contains all questions. Enter the name of that file
d. The system will then execute and output a json file with the answers to the questions. Answers will also be printed on the command line interface.

## Extras

The project also consists of some other .py Python files. These files exercise some of the NLP concepts used to develop the system in individuality. It is kind of fun to see how these concepts help in the development of an integrated system as a whole. Do try executing them too. 
Execute as:
```
python <program name>.py
```

## Built With

* [Python](https://www.python.org/) - The programming language
* [Jupyter Notebook](https://jupyter.org/) - Interactive Development Editor
* [Sublime Text](https://www.sublimetext.com/) - Sophisticated text editor
* [Git](https://git-scm.com/book/en/v1/Getting-Started-About-Version-Control) - Version Control


## Authors

* **Habil Damania**

## Acknowledgement

A big shout out to my friend and project partner Kanika Rawat for helping in developing this system. 


