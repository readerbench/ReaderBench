# ReaderBench Python

## Install
1. `pip3 install --user rbpy-rb` (or clone this repo master branch)
2. run `./install.sh`

You may also need some spacy models which are downloaded through spacy.     
You have to download these spacy models by yourself, using the command:    
`python3 -m spacy download name_of_the_model`   
The logger will also write instructions on which models you need, and how to download them.  

Be careful, you need to have spacy 2.1.3. 
If you change the version of spacy (you had a previous version) you need to reinstall xx_ent_wiki_sm model.

For neural coref errors install it as follows: https://github.com/huggingface/neuralcoref#spacystringsstringstore-size-changed-error

## Usage
For usage (parsing, lemmatization, NER, wordnet, content words, indices etc.)  see file `usage.py`

## Dev instructions

## How to use the logger
In each file you have to initialize the logger:  
```sh
from rb.utils.rblogger import Logger  
logger = Logger.get_logger() 
logger.info("info msg")
logger.warning("warning msg")  
logger.error()
```

