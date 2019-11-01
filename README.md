# ReaderBench Python

## Install
We recommend using virtual environments, as some packages require an exact version.   
If you only want to use the package do the following:

1. `python3 -m venv rbenv` (create virutal environment named rbenv)
2. `source rbenv/bin/activate` (actiavte virtual env)
3. `pip3 uninstall setuptools`
4. `pip3 install setuptools`
5. `pip3 install --upgrade pip`
6. `pip3 install --no-cache-dir rbpy-rb` (install the package)
6. `./install.sh` (install all other things which are not python packages (semantic models))

If you want to contribute to the code base of package:
1. `git clone git@git.readerbench.com:ReaderBench/readerbenchpy.git` 
2. `cd readerbechpy`
3. `python3 -m venv rbenv` (create virutal environment named rbenv)
4. `source rbenv/bin/activate` (actiavte virtual env)
3. `pip3 uninstall setuptools`
4. `pip3 install setuptools`
5. `pip3 install --upgrade pip`
5. `python3 -r requirements.txt` 
6. `./install.sh` (install all other things which are not python packages (semantic models))

You may also need some spacy models which are downloaded through spacy.     
You have to download these spacy models by yourself, using the command:    
`python3 -m spacy download name_of_the_model`   
The logger will also write instructions on which models you need, and how to download them.  

Be careful, you need to have spacy 2.1.3. 
If you change the version of spacy (you had a previous version) you need to reinstall xx_ent_wiki_sm model.

For neural coref errors install it as follows: https://github.com/huggingface/neuralcoref#spacystringsstringstore-size-changed-error

## Usage
For usage (parsing, lemmatization, NER, wordnet, content words, indices etc.)  see file `usage.py`

## Developer instructions

## How to use the logger
In each file you have to initialize the logger:  
```sh
from rb.utils.rblogger import Logger  
logger = Logger.get_logger() 
logger.info("info msg")
logger.warning("warning msg")  
logger.error()
```
## How to push the wheel on pip
```sh
    .\upload_to_pypi.sh
```

