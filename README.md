# ReaderBench Python

## Install
We recommend using virtual environments, as some packages require an exact version.   
If you only want to use the package do the following:  
1. `sudo apt-get install python3-pip, python3-venv, python3.6, python3-dev`    
2. `python3 -m venv rbenv` (create virutal environment named rbenv)
3. `source rbenv/bin/activate` (activate virtual env)
4. `pip3 uninstall setuptools && pip3 install setuptools && pip3 install --upgrade pip && pip3 install --no-cache-dir rbpy-rb`
5. Use it as in: https://git.readerbench.com/ReaderBench/readerbenchpy/blob/master/usage.py  

If you want to contribute to the code base of package:   
1. `sudo apt-get install python3-pip, python3-venv, python3.6, python3-dev`    
2. `git clone git@git.readerbench.com:ReaderBench/readerbenchpy.git && cd readerbenchpy/`  
3. `python3 -m venv rbenv` (create virutal environment named rbenv)
4. `source rbenv/bin/activate` (activate virtual env)
5. `pip3 uninstall setuptools && pip3 install setuptools && pip3 install --upgrade pip`
6. `pip3 install -r requirements.txt` 
7. `python3 -m spacy download xx_ent_wiki_sm`
8. `python3 nltk_download.py`  
Optional: prei-install model for en (otherwise most of the English processings would fail
    and ask to run this command):
9. `sudo python3 -m spacy download en_core_web_lg`


If you want to install spellchecking (hunspell) also you need this non-python libraries:
1. `sudo apt-get install libhunspell-1.6-0 libhunspell-dev hunspell-ro`
2. `pip3 install hunspell`

## Usage
For usage (parsing, lemmatization, NER, wordnet, content words, indices etc.)  see file `usage.py` from 
https://git.readerbench.com/ReaderBench/readerbenchpy    

Check main.py (`python3 main.py --help`) to see main processings available.

## Tips
You may also need some spacy models which are downloaded through spacy.     
You have to download these spacy models by yourself, using the command:    
`python3 -m spacy download name_of_the_model`   (do not install them with sudo if you are in a virtual environment)
The logger will also write instructions on which models you need, and how to download them.  
Be careful, you need to have spacy 2.1.3. 
If you change the version of spacy (you had a previous version) you need to reinstall xx_ent_wiki_sm model.

For neural coref errors install it as follows: https://github.com/huggingface/neuralcoref#spacystringsstringstore-size-changed-error

## Developer instructions

## How to use Bert

Our models are also available in the HuggingFace platform: https://huggingface.co/readerbench 

You can use them directly from HuggingFace:
```
# tensorflow
from transformers import AutoModel, AutoTokenizer, TFAutoModel
tokenizer = AutoTokenizer.from_pretrained("readerbench/RoBERT-base")
model = TFAutoModel.from_pretrained("readerbench/RoBERT-base")
inputs = tokenizer("exemplu de propoziție", return_tensors="tf")
outputs = model(inputs)

# pytorch
from transformers import AutoModel, AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("readerbench/RoBERT-base")
model = AutoModel.from_pretrained("readerbench/RoBERT-base")
inputs = tokenizer("exemplu de propoziție", return_tensors="pt")
outputs = model(**inputs)
```

or from ReaderBench:

```
from rb.core.lang import Lang
from rb.processings.encoders.bert import BertWrapper
from tensorflow import keras

bert_wrapper = BertWrapper(Lang.RO, max_seq_len=128)
inputs, bert_layer = bert_wrapper.create_inputs_and_model()
cls_output = bert_wrapper.get_output(bert_layer, "cls") # or "pool"

# Add decision layer and compile model
# eg. 
# hidden = keras.layers.Dense(..)(cls_output)
# output = keras.layers.Dense(..)(hidden)
# model = keras.Model(inputs=inputs, outputs=[output])
# model.compile(..)

bert_wrapper.load_weights() #must be called after compile

# Process inputs for model
feed_inputs = bert_wrapper.process_input(["text1", "text2", "text3"])
# feed_output = ...
# model.fit(feed_inputs, feed_output, ...)
```

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
1. `rm -r dist/`
2. `pip3 install twine wheel`
3. `./upload_to_pypi.sh`


## How to run rb/core/cscl/csv_parser.py
1. Do the installing steps from contribution
2. run `pip3 install xmltodict`
3. run `EXPORT PYTHONPATH=/add/path/to/repo/readerbenchpy/`
4. add json resources in a `jsons` directory in `readerbenchpy/rb/core/cscl/`
5. run `cd rb/core/cscl/ && python3 csv_parser.py`

## Supported Date Formats
ReaderBench is able to perform conversation analysis from chats and communities. Each utterance must have the time expressed in one of the following formats:
- %Y-%m-%d %H:%M:%S.%f %Z
- %Y-%m-%d %H:%M:%S %Z
- %Y-%m-%d %H:%M %Z
- %Y-%m-%d %H:%M:%S.%f
- %Y-%m-%d %H:%M:%S
- %Y-%m-%d %H:%M
where codifications are extracted from [Python date format codes](https://docs.python.org/3/library/datetime.html#strftime-and-strptime-format-codes).
