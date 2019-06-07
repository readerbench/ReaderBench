# ReaderBench Python

## Install
`pip3 install --user rbpy-rb`
./install.sh

You may also need some spacy models which are downloaded through spacy.     
You have to download these spacy models by yourself, using the command:    
`python3 -m spacy download name_of_the_model`   
The logger will also write instructions on which models you need, and how to download them.  

## Usage

For tokenization, lemmatiozation, pos tagging, use:  
```sh
from rb.parser.spacy_parser import SpacyParser
from rb.core.lang import Lang
from rb.core.document import Document

nlp_ro = SpacyParser.get_instance().get_model(Lang.RO)

test_text_ro = "Am mers repede la magazinul frumos."

# tokenize
docs_ro = nlp_ro(test_text_ro)
# print all attributes of token objects
print(dir(docs_ro[0]))

for token in docs_ro:
    print(token.lemma_, token.is_stop, token.tag_, token.pos_)
```

For indices use:  
```sh
from rb.core.lang import Lang  
from rb.core.document import Document  

doc = Document(Lang.EN, 'This is a sample document. It can contain multiple sentences and paragraphs')
```

See `examples.py` for usage examples.

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

