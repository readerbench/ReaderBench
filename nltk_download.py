from os import getenv, path

import nltk

from rb.core.lang import Lang
from rb.utils.downloader import download_wordnet

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw')

nltk_path = getenv('NLTK_DATA')
if nltk_path is None:
    nltk_path = path.expanduser('~/nltk_data/')
download_wordnet(Lang.RO, nltk_path + "corpora/omw")
