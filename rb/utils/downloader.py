import json
import os
import sys
import zipfile
from typing import List, Union

import wget
from rb.core.lang import Lang
from rb.utils.rblogger import Logger

LINKS = {
    Lang.EN: {
        'models': {
            'coca': {
                'link': "https://nextcloud.readerbench.com/index.php/s/TxamWx6Er9G2wDo/download",
                'version': "https://nextcloud.readerbench.com/index.php/s/npzePeQrpHLE8N6/download"
            },
            'tasa': {
                'link': "https://nextcloud.readerbench.com/index.php/s/CSqbbkxm4En2KKE/download",
                'version': "https://nextcloud.readerbench.com/index.php/s/cXikjDcESTtsJDC/download"
            },
            'enea_tasa': {
                'link': "https://nextcloud.readerbench.com/index.php/s/fd8Ss5NtL6yDoYj/download",
                'version': "https://nextcloud.readerbench.com/index.php/s/5zJxGKxsLSN4YRd/download"
            },
        },
        'spacy': {},
        'aoa': 'https://nextcloud.readerbench.com/index.php/s/estDka8fYiSNWzj/download'
    },
    Lang.RO: {
        'models': {
            'diacritics': "https://nextcloud.readerbench.com/index.php/s/pfC25G64JgxcfZS/download",
            'readme': "https://nextcloud.readerbench.com/index.php/s/g9etLBeSTmKxjM8/download",
        },
        'spacy': {
            'ro_ud_ft_ner': {
                'link': "https://nextcloud.readerbench.com/index.php/s/5mMb98BXkctjXcP/download",
                'version': "https://nextcloud.readerbench.com/index.php/s/gR6zetbDdfnMMEC/download"
            }
        },
        'wordnet': "https://nextcloud.readerbench.com/index.php/s/7tDka2CSGYeJqgC/download"
    },
    Lang.RU: {
        'spacy': {
            'ru_ud_ft': "https://nextcloud.readerbench.com/index.php/s/nbErkdgbxmgiRCG/download"
        }
    },
    Lang.FR: {
        'models': {
            'le_monde_small': {
                'link': "https://nextcloud.readerbench.com/index.php/s/6MsKa9TtmnAxCc4/download",
                'version': "https://nextcloud.readerbench.com/index.php/s/2jLd8ZiCfmMo88X/download"
            },
        },
    },
}

logger = Logger.get_logger()

def download_folder(link: str, destination: str):
    os.makedirs(destination, exist_ok=True)     
    filename = wget.download(link, out=destination, bar=wget.bar_thermometer)
    logger.info('Downloaded {}'.format(filename))
    if zipfile.is_zipfile(filename):
        logger.info('Extracting files from {}'.format(filename))
        with zipfile.ZipFile(filename,"r") as zip_ref:
            zip_ref.extractall(destination)
        os.remove(filename)

def download_file(link: str, destination: str):
    os.makedirs(destination, exist_ok=True)
    filename = wget.download(link, out=destination, bar=wget.bar_thermometer)
    logger.info('Downloaded {}'.format(filename))

def download_model(lang: Lang, name: Union[str, List[str]]) -> bool:
    if isinstance(name, str):
        name = ['models', name]
    if not lang in LINKS:
        logger.info('{} not supported.'.format(lang))
        return False
    path = "/".join(name)
    root = LINKS[lang]
    for key in name:
        if key not in root:
            logger.info('Remote path not found {} ({}).'.format(path, key))
            return False
        root = root[key]
    logger.info("Downloading model {} for {} ...".format(path, lang.value))
    link = root['link'] if isinstance(root, dict) else root
    folder = "resources/{}/{}".format(lang.value, "/".join(name[:-1]))
    download_folder(link, folder)
    return True

def download_spacy_model(lang: Lang, name: str) -> bool:
    return download_model(lang, ['spacy', name])

def download_wordnet(lang: Lang, folder: str) -> bool:
    if lang not in LINKS:
        logger.info('{} not supported.'.format(lang))
        return False
    if 'wordnet' not in LINKS[lang]:
        logger.info('No WordNet found')
        return False
    link = LINKS[lang]['wordnet']
    download_folder(link, folder)

def download_aoa(lang: Lang) -> bool:
    path = "resources/{}/aoa/AoA.csv".format(lang.value)
    if os.path.isfile(path):
        logger.info('File already downloaded')
        return True
    if lang not in LINKS:
        logger.info('{} not supported.'.format(lang))
        return False
    if 'aoa' not in LINKS[lang]:
        logger.info('No AoA found')
        return False
    link = LINKS[lang]['aoa']
    download_file(link, "resources/{}/aoa/".format(lang.value))
        
def check_spacy_version(lang: Lang, name: str) -> bool:
    return check_version(lang, ['spacy', name])

def check_version(lang: Lang, name: Union[str, List[str]]) -> bool:
    if isinstance(name, str):
        name = ['models', name]
    path = "/".join(name)
    folder = "resources/{}/{}".format(lang.value, path)
    try:
        local_version = read_version(folder + "/version.txt")
    except:
        logger.info('Local model {} for {} not found.'.format(path, lang))
        return True
        
    if not lang in LINKS:
        logger.info('{} not supported.'.format(lang))
        return False
    root = LINKS[lang]
    for key in name:
        if key not in root:
            logger.info('Remote path not found {} ({}).'.format(path, key))
            return False
        root = root[key]
    if isinstance(root, dict):
        filename = wget.download(root['version'], out="resources/", bar=wget.bar_thermometer)
        try:
            remote_version = read_version(filename)
        except:
            logger.info('Error reading remote version for {} ({})'.format(path, lang))
            return False
        return newer_version(remote_version, local_version)
    else:
        return True
    
def read_version(filename: str) -> str:
    with open(filename, "r") as f:
        return f.readline()

def newer_version(remote_version: str, local_version: str) -> bool:
    remote_version = remote_version.split(".")
    local_version = local_version.split(".")
    for a, b in zip(remote_version, local_version):
        if int(a) > int(b):
            return True
        if int(a) < int(b):
            return False
    return False   
    
        

if __name__ == "__main__":
    download_model(Lang.EN, 'coca')
