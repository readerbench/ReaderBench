import os
import zipfile
from typing import List, Union
from urllib.request import urlopen

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
            'muse': {
                'link': "https://nextcloud.readerbench.com/index.php/s/68gXrooiYq3CKcg/download",
                'version': "https://nextcloud.readerbench.com/index.php/s/9SkpScAGA4YmM3q/download"
            },
            'chains': {
                'mlp': {
                    "link": "https://nextcloud.readerbench.com/index.php/s/pWmgazgq5bmkC4e/download",
                    "version": "https://nextcloud.readerbench.com/index.php/s/Y3tE8587CbN94GJ/download"
                },
                'linear': {
                    "link": "https://nextcloud.readerbench.com/index.php/s/FLCRBYFfSYLAgyT/download",
                    "version": "https://nextcloud.readerbench.com/index.php/s/8zdybMG7ai5C6ot/download"
                },
            },
        },
        'spacy': {},
        'aoa': 'https://nextcloud.readerbench.com/index.php/s/estDka8fYiSNWzj/download',
        'syllabified_dict': "https://nextcloud.readerbench.com/index.php/s/9zzptWpYT9BbxYS/download",
        'wordlists': {
            'link': 'https://nextcloud.readerbench.com/index.php/s/XyeiJCSripBWpx7/download',
            'version': 'https://nextcloud.readerbench.com/index.php/s/xyyGMqeLwTkBRms/download'
        }
    },
    Lang.RO: {
        'models': {
            'diacritice': {
                'small': {
                    "link": "https://nextcloud.readerbench.com/index.php/s/HbNRckT5LHa4cc4/download",
                    "version": "https://nextcloud.readerbench.com/index.php/s/wtcqmsb6CmpnwdN/download"
                },
                'base': {
                    "link": "https://nextcloud.readerbench.com/index.php/s/Y56BLDLtYZ6WRRa/download",
                    "version": "https://nextcloud.readerbench.com/index.php/s/GRsJP7yFreeicaR/download"
                }
            },
            'sentiment': {
                'small': {
                    "link": "https://nextcloud.readerbench.com/index.php/s/QNYRtoCEQgri499/download",
                    "version": "https://nextcloud.readerbench.com/index.php/s/CF5TmPeEKFNib4H/download"
                },
                'base': {
                    "link": "https://nextcloud.readerbench.com/index.php/s/FwWy9TWNHymJ5Hy/download",
                    "version": "https://nextcloud.readerbench.com/index.php/s/oBLaKLtDMfEaaxG/download"
                },
                'large': {
                    "link": "https://nextcloud.readerbench.com/index.php/s/pHXqLQ8HXar44WT/download",
                    "version": "https://nextcloud.readerbench.com/index.php/s/aEnbRMiscMjRxcL/download"
                },
            },
            'gec': {
                "transformer_768": {
                    "link": "https://nextcloud.readerbench.com/index.php/s/CPAS95MNyZGsKas/download",
                    "version": "https://nextcloud.readerbench.com/index.php/s/GaETKtdRkFmpDDH/download"
                },
                'transformer_64': {
                    'version': 'https://nextcloud.readerbench.com/index.php/s/W9FB2dX5ykHqiLk/download',
                    'link': 'https://nextcloud.readerbench.com/index.php/s/ZrP2mZszCXim4EW/download'
                }
            },
            'readme': {
                "link": "https://nextcloud.readerbench.com/index.php/s/Sj94ysrmDDxX8YH/download",
                "version": "https://nextcloud.readerbench.com/index.php/s/QXd8847qLz5tNAa/download"
            }
        },
        'spacy': {
            'ro_ud_ft_ner': {
                'link': "https://nextcloud.readerbench.com/index.php/s/5mMb98BXkctjXcP/download",
                'version': "https://nextcloud.readerbench.com/index.php/s/gR6zetbDdfnMMEC/download"
            },
            'ud_tags': {
                'link': 'https://nextcloud.readerbench.com/index.php/s/8WEiAWDSP83sBtx/download',
                'version': 'https://nextcloud.readerbench.com/index.php/s/afE4ZYAMoya9Ekp/download'
            }
        },
        'wordnet': "https://nextcloud.readerbench.com/index.php/s/7tDka2CSGYeJqgC/download",
        'syllabified_dict': "https://nextcloud.readerbench.com/index.php/s/opifiTCqXNzRsxF/download",
        'bert':{
            'small': {
                'link': 'https://nextcloud.readerbench.com/index.php/s/WbtmEn9XebfbBFy/download',
                'version': 'https://nextcloud.readerbench.com/index.php/s/zG2qyqbtongNrKW/download'
            },
            'base': {
                'link': 'https://nextcloud.readerbench.com/index.php/s/Hnn68d6QMr8oKxr/download',
                'version': 'https://nextcloud.readerbench.com/index.php/s/mNG8YbSj7QzpSfK/download'
            },
            'large': {
                'link': 'https://nextcloud.readerbench.com/index.php/s/DeH4mBtZFajKRan/download',
                'version': 'https://nextcloud.readerbench.com/index.php/s/MdMKFDFKJ5feMnN/download'
            },
            'multi_cased_base': {
                'link': 'https://nextcloud.readerbench.com/index.php/s/yGY3AoRYWR2McyD/download',
                'version': 'https://nextcloud.readerbench.com/index.php/s/yWetLtdEz2D4i6f/download'
            },
        },
        'wordnet': "https://nextcloud.readerbench.com/index.php/s/7tDka2CSGYeJqgC/download",
        'wordlists': {
            'link': "https://nextcloud.readerbench.com/index.php/s/RPMSmg9AB2NGdXe/download",
            'version':  'https://nextcloud.readerbench.com/index.php/s/ZWJ34FHy5Zwa65F/download'
        },
        'scoring': {
            'link': 'https://nextcloud.readerbench.com/index.php/s/eMri3i2GHLrQZ24/download',
            'version': 'https://nextcloud.readerbench.com/index.php/s/iTETzcskXZGsBWo/download'
        },
        'classifier': {
            'link': 'https://nextcloud.readerbench.com/index.php/s/DKPXSXfXmtC2445/download',
            'version': 'https://nextcloud.readerbench.com/index.php/s/sMMEqkpiMeX4Pbx/download'
        }
    },
    Lang.RU: {
        'models': {
            'rnc_wikipedia': {
                'link': "https://nextcloud.readerbench.com/index.php/s/PJCmkb2focXiMfd/download",
                'version': "https://nextcloud.readerbench.com/index.php/s/d9Jz7XcxRYpJQgq/download"
            },
            'muse': {
                'link': "https://nextcloud.readerbench.com/index.php/s/Q8nxYTfjsZ6YonY/download",
                'version': "https://nextcloud.readerbench.com/index.php/s/2w7yLygTSTbzXAR/download"
            },
        },
        'spacy': {
            'ru_ud_ft': {
                'link': "https://nextcloud.readerbench.com/index.php/s/bWCztgwzdnXowc7/download",
                'version': "https://nextcloud.readerbench.com/index.php/s/TsWjd6KoLjypJQC/download"
            }
        },
        'wordlists': {
            'link': 'https://nextcloud.readerbench.com/index.php/s/erWmo9KQJ4sc2zs/download',
            'version': 'https://nextcloud.readerbench.com/index.php/s/L64godyDAzZwznw/download'
        }
    },
    Lang.FR: {
        'models': {
            'le_monde_small': {
                'link': "https://nextcloud.readerbench.com/index.php/s/T8GfqaQfxinpZtS/download",
                'version': "https://nextcloud.readerbench.com/index.php/s/2jLd8ZiCfmMo88X/download"
            },
            'le_monde': {
                'link': "https://nextcloud.readerbench.com/index.php/s/xSKTSqYQBc2nEgq/download",
                'version': "https://nextcloud.readerbench.com/index.php/s/2LMtn2mP9LHQsaY/download"
            },
            'muse': {
                'link': "https://nextcloud.readerbench.com/index.php/s/f73eNfiWxjDzJx2/download",
                'version': "https://nextcloud.readerbench.com/index.php/s/tmmkHmCnBYZncQj/download"
            },
        },
    },
    Lang.ES: {
        'models': {
            'jose_antonio': {
                'link': "https://nextcloud.readerbench.com/index.php/s/6J9j4NZgmbALxtE/download",
                'version': "https://nextcloud.readerbench.com/index.php/s/D36i64WMJfdRboD/download"
            },
        },
    },
    Lang.NL: {
        'models': {
            'wiki': {
                'link': "https://nextcloud.readerbench.com/index.php/s/7mDZ4s6YmzQCAQw/download",
                'version': "https://nextcloud.readerbench.com/index.php/s/qtNAtLQEaAqEXSM/download"
            },
        },
    },
    Lang.DE: {
        'models': {
            'wiki': {
                'link': "https://nextcloud.readerbench.com/index.php/s/Cdg9cNcFaoFbbk7/download",
                'version': "https://nextcloud.readerbench.com/index.php/s/f5ZtjyQsjSHCTJX/download"
            },
        },
    },
}

logger = Logger.get_logger()

def download(link: str, destination: str) -> str:
    with urlopen(link) as webpage:
        filename = webpage.info().get_filename()
        content = webpage.read()
    with open(os.path.join(destination, filename), 'wb' ) as f:
        f.write(content)
    return os.path.join(destination, filename)
    
def download_folder(link: str, destination: str):
    os.makedirs(destination, exist_ok=True)     
    filename = download(link, destination)
    logger.info('Downloaded {}'.format(filename))
    if zipfile.is_zipfile(filename):
        logger.info('Extracting files from {}'.format(filename))
        with zipfile.ZipFile(filename, "r") as zip_ref:
            zip_ref.extractall(destination)
        os.remove(filename)


def download_file(link: str, destination: str):
    os.makedirs(destination, exist_ok=True)
    filename = download(link, destination)
    logger.info('Downloaded {}'.format(filename))


def download_model(lang: Lang, name: Union[str, List[str]]) -> bool:
    if isinstance(name, str):
        name = ['models', name]
    if lang not in LINKS:
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
    
# TODO check version?
def download_wordlist(lang: Lang) -> bool:
    base_path = os.path.join('resources', lang.value)
    path = os.path.join(base_path, 'wordlists')
    version_path = os.path.join(path, 'version.txt')
    
    if os.path.isfile(version_path) and os.path.isdir(path):
        return True
    if lang not in LINKS:
        logger.info('{} not supported for tags'.format(lang.value))
        return False
    if 'wordlists' not in LINKS[lang]:
        logger.info('No wordlists exist for {}'.format(lang.value))
        return False
    
    link = LINKS[lang]['wordlists']['link']
    logger.info('Downloading wordlists for {} ...'.format(lang.value))
    download_folder(link, base_path)
    logger.info('Downloaded wordlists for {} successfully'.format(lang.value))
    return True

def download_tags(lang: Lang) -> bool:
    path = "resources/{}/{}".format(lang.value, 'spacy')
    full_path = path + '/ud_tags'
    version_path = full_path + '/version.txt'
    if os.path.isfile(version_path):
        return True
    if lang not in LINKS:
        logger.info('{} not supported for tags'.format(lang.value))
        return False
    if 'spacy' not in LINKS[lang]:
        logger.info('Spacy does not exists for {}'.format(lang.value))
        return False
    if 'ud_tags' not in LINKS[lang]['spacy']:
        logger.info('No tags for {}'.format(lang.value))
        return False
    link = LINKS[lang]['spacy']['ud_tags']['link']
    # if you chnage this path you also have to change it in ro_pos_feature_extractor
    logger.info('Downloading tags for {} ...'.format(lang.value))
    
    download_folder(link, path)
    logger.info('Downloaded tags for {} successfully'.format(lang.value))
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

def download_syllabified_dict(lang: Lang) -> bool:
    path = "resources/{}/dict/syllabified_dict.dict".format(lang.value)
    if os.path.isfile(path):
        logger.info('Dictionary already downloaded.')
        return True
    if lang not in LINKS:
        logger.info('{} not supported,'.format(lang))
        return False
    if 'syllabified_dict' not in LINKS[lang]:
        logger.info('No syllabified dictionary found.')
        return False
    link = LINKS[lang]['syllabified_dict']
    download_file(link, "resources/{}/dict".format(lang.value))
    return True

def download_scoring(lang: Lang) -> bool:
    path = "resources/{}/scoring/svr_gamma.p".format(lang.value)
    if os.path.isfile(path):
        logger.info('File already downloaded')
        return True
    if lang not in LINKS:
        logger.info('{} not supported.'.format(lang))
        return False
    if 'scoring' not in LINKS[lang]:
        logger.info('No scoring model found')
        return False
    link = LINKS[lang]['scoring']['link']
    download_file(link, "resources/{}/scoring/".format(lang.value))
    return True


def download_classifier(lang: Lang) -> bool:
    path = "resources/{}/classifier/svr.p".format(lang.value)
    if os.path.isfile(path):
        logger.info('File already downloaded')
        return True
    if lang not in LINKS:
        logger.info('{} not supported.'.format(lang))
        return False
    if 'classifier' not in LINKS[lang]:
        logger.info('No classifier model found')
        return False
    link = LINKS[lang]['classifier']['link']
    download_file(link, "resources/{}/classifier/".format(lang.value))
    return True

def check_spacy_version(lang: Lang, name: str) -> bool:
    return check_version(lang, ['spacy', name])


def check_version(lang: Lang, name: Union[str, List[str]]) -> bool:
    logger.info('Checking version for model {}, {}'.format(name, lang.value))
    if isinstance(name, str):
        name = ['models', name]
    path = "/".join(name)
    folder = "resources/{}/{}".format(lang.value, path)
    try:
        local_version = read_version(folder + "/version.txt")
    except FileNotFoundError:
        logger.info('Local model {} for {} not found.'.format(path, lang))
        return True

    if lang not in LINKS:
        logger.error('{} not supported.'.format(lang))
        return False
    root = LINKS[lang]
    for key in name:
        if key not in root:
            logger.error('Remote path not found {} ({}).'.format(path, key))
            return False
        root = root[key]
    if isinstance(root, dict):
        filename = download(root['version'], "resources/")
        try:
            remote_version = read_version(filename)
        except FileNotFoundError:
            logger.warning('Error reading remote version for {} ({})'.format(path, lang))
            return False
        return newer_version(remote_version, local_version)
    else:
        logger.error('Could not find version link in links json')
        return True


def read_version(filename: str) -> str:
    with open(filename, "r") as f:
        return f.readline()


def newer_version(remote_version: str, local_version: str) -> bool:
    remote_version = remote_version.split(".")
    local_version = local_version.split(".")

    for a, b in zip(remote_version, local_version):
        if int(a) > int(b):
            logger.info('Remote version {} is ahead of local version {}'.format(remote_version, local_version))
            return True
        if int(a) < int(b):
            logger.info('Remote version {} is behind of local version {}'.format(remote_version, local_version))
            return False
    logger.info('Remote version {} is the same as local version {}'.format(remote_version, local_version))
    return False   
    

if __name__ == "__main__":
    download_model(Lang.EN, 'coca')
