import sys
from os import chdir, getcwd, makedirs, rmdir
from shutil import rmtree
from subprocess import check_call
from tempfile import TemporaryDirectory

import setuptools
from setuptools.command.develop import develop
from setuptools.command.install import install


def do_post_install_tasks():
    # download spacy thing
    cwd = getcwd()
    with TemporaryDirectory() as temp_folder:
        chdir(temp_folder)
        check_call(["git", "clone", "https://github.com/explosion/spaCy.git"])
        chdir("spaCy")
        check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        check_call([sys.executable, "setup.py", "build_ext", "--inplace"])      
        check_call([sys.executable, "-m", "pip", "install", "-U", "."])
        chdir("..")
        check_call(["git", "clone", "https://github.com/huggingface/neuralcoref.git"])
        chdir("neuralcoref")
        # check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        check_call([sys.executable, "setup.py", "build_ext", "--inplace"])      
        check_call([sys.executable, "-m", "pip", "install", "-U", "."])
        chdir(cwd)
    check_call([sys.executable, "-m", "spacy", "download", "xx_ent_wiki_sm"])
    # download nltk stuff
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


class PostDevelopmentCommand(develop):
    """Post-instalation for development? mode."""
    def run(self):
        do_post_install_tasks()
        # run
        develop.run(self)


class PostInstallCommand(install):
    """Post-instalation for install mode."""
    def run(self):
        do_post_install_tasks()
        install.run(self)

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='rbpy-rb',
    version='0.9.15',
    author='Woodcarver',
    author_email='batpepastrama@gmail.com',
    description='ReaderBench library written in python',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://git.readerbench.com/ReaderBench/Readerbench-python',
    packages=setuptools.find_packages(),
    include_package_data=True,
    install_requires=[
        'bert-for-tf2>=0.14',
        'blis',
        'boto',
        'boto3',
        'botocore',
        'certifi',
        'chardet',
        'Click',
        'cymem',
        'DAWG-Python',
        'decorator',
        'docopt',
        'docutils',
        'Flask',
        'gensim',
        'googletrans',
        'idna',
        'itsdangerous',
        'Jinja2',
        'jmespath',
        'joblib',
        'MarkupSafe',
        'murmurhash',
        'networkx',
        'nltk==3.4.5',
        'pymorphy2',
        'Pyphen',
        'python-dateutil<2.8.1',
        'requests',
        's3transfer',
        'scipy',
        'sentence-splitter',
        'six',
        'sklearn',
        'smart-open',
        'srsly',
        'tensorflow>=2',
        'tensorflow-hub',
        'transformers',
        'tqdm',
        'urllib3',
        'wasabi',
        'Werkzeug',
        'wget',
        'pyLDAvis',
        'unidecode',
        'xlrd',
        'xmltodict',
      ],
    cmdclass={
        'develop': PostDevelopmentCommand,
        'install': PostInstallCommand,
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent"
    ],
)
