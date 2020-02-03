import setuptools
from setuptools.command.develop import develop
from setuptools.command.install import install
from subprocess import check_call

def do_post_install_tasks():
    # download spacy thing
   
    check_call("pip3 install https://github.com/explosion/spacy-models/releases/download/xx_ent_wiki_sm-2.1.0/xx_ent_wiki_sm-2.1.0.tar.gz".split())
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
    version='0.6.12',
    author='Woodcarver',
    author_email='batpepastrama@gmail.com',
    description='ReaderBench library written in python',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://git.readerbench.com/ReaderBench/Readerbench-python',
    packages=setuptools.find_packages(),
    include_package_data=True,
    dependency_links=['https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.1.0/en_core_web_sm-2.1.0.tar.gz'],
    install_requires=[
        'spacy==2.1.3',
        'pymorphy2-dicts==2.4.393442.3710985',
        'pymorphy2',
        'nltk==3.4.5',
        'gensim==3.8.1',
        'tensorflow==1.14',
        'sklearn',
        'networkx',
        'joblib',
        'blis<0.3.0',
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
        'idna',
        'itsdangerous',
        'Jinja2',
        'jmespath',
        'jsonschema<3.0.0',
        'MarkupSafe',
        'murmurhash',
        'neuralcoref',
        'plac<1.0.0',
        'preshed<2.1.0',
        'Pyphen',
        'python-dateutil<2.8.1',
        'requests',
        's3transfer',
        'scipy',
        'six',
        'smart-open',
        'srsly',
        'thinc<7.1.0',
        'tqdm',
        'urllib3',
        'wasabi',
        'Werkzeug',
        'wget',
        'pyLDAvis',
        'keras',
        'unidecode'
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
