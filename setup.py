import platform
import sys
from os import chdir, getcwd, makedirs, rmdir
from shutil import rmtree
from subprocess import check_call
from tempfile import TemporaryDirectory

import setuptools
from setuptools.command.develop import develop
from setuptools.command.install import install

with open('requirements.txt') as f:
    required = f.read().splitlines()

def do_post_install_tasks():
    check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

    if platform.system() == "Darwin" and platform.processor() == "arm":
        check_call([sys.executable, "-m", "pip", "install", 
            "--pre", "torch", "-f", 
            "https://download.pytorch.org/whl/nightly/cpu/torch_nightly.html"])
        check_call([sys.executable, "-m", "pip", "install", "tensorflow-macos"]) 
        check_call([sys.executable, "-m", "pip", "install", "tensorflow-metal"]) 
        check_call(["brew", "install", "rust"]) 
        cwd = getcwd()
        try:
            import tokenizers
        except:
            with TemporaryDirectory() as temp_folder:
                chdir(temp_folder)
                check_call(["git", "clone", "https://github.com/huggingface/tokenizers"])
                chdir("tokenizers/bindings/python")
                check_call([sys.executable, "-m", "pip", "install", "setuptools_rust"])
                check_call([sys.executable, "setup.py", "install"])      
                chdir(cwd)
        try:
            import transformers
        except:
            check_call([sys.executable, "-m", "pip", "install", "git+https://github.com/huggingface/transformers"])
    else:
        check_call([sys.executable, "-m", "pip", "install", "tensorflow", "pytorch", "transformers"])

    #  download nltk stuff
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
        develop.run(self)
        do_post_install_tasks()

class PostInstallCommand(install):
    """Post-instalation for install mode."""
    def run(self):
        install.run(self)
        do_post_install_tasks()
        
with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='rbpy-rb',
    version='0.11.2',
    python_requires='>=3.6,<3.10',
    author='Woodcarver',
    author_email='batpepastrama@gmail.com',
    description='ReaderBench library written in python',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/readerbench/ReaderBench',
    packages=setuptools.find_packages(),
    include_package_data=True,
    install_requires=required,
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
