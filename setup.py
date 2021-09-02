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
    # cwd = getcwd()
    # with TemporaryDirectory() as temp_folder:
    #     chdir(temp_folder)
    #     check_call(["git", "clone", "https://github.com/huggingface/neuralcoref.git"])
    #     chdir("neuralcoref")
    #     check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    #     check_call([sys.executable, "setup.py", "build_ext", "--inplace"])      
    #     check_call([sys.executable, "-m", "pip", "install", "-U", "."])
    #     chdir(cwd)
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
    version='0.10.30',
    python_requires='>=3.6,<3.9',
    author='Woodcarver',
    author_email='batpepastrama@gmail.com',
    description='ReaderBench library written in python',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://git.readerbench.com/ReaderBench/Readerbench-python',
    packages=setuptools.find_packages(),
    include_package_data=True,
    install_requires=required,
    data_files=[("",["requirements.txt"])],
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
