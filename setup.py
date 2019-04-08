#!/usr/bin/env python
import os
import io
import re
from setuptools import setup, find_packages

def read(*names, **kwargs):
    with io.open(
        os.path.join(os.path.dirname(__file__), *names),
        encoding=kwargs.get("encoding", "utf8")
    ) as fp:
        return fp.read()

def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")

VERSION = find_version('nlpblock', '__init__.py')
long_description = read('README.md')

setup_info = dict(
    name='nlpblock',
    version=VERSION,
    author='Tae Hwan Jung(@graykode)',
    author_email='nlkey2022@gmail.com',
    url='https://github.com/graykode/nlpblock',
    description='Use All NLP models abstracted to block level with Pytorch',
    long_description=long_description,
    long_description_content_type='text/markdown',
    license='MIT',
    install_requires=[ 'tqdm', 'torch', 'numpy', 'torchsummary'],

    # Package info
    packages=find_packages(exclude=('tests',)),
    keywords='pytorch nlp nlpblock',
    python_requires='>=3.5',
)

setup(**setup_info)