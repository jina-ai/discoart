import sys
from os import path

from setuptools import find_packages
from setuptools import setup

if sys.version_info < (3, 7, 0):
    raise OSError(f'DiscoArt requires Python >=3.7, but yours is {sys.version}')

try:
    pkg_name = 'discoart'
    libinfo_py = path.join(pkg_name, '__init__.py')
    libinfo_content = open(libinfo_py, 'r', encoding='utf8').readlines()
    version_line = [l.strip() for l in libinfo_content if l.startswith('__version__')][
        0
    ]
    exec(version_line)  # gives __version__
except FileNotFoundError:
    __version__ = '0.0.0'

try:
    with open('README.md', encoding='utf8') as fp:
        _long_description = fp.read()
except FileNotFoundError:
    _long_description = ''

setup(
    name=pkg_name,
    packages=find_packages(),
    version=__version__,
    include_package_data=True,
    description='Create Disco Diffusion artworks in one line',
    author='Jina AI',
    author_email='hello@jina.ai',
    license='MIT',
    url='https://github.com/jina-ai/discoart',
    download_url='https://github.com/jina-ai/discoart/tags',
    long_description=_long_description,
    long_description_content_type='text/markdown',
    zip_safe=False,
    setup_requires=['setuptools>=18.0', 'wheel'],
    install_requires=[
        'numpy',
        'lpips',
        'ftfy',
        'docarray[common]>=0.14.6,<0.30.0',
        'pyyaml',
        'open_clip_torch',
        'pyspellchecker',
        'jina>=3.7.3',
        'guided-diffusion-sdk',
        'resize-right-sdk',
        'openai-clip',
        'packaging',
        'wandb',
        'timm',
        'protobuf',
    ],
    extras_require={
        'test': [
            'pytest',
            'pytest-timeout',
            'pytest-mock',
            'pytest-cov',
            'pytest-repeat',
            'pytest-reraise',
            'mock',
            'pytest-custom_exit_code',
            'black==22.3.0',
            'torch==1.9.0',
            'torchvision==0.10.0',
        ],
    },
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Unix Shell',
        'Environment :: Console',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
        'Topic :: Database :: Database Engines/Servers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Internet :: WWW/HTTP :: Indexing/Search',
        'Topic :: Scientific/Engineering :: Image Recognition',
        'Topic :: Multimedia :: Video',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    project_urls={
        'Source': 'https://github.com/jina-ai/discoart/',
        'Tracker': 'https://github.com/jina-ai/discoart/issues',
    },
    keywords='discoart diffusion art dalle disco-diffusion generative-art creative-ai cross-modal multi-modal artwork',
)
