import os
import re
import sys
import textwrap
from setuptools import setup, find_packages


if sys.version_info < (3, 6):
    sys.exit(
        textwrap.dedent(
            """
            ======================================
            
            Sorry, HOTTBOX requires Python >= 3.6.
            
            ======================================
            """
        )
    )

ROOT = os.path.dirname(__file__)
VERSION_RE = re.compile(r'''__version__ = ['"]([0-9.]+)['"]''')


def get_version():
    init = open(os.path.join(ROOT, 'hottbox', '__init__.py')).read()
    return VERSION_RE.search(init).group(1)


def readme():
    with open(os.path.join(ROOT, 'README.rst')) as f:
        return f.read()


def install_requires():
    with open(os.path.join(ROOT, 'requirements.txt')) as f:
        return list(f.read().strip().split('\n'))


def extras_require():
    extra_requirements = {
        'tests': [
            'pytest>=5.0.0',
            'pytest-cov>=2.7.1',
        ],
        'docs': [
            'sphinx>=2.1.2',
            'guzzle_sphinx_theme==0.7.11',
            'numpydoc==0.9.1',
            'm2r>=0.2.1'
        ]
    }
    all_requires = [item for sublist in extra_requirements.values() for item in sublist]
    extra_requirements['all'] = all_requires
    return extra_requirements


def do_setup():
    config = dict(
        name='hottbox',
        version=get_version(),
        packages=find_packages(exclude=['docs']),
        url='https://github.com/hottbox/hottbox',
        license='Apache License 2.0',
        author='Ilya Kisil',
        author_email='ilyakisil@gmail.com',
        description='Higher Order Tensors ToolBOX',
        long_description=readme(),
        classifiers=[
            'Development Status :: 3 - Alpha',
            'Topic :: Scientific/Engineering',
            'Intended Audience :: Developers',
            'Intended Audience :: Education',
            'Intended Audience :: Science/Research',
            'License :: OSI Approved :: Apache Software License',
            'Programming Language :: Python :: 3'
        ],
        keywords=['tensor decompositions', 'machine learning'],
        python_requires='>=3.6',
        install_requires=install_requires(),
        extras_require=extras_require(),
        include_package_data=True,
        zip_safe=False
    )

    setup(**config)


if __name__ == "__main__":
    do_setup()
