#! /usr/bin/env python

# Author: Alexander Fabisch <afabisch@googlemail.com>
# License: BSD 3 clause

from setuptools import setup

import batch_gmr


def setup_package():
    setup(
        name="batch-gmr",
        version=batch_gmr.__version__,
        author="Akshay L Chandra",
        author_email="achandrapro@googlemail.com",
        url="https://github.com/acl21/batch-gmr",
        description="Batch Gaussian Mixture Regression",
        long_description=open("README.rd").read(),
        license="new BSD",
        classifiers=[
            "Intended Audience :: Science/Research",
            "Intended Audience :: Developers",
            "Programming Language :: Python",
            "Topic :: Software Development",
            "Topic :: Scientific/Engineering",
            "Operating System :: Microsoft :: Windows",
            "Operating System :: POSIX",
            "Operating System :: Unix",
            "Operating System :: MacOS",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
        ],
        packages=["batch-gmr"],
        install_requires=[
            "numpy==1.24.5",
            "scipy==1.14.1",
            "gmr==1.6.2",
            "torch==2.0.1",
        ],
    )


if __name__ == "__main__":
    setup_package()
