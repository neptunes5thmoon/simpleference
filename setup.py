import os
import sys
from setuptools import find_packages, setup
import warnings

try:
    import z5py
except ModuleNotFoundError as e:
    warnings.warn(
        str(e)
        + " - 'z5py' optional dependency needs to be installed manually, it is not installable via "
        "pip",
        category=UserWarning,
    )
try:
    import libdvid
except ModuleNotFoundError as e:
    warnings.warn(
        str(e)
        + " - 'libdvid' optional dependency needs to be installed manually, it is not installable via "
        "pip",
        category=UserWarning,
    )


NAME = "neptunes5thmoon-simpleference"
DESCRIPTION = "A collection of scripts for building, training and validating Convolutional Neural Networks (CNNs) for Connectomics"
URL = "https://github.com/neptunes5thmoon/simpleference"
EMAIL = "heinrichl@janelia.hhmi.org"
AUTHOR = "Larissa Heinrich"
REQUIRES_PYTHON = ">=3.6"
VERSION = "0.1"

REQUIRED = [
    "numpy",
    "scipy",
    "dill",
    "dask",
    "toolz"
]

EXTRAS = {
    "gunpowder": [
        "gunpowder @ git+https://github.com/neptunes5thmoon/gunpowder@dist_transform_py3"
    ],
    "tensorflow": [
        "tensorflow_gpu<1.15"
    ],
    "pytorch": [
        "torch",
        "torchvision"
    ],
    "hdf5": [
        "h5py"
    ],
}

DEPENDENCY_LINKS = [
    "git+https://github.com/neptunes5thmoon/gunpowder.git@dist_transform_py3#egg=gunpowder",
]

here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, "README.md"), "r") as f:
    LONG_DESCRIPTION = "\n" + f.read()

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=['simpleference'],
    install_requires=REQUIRED,
    extras_require=EXTRAS,
    dependency_links=DEPENDENCY_LINKS,
    include_package_data=True,
    license="BSD-2-Clause",
    classifiers=[
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
)
