import setuptools
from setuptools import find_packages

setuptools.setup(
    name="cabinet-robot",
    version="0.0.1",
    author="Thomas Lips",
    author_email="thomas.lips@ugent.be",
    description="TODO",
    install_requires=[
        "numpy",
        "roslibpy",
    ],
    packages=find_packages(),
)
