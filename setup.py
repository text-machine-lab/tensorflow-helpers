from setuptools import setup, find_packages
setup(
    name = "Tensorflow Helpers",
    version = "0.1.0",
    packages = find_packages(),
    install_requires=['numpy', 'tensorflow'],
    author="Alexey Romanov",
    author_email="aromanov@cs.uml.edu",
    description="different helpers for building Tensorflow models",
    url="https://github.com/text-machine-lab/tensorflow-helpers",
)