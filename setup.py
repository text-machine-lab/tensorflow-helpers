from setuptools import setup, find_packages
setup(
    name = "Tensorflow Helpers",
    version = "0.1.8",
    packages = find_packages(),
    install_requires=['numpy', ], # 'tensorflow'
    extras_require = {'gpu': ['tensorflow-gpu>=0.12.0']},
    author="Alexey Romanov",
    author_email="aromanov@cs.uml.edu",
    description="Different helpers for building Tensorflow models",
    url="https://github.com/text-machine-lab/tensorflow-helpers",
)
