from setuptools import setup, find_packages
setup(
    name = "tensorflow_helpers",
    version = "0.1.12",
    packages = find_packages(),
    install_requires=['numpy', ], # 'tensorflow'
    extras_require = {'gpu': ['tensorflow-gpu>=0.12.0']},
    author="Alexey Romanov",
    author_email="aromanov@cs.uml.edu",
    description="Different helpers for building Tensorflow models",
    url="https://github.com/text-machine-lab/tensorflow-helpers",
)
