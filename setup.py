from setuptools import setup,find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="Ml-Project",
    version="0.1",
    author="Siddharth",
    packages=find_packages(),
    install_requires = requirements,
)