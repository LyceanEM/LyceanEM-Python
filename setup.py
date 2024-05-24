from setuptools import setup
import os
import sys

# move to the directory of the setup.py file

# Read metadata from setup.cfg
def read_setup_cfg():
    from configparser import ConfigParser

    config = ConfigParser()
    config.read("setup.cfg")
    return config["metadata"]

metadata = read_setup_cfg()
print("hi")
setup(
    name=metadata.get("name"),
    version=metadata.get("version"),
    author=metadata.get("author"),
    author_email=metadata.get("author_email"),
    description=metadata.get("description"),
    long_description=metadata.get("long_description"),
    long_description_content_type=metadata.get("long_description_content_type"),
    url=metadata.get("url"),
    packages=metadata.get("packages"),
    python_requires=metadata.get("python_requires"),
    install_requires=metadata.get("install_requires"),
    classifiers=metadata.get("classifiers", "").split("\n"),
    license=metadata.get("license"),
    license_file=metadata.get("license_file"),
    include_package_data=metadata.get("include_package_data", False),
)
print("hi")
os.system("cmake -S ./CUDA_source -B ./CUDA_source/build/ ")
print("hi")
os.system("cmake --build ./CUDA_source/build/")