from setuptools import find_packages  # This line replaces 'from setuptools import setup'
from skbuild import setup


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
    python_requires=metadata.get("python_requires"),
    install_requires=metadata.get("install_requires"),
    classifiers=metadata.get("classifiers", "").split("\n"),
    license=metadata.get("license"),
    license_file=metadata.get("license_file"),
    packages = find_packages(),
    include_package_data=True,
    cmake_install_dir=
        "lyceanem/",

)