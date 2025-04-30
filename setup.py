import setuptools
import os

file_dir = os.path.abspath(os.path.dirname(__file__))
os.chdir(file_dir)

extensions = ['*.txt','*.dat','*.md','*.py','*.yaml']

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="MaPar",
    version="1.0",
    description="Maps to Parameters",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/NoahSailer/MaPar",
    packages=setuptools.find_packages(),
    package_data={'MaPar': extensions, 'MaPar/ckg_cgg': extensions},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=['numpy','scipy'],
)
