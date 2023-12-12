from setuptools import setup, find_packages

with open('README.md', 'r') as fh:
    long_description = fh.read()
setup(
    name="ontosample",
    description="Ontosample is a package that offers different sampling techniques for OWL ontologies.",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "matplotlib>=3.3.4",
        "owlready2>=0.40",
        "torch>=1.7.1",
        "pandas>=1.5.0",
        "sortedcontainers>=2.4.0",
        "owlapy>=0.1.0"],
    author='Alkid Baci',
    author_email='alkid1baci@gmail.com',
    url='https://github.com/alkidbaci/OntoSample',
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: GNU Affero General Public License v3 or later (AGPLv3+)",
        "Topic :: Scientific/Engineering :: Artificial Intelligence"],
    python_requires='>=3.9.18',
    long_description=long_description,
    long_description_content_type="text/markdown",
)
