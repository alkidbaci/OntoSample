from setuptools import setup, find_packages

with open('README.md', 'r') as fh:
    long_description = fh.read()
setup(
    name="ontosample",
    description="Ontosample is a package that offers different sampling techniques for OWL ontologies.",
    version="0.2.5",
    packages=find_packages(),
    install_requires=[
        "matplotlib>=3.3.4",
        "owlready2>=0.40",
        "torch>=1.7.1",
        "pandas>=1.5.0",
        "sortedcontainers>=2.4.0",
        "owlapy==1.3.0",
        "requests>=2.31.0",
        "deap>=1.3.1"],
    author='Alkid Baci',
    author_email='alkid1baci@gmail.com',
    url='https://github.com/alkidbaci/OntoSample',
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering"],
    python_requires='>=3.10.13',
    long_description=long_description,
    long_description_content_type="text/markdown",
)
