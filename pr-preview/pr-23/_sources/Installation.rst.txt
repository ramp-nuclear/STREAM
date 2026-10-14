Installation
============

For the time being, we're working with **conda**. Inside a conda environment, run::
    
    conda env update -f requirements.yml
    conda env update -f tests/requirements.yml  # To run the tests
    conda env update -f docs/requirements.yml  # To build the docs
    pip install .

in the package root, after the project has already been cloned.

TODO: We intend to create a better support using our own conda recipe in the future.
