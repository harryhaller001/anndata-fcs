
.. Licensed under the MIT License
.. _install:


============
Installation
============

.. highlight:: console
.. _setuptools: https://pypi.org/project/setuptools/


For installation of this package you need to have Python 3.8 or newer installed. You can install ``anndata-fcs`` with ``pip``::

    pip install anndata-fcs


Development
-----------

Install development version of `anndata-fcs` with::

    git clone git+https://github.com/harryhaller001/citeseq_to_fcs.git


To setup development environment create python virtual environment::

    python3 -m virtualenv venv
    source venv/bin/activate


Use `make` to setup dependencies::

    make install

    # activate pre-commit
    pre-commit install


Run checks with `make`::

    # Run all checks
    make check

    # Run formatting
    make format

    # Run unit testing with pytest
    make testing

    # Run type checks with mypy
    make typing


Dependencies
------------

Required dependencies:

- `flowio` (Documentation: https://flowio.readthedocs.io/en/latest)
- `anndata` (Documentation: https://anndata.readthedocs.io/en/latest)
- `matplotlib` (Documentation: https://matplotlib.org/stable/)

Optional dependencies:

- For density plots, `scipy` is required.
