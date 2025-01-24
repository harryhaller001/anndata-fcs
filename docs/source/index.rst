
.. Licensed under the MIT License

.. _index:


===========
anndata-fcs
===========

.. image:: https://codecov.io/gh/harryhaller001/anndata-fcs/graph/badge.svg?token=YBZE1HZ4V1
 :target: https://codecov.io/gh/harryhaller001/anndata-fcs

.. image:: https://img.shields.io/pypi/v/anndata-fcs
   :alt: PyPI - Version

.. image:: https://img.shields.io/pypi/l/anndata-fcs
   :alt: PyPI - License

.. image:: https://img.shields.io/pypi/pyversions/anndata-fcs
   :alt: PyPI - Python Version


Converting CITESeq data to FCS file format. `anndata-fcs` implements function to convert from `AnnData` object to
`FlowData` objects, `DataFrame` objects and vice versa. For loading and generation of FCS files `anndata-fcs` uses
the implementation of the `flowio` package.


Get started
-----------

You can install ``anndata-fcs`` with ``pip``::

    pip install anndata-fcs


For more details, see :ref:`install`.


Contact
-------

If you found a bug, please use the `Issue tracker <https://github.com/harryhaller001/citeseq_to_fcs/issues>`_.



.. toctree::
    :caption: Start
    :maxdepth: 4
    :glob:

    install
    vignette

.. toctree::
    :caption: API Documentation
    :maxdepth: 4
    :glob:

    autoapi/anndata_fcs/index
