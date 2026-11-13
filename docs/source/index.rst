
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


Converting between FCS files and ``AnnData`` objects without losing metadata. FCS keywords are mapped to ``var``
and ``uns``, and ``obs``, ``var`` and ``uns`` of ``AnnData`` objects are embedded in FCS files, so both directions
round trip. For reading and writing FCS files, ``anndata-fcs`` uses the ``flowio`` package.

.. code-block:: python

    from anndata_fcs import read_fcs, write_fcs

    adata = read_fcs("sample.fcs")
    write_fcs(adata, "copy.fcs")

See :ref:`data_model` for the layout.


Get started
-----------

You can install ``anndata-fcs`` with ``uv``::

    uv add anndata-fcs


For more details, see :ref:`install`.


Contact
-------

If you found a bug, please use the `Issue tracker <https://github.com/harryhaller001/anndata-fcs/issues>`_.



.. toctree::
    :caption: Start
    :maxdepth: 4
    :glob:

    install
    data_model
    vignette

.. toctree::
    :caption: API Documentation
    :maxdepth: 4
    :glob:

    autoapi/anndata_fcs/index
