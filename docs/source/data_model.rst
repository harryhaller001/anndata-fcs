.. Licensed under the MIT License
.. _data_model:


==========
Data model
==========

``anndata-fcs`` converts between AnnData objects and FCS files without losing metadata. This page describes where
each piece of information is stored.


FCS to AnnData
--------------

:func:`anndata_fcs.read_fcs` creates the following layout:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - AnnData
     - FCS
   * - ``X``
     - Event data as stored in the file, in the stored data type (``float32`` for ``$DATATYPE/F``, ``uint`` for
       ``I``).
   * - ``layers["preprocessed"]``
     - Only with ``preprocess=True``: event data with gain (``$PnG``), log (``$PnE``) and time (``$TIMESTEP``)
       scaling applied. ``X`` and ``uns["fcs"]`` are not modified.
   * - ``var_names``
     - Channel names (``$PnN``).
   * - ``var`` columns ``$PnS``, ``$PnR``, ``$PnE``, ...
     - Per-channel keywords as strings, one column per keyword: ``$PnS`` holds ``$P1S``, ``$P2S``, ..., and
       non-standard keywords get a column as well (e.g. ``$PnDISPLAY`` for ``P1DISPLAY``, ...). Channels without the
       keyword hold missing values.
   * - ``uns["fcs"]["text"]``
     - Global keywords of the TEXT segment as raw strings, with the names parsed by FlowIO (lower case, without
       ``$``), e.g. ``cyt``, ``date``, ``spill``.
   * - ``uns["fcs"]["analysis"]``
     - Keywords of the ANALYSIS segment.
   * - ``uns["fcs"]["version"]``, ``["filename"]``, ``["schema_version"]``
     - FCS version and file name of the source, and the version of this layout.

Keywords describing the binary layout (``$BEGINDATA``, ``$DATATYPE``, ``$BYTEORD``, ``$PnB``, ...) are not stored,
since they are regenerated when writing.

Since per-channel keywords are ``var`` columns, they stay with their channel when channels are subset, reordered or
renamed, e.g. marker labels are ``adata.var["$PnS"]``.

Keywords containing ``/`` are stored with ``/`` escaped as ``%2F`` (and ``%`` as ``%25``), because h5ad files do not
allow ``/`` in keys or column names.


AnnData to FCS
--------------

:func:`anndata_fcs.write_fcs` stores:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - AnnData
     - FCS
   * - ``X``
     - Event data as 32-bit float, the only data type FlowIO can write. Channels on log scale are converted to
       linear scale.
   * - ``var_names``
     - ``$PnN``.
   * - ``var`` columns ``$PnS``, ``$PnR``, ``$PnE``, ...
     - Per-channel keywords. Values are written as strings, missing values are not written. Set marker labels with
       e.g. ``adata.var["$PnS"] = adata.var["gene_ids"]``. ``$PnR`` defaults to the maximum of 262144 and the channel
       maximum.
   * - ``uns["fcs"]["text"]``
     - Global keywords. ``$SPILLOVER`` is restricted to channels present in ``var_names``, since it refers to
       channels by ``$PnN``. ``$ORIGINALITY`` is set to ``DataModified`` and ``$LAST_MODIFIED`` to the current time.
   * - ``obs``, ``obs_names``
     - JSON encoded in the ``ANNDATA_FCS_OBS`` keyword (``store_obs=True``).
   * - Other ``var`` columns
     - JSON encoded in the ``ANNDATA_FCS_VAR`` keyword.
   * - Other ``uns`` entries
     - JSON encoded in the ``ANNDATA_FCS_UNS`` keyword. Supported: numbers, strings, booleans, ``None``, lists,
       dictionaries with string keys, NumPy scalars and arrays.
   * - Position of each observation
     - Synthetic ``barcode_rank`` channel (``index_channel``). It allows restoring ``obs`` for files containing a
       subset of the events, e.g. events exported from a gate.

``obs`` and ``var`` columns may be numeric, boolean, string or categorical (including nullable pandas dtypes).

Only ``X`` is written. ``layers["preprocessed"]`` is not written, since it is computed from ``X`` and the keywords.
``raw`` is not written either. For scanpy objects holding the counts in ``raw``, write ``adata.raw.to_adata()``.

Any information that is lost when writing is reported by an :class:`anndata_fcs.FCSConversionWarning`. To raise an
error instead, turn the warning into an exception:

.. code-block:: python

    import warnings

    from anndata_fcs import FCSConversionWarning, write_fcs

    with warnings.catch_warnings():
        warnings.simplefilter("error", FCSConversionWarning)
        write_fcs(adata, "out.fcs")

.. warning::

   Use :func:`anndata_fcs.write_fcs` to write FCS files. :meth:`flowio.FlowData.write_fcs` only keeps the
   ``$CYT``, ``$DATE`` and ``$SPILLOVER`` keywords by default.
