.. Licensed under the MIT License
.. _data_model:


==========
Data model
==========

``anndata-fcs`` converts between AnnData objects and FCS files without losing metadata. This page describes where
each piece of information is stored.


FCS to AnnData
--------------

:func:`anndata_fcs.read_fcs` and :func:`anndata_fcs.fcs_to_anndata` create the following layout:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - AnnData
     - FCS
   * - ``X``
     - Event data in the stored data type (``float32`` for ``$DATATYPE/F``, ``uint`` for ``I``). With
       ``preprocess=True``, gain (``$PnG``), log (``$PnE``) and time (``$TIMESTEP``) scaling is applied.
   * - ``var_names``
     - Channel names (``$PnN``).
   * - ``var["pn<suffix>"]``
     - Per-channel keywords as raw strings, e.g. ``$PnS`` → ``pns``, ``$PnR`` → ``pnr``, ``$PnV`` → ``pnv``,
       ``P1DISPLAY`` → ``pndisplay``. Missing keywords are missing values.
   * - ``uns["fcs"]["text"]``
     - All other keywords of the TEXT segment (lower case, without ``$``), e.g. ``cyt``, ``date``, ``spill``.
   * - ``uns["fcs"]["analysis"]``
     - Keywords of the ANALYSIS segment.
   * - ``uns["fcs"]["source"]``
     - Keywords describing the binary layout (``$BEGINDATA``, ``$DATATYPE``, ``$BYTEORD``, ``$PnB``, ...). For
       reference only, they are regenerated when writing.
   * - ``uns["fcs"]["version"]``, ``["filename"]``, ``["preprocessed"]``, ``["schema_version"]``
     - FCS version and file name of the source, whether ``X`` was preprocessed, and the version of this layout.

Keywords containing ``/`` are stored with ``/`` escaped as ``%2F`` (and ``%`` as ``%25``), because h5ad files do not
allow ``/`` in keys.


AnnData to FCS
--------------

:func:`anndata_fcs.write_fcs` and :func:`anndata_fcs.anndata_to_fcs` store:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - AnnData
     - FCS
   * - ``X``
     - Event data as 32-bit float, the only data type FlowIO can write. Channels on log scale are converted to linear
       scale.
   * - ``var_names``, ``var["pn<suffix>"]``
     - ``$PnN`` and per-channel keywords. Use ``pns_key`` to take marker labels (``$PnS``) from another column.
       ``$PnR`` defaults to the maximum of 262144 and the channel maximum.
   * - ``uns["fcs"]["text"]``
     - Global keywords.
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

.. warning::

   Use :func:`anndata_fcs.write_fcs` to write FCS files. :meth:`flowio.FlowData.write_fcs` only keeps the
   ``$CYT``, ``$DATE`` and ``$SPILLOVER`` keywords by default.
