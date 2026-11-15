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
   * - ``raw``
     - Unprocessed event data in the stored data type (``float32`` for ``$DATATYPE/F``, ``uint`` for ``I``) and a
       copy of ``var``.
   * - ``X``
     - Unprocessed event data, like ``raw.X``. With ``preprocess=True``, gain (``$PnG``), log (``$PnE``) and time
       (``$TIMESTEP``) scaling is applied. Only ``X`` is modified: ``uns["fcs"]`` keeps the keywords of the file and
       describes ``raw``.
   * - ``var_names``
     - Channel names (``$PnN``).
   * - ``uns["fcs"]["text"]``
     - Keywords of the TEXT segment as raw strings, with the names parsed by FlowIO (lower case, without ``$``).
       This includes global keywords, e.g. ``cyt``, ``date``, ``spill``, and per-channel keywords, e.g. ``p1n``,
       ``p1s``, ``p1r``, ``p1display``.
   * - ``uns["fcs"]["analysis"]``
     - Keywords of the ANALYSIS segment.
   * - ``uns["fcs"]["source"]``
     - Keywords describing the binary layout (``$BEGINDATA``, ``$DATATYPE``, ``$BYTEORD``, ``$PnB``, ...). For
       reference only, they are regenerated when writing.
   * - ``uns["fcs"]["version"]``, ``["filename"]``, ``["preprocessed"]``, ``["schema_version"]``
     - FCS version and file name of the source, whether ``X`` was preprocessed, and the version of this layout.

``var`` has no columns for FCS files not written by ``anndata-fcs``. Use the per-channel keywords in
``uns["fcs"]["text"]`` to add columns, e.g. marker labels (``$PnS``):

.. code-block:: python

    text = adata.uns["fcs"]["text"]
    adata.var["marker"] = [text.get(f"p{n}s") for n in range(1, adata.n_vars + 1)]

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
   * - ``raw.X`` (default) or ``X``
     - Event data as 32-bit float, the only data type FlowIO can write. ``raw`` is written if present, use
       ``use_raw=False`` to write ``X`` instead. Channels on log scale are converted to linear scale. If ``X`` is
       written and was preprocessed (``uns["fcs"]["preprocessed"]``), ``$PnE``, ``$PnG``, ``$PnR`` and ``$TIMESTEP``
       are adjusted to describe the preprocessed values.
   * - ``raw.var_names`` (or ``var_names`` with ``use_raw=False``)
     - ``$PnN``.
   * - ``uns["fcs"]["text"]``
     - Keywords. Per-channel keywords (``p<n>s``, ``p<n>r``, ...) are assigned to the written channels by their
       name (``p<n>n``), so they follow their channel if channels are removed or reordered. Per-channel keywords that
       cannot be assigned are not written. Use ``pns_key`` to take marker labels (``$PnS``) from a ``var`` column.
       ``$PnR`` defaults to the maximum of 262144 and the channel maximum. ``$SPILLOVER`` is restricted to channels present in ``var_names``, since it refers to
       channels by ``$PnN``. ``$ORIGINALITY`` is set to ``DataModified`` and ``$LAST_MODIFIED`` to the current time.
   * - ``obs``, ``obs_names``
     - JSON encoded in the ``ANNDATA_FCS_OBS`` keyword (``store_obs=True``).
   * - ``var`` (``raw.var``) columns
     - JSON encoded in the ``ANNDATA_FCS_VAR`` keyword.
   * - Other ``uns`` entries
     - JSON encoded in the ``ANNDATA_FCS_UNS`` keyword. Supported: numbers, strings, booleans, ``None``, lists,
       dictionaries with string keys, NumPy scalars and arrays.
   * - Position of each observation
     - Synthetic ``barcode_rank`` channel (``index_channel``). It allows restoring ``obs`` for files containing a
       subset of the events, e.g. events exported from a gate.

``obs`` and ``var`` columns may be numeric, boolean, string or categorical (including nullable pandas dtypes).

Only one of ``X`` and ``raw`` is written. If they hold different event data (e.g. preprocessed or transformed ``X``,
or a subset of channels), a :class:`anndata_fcs.FCSConversionWarning` is emitted for the one that is not written.

.. warning::

   Use :func:`anndata_fcs.write_fcs` to write FCS files. :meth:`flowio.FlowData.write_fcs` only keeps the
   ``$CYT``, ``$DATE`` and ``$SPILLOVER`` keywords by default.
