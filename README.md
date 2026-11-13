# anndata-fcs

![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/harryhaller001/anndata-fcs/testing.yml)
![PyPI - Version](https://img.shields.io/pypi/v/anndata-fcs)
![PyPI - License](https://img.shields.io/pypi/l/anndata-fcs)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/anndata-fcs)


Converting between FCS files and `AnnData` objects without losing metadata. FCS keywords are mapped to `var` and
`uns`, and `obs`, `var` and `uns` of `AnnData` objects are embedded in FCS files, so both directions round trip. For
reading and writing FCS files, `anndata-fcs` uses the `flowio` package.

### Get started

You can install `anndata-fcs` with `pip`:

`pip install anndata-fcs`


### Example

```python
import scanpy as sc

from anndata_fcs import read_fcs, write_fcs

# Load CITE-seq data
adata = sc.read_10x_h5("5k_pbmc_protein_v3_nextgem_filtered_feature_bc_matrix.h5", gex_only=False)
adata_citeseq = adata[:, adata.var["feature_types"] == "Antibody Capture"].copy()

# Write FCS file. obs, var and uns are embedded, a "barcode_rank" channel maps events to cells.
write_fcs(adata_citeseq, "citeseq.fcs")

# Read FCS file (also files from cytometers or gates exported from flow cytometry software)
adata_restored = read_fcs("citeseq.fcs")
```

Channel keywords (`$PnS`, `$PnR`, `$PnV`, ...) are stored as `var` columns (`pns`, `pnr`, `pnv`, ...) and all other
keywords in `adata.uns["fcs"]["text"]`. See the [data model](https://harryhaller001.github.io/anndata-fcs/data_model.html)
for details.

> Use `anndata_fcs.write_fcs` to write files. `flowio.FlowData.write_fcs` only keeps a few keywords by default.


### Development

Install development version of `anndata-fcs` with `git`:

`git clone https://github.com/harryhaller001/anndata-fcs.git`


To setup development environment create python virtual environment:

```bash
python3 -m virtualenv venv
source venv/bin/activate
```


Use `make` to setup dependencies:

```bash
make install

# activate pre-commit
pre-commit install
```

Run checks with `make`:

```bash
# Run all checks
make check

# Run formatting
make format

# Run unit testing with pytest
make testing

# Run type checks with ty
make typing
```

### Dependencies

- `flowio>=1.4` (Documentation: https://flowio.readthedocs.io/en/latest)
- `anndata` (Documentation: https://anndata.readthedocs.io/en/latest)
- `numpy`, `pandas`, `scipy`


### License

Please refer to the [LICENSE](https://github.com/harryhaller001/anndata-fcs/blob/main/LICENSE) file
