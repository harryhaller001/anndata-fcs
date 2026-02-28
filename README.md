# anndata-fcs

![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/harryhaller001/anndata-fcs/testing.yml)
![PyPI - Version](https://img.shields.io/pypi/v/anndata-fcs)
![PyPI - License](https://img.shields.io/pypi/l/anndata-fcs)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/anndata-fcs)


Converting CITESeq data to FCS file format. `anndata-fcs` implements function to convert from `AnnData` object to
`FlowData` objects, `DataFrame` objects and vice versa. For loading and generation of FCS files `anndata-fcs` uses
the implementation of the `flowio` package.

### Get started

You can install `anndata-fcs` with `pip`:

`pip install anndata-fcs`


### Example

```python
# Load anndata object
adata = sc.read_10x_h5(
    filename="5k_pbmc_protein_v3_nextgem_filtered_feature_bc_matrix.h5",
    gex_only=False,
)

# Filter citeseq data
adata_citeseq = adata[:, adata.var["feature_types"] == "Antibody Capture"]

# Convert data to fcs format
fcs_data = anndata_to_fcs(adata_citeseq)

# Save fcs file
fcs_data.write_fcs("citeseq.fcs")
```


### Development

Install development version of `anndata-fcs` with `git`:

`git clone git+https://github.com/harryhaller001/citeseq_to_fcs.git`


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

# Run type checks with mypy
make typing
```

### Dependencies

Required dependencies:

- `flowio` (Documentation: https://flowio.readthedocs.io/en/latest)
- `anndata` (Documentation: https://anndata.readthedocs.io/en/latest)
- `matplotlib` (Documentation: https://matplotlib.org/stable/)

Optional dependencies:

- For density plots, `scipy` is required.


### License

Please refer to the [LICENSE](https://github.com/harryhaller001/citeseq_to_fcs/blob/main/LICENSE) file
