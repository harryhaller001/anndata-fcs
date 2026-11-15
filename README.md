# anndata-fcs

![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/harryhaller001/anndata-fcs/testing.yml)
![PyPI - Version](https://img.shields.io/pypi/v/anndata-fcs)
![PyPI - License](https://img.shields.io/pypi/l/anndata-fcs)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/anndata-fcs)


Converting between FCS files and `AnnData` objects without losing metadata. Per-channel FCS keywords are stored as `var`
columns (`$PnS`, `$PnR`, ...) and global keywords in `uns["fcs"]`, and `obs` and `var` of `AnnData` objects are embedded
in FCS files. For reading and writing FCS files, `anndata-fcs` uses the `flowio` package.

### Get started

You can install `anndata-fcs` with `uv`:

`uv add anndata-fcs`


### Example

```python
from anndata_fcs import read_fcs, write_fcs

# Read FCS file
adata = read_fcs("input.fcs")

# Write FCS file
write_fcs(adata, "output.fcs")
```


### Development

Install development version of `anndata-fcs` with `git`:

`git clone https://github.com/harryhaller001/anndata-fcs.git`


To setup development environment create python virtual environment:

```bash
uv venv
```


Use `make` to setup dependencies:

```bash
make install
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
- `numpy`
- `pandas`


### License

MIT
