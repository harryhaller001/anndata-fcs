import os
import sys

import scanpy as sc

BASEDIR = os.path.dirname(__file__)

# Check if output file already exists
output_filename = os.path.join(os.path.dirname(BASEDIR), "test", "testing.h5ad")
if os.path.isfile(output_filename) is True:
    print("Testing anndata checkpoint already exists.")
    sys.exit(0)

# Check if input file present
input_filename = os.path.join(BASEDIR, "5k_pbmc_protein_v3_nextgem_filtered_feature_bc_matrix.h5")
if os.path.isfile(input_filename) is False:
    print("Input file does not exist.")
    sys.exit(1)

# Load anndata object from rawdata
adata = sc.read_10x_h5(
    input_filename,
    gex_only=False,
)
adata.var_names_make_unique()

# Isolate CITESeq
adata = adata[:, adata.var["feature_types"] == "Antibody Capture"]

# Subsample to 1k cells
sc.pp.subsample(adata, n_obs=1000, random_state=0)

# Save checkpoint
adata.write_h5ad(output_filename)
print(adata)
