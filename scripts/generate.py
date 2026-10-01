"""Generate the dataset with the LBM solver. (Owner: Aryan)

Usage: python scripts/generate.py --n 1500 --workers 8 --seed 0
           [--stratify-by kind,re] [--out DIR]

Writes DATA_DIR/dataset.npz, meta.json and norm.json (schema: contract.py
and scripts/make_fake_dataset.py). Deterministic from --seed. Samples that
do not converge or have tau < TAU_MIN are dropped and counted in meta.json.
Each sample's blockage ratio D / CHANNEL_HEIGHT is stored in meta.json.
--stratify-by kind,re (default) splits within each (kind, Re bucket) with
SPLIT_FRACTIONS; the per-bucket split counts are written to meta.json.
Triangles (OOD_KINDS) only ever go to idx_ood.
"""

if __name__ == "__main__":
    raise NotImplementedError("solver session")
