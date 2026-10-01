import numpy as np

from windtunnel import dataset


def test_stratified_split_hits_fractions_with_small_groups():
    rng = np.random.default_rng(0)
    meta = [{"kind": k, "re": float(re)} for k in ("circle", "ellipse", "rectangle")
            for re in rng.uniform(10, 40, 20)] + [{"kind": "triangle", "re": 20.0}] * 6
    split, _ = dataset.split_indices(meta, np.random.default_rng(1))
    assert (len(split["val"]), len(split["test"]), len(split["ood"])) == (6, 6, 6)
    assert len(split["train"]) == 48
    allidx = np.concatenate([split[k] for k in ("train", "val", "test", "ood")])
    assert len(np.unique(allidx)) == len(meta)
