import numpy as np
import pandas as pd
import pyfaidx
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader
from spliceai2.dataset import VariantDataset, _one_hot_encode
from spliceai2.model import SpliceAI2, VariantAnnotator
from spliceai2.saturate import (
    SaturationAnnotator, _gather_upsampled, _patch_geometry, _pool_upsampled
)

IN_LEN = 196608


def _random_model(seed):
    # Larger weights than the default init so that variant effects are
    # well above floating point noise
    torch.manual_seed(seed)
    model = SpliceAI2(num_features=8, num_blocks=20, crop_len=131072)
    for m in model.modules():
        if isinstance(m, nn.Conv1d):
            nn.init.normal_(m.weight, std=0.2)
            nn.init.normal_(m.bias, std=0.2)
    return model.eval()


def _random_seq(length, seed):
    rng = np.random.default_rng(seed)
    return "".join(rng.choice(list("ACGT"), length))


def _assert_topk_close(result, expected, col, k=10, atol=1e-5):
    # Tied delta scores can swap ranks, or swap entries in and out of the
    # top k, under floating point rounding. Delta scores are compared in rank
    # order, and entries are compared as sets, excluding ties with the k-th.
    for effect in dict.fromkeys("_".join(c.split("_")[:2]) for c in col):
        idxs = [i for i, c in enumerate(col) if c.startswith(f"{effect}_")]
        result_topk = result[:, idxs].reshape(len(result), -1, k)
        expected_topk = expected[:, idxs].reshape(len(expected), -1, k)
        np.testing.assert_allclose(
            result_topk[:, 0], expected_topk[:, 0], rtol=0, atol=atol
        )
        for res, exp in zip(result_topk, expected_topk):
            cutoff = max(res[0, -1], exp[0, -1]) + atol
            res, exp = res[:, res[0] > cutoff], exp[:, exp[0] > cutoff]
            np.testing.assert_allclose(
                res[:, np.lexsort(res[1:][::-1])],
                exp[:, np.lexsort(exp[1:][::-1])],
                rtol=0,
                atol=atol
            )


@pytest.mark.parametrize("pool", [1, 2, 4, 8, 16])
def test_gather_upsampled(pool):
    z = torch.randn(2, 3, 40)
    up = nn.Upsample(scale_factor=pool, mode="linear")(z)
    pos = torch.randint(2 * pool, 38 * pool, (2, 50))
    cell_lo = 7
    torch.testing.assert_close(
        _gather_upsampled(z, cell_lo, pool, pos + cell_lo * pool),
        up.gather(2, pos.unsqueeze(1).expand(-1, 3, -1))
    )


@pytest.mark.parametrize("pool,ratio", [(1, 1), (1, 2), (2, 1), (2, 2), (16, 1)])
def test_pool_upsampled(pool, ratio):
    z = torch.randn(2, 3, 40)
    up = nn.Upsample(scale_factor=pool, mode="linear")(z)
    torch.testing.assert_close(
        _pool_upsampled(z, pool, ratio), nn.AvgPool1d(pool * ratio)(up)
    )


def test_forward_2d_fused():
    model = _random_model(0)
    ftrs_2d = [torch.randn(2, 8, 64) for _ in range(20)]
    idxs = torch.randint(0, IN_LEN, (2, 64)).sort(dim=1).values
    with torch.inference_mode():
        torch.testing.assert_close(
            model.forward_2d_fused(ftrs_2d, idxs),
            model.forward_2d_gathered(ftrs_2d, idxs)
        )


@pytest.mark.parametrize("var_idx", [IN_LEN // 2, IN_LEN // 2 - 1])
def test_patch_geometry_covers_alt_changes(var_idx):
    model = _random_model(0)
    x_ref = torch.tensor(
        _one_hot_encode(_random_seq(IN_LEN, 0), "GRCh38")[None]
    ).float()
    x_alt = x_ref.clone()
    x_alt[0, :4, var_idx] = x_ref[0, :4, var_idx].roll(1)
    with torch.inference_mode():
        ftr_ref = model.forward_embed(x_ref)
        ftr_alt = model.forward_embed(x_alt)
        for b, geom in enumerate(_patch_geometry(model, IN_LEN, var_idx)):
            u_ref = model._downsamplers[b](ftr_ref)
            u_alt = model._downsamplers[b](ftr_alt)
            z_ref, ftr_ref = model.forward_block(b, ftr_ref)
            z_alt, ftr_alt = model.forward_block(b, ftr_alt)
            for ref, alt, (lo, hi) in [
                (u_ref, u_alt, geom.patch),
                (z_ref, z_alt, geom.changed),
                (ftr_ref, ftr_alt, geom.dirty)
            ]:
                changed = (alt != ref).any(dim=1)[0].nonzero()
                assert len(changed) > 0
                assert lo <= changed.min()
                assert changed.max() < hi


@pytest.mark.parametrize("strand", ["+", "-"])
def test_saturation_matches_variant_annotator(tmp_path, strand):
    start = IN_LEN // 2 + 20
    seq = _random_seq(IN_LEN + 64, 1)
    seq = seq[:start + 2] + "N" + seq[start + 3:]
    fasta_file = tmp_path / "test.fa"
    fasta_file.write_text(">chrT\n" + "\n".join(
        seq[i:i + 60] for i in range(0, len(seq), 60)
    ) + "\n")
    models = [_random_model(1), _random_model(2)]

    preds = list(SaturationAnnotator(models, IN_LEN, 1024).predict_range(
        pyfaidx.Fasta(str(fasta_file)),
        "chrT",
        start,
        start + 5,
        strand,
        batch_size=4,
        chunk_len=4
    ))
    var_df = pd.concat([p[0] for p in preds], ignore_index=True)
    result = np.concatenate([p[1] for p in preds])
    col = preds[0][2]
    assert len(var_df) == 15
    assert start + 3 not in var_df["pos"].values
    assert (var_df["ref"] != var_df["alt"]).all()

    var_dl = DataLoader(
        VariantDataset(
            var_df.assign(assembly="GRCh38"), str(fasta_file), IN_LEN
        ),
        batch_size=5
    )
    var_annotator = VariantAnnotator(models, 1024).eval()
    with torch.inference_mode():
        expected = torch.cat(
            [var_annotator.predict_step(batch)[0] for batch in var_dl]
        ).numpy()
    assert expected[:, col.index("donor_gain_delta_score_0")].min() > 1e-4
    _assert_topk_close(result, expected, col)
