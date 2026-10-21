import argparse
import contextlib
import gzip
from collections import namedtuple
import numpy as np
import pandas as pd
import pyfaidx
import torch
from spliceai2.dataset import _one_hot_encode
from spliceai2.model import SpliceAI2, VariantAnnotator
from spliceai2.score import format_scores

# Per-block geometry of an SNV patch, in coordinates of the per-variant
# window: `slice` is the input range recomputed for the block, `dirty` the
# full-resolution range where alt can differ from ref after the block, and
# `cells` the pooled range kept for gathering junction features.
_BlockGeometry = namedtuple(
    "_BlockGeometry", ["pool", "slice", "dirty", "cells"]
)


def _round_up(x, unit):
    return -(-x // unit) * unit


def _patch_geometry(model, in_len, var_idx):
    hparams = model.hparams
    unit = max(hparams.pool[:hparams.num_blocks])
    dirty = (var_idx, var_idx + 1)
    geometry = []
    for b in range(hparams.num_blocks):
        pool = hparams.pool[b]
        half_width = (hparams.kernel_size // 2) * hparams.dilation[b]
        cell_lo, cell_hi = dirty[0] // pool, -(-dirty[1] // pool)
        slice_ = (
            (cell_lo - 2 * half_width - 2) * pool // unit * unit,
            _round_up((cell_hi + 2 * half_width + 2) * pool, unit)
        )
        assert slice_[0] >= 0 and slice_[1] <= in_len
        dirty = (
            (cell_lo - half_width - 1) * pool,
            (cell_hi + half_width + 1) * pool
        )
        cells = (cell_lo - half_width - 2, cell_hi + half_width + 2)
        geometry.append(_BlockGeometry(pool, slice_, dirty, cells))
    return geometry


def _slice_windows(x, starts, length):
    return torch.stack([x[..., s:s + length] for s in starts])


def _gather_upsampled(z, cell_lo, pool, pos):
    # Matches nn.Upsample(scale_factor=pool, mode="linear") at positions pos
    # of the upsampled sequence, given pooled cells z starting at cell_lo
    src = ((pos + 0.5) * (1 / pool) - 0.5).clamp(min=0)
    idx_0 = src.floor().long()
    lam = (src - idx_0).unsqueeze(1)
    idx_0 = (idx_0 - cell_lo).clamp(0, z.shape[2] - 1)
    idx_1 = (idx_0 + 1).clamp(max=z.shape[2] - 1)
    z = z.float()
    z_0 = z.gather(2, idx_0.unsqueeze(1).expand(-1, z.shape[1], -1))
    z_1 = z.gather(2, idx_1.unsqueeze(1).expand(-1, z.shape[1], -1))
    return (1 - lam) * z_0 + lam * z_1


class SaturationAnnotator(VariantAnnotator):
    """Scores all SNVs in a range, sharing ref passes and patching alts.

    An SNV changes a single input column, and every op in the trunk is
    pointwise or local, so after each block alt differs from ref only inside
    a bounded window. Ref streams are computed once per pooling phase over a
    long window, and each alt is recomputed block by block only where it can
    differ. Results match VariantAnnotator up to floating point rounding.
    """

    def __init__(self, models, in_len, out_2d_len, k=10):
        super().__init__(models, out_2d_len, k)
        hparams = models[0].hparams
        for model in models:
            assert all(
                model.hparams[key] == hparams[key]
                for key in (
                    "num_blocks", "kernel_size", "dilation", "pool", "crop_len"
                )
            )
        self._in_len = in_len
        self._num_blocks = hparams.num_blocks
        self._unit = max(hparams.pool[:hparams.num_blocks])
        self._out_lo = hparams.crop_len // 2
        self._out_1d_len = in_len - hparams.crop_len
        self._geometry = {
            1: _patch_geometry(models[0], in_len, in_len // 2),
            -1: _patch_geometry(models[0], in_len, in_len // 2 - 1)
        }

    def forward_ref(self, x, offsets, strand):
        geometry = self._geometry[strand]
        out_hi = self._out_lo + self._out_1d_len
        extents = []
        for t in range(self._num_blocks + 1):
            bounds = []
            if t < self._num_blocks:
                bounds.append(geometry[t].slice)
            if t > 0:
                bounds.append((self._out_lo, out_hi))
            extents.append((
                min(offsets) + min(lo for lo, _ in bounds),
                max(offsets) + max(hi for _, hi in bounds)
            ))
        logits_lo = min(offsets) + self._out_lo
        logits_hi = max(offsets) + out_hi

        states = []
        for model in self._models:
            stream, logits = [], []
            ftr = model.forward_embed(x)
            for t, (lo, hi) in enumerate(extents):
                if t > 0:
                    _, ftr = model.forward_block(t - 1, ftr)
                    logits.append(model.forward_1d_block(
                        t - 1, ftr[..., logits_lo:logits_hi]
                    )[0].float())
                stream.append(ftr[0, :, lo:hi].clone())
            logits = torch.stack(logits)
            states.append({
                "stream": stream,
                "stream_lo": [lo for lo, _ in extents],
                "logits": logits,
                "logits_sum": logits.sum(dim=0),
                "logits_lo": logits_lo
            })
        return states

    def forward(self, states, offsets, x_alt, strand):
        geometry = self._geometry[strand]
        var_idx = self._in_len // 2 - (strand == -1)
        out_lo, out_len = self._out_lo, self._out_1d_len

        prob_ss_ref, prob_ss_alt, z_alt = [], [], []
        for model, state in zip(self._models, states):
            stream, stream_lo = state["stream"], state["stream_lo"]
            logits, logits_lo = state["logits"], state["logits_lo"]
            out_ss_ref = _slice_windows(
                state["logits_sum"],
                [o + out_lo - logits_lo for o in offsets],
                out_len
            )
            out_ss_alt = out_ss_ref.clone()
            patch = model.forward_embed(x_alt)
            dirty = (var_idx, var_idx + 1)
            z_alt.append([])
            for b, geom in enumerate(geometry):
                slice_lo, slice_hi = geom.slice
                ftr = _slice_windows(
                    stream[b],
                    [o + slice_lo - stream_lo[b] for o in offsets],
                    slice_hi - slice_lo
                )
                ftr[..., dirty[0] - slice_lo:dirty[1] - slice_lo] = patch
                z, ftr = model.forward_block(b, ftr)
                dirty = geom.dirty
                patch = ftr[..., dirty[0] - slice_lo:dirty[1] - slice_lo]
                z_alt[-1].append(z[
                    ...,
                    geom.cells[0] - slice_lo // geom.pool:
                    geom.cells[1] - slice_lo // geom.pool
                ])

                head_lo = max(dirty[0], out_lo)
                head_hi = min(dirty[1], out_lo + out_len)
                if head_lo < head_hi:
                    out_ss_alt[..., head_lo - out_lo:head_hi - out_lo] += (
                        model.forward_1d_block(
                            b,
                            patch[..., head_lo - dirty[0]:head_hi - dirty[0]]
                        ).float()
                        - _slice_windows(
                            logits[b],
                            [o + head_lo - logits_lo for o in offsets],
                            head_hi - head_lo
                        )
                    )
            prob_ss_ref.append(out_ss_ref.sigmoid())
            prob_ss_alt.append(out_ss_alt.sigmoid())
        prob_ss_ref = torch.stack(prob_ss_ref).mean(dim=0).float()
        prob_ss_alt = torch.stack(prob_ss_alt).mean(dim=0).float()

        subset_idxs = (
            torch.max(prob_ss_ref, prob_ss_alt).
            amax(dim=1).
            topk(self._out_2d_len, dim=1)[1].
            sort(dim=1).
            values
        )
        idxs = torch.arange(out_len, device=subset_idxs.device)
        dist_1d = (
            out_len // 2 - 1 - idxs if strand == -1 else idxs - out_len // 2
        ).expand(len(offsets), -1)
        dist_2d = dist_1d.gather(1, subset_idxs)

        win_pos = out_lo + subset_idxs
        offsets = torch.tensor(offsets, device=subset_idxs.device).unsqueeze(1)
        prob_jxn_ref, prob_jxn_alt = [], []
        for model, state, z_alt_model in zip(self._models, states, z_alt):
            ftrs_ref, ftrs_alt = [], []
            for b, geom in enumerate(geometry):
                ftr_ref = state["stream"][b + 1][
                    :, offsets + win_pos - state["stream_lo"][b + 1]
                ].transpose(0, 1)
                in_patch = (
                    (win_pos >= geom.dirty[0]) & (win_pos < geom.dirty[1])
                )
                ftr_alt = torch.where(
                    in_patch.unsqueeze(1),
                    _gather_upsampled(
                        z_alt_model[b], geom.cells[0], geom.pool, win_pos
                    ).to(ftr_ref.dtype),
                    ftr_ref
                )
                ftrs_ref.append(ftr_ref)
                ftrs_alt.append(ftr_alt)
            prob_jxn_ref.append(
                model.forward_2d_fused(ftrs_ref, subset_idxs).sigmoid()
            )
            prob_jxn_alt.append(
                model.forward_2d_fused(ftrs_alt, subset_idxs).sigmoid()
            )
        prob_jxn_ref = torch.stack(prob_jxn_ref).mean(dim=0).float()
        prob_jxn_alt = torch.stack(prob_jxn_alt).mean(dim=0).float()

        return (
            prob_ss_ref,
            prob_ss_alt,
            dist_1d,
            prob_jxn_ref,
            prob_jxn_alt,
            dist_2d
        )

    def predict_step(self, batch):
        return self._summarize(*self.forward(*batch))

    def predict_range(
            self,
            fasta,
            chrom,
            start,
            end,
            strand,
            assembly="GRCh38",
            batch_size=32,
            chunk_len=16384
    ):
        half = self._in_len // 2
        chrom_len = len(fasta[chrom])
        if start - 1 - half < 0 or end - 1 + half > chrom_len:
            raise ValueError(
                f"{chrom}:{start}-{end} needs {half} nt of flanking sequence "
                f"on each side within the chromosome (length {chrom_len})"
            )
        for chunk_start in range(start, end + 1, chunk_len):
            chunk_end = min(chunk_start + chunk_len - 1, end)
            pred = self._predict_chunk(
                fasta, chrom, chunk_start, chunk_end, strand, assembly,
                batch_size
            )
            if pred is not None:
                yield pred

    def _predict_chunk(
            self, fasta, chrom, start, end, strand, assembly, batch_size
    ):
        # Variants are grouped by the phase of their window start relative
        # to the pooling grid; each group shares one ref pass whose grid is
        # aligned with every variant window in the group.
        unit = self._unit
        strand_sign = -1 if strand in ("-", -1) else 1
        num_pos = _round_up(end - start + 1, unit)
        seq_lo = start - 1 - self._in_len // 2 - unit
        seq_hi = seq_lo + num_pos + self._in_len + 2 * unit
        chrom_len = len(fasta[chrom])
        seq = (
            "N" * (max(seq_lo, 0) - seq_lo)
            + fasta[chrom][max(seq_lo, 0):min(seq_hi, chrom_len)].seq.upper()
            + "N" * (seq_hi - min(seq_hi, chrom_len))
        )

        phases = {}
        for pos in range(start, end + 1):
            ref = seq[pos - 1 - seq_lo]
            if ref not in ("A", "C", "G", "T"):
                continue
            win_start = (
                pos - start + unit if strand_sign == 1
                else start - pos + num_pos + unit
            )
            phases.setdefault(win_start % unit, []).append(
                (pos, ref, win_start - win_start % unit)
            )
        if not phases:
            return None

        x = _one_hot_encode(seq, assembly)
        if strand_sign == -1:
            x[:4] = x[:4][::-1, ::-1]
        window_len = num_pos + self._in_len + unit
        rows, results = [], []
        autocast = (
            torch.autocast("cuda", dtype=torch.float16)
            if self.device.type == "cuda" else contextlib.nullcontext()
        )
        with torch.inference_mode(), autocast:
            x = torch.tensor(x, dtype=torch.float32, device=self.device)
            for phase, positions in sorted(phases.items()):
                states = self.forward_ref(
                    x[None, :, phase:phase + window_len],
                    [offset for _, _, offset in positions],
                    strand_sign
                )
                variants = [
                    (pos, ref, alt, offset)
                    for pos, ref, offset in positions
                    for alt in ("A", "C", "G", "T") if alt != ref
                ]
                for i in range(0, len(variants), batch_size):
                    batch = variants[i:i + batch_size]
                    x_alt = np.stack([
                        _one_hot_encode(alt, assembly)
                        for _, _, alt, _ in batch
                    ])
                    if strand_sign == -1:
                        x_alt[:, :4] = x_alt[:, :4][:, ::-1]
                    result, col = self.predict_step((
                        states,
                        [offset for _, _, _, offset in batch],
                        torch.tensor(
                            x_alt, dtype=torch.float32, device=self.device
                        ),
                        strand_sign
                    ))
                    results.append(result.float().cpu())
                    rows += [
                        (chrom, pos, ref, alt, strand)
                        for pos, ref, alt, _ in batch
                    ]
                del states

        var_df = pd.DataFrame(
            rows, columns=["chrom", "pos", "ref", "alt", "strand"]
        )
        return var_df, torch.cat(results).numpy(), col


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_folders", nargs="+")
    parser.add_argument("--fasta_file")
    parser.add_argument("--chrom")
    parser.add_argument("--start", type=int)
    parser.add_argument("--end", type=int)
    parser.add_argument("--strand", choices=["+", "-"])
    parser.add_argument("--out_file")
    parser.add_argument("--assembly", default="GRCh38")
    parser.add_argument("--in_len", type=int, default=196608)
    parser.add_argument("--out_2d_len", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--chunk_len", type=int, default=16384)
    args = parser.parse_args()

    print(args.model_folders)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = [
        SpliceAI2.load_from_checkpoint(
            f"{model_folder}/model.ckpt", map_location=device
        )
        for model_folder in args.model_folders
    ]
    annotator = SaturationAnnotator(models, args.in_len, args.out_2d_len)
    annotator = annotator.to(device).eval()
    fasta = pyfaidx.Fasta(args.fasta_file)

    open_fn = gzip.open if args.out_file.endswith(".gz") else open
    with open_fn(args.out_file, "wt") as out_file:
        for i, (var_df, result, col) in enumerate(annotator.predict_range(
                fasta,
                args.chrom,
                args.start,
                args.end,
                args.strand,
                args.assembly,
                args.batch_size,
                args.chunk_len
        )):
            var_df = format_scores(var_df, result, col)
            var_df = var_df.sort_values(["pos", "alt"])
            var_df.to_csv(out_file, sep="\t", index=False, header=i == 0)
            print(f"{args.chrom}:{var_df.pos.min()}-{var_df.pos.max()}")


if __name__ == "__main__":
    main()
