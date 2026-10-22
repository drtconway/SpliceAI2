"""Writes a spliceai2 input tsv with every 1 nt deletion in a range.

Deletions are written VCF style, anchored on the preceding base: deleting
position p gives pos = p - 1, ref = the bases at p - 1 and p, and alt = the
base at p - 1. Positions where either base is not A, C, G or T are skipped.
"""
import argparse
import pyfaidx


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fasta_file", required=True)
    parser.add_argument("--chrom", required=True)
    parser.add_argument(
        "--start", type=int, required=True,
        help="first deleted position (1-based)"
    )
    parser.add_argument(
        "--end", type=int, required=True,
        help="last deleted position (1-based, inclusive)"
    )
    parser.add_argument("--strand", choices=["+", "-"], required=True)
    parser.add_argument("--out_file", required=True)
    args = parser.parse_args()

    # seq[i] is the base at position start - 1 + i
    fasta = pyfaidx.Fasta(args.fasta_file)
    seq = fasta[args.chrom][args.start - 2:args.end].seq.upper()
    with open(args.out_file, "w") as f:
        f.write("chrom\tpos\tref\talt\tstrand\n")
        for i in range(1, len(seq)):
            ref = seq[i - 1:i + 1]
            if set(ref) <= set("ACGT"):
                f.write(
                    f"{args.chrom}\t{args.start + i - 2}\t{ref}\t{ref[0]}\t"
                    f"{args.strand}\n"
                )


if __name__ == "__main__":
    main()
