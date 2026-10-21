#!/usr/bin/awk -f
# Converts the protein coding gene rows of a GENCODE GTF into the gene list
# for main.nf: chrom, start, end, strand, name (tab separated, no header),
# where name is the gene ID without its version, e.g.
#   gzip -dc gencode.v48.basic.annotation.gtf.gz |
#       awk -f scripts/gencode_genes.awk > genes.tsv

BEGIN { FS = OFS = "\t" }

$3 == "gene" && $9 ~ /gene_type "protein_coding"/ {
    match($9, /gene_id "[^"]+"/)
    id = substr($9, RSTART + 9, RLENGTH - 10)
    sub(/\..*/, "", id)
    print $1, $4, $5, $7, id
}
