// Scan of every 1 nt deletion over a list of gene ranges with spliceai2.
//
// Genes are split into blocks of block_len deleted positions, and each block
// is scored by one task: scripts/deletions.py writes the block's deletions as
// a spliceai2 input tsv, and spliceai2 scores them. As in main.nf, outputs
// are hard links to the task outputs in the work folder, and blocks whose
// output already exists in the output folder are skipped.

params.genes = "${projectDir}/gencode-50-genes.tsv"
params.fasta = "${System.getProperty('user.home')}/data/hg38/hg38_primary.fa"
params.models = "${projectDir}/SpliceAI2_models/spliceai2_multispecies_13M_01 ${projectDir}/SpliceAI2_models/spliceai2_multispecies_13M_02"
params.python = "${projectDir}/.venv/bin/python"
params.spliceai2 = "${projectDir}/.venv/bin/spliceai2"
params.block_len = 10000
params.in_len = 196608
params.compile = false
params.max_retries = 1
params.dels_outdir = "${params.outdir}/dels"

// Splits a gene into blocks of deleted positions, clipped to the positions
// whose deletion has in_len / 2 nt of flanking sequence on each side within
// the chromosome. Deletions are anchored on the preceding base, so the lower
// bound is one more than for substitutions.
def geneBlocks(row, chromLens) {
    def (chrom, start, end, strand, name) = row
    def half = (params.in_len as long).intdiv(2)
    def blockLen = params.block_len as int
    def lo = Math.max(start as long, half + 2)
    def hi = Math.min(end as long, chromLens[chrom] + 1 - half)
    if (lo > hi) {
        log.warn "Skipping ${name} (${chrom}:${start}-${end}): too close to the chromosome end"
        return []
    }
    if (lo != (start as long) || hi != (end as long)) {
        log.warn "Clipping ${name} (${chrom}:${start}-${end}) to ${chrom}:${lo}-${hi}"
    }
    return (lo..hi).step(blockLen).collect { s ->
        [name, chrom, s, Math.min(s + blockLen - 1, hi), strand]
    }
}

process DELETIONS {
    tag "${name}:${start}-${end}"
    errorStrategy { task.attempt <= (params.max_retries as int) ? 'retry' : 'ignore' }
    publishDir params.dels_outdir, mode: 'link'

    input:
    tuple val(name), val(chrom), val(start), val(end), val(strand)

    output:
    path "${name}/${name}.${start}-${end}.dels.tsv.gz"

    script:
    def prefix = "${name}/${name}.${start}-${end}.dels"
    """
    mkdir ${name}
    ${params.python} ${projectDir}/scripts/deletions.py \\
        --fasta_file ${params.fasta} \\
        --chrom ${chrom} --start ${start} --end ${end} --strand '${strand}' \\
        --out_file ${prefix}.tsv
    if [ \$(wc -l < ${prefix}.tsv) -gt 1 ]; then
        ${params.spliceai2} \\
            --model_folders ${params.models} \\
            --var_tsv_file ${prefix}.tsv \\
            --fasta_file ${params.fasta} \\
            --in_len ${params.in_len} \\
            ${params.compile ? '' : '--no_compile'}
        gzip -c ${prefix}.tsv.spliceai2 > ${prefix}.tsv.gz
    else
        # No scorable deletions (e.g. all N), so spliceai2 has nothing to do
        gzip -c ${prefix}.tsv > ${prefix}.tsv.gz
    fi
    """
}

workflow {
    def chromLens = file("${params.fasta}.fai").readLines().collectEntries { line ->
        def fields = line.split('\t')
        [fields[0], fields[1] as long]
    }

    blocks = channel.fromPath(params.genes)
        .splitCsv(sep: '\t')
        .flatMap { row -> geneBlocks(row, chromLens) }
        .filter { name, _chrom, start, end, _strand ->
            !file("${params.dels_outdir}/${name}/${name}.${start}-${end}.dels.tsv.gz").exists()
        }

    DELETIONS(blocks)
}
