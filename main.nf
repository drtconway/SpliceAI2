// Saturation mutagenesis over a list of gene ranges with spliceai2-saturate.
//
// Genes are split into tiles, and each tile is scored by one task. Tasks run
// one at a time because a single task saturates the GPU. Outputs are hard
// links to the task outputs in the work folder, so -resume works without
// doubling disk use. Tiles whose output already exists in the output folder
// are also skipped, with or without -resume.

params.genes = "${projectDir}/gencode-50-genes.tsv"
params.fasta = "${System.getProperty('user.home')}/data/hg38/hg38_primary.fa"
params.models = "${projectDir}/SpliceAI2_models/spliceai2_multispecies_13M_01 ${projectDir}/SpliceAI2_models/spliceai2_multispecies_13M_02"
params.saturate = "${projectDir}/.venv/bin/spliceai2-saturate"
params.tile_len = 49152
params.batch_size = 32
params.in_len = 196608
params.max_retries = 1

// Splits a gene into tiles, clipped to the positions that have in_len / 2 nt
// of flanking sequence on each side within the chromosome
def geneTiles(row, chromLens) {
    def (chrom, start, end, strand, name) = row
    def half = (params.in_len as long).intdiv(2)
    def tileLen = params.tile_len as int
    def lo = Math.max(start as long, half + 1)
    def hi = Math.min(end as long, chromLens[chrom] + 1 - half)
    if (lo > hi) {
        log.warn "Skipping ${name} (${chrom}:${start}-${end}): too close to the chromosome end"
        return []
    }
    if (lo != (start as long) || hi != (end as long)) {
        log.warn "Clipping ${name} (${chrom}:${start}-${end}) to ${chrom}:${lo}-${hi}"
    }
    return (lo..hi).step(tileLen).collect { s ->
        [name, chrom, s, Math.min(s + tileLen - 1, hi), strand]
    }
}

process SATURATE {
    tag "${name}:${start}-${end}"
    maxForks 1
    errorStrategy { task.attempt <= (params.max_retries as int) ? 'retry' : 'ignore' }
    publishDir params.outdir, mode: 'link'

    input:
    tuple val(name), val(chrom), val(start), val(end), val(strand)

    output:
    path "${name}/${name}.${start}-${end}.tsv.gz"

    script:
    """
    mkdir ${name}
    ${params.saturate} \\
        --model_folders ${params.models} \\
        --fasta_file ${params.fasta} \\
        --chrom ${chrom} --start ${start} --end ${end} --strand '${strand}' \\
        --batch_size ${params.batch_size} \\
        --in_len ${params.in_len} \\
        --out_file ${name}/${name}.${start}-${end}.tsv.gz
    """
}

workflow {
    def chromLens = file("${params.fasta}.fai").readLines().collectEntries { line ->
        def fields = line.split('\t')
        [fields[0], fields[1] as long]
    }

    tiles = channel.fromPath(params.genes)
        .splitCsv(sep: '\t')
        .flatMap { row -> geneTiles(row, chromLens) }
        .filter { name, _chrom, start, end, _strand ->
            !file("${params.outdir}/${name}/${name}.${start}-${end}.tsv.gz").exists()
        }

    SATURATE(tiles)
}
