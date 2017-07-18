# Take genes.csv file and run shmulateSeq on each sequence
library(shazam)

arg <- commandArgs(TRUE)
gene_file <- arg[1]
gene_freq_file <- arg[2]
tot_taxa <- as.numeric(arg[3])
seed <- as.numeric(arg[4])
min_pct_mut <- as.numeric(arg[5])
max_pct_mut <- as.numeric(arg[6])
output_file <- arg[7]

set.seed(seed)

run_shmulate <- function(germline) {
    # Mutate sequences based on uniform random number of mutations b/t 5--20%
    # of sequence length.
    seq_len <- nchar(germline)
    num_muts <- floor(runif(1, min=min_pct_mut * seq_len, max =max_pct_mut * seq_len))
    tolower(shmulateSeq(toupper(germline), num_muts))
}

genes <- read.csv(gene_file, stringsAsFactors=FALSE)
gene_freqs <- read.csv(gene_freq_file, stringsAsFactors=FALSE)

genes <- merge(genes, gene_freqs, by="germline_name")

seqs <- apply(genes, 1, function(gene_data) {
  gene_name <- gene_data[1]
  gene_seq <- gene_data[2]
  gene_freq <- as.numeric(gene_data[3])

  n_germ_taxa <- floor(tot_taxa * gene_freq + 1)
  print(paste(gene_name, n_germ_taxa))
  mutated_seqs <- replicate(n_germ_taxa, run_shmulate(gene_seq))
  seq_names <- paste0(gene_name, '-', seq(n_germ_taxa))
  data.frame(
    germline_name=gene_name,
    locus='',
    species='',
    # each sequence is one clonal family
    clonal_family=seq_names,
    group='',
    subject='',
    sequence_name=seq_names,
    sequence=mutated_seqs
  )
})
seq_data_frame <- do.call("rbind", seqs)

write.csv(
    seq_data_frame,
    file=output_file,
    quote=FALSE,
    row.names=FALSE
)
warnings()
