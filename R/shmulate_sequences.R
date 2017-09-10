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
    tolower(shmulateSeq(toupper(germline), num_muts, targetingModel = MK_RS5NF))
}

genes <- read.csv(gene_file, stringsAsFactors=FALSE)
gene_freqs <- read.csv(gene_freq_file, stringsAsFactors=FALSE)
genes <- merge(genes, gene_freqs, by="germline_name")
taxa_per_genes <- data.frame(
  germline_name=gene_freqs$germline_name,
  n_taxa=rmultinom(1, tot_taxa, prob=gene_freqs$freq)
)
genes <- merge(genes, taxa_per_genes, by="germline_name")

seqs <- apply(genes, 1, function(gene_data) {
  gene_name <- gene_data[1]
  gene_seq <- gene_data[2]
  n_germ_taxa <- as.numeric(gene_data[5])

  mutated_seqs <- replicate(n_germ_taxa, run_shmulate(gene_seq))
  seq_names <- paste0(gene_name, '-', seq(n_germ_taxa))
  if (n_germ_taxa > 0) {
    data.frame(
      germline_name=rep(gene_name, n_germ_taxa),
      # each sequence is one clonal family
      clonal_family=seq_names,
      sequence_name=seq_names,
      sequence=mutated_seqs
    )
  } else {
    data.frame()
  }
})
seq_data_frame <- do.call("rbind", seqs[lengths(seqs) > 0])

write.csv(
    seq_data_frame,
    file=output_file,
    quote=FALSE,
    row.names=FALSE
)
warnings()
