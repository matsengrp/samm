# Take genes.csv file and run shmulateSeq on each sequence
source('shmulate/R/SHMulate.R')
source('shmulate/R/SHMulate_Functions.R')

arg <- commandArgs(TRUE)
gene_file <- arg[1]
seed <- arg[2]
min_pct_mut <- as.numeric(arg[3])
max_pct_mut <- as.numeric(arg[4])
output_file <- arg[5]

set.seed(seed)

run_shmulate <- function(germline) {
    # Mutate sequences based on uniform random number of mutations b/t 5--20%
    # of sequence length.
    seq_len <- nchar(germline)
    num_muts <- floor(runif(1, min=min_pct_mut * seq_len, max =max_pct_mut * seq_len))
    tolower(shmulateSeq(toupper(germline), num_muts))
}

genes <- read.csv(gene_file, stringsAsFactors=FALSE)

seqs <- unname(sapply(genes$germline_sequence, function(seq) run_shmulate(seq)))
seq_names <- unname(sapply(genes$germline_name, function(name) paste0(name, '-', runif(1))))

seq_data_frame <- 
    data.frame(germline_name=genes$germline_name,
               locus='',
               species='',
               clonal_family=genes$germline_name,
               group='',
               subject='',
               sequence_name=seq_names,
               sequence=seqs)

write.csv(
    seq_data_frame,
    file=output_file,
    quote=FALSE,
    row.names=FALSE
)

