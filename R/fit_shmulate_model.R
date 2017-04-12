# Run shmulate given command line arguments
source('shmulate/R/SHMulate.R')
source('shmulate/R/SHMulate_Functions.R')

# Packages needed:
# igraph
# plyr
# seqinr
# shazam
# Subset example data to one isotype and sample as a demo

arg <- commandArgs(TRUE)
seq.file <- arg[1]
gene.file <- arg[2]
output.file <- arg[3]

seq <- read.csv(seq.file)
genes <- read.csv(gene.file)
seq_genes <- merge(seq, genes, by="germline_name")

orig_num_seqs <- nrow(seq_genes)
filter_mask <- slideWindowDb(db = seq_genes, sequenceColumn="sequence_name", germlineColumn="germline_sequence", mutThresh=6, windowSize=10)
print(paste("Original number of sequences", orig_num_seqs))
print(paste("Post-Filter number of sequences", orig_num_seqs - sum(filter_mask)))

shm_model_type <- "RS"
# Create model using only silent mutations
# shm_model_type <- "S"

sub_model <- createSubstitutionMatrix(
    seq_genes,
    model=shm_model_type,
    sequenceColumn="sequence",
    germlineColumn="germline_sequence",
    vCallColumn="germline_name",
)

mut_model <- createMutabilityMatrix(
    seq_genes,
    sub_model,
    model=shm_model_type,
    sequenceColumn="sequence",
    germlineColumn="germline_sequence",
    vCallColumn="germline_name",
)

target_model <- createTargetingMatrix(sub_model, mut_model)

write.csv(
    t(target_model),
    file=paste(output.file, "_target.csv", sep=""),
    quote=FALSE
)

write.csv(
    t(sub_model),
    file=paste(output.file, "_sub.csv", sep=""),
    quote=FALSE
)

write.csv(
    t(mut_model),
    file=paste(output.file, "_mut.csv", sep=""),
    quote=FALSE
)
