# Run shmulate given command line arguments
source('shmulate/R/SHMulate.R')
source('shmulate/R/SHMulate_Functions.R')

customSlideWindowDb <- function(db, sequenceColumn="sequence", germlineColumn="germline_sequence", mutThresh=6, windowSize=10) {
  db_filter <- sapply(1:nrow(db), function(i) {
    sequence <- toupper(db[i, sequenceColumn])
    germline <- toupper(db[i,germlineColumn])
    slideWindowSeq(inputSeq = sequence, germlineSeq = germline, mutThresh = mutThresh, windowSize = windowSize)
  })
}

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
print(paste("Original number of sequences", orig_num_seqs))

# Filter sequences
# filter_mask <- customSlideWindowDb(db = seq_genes)
# seq_genes <- seq_genes[filter_mask,]
# print(paste("Post-Filter number of sequences", nrow(seq_genes)))

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
