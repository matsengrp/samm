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

shm_model_type <- "RS"
# Create model using only silent mutations
# shm_model_type <- "S"

sub <- createSubstitutionMatrix(
    seq_genes,
    model=shm_model_type,
    sequenceColumn="sequence",
    germlineColumn="germline_sequence",
    vCallColumn="germline_name",
    multipleMutation=c("independent"),
    returnModel=c("5mer"),
)
write.csv(
    t(sub),
    file=output.file,
    quote=FALSE
)
