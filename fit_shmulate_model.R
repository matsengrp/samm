# Run shmulate given command line arguments
source('shmulate/R/SHMulate.R')
source('shmulate/R/SHMulate_Functions.R')

# Packages needed:
# igraph
# plyr
# seqinr
# shazam
# Subset example data to one isotype and sample as a demo

data(ExampleDb, package="alakazam")
db <- subset(ExampleDb, ISOTYPE == "IgA" & SAMPLE == "-1h")
# print(head(db))

db$SEQUENCE_IMGT <- paste("AA", substr(db$SEQUENCE_IMGT, 1, 20))
db$GERMLINE_IMGT_D_MASK <- paste("TT", substr(db$GERMLINE_IMGT_D_MASK, 1, 20))
db$V_CALL[seq(10)] <- "Asdfadsf"
db$V_CALL[seq(11, nrow(db))] <- "asjdfklajsdf"
print(db$V_CALL)
print(head(db[db$SEQUENCE_IMGT != db$GERMLINE_IMGT_D_MASK, c("SEQUENCE_IMGT", "GERMLINE_IMGT_D_MASK")]))

sub <- createSubstitutionMatrix(db, model="RS", returnModel="1mer_raw")
print(sub)
#
# arg <- commandArgs(TRUE)
# seq.file <- arg[1]
# gene.file <- arg[2]
#
# seq <- read.csv(seq.file)
# genes <- read.csv(gene.file)
#
# seq_genes <- merge(seq, genes, by="germline_name")
# seq_genes$sequence <- toupper(seq_genes$sequence)
# seq_genes$germline_sequence <- toupper(seq_genes$germline_sequence)
#
# seq_genes$sequence <- substr(seq_genes$sequence, 1, 20)
# seq_genes$germline_sequence <- substr(seq_genes$germline_sequence, 1, 20)
#
# print(head(seq_genes[seq_genes$sequence != seq_genes$germline_sequence,]))
#
# # # Create model using both replacement and silent mutations
# shm_model_type <- "RS"
# # Create model using only silent mutations
# # shm_model_type <- "S"
#
# sub <- createSubstitutionMatrix(
#     seq_genes,
#     model=shm_model_type,
#     sequenceColumn="sequence",
#     germlineColumn="germline_sequence",
#     vCallColumn="germline_name",
#     multipleMutation=c("independent"),
#     returnModel=c("1mer_raw"),
# )
# print(sub)
# # print(sub[,seq(4)])
