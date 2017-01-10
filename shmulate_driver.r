# Run shmulate given command line arguments
source('shmulate/R/SHMulate.R')
source('shmulate/R/SHMulate_Functions.R')

# Packages needed:
# igraph
# plyr
# seqinr
# shazam

arg <- commandArgs(TRUE)                       
n.taxa <- as.numeric(arg[1])
seq.file <- arg[2]
germline <- arg[3]
group <- arg[4]
n.mutes <- arg[5]

set.seed(arg[6])

glc <- tolower(s2c(germline))
n.nuc <- length(glc)
dat <- array('n', dim=c(n.taxa,n.nuc))
gl <- toupper(c2s(glc))

seqs <- data.frame()
for(taxon in 1:n.taxa) {
    seqs[taxon,'sequence'] <- shmulateSeq(gl, n.mutes)
}
dat <- t(sapply(1:nrow(seqs), function(u) s2c(tolower(seqs[u,'sequence']))))
outdat <- lapply(1:nrow(dat), function(u) dat[u,])

# output file should be same as input file
write.fasta(sequences = outdat, file.out = seq.file, names = paste(group, '-Sequence', 1:n.taxa, sep=''))

