library(shazam)

# MK_RS5NF - S5F params for mice
mut = attr(MK_RS5NF, "mutability")
mut_df = data.frame(motif=names(mut), mutability=mut)
mut_df = mut_df[-grep('N', mut_df$motif),]
write.table(mut_df, "R/shmulate_params/mut_mouse.csv", row.names=F)

sub = attr(MK_RS5NF, "substitution")
sub_df = data.frame(
  motif=colnames(sub),
  t(sub)
)
# filter out things with N
sub_df = sub_df[-grep('N', sub_df$motif), c("motif", "A", "C", "G", "T")]
sub_df[is.na(sub_df)] = 0
write.table(sub_df, "R/shmulate_params/sub_mouse.csv", row.names=F)
