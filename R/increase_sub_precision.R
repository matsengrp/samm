# increase precision of substitution parameters

options(digits=20)
df = read.table('R/shmulate_params/substitution.csv', header=TRUE)
df.normalized = data.frame(t(apply(df[,2:5], 1, function(x) x / sum(x))))
df.normalized = cbind.data.frame(motif=df$motif, df.normalized)
write.table(df.normalized, 'R/shmulate_params/substitution.csv', row.names=FALSE)

