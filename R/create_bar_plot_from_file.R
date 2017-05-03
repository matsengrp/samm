# Plot bar chart

# Dependencies for plots
library(methods)
library(ggplot2)
library(dplyr)
library(lazyeval)
library(alakazam)
library(gridExtra)
source('R/plotBarchart.R')

arg <- commandArgs(TRUE)
data_path <- arg[1]

# comma-separated motif lengths list
motif_str <- arg[2]
output_file <- arg[3]

motif_lens <- as.integer(unlist(strsplit(motif_str, ',')))

# Read data and convert to format plotBarchart wants

raw_data <- read.table(data_path, sep=',')
log_mutabilities <- unlist(raw_data['V2'])
names(log_mutabilities) <- gsub('N', 'Z', unlist(raw_data['V1']))
#names(log_mutabilities) <- unlist(raw_data['V1'])
#log_mutabilities <- log_mutabilities[!grepl("N", names(log_mutabilities))]

# We're plotting mutabilities, so we exponentiate and scale them
filtered_mutes <- exp(log_mutabilities) /
    sum(exp(log_mutabilities), na.rm=TRUE)

# Plot for multiple nucleotides
pdf(output_file, width=20, height=5)
center_nuc <- c('A', 'T', 'G', 'C')
plot_list <- plotBarchart(filtered_mutes, center_nuc, 'bar', bar.size=.25)
do.call('grid.arrange', args = c(plot_list, ncol = length(plot_list)))
dev.off()
#ggsave(file=output_file, plot=do.call('grid.arrange', args = c(plot_list, ncol = length(plot_list))), width=20, height=5, dpi=500)

