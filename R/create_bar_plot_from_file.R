# Plot bar chart

# Dependencies for plots
library(methods)
library(ggplot2)
library(dplyr)
library(lazyeval)
library(alakazam)
library(gridExtra)
source('R/plotBarchart.R')

data_path <- '/fh/fast/matsen_e/dshaw/samm/cui_3mer.csv'
motif_lens <- c(3)
#data_path <- '/fh/fast/matsen_e/dshaw/samm/cui_357mer.csv'
#motif_lens <- c(3,5,7)
arg <- commandArgs(TRUE)
data_path <- arg[1]

# comma-separated motif lengths list
motif_str <- arg[2]
output_file <- arg[3]

motif_lens <- as.integer(unlist(strsplit(motif_str, ',')))

# Read data and convert to format plotBarchart wants
raw_data <- read.table(data_path, sep=',')
log_mutabilities <- unlist(raw_data['V2'])
names(log_mutabilities) <- unlist(raw_data['V1'])
log_mutabilities <- log_mutabilities[!grepl("N", names(log_mutabilities))]

# For hierarchical motif model, get the values of the max-mer by adding
if (length(motif_lens) > 1) {
    max_len <- max(motif_lens)
    filtered_log_mutes <-
        log_mutabilities[
            sapply(names(log_mutabilities),
                   function(mute_name) nchar(mute_name) == max_len)
            ]
    for (mute in names(filtered_log_mutes)) {
        for (sub_len in head(motif_lens, -1)) {
            filtered_log_mutes[mute] <-
                filtered_log_mutes[mute] +
                log_mutabilities[substr(mute,
                                        max_len %/% 2 - sub_len %/% 2,
                                        max_len %/% 2 + sub_len %/% 2)]
        }
    }
    # We're plotting mutabilities, so we exponentiate and scale them
    filtered_mutes <- exp(filtered_log_mutes) /
        sum(exp(filtered_log_mutes), na.rm=TRUE)
} else {
    # We're plotting mutabilities, so we exponentiate and scale them
    filtered_mutes <- exp(log_mutabilities) /
        sum(exp(log_mutabilities), na.rm=TRUE)
}

# Plot for multiple nucleotides
svg(output_file, width=20, height=5)
center_nuc <- c('A', 'T', 'G', 'C')
plot_list <- plotBarchart(filtered_mutes, center_nuc, 'bar', bar.size=.25)
do.call('grid.arrange', args = c(plot_list, ncol = length(plot_list)))
dev.off()

