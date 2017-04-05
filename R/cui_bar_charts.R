# Test bar plot

# Dependencies for plots
library(ggplot2)
library(dplyr)
library(lazyeval)
library(alakazam)
library(gridExtra)

source('R/plot_barchart.R')

## Uncomment output you want to plot
## 3mer model (basic proof of concept)
#data_path <- '/fh/fast/matsen_e/dshaw/samm/cui_3mer.csv'
#motif_lens <- c(3)
## 5mer model (calculated from S5F)
#data_path <- '/fh/fast/matsen_e/dshaw/samm/s5f_log_mutability.csv'
#motif_lens <- c(5)
# 3-5-7mer model (basic proof of concept)
data_path <- '/fh/fast/matsen_e/dshaw/samm/cui_357mer.csv'
motif_lens <- c(3,5,7)

# Read data and convert to format plot_barchart wants
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
            filtered_log_mutes[mute] <- filtered_log_mutes[mute] + log_mutabilities[substr(mute, max_len %/% 2 - sub_len %/% 2, max_len %/% 2 + sub_len %/% 2)]
        }
    }
    # We're plotting mutabilities, so we exponentiate and scale them
    filtered_mutes <- exp(filtered_log_mutes) / sum(exp(filtered_log_mutes), na.rm=TRUE)
} else {
    # We're plotting mutabilities, so we exponentiate and scale them
    filtered_mutes <- exp(log_mutabilities) / sum(exp(log_mutabilities), na.rm=TRUE)
}

# Plot for one nucleotide
center_nuc <- 'A'
plot_list <- plot_barchart(filtered_mutes, center_nuc, 'bar')
do.call('grid.arrange', args = c(plot_list, ncol = length(plot_list)))

# Plot for multiple nucleotides
center_nuc <- c('A', 'T', 'G', 'C')
plot_list <- plot_barchart(filtered_mutes, center_nuc, 'bar')
do.call('grid.arrange', args = c(plot_list, ncol = length(plot_list)))

