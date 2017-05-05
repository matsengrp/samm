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

log_mut_lower <- unlist(raw_data['V3'])
log_mut_upper <- unlist(raw_data['V4'])
names(log_mutabilities) <- unlist(raw_data['V1'])

# Plot for multiple nucleotides
center_nuc <- c('A', 'T', 'G', 'C')
y_lim <- c(
    floor(min(log_mut_lower)),
    ceiling(max(log_mut_upper))
)
plot_list <- plotBarchart(log_mutabilities, log_mut_lower, log_mut_upper, center_nuc, 'bar', bar.size=.25, y_lim = y_lim, rect_height = 0.45)
image <- do.call('grid.arrange', args = c(plot_list, ncol = length(plot_list)))
ggsave(file=output_file, plot=image, width=30, height=8)
