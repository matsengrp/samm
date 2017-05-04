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
names(log_mutabilities) <- unlist(raw_data['V1'])
log_mutabilities <- log_mutabilities[!grepl("N", names(log_mutabilities))]

# Plot for multiple nucleotides
png(filename=output_file,
    units="in",
    width=20,
    height=5,
    pointsize=12,
    res=128)
center_nuc <- c('A', 'T', 'G', 'C')
y_lim <- c(
    floor(min(log_mutabilities)),
    ceiling(max(log_mutabilities))
)
plot_list <- plotBarchart(log_mutabilities, center_nuc, 'bar', bar.size=.25, y_lim = y_lim, rect_height = 0.45)
do.call('grid.arrange', args = c(plot_list, ncol = length(plot_list)))
dev.off()
