# Plot bar chart

# Dependencies for plots
library(methods)
library(ggplot2)
library(dplyr)
library(lazyeval)
library(gridExtra)
library(seqinr)
library(tidyr)
source('R/plotBarchart.R')

arg <- commandArgs(TRUE)
data_path <- arg[1]

# comma-separated motif lengths list
motif_str <- arg[2]
output_file <- arg[3]
target_nucs <- unlist(strsplit(arg[4], ','))

motif_lens <- as.integer(unlist(strsplit(motif_str, ',')))

# Read data and convert to format plotBarchart wants, and convert -inf to 0
raw_data <- read.table(data_path, sep=',', header=TRUE, stringsAsFactors=FALSE)
raw_data <- do.call(data.frame, lapply(raw_data, function(x) replace(x, is.infinite(x), NA)))

#replace target "N" with "Z" so that the plots are properly ordered
raw_data['motif'] <- apply(raw_data['motif'], 2, function(motif) gsub('N', 'Z', motif))

# Plot for multiple nucleotides
center_nucs <- c('A', 'T', 'G', 'C')

# Change the following to adjust y-limits in the plot
y_lim <- c(
    floor(min(raw_data['theta_lower'], na.rm=TRUE)),
    ceiling(max(raw_data['theta_upper'], na.rm=TRUE))
)
plot_list <- plotBarchart(raw_data,
                          nucleotides=center_nucs,
                          target=target_nucs,
                          style='bar',
                          bar.size=.25,
                          y_lim=y_lim,
                          rect_height=0.6)
image <- do.call('grid.arrange', args = c(plot_list, ncol = max(1, length(center_nucs)/2)))
if (length(center_nucs) > 1) {
    ggsave(file=output_file, plot=image, width=15, height=8*length(target_nucs) * 2)
} else {
    ggsave(file=output_file, plot=image, width=8, height=8*length(target_nucs))
}
