# Project documentation for shm
#
# @author     Mohamed Uduman, Gur Yaari, Namita Gupta
# @copyright  Copyright 2014 Kleinstein Lab, Yale University. All rights reserved
# @license    Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported
# @version    0.1.0
# @date       2015.08.28


#' The SHMulate package
#'
#' Provides tools for simulating Immunoglobulin (Ig) Somatic HyperMutation
#' (SHM).
#'
#' @references
#' \enumerate{
#'   \item  Yaari G, et al. Models of somatic hypermutation targeting and substitution based
#'            on synonymous mutations from high-throughput immunoglobulin sequencing data.
#'            Front Immunol. 2013 4:358.
#'  }
#'
#' @seealso
#' The Change-O suite of tools includes three separate R packages: \link[alakazam]{alakazam},
#' \link[tigger]{tigger}, and \link[shm]{shm}.
#'
#' @name     SHMulate
#' @docType  package
#' @import   igraph
#' @import   plyr
#' @import   shm
#' @importFrom  seqinr     c2s
#' @importFrom  seqinr     s2c
NULL


#### Constants ####

# Nucleotides
NUCLEOTIDES <- c("A","C","G","T")
NUC_N <- c("A","C","G","T","N")

# Map codons to amino acids
AMINO_ACIDS <- c("F", "F", "L", "L", "S", "S", "S", "S", "Y", "Y", "*", "*", "C", "C",
                 "*", "W", "L", "L", "L", "L", "P", "P", "P", "P", "H", "H", "Q", "Q",
                 "R", "R", "R", "R", "I", "I", "I", "M", "T", "T", "T", "T", "N", "N",
                 "K", "K", "S", "S", "R", "R", "V", "V", "V", "V", "A", "A", "A", "A",
                 "D", "D", "E", "E", "G", "G", "G", "G")
names(AMINO_ACIDS) <- c("TTT", "TTC", "TTA", "TTG", "TCT", "TCC", "TCA", "TCG", "TAT",
                        "TAC", "TAA", "TAG", "TGT", "TGC", "TGA", "TGG", "CTT", "CTC",
                        "CTA", "CTG", "CCT", "CCC", "CCA", "CCG", "CAT", "CAC", "CAA",
                        "CAG", "CGT", "CGC", "CGA", "CGG", "ATT", "ATC", "ATA", "ATG",
                        "ACT", "ACC", "ACA", "ACG", "AAT", "AAC", "AAA", "AAG", "AGT",
                        "AGC", "AGA", "AGG", "GTT", "GTC", "GTA", "GTG", "GCT", "GCC",
                        "GCA", "GCG", "GAT", "GAC", "GAA", "GAG", "GGT", "GGC", "GGA",
                        "GGG")

library(igraph)
library(shazam)
library(plyr)
library(seqinr)

