#' Creates an igraph object from edge and node data.frames
#'
#' @param   edge.df   data.frame of edges with columns [from, to, weight]
#' @param   node.df   data.frame of nodes with columns [taxa, seq]
#' @return  an igraph object with added vertex annotations
#' @export
getGraph <- function(edge_df, node_df) {
    # Create igraph object
    g_obj <- graph.data.frame(edge_df, directed=T)
    V(g_obj)$number <- match(V(g_obj)$name, node_df$taxa)
    sub_df <- node_df[V(g_obj)$number, ]
    # Add annotations from node_df to graph
    for(col in names(sub_df)) {
        g_obj <- set.vertex.attribute(g_obj, name=col, value=sub_df[,col])
    }
    return(g_obj)
}


#' Identifies clonal consensus sequences
#'
#' \code{calcClonalConsensus} identifies and returns the consensus sequence of a
#' group of sequences.
#'
#' @param   inputSeqs           clonal input sequences.
#' @param   germlineSeq         clone germline sequence.
#'
#' @return   A clonal consensus sequences.
#'
#' @details
#' For sequences identified to be part of the same clone, this function defines an
#' effective sequence that will be representative for all mutations in the clone. Each
#' position in this consensus (or effective) sequence is created by a weighted sampling
#' of each mutated base (and non "N", "." or "-" characters) from all the sequences in
#' the clone.
#'
#' For example, in a clone with 5 sequences that have a C at position 1, and 5 sequences
#' with a T at this same position, the consensus sequence will have a C 50\%  and T 50\%
#' of the time it is called.
#' @export
calcClonalConsensus <- function(inputSeqs, germlineSeq){

    # Find length of shortest input sequence
    # This is used to trim all the sequences to that length
    len_inputSeqs <- sapply(inputSeqs, nchar)
    len_shortest <- min(len_inputSeqs, na.rm=TRUE)

    #Find the length of the longest germline sequence
    len_germlineSeq <- sapply(germlineSeq, nchar)
    len_longest <- max(len_germlineSeq, na.rm=TRUE)
    germlineSeq <- germlineSeq[(which(len_longest==len_germlineSeq))[1]]

    # Identify the consensus sequence
    # TODO: Figure out the T/F
    charInputSeqs <- sapply(inputSeqs, function(x){ s2c(x)[1:len_shortest] })
    charGLSeq <- s2c(germlineSeq)
    matClone <- sapply(1:len_shortest, function(i) {

        # Identify the nucleotides (in seqs and germline) at the current position
        posNucs <- unique(charInputSeqs[i,])
        posGL <- charGLSeq[i]
        error <- FALSE

        # If the current position is a gap in both germline and the sequence,
        # return a gap
        if(posGL=="-" & sum(!(posNucs%in%c("-","N")))==0) {
            return(c("-",error))
        }

        # If all the sequences in the clone have the same nucleotide at the current
        # position, return the value at the current positions
        if(length(posNucs)==1) {
            return(c(posNucs[1],error))
        } else {
            # if the nucleotides at the current position are not all the same

            # TODO: The error message is not doing anything currently...
            if("N"%in%posNucs) {
                error=TRUE
            }

            # If the current nucleotide matches germline, return germline
            if(sum(!posNucs[posNucs!="N"]%in%posGL)==0) {
                return( c(posGL,error) )
            } else {
                # If we look at all nodes (including terminal nodes), sample a nucleotide from the possible
                # nucleotides in the clonal sequences at this position
                return( c(sample(charInputSeqs[i,charInputSeqs[i,]!="N" & charInputSeqs[i,]!=posGL],1),error) )
            }
        }
        if(error==TRUE) { warning("Error while attempting to collapse by clone!") }
    })
    return(c2s(matClone[1,]))
}


#' Calculate targeting probability along sequence
#'
#' @param germlineSeq     input sequence
#' @param targetingModel  model underlying SHM to be used
#'
#' @return matrix of probabilities of each nucleotide being mutated to any other nucleotide.
#'
#' @details
#' Applies the targeting model to the input sequence to determine the probability
#' that a given nucleotide at a given position will mutate to any of the other
#' nucleotides. This is calculated for every position and every possible mutation.
calculateTargeting <- function(germlineSeq, targetingModel=shazam::HS5FModel) {

    s_germlineSeq <- germlineSeq
    c_germlineSeq <- s2c(s_germlineSeq)

    # Removing IMGT gaps (they should come in threes)
    # After converting ... to XXX any other . is not an IMGT gap & will be treated like N
    gaplessSeq <- gsub("\\.\\.\\.", "XXX", s_germlineSeq)
    #If there is a single gap left convert it to an N
    gaplessSeq <- gsub("\\.", "N", gaplessSeq)

    # Re-assigning s_germlineSeq (now has all "." that are not IMGT gaps converted to Ns)
    s_germlineSeq <- gsub("XXX", "...", gaplessSeq)
    c_germlineSeq <- s2c(s_germlineSeq)
    # Matrix to hold targeting values for each position in c_germlineSeq
    germlineSeqTargeting <- matrix(NA,
                                   ncol=nchar(s_germlineSeq),
                                   nrow=length(NUC_N),
                                   dimnames=list(NUC_N, c_germlineSeq))

    # Now remove the IMGT gaps so that the correct 5mers can be made to calculate
    # targeting. e.g.
    # GAGAAA......TAG yields: "GAGAA" "AGAAA" "GAAAT" "AAATA" "AATAG"
    # (because the IMGT gaps are NOT real gaps in sequence!!!)
    gaplessSeq <- gsub("\\.\\.\\.", "", s_germlineSeq)
    gaplessSeqLen <- nchar(gaplessSeq)

    #Slide through 5-mers and look up targeting
    gaplessSeq <- paste("NN",gaplessSeq,"NN",sep="")
    gaplessSeqLen <- nchar(gaplessSeq)
    pos<- 3:(gaplessSeqLen-2)
    subSeq =  substr(rep(gaplessSeq,gaplessSeqLen-4),(pos-2),(pos+2))
    germlineSeqTargeting_gapless <- sapply(subSeq,function(x){
        targetingModel@targeting[NUC_N,x]
    })
    germlineSeqTargeting[,c_germlineSeq!="."] <- germlineSeqTargeting_gapless

    # Set self-mutating targeting values to be NA
    mutatingToSelf <- colnames(germlineSeqTargeting)
    # print(mutatingToSelf[!(mutatingToSelf%in%NUCLEOTIDES)])
    mutatingToSelf[!(mutatingToSelf%in%NUCLEOTIDES)] <- "N"
    tmp <- sapply( 1:ncol(germlineSeqTargeting), function(pos){
        germlineSeqTargeting[ mutatingToSelf[pos],pos ] <<- NA })

    germlineSeqTargeting[!is.finite(germlineSeqTargeting)] <- NA
    return(germlineSeqTargeting[NUCLEOTIDES,])
}


#' Create all codons one mutation away from input codon.
#'
#' @param codon   starting codon to which mutations are added
#'
#' @return a vector of codons.
#'
#' @details
#' All codons one mutation away from the input codon are generated.
allCodonMuts <- function(codon) {
	codon_char <- s2c(codon)
	matCodons <- t(array(codon_char, dim=c(3,12)))
	matCodons[1:4, 1] <- NUCLEOTIDES
	matCodons[5:8, 2] <- NUCLEOTIDES
	matCodons[9:12,3] <- NUCLEOTIDES
	return(apply(matCodons,1,c2s))
}


#' Asses if a mutation is R or S based on codon information.
#'
#' @param codonFrom     starting codon
#' @param codonTo       codon with mutation
#'
#' @return type of mutation, one of "S", "R", "Stop", or NA.
mutationType <- function(codonFrom,codonTo) {
	if(is.na(codonFrom) | is.na(codonTo) | is.na(AMINO_ACIDS[codonFrom]) | is.na(AMINO_ACIDS[codonTo])) {
		mutationType <- NA
	} else {
		mutationType <- "S"
		if(AMINO_ACIDS[codonFrom] != AMINO_ACIDS[codonTo]) {
			mutationType <- "R"
		}
		if(AMINO_ACIDS[codonFrom]=="*" | AMINO_ACIDS[codonTo]=="*") {
			mutationType <- "Stop"
		}
	}
	return(mutationType)
}


#' Generate codon table
#'
#' @return matrix with all codons as row and column names
#' and the type of mutation as the corresponding value in the matrix.
#'
#' @details
#' First generates all informative codons and determines types of mutations.
#' Next generates uninformative codons (having either an N or a gap "-"
#' character) and sets the mutation type as NA.
computeCodonTable <- function() {
    # Initialize empty data.frame
    codon_table <- as.data.frame(matrix(NA,ncol=64,nrow=12))
    # Pre-compute every codon
    counter <- 1
    for(pOne in NUCLEOTIDES) {
        for(pTwo in NUCLEOTIDES) {
            for(pThree in NUCLEOTIDES) {
                codon <- paste0(pOne,pTwo,pThree)
                colnames(codon_table)[counter] <- codon
                counter <- counter + 1
                all_muts <- allCodonMuts(codon)
                codon_table[,codon] <- sapply(all_muts, function(x) { mutationType(x, codon) })
            }
        }
    }
    # Set codons with N or - to be NA
    chars <- c("N","A","C","G","T", "-")
    for(n1 in chars) {
        for(n2 in chars) {
            for(n3 in chars) {
                if(n1=="N" | n2=="N" | n3=="N" | n1=="-" | n2=="-" | n3=="-") {
                    codon_table[,paste0(n1,n2,n3)] <- rep(NA,12)
                }
            }
        }
    }
    return(as.matrix(codon_table))
}


#' Compute the mutations types
#'
#' @param seq   sequence for which to compute mutation types
#'
#' @return matrix of mutation types for each position in the sequence.
#'
#' @details
#' For each position in the input sequence, use the codon table to
#' determine what types of mutations are possible. Returns matrix
#' of all possible mutations and corresponding types.
computeMutationTypes <- function(seq){
    codon_table <- computeCodonTable()
    len_seq <- nchar(seq)
    codons <- sapply(seq(1, len_seq, by=3),
                    function(x) {substr(seq,x,x+2)})
    mut_types <- matrix(unlist(codon_table[,codons]),
                       ncol=len_seq, nrow=4, byrow=F)
    dimnames(mut_types) <-  list(NUCLEOTIDES, 1:(ncol(mut_types)))
    return(mut_types)
}


#' Find encompassing codon
#'
#' @param nuc_pos    position for which codon is to be found
#' @param frame      reading frame in which to determine codon
#'
#' @return vector of positions of codon encompassing input position.
#'
#' @details
#' Given a nuclotide position, find the positions of the three nucleotides
#' that encompass the codon in the given reading frame of the sequence.
# e.g. nuc 86 is part of nucs 85,86,87
getCodonPos <- function(nuc_pos, frame=0) {
	codon_num <- (ceiling((nuc_pos + frame) / 3)) * 3
	codon <- (codon_num-2):codon_num
	return(codon)
}


#' Pick a position to mutate
#'
#' @param sim_len       length of sequence in which mutation is being simulated
#' @param targeting     probabilities of each position in the sequence being mutated
#' @param positions     vector of positions which have already been mutated
#'
#' @return list of position being mutated and updated vector of mutated positions.
#'
#' @details
#' Sample positions in the sequence to mutate given targeting probability
#' until a new position is selected. This new position is then added to the
#' vector of mutated positions and returned.
sampleMut <- function(sim_len, targeting, positions) {
	pos <- 0
	# Sample mutations until new position is selected
	while(pos %in% positions) {
		# Randomly select a mutation
		mut <- sample(1:(4*sim_len), 1, replace=F, prob=as.vector(targeting))
		pos <- ceiling(mut/4)
	}
	return(list(mut=mut, pos=pos))
}


#' Simulate mutations in a single sequence
#'
#' @param input_seq    sequence in which mutations are to be introduced
#' @param num_muts     number of mutations to be introduced into input sequence
#'
#' @return mutated sequence.
#'
#' @details
#' Generates mutations in sequence one by one while updating targeting
#' probability of each position after each mutation.
#' @export
shmulateSeq <- function(input_seq, num_muts) {
    # Trim sequence to last codon
    if(getCodonPos(nchar(input_seq))[3] > nchar(input_seq)) {
        sim_seq <- substr(input_seq, 1, getCodonPos(nchar(input_seq))[1]-1)
    } else {
        sim_seq <- input_seq
    }
    sim_seq <- gsub("\\.", "-", sim_seq)
    sim_len <- nchar(sim_seq)

    # Calculate possible mutations (given codon table)
    mutation_types <- computeMutationTypes(sim_seq)
    # Calculate probabilities of mutations at each position given targeting
    ## Internal shm function in MutationProfiling.R
    targeting <- calculateTargeting(sim_seq)
    targeting[is.na(targeting)] <- 0
    # Make probability of stop codon 0
    targeting[mutation_types=="Stop"] <- 0

    # Initialize counters
    total_muts <- 0
    positions <- numeric(num_muts)

    while(total_muts < num_muts) {
        # Get position to mutate and update counters
        mutpos <- sampleMut(sim_len, targeting, positions)
        total_muts <- total_muts + 1
        positions[total_muts] <- mutpos$pos

        # Implement mutation in simulation sequence
        mut_nuc <- 4 - (4*mutpos$pos - mutpos$mut)
        sim_char <- s2c(sim_seq)
        sim_char[mutpos$pos] <- NUCLEOTIDES[mut_nuc]
        sim_seq <- c2s(sim_char)

        # Update targeting
        lower <- max(mutpos$pos-4, 1)
        upper <- min(mutpos$pos+4, sim_len)
        targeting[,lower:upper] <- calculateTargeting(substr(sim_seq, lower, upper))
        targeting[is.na(targeting)] <- 0

        # Update possible mutations
        lower <- getCodonPos(lower)[1]
        upper <- getCodonPos(upper)[3]
        mutation_types[,lower:upper] <- computeMutationTypes(substr(sim_seq, lower, upper))
        # Make probability of stop codon 0
        if(any(mutation_types[,lower:upper]=="Stop", na.rm=T)) {
            targeting[,lower:upper][mutation_types[,lower:upper]=="Stop"] <- 0
        }
    }
    return(sim_seq)
}


#' Simulate sequences to populate a tree
#'
#' shmulateTree returns a set of simulated sequences generated from an input sequence and an
#' igraph object. The input sequence is used to replace the founder node of the igraph lineage
#' tree and sequences are simulated with mutations corresponding to edge weights in the tree.
#' Sequences will not be generated for groups of nodes that are specified to be excluded.
#'
#' @param input_seq   sequence in which mutations are to be introduced.
#' @param graph       igraph object with vertex annotations whose edges are to be recreated.
#' @param field       annotation field to use for both unweighted path length exclusion and
#'                    consideration as a founder node. if NULL do not exclude any nodes.
#' @param exclude     vector of annotation values in the given field to exclude from potential
#'                    founder set. If NULL do not exclude any nodes. Has no effect if field=NULL.
#' @param jun_frac    fraction of characters in the junction region to add proportional number
#'                    of trunk mutations to the sequence.
#'
#' @return a data.frame of simulated sequences.
#' @export
shmulateTree <- function(input_seq, graph, founder_name = 'Node1', field=NULL, exclude=NULL, jun_frac=NULL, omega = 1) {
    # Determine founder (mrca) of lineage tree
    # Get adjacency matrix
    adj <- get.adjacency(graph, sparse=F)
    # Get names of nodes for which sequences are not to be returned
    skip_names <- c()
    if (!is.null(field)) {
        g <- get.vertex.attribute(graph, name = field)
        g_names <- get.vertex.attribute(graph, name = 'name')
        skip_names <- g_names[g %in% exclude]
    }
    # Create data.frame to hold simulated sequences
    sim_tree <- data.frame('name'=founder_name,
                           'sequence'=input_seq, 'distance'=0,
                           stringsAsFactors=F)
    parent_nodes <- founder_name
    nchil <- sum(adj[parent_nodes,]>0)
    # Add trunk mutations proportional to fraction of sequence in junction
    if(!is.null(jun_frac)) {
        adj[parent_nodes,] <- round(adj[parent_nodes,]*(1+jun_frac))
    }
    while(nchil>0) {
        new_parents <- c()
        # Loop through parent-children combos
        for(p in parent_nodes) {
            children <- colnames(adj)[adj[p,]>0]
            for(ch in children) {
                # Add child to new parents
                new_parents <- union(new_parents, ch)
                # Simulate sequence for that edge
                seq <- shmulateSeq(sim_tree$sequence[sim_tree$name==p],
                                   adj[p,ch], omega)
                new_node <- data.frame('name'=ch, 'sequence'=seq,
                                       'distance'=adj[p,ch])
                # Update output data.frame
                sim_tree <- rbind.fill(sim_tree, new_node)
            }
        }
        # Re-calculate number of children
        parent_nodes <- new_parents
        nchil <- sum(adj[parent_nodes,]>0)
    }
    # Remove sequences that are to be excluded
    sim_tree <- subset(sim_tree, !(name %in% skip_names))
    return(sim_tree)
}
