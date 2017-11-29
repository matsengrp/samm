plotBarchart <- function (model,
                          model_lower=NULL, model_upper=NULL,
                          nucleotides = c("A", "C", "G", "T"),
                          target = c("N"),
                          style = c("hedgehog", "bar"), size = 2,
                          bar.size = 0, y_lim = c(-5, 5), rect_height=0.5)
{
    # Plots bar chart of mutabilities
    #
    # Args
    #   model: mutability vector or TargetingModel object
    #          mutability vector is a vector with motifs as column names
    #   nucleotides: center nucleotide; one of "A", "C", "G" or "T",
    #                or a vector of any combination thereof
    #   style: one of "hedgehog" or "bar"; the function name indicates you should probably use
    #          "bar" since larger motif lengths take quite a while to make hedgehog plots
    #   size: base text size for labels
    #   bar.size: size of bars; as motif length increases bars get more difficult to see,
    #             so increase this for larger motifs
    #
    # Returns:
    #   list of ggplot elements corresponding to plots for each center nucleotide

    nucleotides <- toupper(nucleotides)
    style <- match.arg(style)

    # Processing data
    if (is(model, "TargetingModel")) {
        mut_scores <- model@mutability
        mut_words <- names(mut_scores)
    }
    else if (is(model, 'data.frame')) {
        mut_scores <- model[,'theta']
        model_lower <- model[,'theta_lower']
        model_upper <- model[,'theta_upper']
        # for some reason unname is needed so mut_df can be created
        mut_words <- unname(model[,'motif'])
    }
    else if (is(model, 'vector')) {
        mut_scores <- model
        mut_words <- names(model)
    }
    else if (!is(model, "vector") | !is(model, 'data.frame')) {
        stop("Input must be either a mutability vector, mutability data frame or TargetingModel object.")
    }

    if (is.null(model_lower)) {
        model_lower <- mut_scores
    }

    if (is.null(model_upper)) {
        model_upper <- mut_scores
    }

    mut_scores <- mut_scores[!grepl("N", mut_words)]
    mut_scores[!is.finite(mut_scores)] <- 0
    mut_positions <- as.data.frame(t(sapply(mut_words, s2c)))
    motif_len <- ncol(mut_positions)
    if (!(motif_len %% 2) | !(motif_len > 1)) {
        stop("Motif length must be odd and greater than one.")
    }
    data_cols <- paste0("pos", 1:motif_len)
    left_motif_len <- motif_len %/% 2

    center_nuc_col <- paste0('pos', left_motif_len+1)
    flank_cols <- data_cols[-left_motif_len-1]

    colnames(mut_positions) <- data_cols
    mut_df <- data.frame(
        word = mut_words,
        score = mut_scores,
        lower = model_lower,
        upper = model_upper,
        mut_positions,
        target = sapply(raw_data['target'], function(nuc) c('N', 'A', 'C', 'G', 'T')[nuc+1])
    )

    # Setting up plotting environment
    base_theme <- theme_bw() +
        theme(panel.margin = unit(0, "lines"),
              panel.background = element_blank()) +
        theme(axis.text = element_text(margin = unit(0, "lines"))) +
        theme(text = element_text(size = 10 * size),
              title = element_text(size = 10 * size),
              legend.margin = unit(0, "lines"),
              legend.background = element_blank())
    score_offset <- 0
    score_scale <- 15
    text_offset <- - rect_height * (motif_lens + 0.5) - 0.25
    motif_colors <- setNames(c("#4daf4a", "#e41a1c", "#094d85",
        "#999999"), c("WA/TW", "WRC/GYW", "SYC/GRS", "Neutral"))
    dna_colors <- setNames(c("#7bce77", "#ff9b39", "#f04949",
                             "#5796ca", "#c4c4c4"), c("A", "C", "G", "T", "N"))

    # Recognizing known hot/cold spots
    mut_df$motif <- "Neutral"
    if (motif_len == 3) {
        mut_df$motif[grepl("([AT]A.)|(.T[AT])", mut_df$word,
                           perl = TRUE)] <- "WA/TW"
        grep_levels <- c('WA/TW', 'Neutral')
    } else {
        grep_exp <- list(
            c('.[AT]A..','..T[AT].', 'WA/TW'),
            c('[AT][GA]C..','..G[CT][AT]', 'WRC/GYW'),
            c('[CG][CT]C..','..G[GA][CG]', 'SYC/GRS'))
        # number of dots we need to add to each end
        n_extra <- (motif_len - 5)/2
        for (grep_val in grep_exp) {
            combined_grep <- paste0('(',
                                    rep('.', n_extra), grep_val[1], rep('.', n_extra), ')|(',
                                    rep('.', n_extra), grep_val[2], rep('.', n_extra), ')')
            mut_df$motif[grepl(combined_grep, mut_df$word,
                               perl = TRUE)] <- grep_val[3]
        }

        grep_levels <- c('WA/TW', 'WRC/GYW', 'SYC/GRS', 'Neutral')
    }
    mut_df$motif <- factor(mut_df$motif, levels = grep_levels)
    mut_df <- mut_df[mut_df[,center_nuc_col] %in% nucleotides, ]

    # Generate plots for each nucleotide of interest
    plot_list <- list()
    for (target_nuc in target) {
        for (center_nuc in nucleotides) {
            sub_df <- mut_df[mut_df[,center_nuc_col] == center_nuc & mut_df$target == target_nuc, ]
            if ((center_nuc %in% c("A", "C") & left_motif_len == motif_len %/% 2) | left_motif_len < motif_len %/% 2) {
                # 3' for A/C or offset
                sub_df <- arrange_(sub_df, .dots = flank_cols)
                sub_df$x <- -.5 + 1:nrow(sub_df)
            }
            else if ((center_nuc %in% c("G", "T") & left_motif_len == motif_len %/% 2) | left_motif_len > motif_len %/% 2) {
                # 5' for G/T or offset
                sub_df <- arrange_(sub_df, .dots = rev(flank_cols))
                sub_df$x <- -.5 + 1:nrow(sub_df)
            }
            else {
                stop("Invalid nucleotide choice")
            }

            # Create coordinates for nucleotide rectangles
            sub_melt <- sub_df %>%
                gather_("pos", "char", colnames(mut_positions)) %>%
                select_(.dots = c("x", "pos", "char"))
            sub_melt$pos <- as.numeric(gsub("pos", "", sub_melt$pos))
            sub_text <- list()
            for (i in 1:motif_len) {
                # Run-length encoding for rectangle sizes
                nuc_rle <- rle(sub_melt$char[sub_melt$pos == i])
                rect_max <- cumsum(nuc_rle$lengths)
                rect_min <- rect_max - diff(c(0, rect_max))
                if (length(rect_max) > 1) {
                    text_x <- rect_max - diff(c(0, rect_max))/2
                }
                else {
                    text_x <- rect_max/2
                }
                # Data frame of nucleotides/rectangle sizes
                tmp_df <- data.frame(text_x = text_x,
                                     text_y = i * rect_height,
                                     text_label = factor(nuc_rle$values,
                                                         levels = names(dna_colors)),
                                     rect_min = rect_min,
                                     rect_max = rect_max)
                sub_text[[i]] <- tmp_df
            }
            sub_melt$pos <- sub_melt$pos + text_offset

            # Shorten/lengthen the motif/text rectangles based on how many we have
            motif_offset <- y_lim[1]
            sub_text <- lapply(
                sub_text,
                function(x) {
                    mutate_(x,
                                   text_y = interp(~y + text_offset + motif_offset,
                                                   y = as.name("text_y")))
                })

            sub_rect <- bind_rows(sub_text) %>%
                mutate_(rect_width = interp(~y - x,
                                            x = as.name("rect_min"),
                                            y = as.name("rect_max")),
                        ymin = interp(~y - .5 * rect_height, y = as.name("text_y")),
                        ymax = interp(~y + .5 * rect_height, y = as.name("text_y")))

            # Finally begin plotting; set up axes/scales/theme and
            # plot nucleotide rectangles
            p1 <- ggplot(sub_df) +
                base_theme +
                xlab("") +
                ylab("") +
                scale_color_manual(name = "Motif",
                                   values = c(motif_colors, dna_colors),
                                   breaks = names(motif_colors)) +
                scale_fill_manual(name = "",
                                  values = c(motif_colors, dna_colors),
                                  guide = FALSE) +
                geom_rect(data = sub_rect,
                          mapping = aes_string(xmin = "rect_min",
                                               xmax = "rect_max",
                                               ymin = "ymin",
                                               ymax = "ymax",
                                               fill = "text_label",
                                               color = "text_label"),
                          size = 0.5 * size,
                          alpha = 1,
                          show.legend = FALSE)

            # Plot nucleotide characters in their corresponding rectangles
            p1 <- p1 +
                geom_text(data = sub_text[[left_motif_len+1]],
                          mapping = aes_string(x = "text_x",
                                               y = "text_y",
                                               label = "text_label"),
                          color = "black",
                          hjust = 0.5,
                          vjust = 0.5,
                          size = 3 * size,
                          fontface = 2)

            # Only plot at most two levels of text---otherwise a little busy
            if ((center_nuc %in% c("A", "C") & left_motif_len == motif_len %/% 2)) {
                for (flank in 1:min(left_motif_len, 2)) {
                    p1 <- p1 +
                        geom_text(data = sub_text[[flank]],
                                  mapping = aes_string(x = "text_x",
                                                       y = "text_y",
                                                       label = "text_label"),
                                  color = "black",
                                  hjust = 0.5,
                                  vjust = 0.5,
                                  size = 2 * size)
                }
            }
            else if ((center_nuc %in% c("G", "T") & left_motif_len == motif_len %/% 2)) {
                for (flank in rev(1 + motif_len - 1:min(left_motif_len, 2))) {
                    p1 <- p1 +
                        geom_text(data = sub_text[[flank]],
                                  mapping = aes_string(x = "text_x",
                                                       y = "text_y",
                                                       label = "text_label"),
                                  color = "black",
                                  hjust = 0.5,
                                  vjust = 0.5,
                                  size = 2 * size)
                }
            }

            # Now plot hedgehog or barchart
            if (style == "hedgehog") {
                # ggplot special sauce for hedgehog plot
                y_limits <- c(text_offset - 1, score_scale + score_offset)
                p1 <- p1 +
                    theme(plot.margin = unit(c(0, 0, 0, 0), "lines"),
                          panel.grid = element_blank(),
                          panel.border = element_blank(),
                          axis.title = element_blank(),
                          axis.text = element_blank(),
                          axis.ticks = element_blank(),
                          legend.direction = "horizontal",
                          legend.justification = c(0.5, 1),
                          legend.position = c(0.5, 1)) +
                    guides(color = guide_legend(override.aes = list(linetype = 1,
                                                                    size = 2 * size))) +
                    scale_x_continuous(expand = c(0, 0)) +
                    scale_y_continuous(limits = y_limits,
                                       expand = c(0, 0)) +
                    coord_polar(theta = "x") +
                    geom_segment(data = sub_df,
                                 mapping = aes_string(x = "x",
                                                      xend = "x",
                                                      yend = "score",
                                                      color = "motif"),
                                 y = score_offset,
                                 size = 0.75 * size)
            }
            else if (style == "bar") {
                # ggplot barchart
                y_breaks <- seq(y_lim[1], y_lim[2], 1)
                y_limits <- c(text_offset + motif_offset, y_lim[2] + score_offset)
                sub_colors <- motif_colors[names(motif_colors) %in% sub_df$motif]
                p1 <- p1 +
                    theme(plot.margin = unit(c(1, 1, 1, 1), "lines"),
                          panel.grid = element_blank(),
                          panel.border = element_rect(color = "black"),
                          axis.text.x = element_blank(),
                          axis.ticks.x = element_blank(),
                          legend.position = "top") +
                    guides(color = guide_legend(override.aes = list(fill = sub_colors,
                                                                    linetype = 0))) +
                    ylab("Theta") +
                    scale_x_continuous(expand = c(0, 1)) +
                    scale_y_continuous(limits = y_limits,
                                       breaks = y_breaks,
                                       expand = c(0, 0.5)) +
                    geom_bar(data = sub_df,
                             mapping = aes_string(x = "x",
                                                  y = "score",
                                                  fill = "motif",
                                                  color = "motif"),
                             stat = "identity",
                             position = "identity",
                             size = bar.size,
                             width = 0.7) +
                    geom_errorbar(
                        #data = sub_df[sub_df$lower != 0,],
                        data = sub_df,
                        mapping = aes_string(x = "x",
                                             ymin = "lower",
                                             ymax = "upper"),
                        width=0,
                        size=0.25
                    )
            }

            # Add plots to list
            plot_list[[paste0(center_nuc, '->', target_nuc)]] <- p1
        }
    }
    return(plot_list)
}
