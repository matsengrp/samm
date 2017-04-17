plotBarchart <- function (model, nucleotides = c("A", "C", "G", "T"),
                          style = c("hedgehog", "bar"), size = 1,
                          bar.size = 0, ...)
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
    if (is(model, "TargetingModel")) {
        model <- model@mutability
    }
    else if (!is(model, "vector")) {
        stop("Input must be either a mutability vector or TargetingModel object.")
    }
    # Processing data
    mut_scores <- model[!grepl("N", names(model))]
    mut_scores[!is.finite(mut_scores)] <- 0
    mut_words <- names(mut_scores)
    mut_positions <- as.data.frame(t(sapply(mut_words, seqinr::s2c)))
    motif_len <- ncol(mut_positions)
    if (!(motif_len %% 2) | !(motif_len > 1)) {
        stop("Motif length must be odd and greater than one.")
    }
    data_cols <- paste0("pos", 1:motif_len)
    motif_half_len <- motif_len/2 - .5
    center_nuc_col <- paste0('pos', motif_half_len+1)
    flank_cols <- data_cols[-motif_half_len-1]
    colnames(mut_positions) <- data_cols
    mut_df <- data.frame(word = mut_words, score = mut_scores, mut_positions)

    # Setting up plotting environment
    base_theme <- theme_bw() +
                  theme(panel.margin = grid::unit(0, "lines"),
                        panel.background = element_blank()) +
                  theme(axis.text = element_text(margin = grid::unit(0, "lines"))) +
                  theme(text = element_text(size = 10 * size),
                        title = element_text(size = 10 * size),
                        legend.margin = grid::unit(0, "lines"),
                        legend.background = element_blank())
    score_offset <- 0
    score_scale <- 15
    text_offset <- -5.6
    motif_colors <- setNames(c("#4daf4a", "#e41a1c", "#094d85",
        "#999999"), c("WA/TW", "WRCY/RGYW", "SYC/GRS", "Neutral"))
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

    # Get max score for y axis limits
    score_max <- max(mut_df$score, na.rm = TRUE)
    .transform_score <- function(x) {
        x/score_max * score_scale + score_offset
    }
    .invert_score <- function(x) {
        (x - score_offset)/score_scale * score_max
    }
    mut_df$score <- .transform_score(mut_df$score)

    # Generate plots for each nucleotide of interest
    plot_list <- list()
    for (center_nuc in nucleotides) {
        sub_df <- mut_df[mut_df[,center_nuc_col] == center_nuc, ]
        if (center_nuc %in% c("A", "C")) {
            # 3' for A/C
            sub_df <- dplyr::arrange_(sub_df, .dots = flank_cols)
            sub_df$x <- -.5 + 1:nrow(sub_df)
        }
        else if (center_nuc %in% c("G", "T")) {
            # 5' for G/T
            sub_df <- dplyr::arrange_(sub_df, .dots = rev(flank_cols))
            sub_df$x <- -.5 + 1:nrow(sub_df)
        }
        else {
            stop("Invalid nucleotide choice")
        }

        # Create coordinates for nucleotide rectangles
        sub_melt <- sub_df %>%
                    tidyr::gather_("pos", "char", colnames(mut_positions)) %>%
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
                                 text_y = i,
                                 text_label = factor(nuc_rle$values,
                                                     levels = names(dna_colors)),
                                 rect_min = rect_min,
                                 rect_max = rect_max)
            sub_text[[i]] <- tmp_df
        }
        sub_melt$pos <- sub_melt$pos + text_offset

        # Shorten/lengthen the motif/text rectangles based on how many we have
        motif_offset <- 5 - motif_len
        sub_text <- lapply(
            sub_text,
            function(x) {
                dplyr::mutate_(x,
                               text_y = interp(~y + text_offset + motif_offset,
                                               y = as.name("text_y")))
            })

        sub_rect <- dplyr::bind_rows(sub_text) %>%
                    mutate_(rect_width = interp(~y - x,
                                                x = as.name("rect_min"),
                                                y = as.name("rect_max")),
                            ymin = interp(~y - .5, y = as.name("text_y")),
                            ymax = interp(~y + .5, y = as.name("text_y")))

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
              geom_text(data = sub_text[[motif_half_len+1]],
                        mapping = aes_string(x = "text_x",
                                             y = "text_y",
                                             label = "text_label"),
                        color = "black",
                        hjust = 0.5,
                        vjust = 0.5,
                        size = 3 * size,
                        fontface = 2)

        # Only plot at most two levels of text---otherwise a little busy
        if (center_nuc %in% c("A", "C")) {
            for (flank in 1:min(motif_half_len, 2)) {
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
        else if (center_nuc %in% c("G", "T")) {
            for (flank in rev(1 + motif_len - 1:min(motif_half_len, 2))) {
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
                  theme(plot.margin = grid::unit(c(0, 0, 0, 0), "lines"),
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
            y_breaks <- seq(score_offset, score_scale + score_offset, 1)
            y_limits <- c(-.01 + text_offset + motif_offset + .5, score_scale + score_offset)
            sub_colors <- motif_colors[names(motif_colors) %in% sub_df$motif]
            p1 <- p1 +
                  theme(plot.margin = grid::unit(c(1, 1, 1, 1), "lines"),
                        panel.grid = element_blank(),
                        panel.border = element_rect(color = "black"),
                        axis.text.x = element_blank(),
                        axis.ticks.x = element_blank(),
                        legend.position = "top") +
                  guides(color = guide_legend(override.aes = list(fill = sub_colors,
                                                                  linetype = 0))) +
                  ylab("Mutability") +
                  scale_x_continuous(expand = c(0, 1)) +
                  scale_y_continuous(limits = y_limits,
                                     breaks = y_breaks,
                                     expand = c(0, 0.5),
                                     labels = function(x)
                                         scales::scientific(.invert_score(x))) +
                  geom_bar(data = sub_df,
                           mapping = aes_string(x = "x",
                                                y = "score",
                                                fill = "motif",
                                                color = "motif"),
                           stat = "identity",
                           position = "identity",
                           size = bar.size,
                           width = 0.7)
        }

        # Add plots to list
        p1 <- p1 + do.call(theme, list(...))
        plot_list[[center_nuc]] <- p1
    }
    return(plot_list)
}
