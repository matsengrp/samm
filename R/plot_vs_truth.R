# plot fitted vs truth

arg <- commandArgs(TRUE)
true_data_path <- arg[1]
fitted_data_path <- arg[2]
plot_path <- arg[3]
plot_title <- arg[4]
x_str <- arg[5]
y_str <- arg[6]

getLogMutabilitiesFromPath <- function(path) {
    # what it says on the tin
    raw_data <- read.table(path, sep=',')
    log_mutabilities <- unlist(raw_data['V2'])
    names(log_mutabilities) <- unlist(raw_data['V1'])
    log_mutabilities <- log_mutabilities[!grepl("N", names(log_mutabilities))]
    return(log_mutabilities)
}

# We're plotting mutabilities, so we exponentiate and scale them
true_log_mutabilities <- getLogMutabilitiesFromPath(true_data_path)
true_scaled <- true_log_mutabilities -
    log(sum(exp(true_log_mutabilities), na.rm=TRUE))

fitted_log_mutabilities <- getLogMutabilitiesFromPath(fitted_data_path)
fitted_scaled <- fitted_log_mutabilities -
    log(sum(exp(fitted_log_mutabilities), na.rm=TRUE))

png(plot_path)

plot(true_scaled,
     fitted_scaled,
     main = plot_title,
     xlab = x_str,
     ylab = y_str)
abline(0,1)

dev.off()
