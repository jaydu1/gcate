library(tidyverse)
library(cate)
library(qvalue)


library(reticulate)
# version <- "3.8.1"
# install_python(version)
# py_install("numpy")
# use_virtualenv("~/.virtualenvs/r-reticulate")
np <- import("numpy")

n_list <- c(100, 250)
r_list <- c(2, 10)

d <- 2
path_load <- sprintf('data/ex2/poisson/')
path_result <- sprintf('result/ex2/poisson/')
dir.create(path_result, recursive=TRUE, showWarnings = FALSE)
for(n in n_list){
    for(r in r_list){
        for(seed in c(0:99)){
            cat(n, r, seed, '\n')

            npz1 <- np$load(sprintf("%sn_%d_r_%d_seed_%d.npz",path_load,n,r,seed))
            npz1$files
            Y <- as.matrix(npz1$f[["Y"]])
            X <- data.frame(npz1$f[["X"]])[,-1, drop=FALSE]
            B <- npz1$f[["B"]][,d+1]


            factor.num <- est.confounder.num(
                ~ X3 | . - X3 + 0, X, log2(Y+1),
                method = "bcv", bcv.plot = FALSE, rmax = 12, nRepeat = 20)

            for(calibrate in c(TRUE,FALSE)){
                cate.results <- cate(~ X3 | . - X3 + 0, X, log2(Y+1), r = factor.num$r, calibrate=calibrate)

                p <- dim(Y)[1]
                t <- rep(NA, p)
                t[colSums(Y)!=0] <- cate.results$beta.t
                beta_hat <- rep(NA, p)
                beta_hat[colSums(Y)!=0] <- cate.results$beta

                p_values <- 2*pnorm(q=abs(t), lower.tail=FALSE)

                df <- data.frame(
                    'signals'=(B!=0.),
                    'beta_hat'=beta_hat,
                    'z_scores'=t,
                    'p_values'=p_values,
                    'q_values'=qvalue(p_values)$qvalues, check.names=F
                )
                names(df) <- c('signals', 'beta_hat', 'z_scores', 'p_values', 'q_values')

                if(calibrate){
                    filename <- sprintf('%scate_%d_%d_%d.csv', path_result, n, r, seed)
                }else{
                    filename <- sprintf('%scate_raw_%d_%d_%d.csv', path_result, n, r, seed)
                }
                write.csv(df, filename, row.names = F)
            }
        }
    }
}