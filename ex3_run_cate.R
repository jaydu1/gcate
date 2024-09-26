library(tidyverse)
library(tibble)
library(cate)
library(qvalue)


n_list <- c(100, 200)
r <- 3
path_data <- 'data/ex3/'
path_result <- 'result/ex3/'
dir.create(path_result, recursive=TRUE, showWarnings = FALSE)
for(n in n_list){
    for(seed in c(0:99)){
        cat(n, seed, '\n')

        Y <- read.csv(sprintf("%s%d_%d_Y.csv",path_data,n,seed))
        Y <- as.matrix(Y)
        Y <- t(apply(Y, 1, function(x){x/sum(x)*10^4}))
        B <- read.csv(sprintf("%s%d_%d_B.csv",path_data,n,seed))[,1]
        df_X <- read.csv(sprintf("%s%d_%d_X.csv",path_data,n,seed))[,-2]
        id_genes <- (colSums(Y!=0)>=10)

        factor.num <- est.confounder.num(
            ~ X | . - X + 0, df_X, log2(Y[,id_genes]+1),
            method = "bcv", bcv.plot = FALSE, rmax = 5, nRepeat = 20)

        for(calibrate in c(TRUE,FALSE)){

            df <- tryCatch({
                cate.results <- cate(~ X | . - X + 0, df_X, log2(Y[,id_genes]+1), r = factor.num$r, calibrate = calibrate)

                p <- dim(Y)[1]
                t <- cate.results$beta.t
                beta_hat <- cate.results$beta

                p_values <- 2*pnorm(q=abs(t), lower.tail=FALSE)

                df <- data.frame(
                    'signals'=(B[id_genes]!=0.),
                    'beta_hat'=beta_hat,
                    'z_scores'=t,
                    'p_values'=p_values,
                    'q_values'=qvalue(p_values)$qvalues, check.names=F
                )
                df
            },
            error = function(e) {
                p <- sum(colSums(Y!=0)>0)
                df <- data.frame(
                    'signals'=rep(NA, p),
                    'beta_hat'=rep(NA, p),
                    'z_scores'=rep(NA, p),
                    'p_values'=rep(NA, p),
                    'q_values'=rep(NA, p), check.names=F
                )
                df
            })
            
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
