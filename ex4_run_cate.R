library(tidyverse)
library(cate)
library(qvalue)


library(reticulate)
# version <- "3.8.1"
# install_python(version)
# py_install("numpy")
# use_virtualenv("~/.virtualenvs/r-reticulate")
np <- import("numpy")


cell_types <- c('T4', 'cM', 'B', 'T8', 'NK')


path_load <- sprintf('data/lupus/')
path_result <- sprintf('result/lupus/')
dir.create(path_result, recursive=TRUE, showWarnings = FALSE)


##############################################################################
#
# CATE on 2000 highly variable genes
#
##############################################################################
for(cell_type in cell_types){
    cat(cell_type, '\n')

    npz1 <- np$load(sprintf("%sdata_lupus_%s.npz", path_load, cell_type))
    
    Y <- as.matrix(npz1$f[["Y"]])
    log_libsize <- log(rowSums(Y))
    id_genes <- (colSums(Y!=0)>=10)
    Y <- Y[,id_genes]
    Y <- t(apply(Y, 1, function(x){x/sum(x)*10^4}))
    X <- as.matrix(npz1$f[["X"]])
    X <- cbind(rep(1, length(log_libsize)), log_libsize, X[,-c(1:5)])
    X <- data.frame(X)
    colnames(X) <- c('X1','X2', 'X3')
    cat(dim(Y), dim(X))
    
    factor.num <- est.confounder.num(
        ~ X3 | . - X3, X, log2(Y+1),
        method = "bcv", bcv.plot = FALSE, rmax = 10, nRepeat = 20)
    cat(factor.num$r)

    # estimation is infeasible for B cell type
    if(cell_type == 'B'){
        X <- X[,-2]
    }

    for(calibrate in c(TRUE,FALSE)){
        cate.results <- cate(~ X3 | . - X3, X, log2(Y+1), r = factor.num$r, calibrate=calibrate)
        
        p <- dim(Y)[1]
        t <- cate.results$beta.t
        beta_hat <- cate.results$beta
        
        p_values <- 2*pnorm(q=abs(t), lower.tail=FALSE)
        
        df <- data.frame(
            'beta_hat'=beta_hat,
            'z_scores'=t,
            'p_values'=p_values,
            'q_values'=qvalue(p_values)$qvalues, check.names=F
        )
        names(df) <- c('beta_hat', 'z_scores', 'p_values', 'q_values')
        
        if(calibrate){
            filename <- sprintf('%scate_%s.csv', path_result, cell_type)
        }else{
            filename <- sprintf('%scate_raw_%s.csv', path_result, cell_type)
        }
        write.csv(df, filename, row.names = F)
    }
}







##############################################################################
#
# CATE on 250 highly variable genes
#
##############################################################################
r <- 5
for(cell_type in cell_types){
    cat(cell_type, '\n')

    npz1 <- np$load(sprintf("%sdata_lupus_%s_250.npz",path_load,cell_type))
    
    Y <- as.matrix(npz1$f[["Y"]])
    log_libsize <- log(rowSums(Y))
    id_genes <- (colSums(Y!=0)>=5)
    Y <- Y[,id_genes]
    Y <- t(apply(Y, 1, function(x){x/sum(x)*10^4}))
    X <- as.matrix(npz1$f[["X"]])
    X <- cbind(rep(1, length(log_libsize)), log_libsize, X[,-c(1:5)])
    X <- data.frame(X)
    colnames(X) <- c('X1','X2', 'X3')
    cat(dim(Y), dim(X))
    
    for(calibrate in c(TRUE,FALSE)){
        cate.results <- cate(~ X3 | . - X3, X, log2(Y+1), r = r, calibrate=calibrate)
        
        p <- dim(Y)[1]
        t <- cate.results$beta.t
        beta_hat <- cate.results$beta
        
        p_values <- 2*pnorm(q=abs(t), lower.tail=FALSE)
        
        df <- data.frame(
            'beta_hat'=beta_hat,
            'z_scores'=t,
            'p_values'=p_values,
            'q_values'=qvalue(p_values)$qvalues, check.names=F
        )
        names(df) <- c('beta_hat', 'z_scores', 'p_values', 'q_values')
        
        if(calibrate){
            filename <- sprintf('%scate_%s.csv', path_result, cell_type)
        }else{
            filename <- sprintf('%scate_raw_%s.csv', path_result, cell_type)
        }
        write.csv(df, filename, row.names = F)
    }
}
