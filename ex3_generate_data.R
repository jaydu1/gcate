suppressPackageStartupMessages({
    library(splatter)
    library(scater)
})


n_list <- c(100, 200)
args <- commandArgs(trailingOnly = TRUE)
n <- n_list[1+as.numeric(args[1])]

path_data <- "data/ex3/"
dir.create(path_data, recursive=TRUE, showWarnings = FALSE)
for(seed in c(0:99)){
    cat(n, seed, '\n')
    set.seed(seed)
    
    # Simulate data using estimated parameters
    sim <- splatSimulate(
            group.prob = c(0.5, 0.5), mean.shape=0.05, mean.rate=2,
            nGenes=10000, seed=seed, de.prob=0.05, de.facLoc=1., de.facScale=0.1, de.downProb=0.5,
             lib.loc=10, lib.scale=.2,
             dropout.type="experiment", dropout.mid=20, dropout.shape=0.001,
             bcv.common = 0.2, bcv.df = 100,
             batchCells = rep(n/4, 4), batch.facLoc=1., batch.facScale=.2,
             method = "groups", verbose=F)

    cat(mean(counts(sim)!=0.), '\n')
    cat(sum(rowSums(counts(sim)>0)>10), '\n')
    cat(mean(counts(sim)[rowSums(counts(sim)>0)>10,]!=0.), '\n')
    
    signal <- (rowData(sim)$DEFacGroup1 !=1) | (rowData(sim)$DEFacGroup2 !=1)
    cat(sum(signal[rowSums(counts(sim)>0)>10])/sum(rowSums(counts(sim)>0)>10), '\n')
    
    hist(rowData(sim)$DEFacGroup1[rowData(sim)$DEFacGroup1!=1])
    
    
    
    
    Y <- as.matrix(t(counts(sim)))
    B <- signal
    B <- B[colSums(Y>0)>10]
    Y <- Y[,colSums(Y>0)>10]
    X <- 2 * (colData(sim)$Group == 'Group1') - 1
    X <- data.frame(intercept=rep(1,length(X)), log_libsize=log(rowSums(Y)), X=X)
    Z <- model.matrix(~0+colData(sim)$Batch)[,-4]
    colnames(Z) <- c('batch_1','batch_2','batch_3')
    Z <- data.frame(Z)
    write.csv(Y, sprintf("%s%d_%d_Y.csv",path_data,n,seed), row.names=F)
    write.csv(B, sprintf("%s%d_%d_B.csv",path_data,n,seed), row.names=F)
    write.csv(X, sprintf("%s%d_%d_X.csv",path_data,n,seed), row.names=F)
    write.csv(Z, sprintf("%s%d_%d_Z.csv",path_data,n,seed), row.names=F)
}
