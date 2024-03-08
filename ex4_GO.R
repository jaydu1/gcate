library(clusterProfiler)
library(org.Hs.eg.db)
library(AnnotationDbi)
library(rrvgo)



df <- read.csv('result/lupus_discoveries_T4.csv', row.names=1)
df <- (df=='True')
head(df)


######################################################################
#
# GLM-oracle
#
######################################################################
genes_to_test <- sort(row.names(df)[df[,1]])
cat(length(genes_to_test)) # 72
GO_results <- enrichGO(gene = genes_to_test, OrgDb = "org.Hs.eg.db", keyType = "SYMBOL", ont = "BP")
as.data.frame(GO_results)
pdf("lupus_T4_glm.pdf", width = 8, height = 4)
plot(barplot(GO_results, showCategory = 10))
dev.off()



go_analysis <- as.data.frame(GO_results)
simMatrix <- calculateSimMatrix(go_analysis$ID,
                                orgdb="org.Hs.eg.db",
                                ont="BP",
                                method="Rel")
scores <- setNames(-log10(go_analysis$qvalue), go_analysis$ID)
reducedTerms <- reduceSimMatrix(simMatrix,
                                scores,
                                threshold=0.7,
                                orgdb="org.Hs.eg.db")

pdf("lupus_T4_glm_treemap.pdf", width = 8, height = 4)
treemapPlot(reducedTerms)
dev.off()



######################################################################
#
# Genes that are significant by both GLM-oracle and GCATE
#
######################################################################
genes_to_test <- sort(row.names(df)[df[,1] & df[,5]])
cat(length(genes_to_test)) # 15
GO_results <- enrichGO(gene = genes_to_test, OrgDb = "org.Hs.eg.db", keyType = "SYMBOL", ont = "BP")
GO_results <- as.data.frame(GO_results)
pdf("lupus_T4_common.pdf", width = 8, height = 4)
plot(barplot(GO_results, showCategory = 10))
dev.off()


go_analysis <- as.data.frame(GO_results)
simMatrix <- calculateSimMatrix(go_analysis$ID,
                                orgdb="org.Hs.eg.db",
                                ont="BP",
                                method="Rel")
scores <- setNames(-log10(go_analysis$qvalue), go_analysis$ID)
reducedTerms <- reduceSimMatrix(simMatrix,
                                scores,
                                threshold=0.7,
                                orgdb="org.Hs.eg.db")

pdf("lupus_T4_common_treemap.pdf", width = 8, height = 4)
treemapPlot(reducedTerms)
dev.off()



######################################################################
#
# Genes that are significant by CATE but insignificant by GLM-oracle
#
######################################################################

genes_to_test <- row.names(df)[(!df[,1]) & df[,4]]
cat(length(genes_to_test)) # 164
GO_results <- enrichGO(gene = genes_to_test, OrgDb = "org.Hs.eg.db", keyType = "SYMBOL", ont = "BP")
as.data.frame(GO_results)
pdf("lupus_T4_cate.pdf", width = 7, height = 9)
plot(barplot(GO_results, showCategory = 15))
dev.off()

go_analysis <- as.data.frame(GO_results)
simMatrix <- calculateSimMatrix(go_analysis$ID,
                                orgdb="org.Hs.eg.db",
                                ont="BP",
                                method="Rel")
scores <- setNames(-log10(go_analysis$qvalue), go_analysis$ID)
reducedTerms <- reduceSimMatrix(simMatrix,
                                scores,
                                threshold=0.7,
                                orgdb="org.Hs.eg.db")

pdf("lupus_T4_cate_treemap.pdf", width = 8, height = 4)
treemapPlot(reducedTerms)
dev.off()
