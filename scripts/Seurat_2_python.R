library(tidyverse)
library(sparseMatrixStats)
library(Seurat)
library(Matrix)
library(glue)


load("data/SRP050054.seurat.Rdata")
#scEiaD
mat = scEiaD@assays$RNA@counts ## genes x cells 
metadata = as_tibble(scEiaD@meta.data)
dim(mat) ## genes x samples
exp_in_ncells = rowSums(mat >0)
quantile(exp_in_ncells, seq(0,1,.1))
keep = (exp_in_ncells > 10) & (exp_in_ncells < 3000)

mat_expfilt = mat[keep,]
dim(mat_expfilt)
gene_var = rowVars(mat_expfilt)
co = gene_var %>% sort %>% tail(5001) %>% head(1)

mat_hvg = t(as.matrix(mat_expfilt[gene_var > co,]))
mode(mat_hvg) <- "integer" ## hacking it back down to integer counts 

mat_final = Matrix(mat_hvg[rowSums(mat_hvg >0) > 100,])

metadata = metadata %>% filter(Barcode %in% rownames(mat_final))

writeMM(mat_final, "mackosko_scrnaseq_preprocessed.mtx")
write(rownames(mat_final), "mackosko_scrnaseq_preprocessed.mtx.rownames", sep = "\n")
write(colnames(mat_final), "mackosko_scrnaseq_preprocessed.mtx.colnames", sep = "\n")
write_csv(metadata, "mackosko_preprocessed_metadata.csv")


load("data/counts.Rdata")
