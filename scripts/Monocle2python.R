library(monocle3)
library(tidyverse)
library(Matrix)
library(matrixStats)
library(glue)
##(genes x cells )
allcounts <- readRDS("/data/vss2134/scTopic/data/GSM4150378_sciPlex3_cds_all_cells.RDS")
keep = !grepl("ENSMUS", rownames(allcounts))
allcounts <- allcounts[keep,]

allcounts <- t(allcounts@assays@data$counts) ## (cells x genes)


avail_proteins = list.files("/data/vss2134/scTopic/data/protein_embedding/GRCH38/") %>% str_remove_all("\\.pickle")
protein_seq_md = read_csv("/data/vss2134/scTopic/data/grch38_protein_seqs.csv") %>% filter(transcript_id %in% avail_proteins)
count_genes = colnames(allcounts) %>% str_remove_all("\\.\\d+$")
sum(protein_seq_md$gene_id %in% count_genes)
allcounts = allcounts[,count_genes %in% protein_seq_md$gene_id]
keep_cells = rowSums(allcounts) > 0
keep_genes = colSums(allcounts) > 0
allcounts = allcounts[keep_cells,keep_genes] 
## filter cells by number of expressed genes 
n_genes_exp_in_cell= rowSums(allcounts > 0)
quantile(n_genes_exp_in_cell, seq(0.05,.95,.1))
## keep cells with at least 500 expressed genes, and remove cells with more than 2500 expressed genes 
keep_nge = (n_genes_exp_in_cell > 500 )&(n_genes_exp_in_cell < 2500)
sum(keep_nge)
allcounts = allcounts[keep_nge, ]

## filter genes that are expressed in many cells 
n_cells_exp_gene = colSums(allcounts > 0)
quantile(n_cells_exp_gene, seq(0.05,.95,.1))
## drop genes expressed in more than 100K cells and detected at least 25 times 
keep_nce = (n_cells_exp_gene <=100000) & (n_cells_exp_gene > 25)
allcounts = allcounts[,keep_nce]

## final matrix 
dim(allcounts)

## split up matrices by highly variable genes -> 5K, 10K, all
gene_var = tibble(gene=colnames(allcounts), var = colVars(allcounts)) %>% arrange()

all_counts_5k = allcounts[,colnames(allcounts) %in% tail(gene_var$gene, 5000)]
all_counts_10k = allcounts[,colnames(allcounts) %in% tail(gene_var$gene, 10000)]

write_matrix = function(x, stem){
    system(glue("mkdir -p {stem}"))
    writeMM(x, glue("{stem}/counts.mtx" ))
    rownames(x) %>% write(glue("{stem}/counts.mtx.rownames"), sep = "\n")
    colnames(x) %>% write(glue("{stem}/counts.mtx.colnames"), sep = "\n")
}
write_matrix(all_counts_5k, "/data/vss2134/scTopic/data/sciplex_counts_5kg")
write_matrix(all_counts_10k, "/data/vss2134/scTopic/data/sciplex_counts_10kg")
write_matrix(allcounts, "/data/vss2134/scTopic/data/sciplex_counts_all")

