library(monocle3)
library(tidyverse)
library(Matrix)
##(genes x cells )
allcounts <- readRDS("/data/vss2134/scTopic/data/GSM4150378_sciPlex3_cds_all_cells.RDS")
keep = !grepl("ENSMUS", rownames(allcounts))
allcounts <- allcounts[keep,]

allcounts <- t(allcounts@assays@data$counts) ## (cells x genes)

writeMM(allcounts, "/data/vss2134/scTopic/data/sciplex_all_counts.mtx")
rownames(allcounts) %>% write("/data/vss2134/scTopic/data/sciplex_all_counts.mtx.rownames", sep = "\n")
colnames(allcounts) %>% write("/data/vss2134/scTopic/data/sciplex_all_counts.mtx.colnames", sep = "\n")
