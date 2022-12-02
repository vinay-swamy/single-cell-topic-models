#%%
import pandas as pd
from Bio import SeqIO
from gtfparse import read_gtf

def fasta2df(fasta_file):
    with open(fasta_file) as infasta:
        fa = SeqIO.parse(infasta, "fasta")
        res = []
        for record in fa:
            transcript_id = record.id.split(".")[0]
            aa = str(record.seq)
            res.append((transcript_id,aa, len(aa) ))

    return pd.DataFrame(res, columns=["protein_id",  "protein_seq", "aa_seq_len"])


appris_principal_isos = pd.read_csv("/data/vss2134/scTopic/data/annotation/appris_data.principal.txt", 
                                    names = ['gene_name',"gene_id", "transcript_id","ccds_id", "appris_prio", "MANE"], sep = "\t", skiprows=1)
#%%

transcript_seqs = fasta2df("/data/vss2134/scTopic/data/annotation/gencode.v42.pc_translations.fa")
#%%
gtf = read_gtf("/data/vss2134/scTopic/data/annotation/gencode.v42.annotation.gtf").assign(
    transcript_id = lambda x: x.transcript_id.str.replace(r"\.\d+$", ""),
    protein_id = lambda x: x.protein_id.str.replace(r"\.\d+$", "")
)

#%%
tx2p = gtf.query("feature == 'transcript'").query("transcript_type == 'protein_coding'")[['transcript_id', "protein_id"]]
tx2strand = gtf[gtf['feature'] == "transcript"][['transcript_id', "strand"]].drop_duplicates()
#%%
transcript_seqs = transcript_seqs.merge(tx2p)


#%%
appris_1to1 = appris_principal_isos.assign(
        prio_level = lambda x: x.appris_prio.str.split(":").apply(lambda x: x[1]),
        prio_type = lambda x: x.appris_prio.str.split(":").apply(lambda x: x[0])
    ).pipe(
        lambda x: x[x['prio_type'] == "PRINCIPAL"]
    ).sort_values(
        "prio_level"
    ).drop_duplicates(
        "gene_name"
    ).merge(
        tx2strand, how = 'inner'
    ).pipe(
        lambda x: x[x.transcript_id.isin(transcript_seqs['transcript_id'])]
    ).merge(transcript_seqs).assign(sampleid = "GRCH38")

# %%
appris_1to1.to_csv("/data/vss2134/scTopic/data/grch38_protein_seqs.csv", index = False)
# %%
