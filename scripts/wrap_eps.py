import subprocess as sp 
import sys 
import pandas as pd 
import time 

# #%%



dataset = sys.argv[1]
outpath = sys.argv[2]
bs = 1
#%%

code = 111
while code !=0:
    print(f"wrapper::running with batchsize {bs}")
    print("wrapper::\trunning seqs")
    print(f"python /data/vss2134/scTopic/scripts/embed_protein_seqs.py {dataset} {bs} {outpath}")
    cp_short = sp.run(f"python /data/vss2134/scTopic/scripts/embed_protein_seqs.py {dataset} {bs} {outpath}",
                       shell = True)
    #time.sleep(5)
    
 
    if (cp_short.returncode != 0 ):
        bs = int(bs/2)
    else:
        sys.exit()
    if bs < 2:
        sys.exit()