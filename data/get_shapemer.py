#!/usr/bin/env python3
"""
Author: Joe Zhang
Student ID: 1206052
Script for extracting shapemers for each protein from pdb files in Alphafold database according to the CAFA5 training protein ID lists.
"""

import os
import sys
import pandas as pnd
from time import time
from geometricus import get_invariants_for_structures, Geometricus, SplitInfo, SplitType
import numpy as np
import requests
import time
import csv
import multiprocessing as mp

def get_entries(file = "train_sequences.fasta"):
    '''
    infile: input fasta file.
    outfile: output file only contain entries.
    this function can extract entries from fasta file 
    '''
    entries = []
    infile = "/lustre/BIF/nobackup/zhang408/cafa5/Train/" + file
    with open(infile) as fl_in:
        lines = fl_in.readlines()
    entry_lines = list(filter(lambda line: line.startswith(">"), lines))
    for line in entry_lines:
        entries.append(line.split(" ")[0][1:])
    print(f"There are {len(entries)}")
    return entries

def download_pdb(entry):
    '''
    entry: a string of protein entry in uniprot
    The function would download pdb file from corresponding alphafold database entry. 
    '''
    response = requests.get(f"http://alphafold.ebi.ac.uk/files/AF-{entry}-F1-model_v4.pdb")
    print(response.status_code)
    print(f"{entry}.pdb")
    if response.status_code == 200:
        with open(f"{entry}.pdb", "w") as fl:
            fl.write(response.text)
            return 0
    elif response.status_code == 404:
        print(f"{entry} is not found in alphafold database")
        return 1

def get_geometricus(pdb_file,_type:str,length: int, resolution:int):
    """
    pdb_file: the pdb file which is downloaded from website
    This function would return geometricus embedding results.
    """
    _types = ["KMER","RADIUS"]
    if _type not in _types:
        raise ValueError(f"{_type} is not valid")
    if _type == "KMER":
        split_type = SplitType.KMER
    elif _type == "RADIUS":
        split_type = SplitType.RADIUS
    invariants, _ = get_invariants_for_structures([pdb_file], n_threads=4,
                                              split_infos=[SplitInfo(split_type, length)],moment_types=["O_3", "O_4", "O_5", "F"])
    shapemer_class = Geometricus.from_invariants(invariants, protein_keys=[pdb_file],resolution = resolution) 

    return shapemer_class

def transcript(file:str, _type:str, size: int, resolution:int):
    """
    _type: shapemer type. KMER or RADIUS
    size: length of the KMER or RADIUS
    resolution: coraseness of the shapemer
    combine all the function above.
    write shapemer results to the targeted file
    """
    entries = get_entries(file)
    unfound = []
    with open(f"train_shapemers_{_type}_{size}_{resolution}", "w") as outfl:
        for entry in entries:         
            flag = download_pdb(entry)
            if flag  == 0:
                pdb = f"{entry}.pdb"
                outfl.write(f">{entry}\n")
                writer = csv.writer(outfl)
                shapemers  = get_geometricus(pdb,_type,size,resolution).proteins_to_shapemers
                for MI in list(shapemers.values())[0]:
                    writer.writerow(MI)
                if os.path.exists(pdb):
                    os.remove(pdb)
            elif flag == 1:
                unfound.append(entry)
                continue 

def main():
    _type = ["KMER","RADIUS"]
    size_KMER = [8,16]
    size_RADIUS = [5,10]
    resolution = [1.,2.,3.]

    p1 = mp.Process(target=transcript, args = ("train_sequences.fasta","KMER", 8, 1))
    p2 = mp.Process(target=transcript, args = ("train_sequences.fasta1","KMER", 8, 2)) 
    p3 = mp.Process(target=transcript,args = ("train_sequences.fasta","KMER", 8, 3))
    #p4 = mp.Process(target=transcript, args = ("train_sequences.fasta","KMER", 16, 1))
    #p5 = mp.Process(target=transcript, args=("train_sequences.fasta","KMER", 16, 2))
    #p6 = mp.Process(target=transcript,args=("train_sequences.fasta","KMER", 16, 3))
    #p01 = mp.Process(target=transcript("train_sequences.fasta","RADIUS", 5, 1))
    #p02 = mp.Process(target=transcript("train_sequences.fasta1","RADIUS", 5, 2)) 
    #p03 = mp.Process(target=transcript("train_sequences.fasta2","RADIUS", 5, 3))
    #p04 = mp.Process(target=transcript("train_sequences.fasta3","RADIUS", 10, 1))
    #p05 = mp.Process(target=transcript("train_sequences.fasta4","RADIUS", 10, 2))
    #p06 = mp.Process(target=transcript("train_sequences.fasta5","RADIUS", 10, 3))

    p1.start()
    p2.start()
    p3.start()
    #p4.start()
    #p5.start()
    #p6.start()
    #p01.start()
    #p02.start()
    #p03.start()
    #p04.start()
    #p05.start()
    #p06.start()

    p1.join()
    p2.join() 
    p3.join()
    #p4.join()
    #p5.join()
    #p6.join()
    #p01.join()
    #p02.join() 
    #p03.join()
    #p04.join()
    #p05.join()
    #p06.join()

    
    


  


if __name__ == "__main__":
    main()
