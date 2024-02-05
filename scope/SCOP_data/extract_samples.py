import os
import requests
import sys
from geometricus import get_invariants_for_structures, Geometricus, SplitInfo, SplitType
import csv
import multiprocessing as mp
import pickle

def extract(file_name):
    CLs = ['1000000', '1000001','1000002', '1000003']
    pros = {}
    for cl in CLs:
        pros[cl] = []

    with open(file_name, "r" ) as input:
        records = input.readlines()
        for record in records:
            if not record.startswith("#") :
                CL = record.rstrip().split(" ")[-1].split(",")[1][3:]
                if CL in CLs and (len(pros[CL]) < 10000):
                    pro = record.split(" ")[1]
                    response = requests.get(f"https://files.rcsb.org/view/{pro}.pdb")
                    response1 = requests.get(f"https://www.rcsb.org/fasta/entry/{pro}/display")
                    lines = [line for line in response1.text.rstrip().split('\n') if line.startswith('>')]
                    if response.status_code == 200 and len(lines) == 1:
                        pros[CL].append(pro)
    with open("pdb_classes_10000", "w") as outfl:
        for cl in pros:
            for pro in pros[cl]:
                outfl.write(f"{pro} {cl}\n")
    return pros

def seq_pdb(pros):
    count = 0
    print("1")
    with open(f"samples_10000.fasta", "w") as fl:
        print("@")
        for key in pros:
            for pro in pros[key]:
                response = requests.get(f"https://www.rcsb.org/fasta/entry/{pro}/display")
                if response.status_code == 200:
                    fl.write(response.text)
                    count += 1 
                else:
                    print(f"{pro} is not found")
        print(count)

def MI(pros,_type = 'KMER', length = 8, resolution = 1):
    _types = ["KMER","RADIUS"]
    with open(f"sample_MI_10000", "a") as outfl:
        for key in pros:
            for pro in pros[key]:
                response = requests.get(f"https://files.rcsb.org/view/{pro}.pdb")
                with open(f"{pro}.pdb", "w") as fl:
                    fl.write(response.text)
                    pdb = f"{pro}.pdb"
                if _type == "KMER":
                    split_type = SplitType.KMER
                elif _type == "RADIUS":
                    split_type = SplitType.RADIUS
                invariants, _ = get_invariants_for_structures([pdb], n_threads=4,
                                                        split_infos=[SplitInfo(split_type, length)],moment_types=["O_3", "O_4", "O_5", "F"])
                shapemer_class = Geometricus.from_invariants(invariants, protein_keys=[pdb],resolution = resolution).proteins_to_shapemers 
                if os.path.exists(pdb):
                    os.remove(pdb)
                    outfl.write(f">{pro}\n")
                    writer = csv.writer(outfl)
                    for MI in list(shapemer_class.values())[0]:
                        writer.writerow(MI)
                else:
                    with open('unfound.txt',"a") as unfl:
                        unfl.write(f"the protein {pro} do not have geometricus\n")
    return 0

def count_matrix(file,_type = 'KMER', length = 8, resolution = 1,output= 'count_matrix_10000.pkl'):
    with open(file) as fl:
        if _type == "KMER":
            split_type = SplitType.KMER
        elif _type == "RADIUS":
            split_type = SplitType.RADIUS
        proteins = []
        lines = fl.readlines()
        for line in lines:
            proteins.append(line.split(' ')[0])
        print(len(proteins))
    invariants, _ = get_invariants_for_structures(proteins, n_threads=4,
                                                        split_infos=[SplitInfo(split_type, length)],moment_types=["O_3", "O_4", "O_5", "F"])
    shapemer_class = Geometricus.from_invariants(invariants, protein_keys=proteins,resolution = resolution).proteins_to_shapemers 
    shapemer_count_matrix = shapemer_class.get_count_matrix()
    print(shapemer_count_matrix.shape)
    file = open(output, 'wb')
    pickle.dump(shapemer_count_matrix, file)
    file.close()

if __name__ == "__main__":
    pros = extract("scop-cla-latest.txt")
    print(pros)
    seq_pdb(pros)
    #MI(pros, 'KMER', 8, 1)
    #count_matrix('/lustre/BIF/nobackup/zhang408/data_prepare/scope/data_10000/pdb_classes_10000', 'KMER', 8, 1, 'count_matrix_10000.pkl')