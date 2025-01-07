import os

def run_postprocessing(diffmod_table_path,out_dir):
    outfile_path=os.path.join(out_dir,"majority_direction_kmer_diffmod.table")

    with open(diffmod_table_path,"r") as file,open(outfile_path,"w") as outfile:
        header = file.readline()
        entries = file.readlines()
        outfile.write(header)
        header = header.strip().split(',')
        kmer_ind,dir_ind = header.index('kmer'),header.index('mod_assignment')    
        dict={}

        for line in entries:
            ln = line.strip().split(",")
            if ln[kmer_ind] not in dict:
                dict[ln[kmer_ind]] = {ln[dir_ind]:1}
            else:
                if ln[dir_ind] not in dict[ln[kmer_ind]]:
                    dict[ln[kmer_ind]][ln[dir_ind]]=1
                else:
                    dict[ln[kmer_ind]][ln[dir_ind]]+=1
        for k in dict:
            if len(dict[k]) > 1:  ##consider one modification type per k-mer
                if dict[k]['higher'] <= dict[k]['lower']: ##choose the majority
                    dict[k]['choose']='lower'
                else:
                    dict[k]['choose']='higher'
            else:
                dict[k]['choose']=list(dict[k].keys())[0]
        for line in entries:
            ln = line.strip().split(",")
            if ln[dir_ind] == dict[ln[kmer_ind]]['choose']:
                outfile.write(line)

def postprocessing(args):
    diffmod_dir = args.diffmod_dir
    diffmod_table_path = os.path.join(diffmod_dir,"diffmod.table")
    run_postprocessing(diffmod_table_path,diffmod_dir)

