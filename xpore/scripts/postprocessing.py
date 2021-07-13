import argparse
import os

def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    required = parser.add_argument_group('required arguments')
    # Required arguments
    required.add_argument('--diffmod_dir', dest='diffmod_dir', help='diffmod directory path, the output from xpore-diffmod.',required=True)

    return parser.parse_args()

def postProcessing(diffmod_table_path,out_dir):
    file=open(diffmod_table_path,"r")
    header=file.readline()
    entries=file.readlines()
    outfile_path=os.path.join(out_dir,"majority_direction_kmer_diffmod.table")
    outfile=open(outfile_path,"w")
    outfile.write(header)
    header=header.strip().split(',')
    kmer_ind,dir_ind=header.index('kmer'),header.index('mod_assignment')    
    dict={}
    for ln in entries:
        l=ln.strip().split(",")
        if l[kmer_ind] not in dict:
            dict[l[kmer_ind]]={l[dir_ind]:1}
        else:
            if l[dir_ind] not in dict[l[kmer_ind]]:
                dict[l[kmer_ind]][l[dir_ind]]=1
            else:
                dict[l[kmer_ind]][l[dir_ind]]+=1
    for k in dict:
        if len(dict[k]) > 1:  ##consider one modification type per k-mer
            if dict[k]['higher'] <= dict[k]['lower']: ##choose the majority
                dict[k]['choose']='lower'
            else:
                dict[k]['choose']='higher'
        else:
            dict[k]['choose']=list(dict[k].keys())[0]
    for ln in entries:
        l=ln.strip().split(",")
        if l[dir_ind] == dict[l[kmer_ind]]['choose']:
            outfile.write(ln)
    outfile.close()

def main():
    args = get_args()
    diffmod_dir = args.diffmod_dir
    diffmod_table_path = os.path.join(diffmod_dir,"diffmod.table")
    postProcessing(diffmod_table_path,diffmod_dir)

if __name__ == '__main__':
    """
    Usage:
        xpore-postprocessing --diffmod_dir DIFFMOD_DIR
    """
    main()
