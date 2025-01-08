import numpy as np
import pandas
import os
import multiprocessing 
import ujson
from collections import defaultdict
import csv

from . import helper
from ..diffmod.configurator import Configurator
from ..diffmod.gmm import GMM
from ..diffmod import io
from ..diffmod.statstest import StatsTest
        
def execute(idx, data_dict, data_info, method, criteria, model_kmer, prior_params, out_paths, save_models,locks):
    """
    Run the model on each position across the given idx.
    """
    data = io.load_data(idx,data_dict,min_count=criteria['readcount_min'],max_count=criteria['readcount_max'],pooling=method['pooling']) 
    models = dict()
    for key, data_at_pos in data.items(): # For each position
        idx, pos, kmer = key
        kmer_signal = {'mean': model_kmer.loc[kmer,'model_mean'],'std': model_kmer.loc[kmer,'model_stdv']}
        kmer_signal['tau'] = 1./(kmer_signal['std']**2)
        _y_mean = data_at_pos['y'].mean()
        _y_tau = 1./(data_at_pos['y'].std()**2)

        K = 2
        if method['pooling']:
            n_groups = len(data_at_pos['condition_names'])
        else:
            n_groups = len(data_at_pos['run_names'])

        ### Set up priors.
        priors = {'mu_tau':defaultdict(list),'w':dict()}

        for k in range(K):
            priors['mu_tau']['location'] += [kmer_signal['mean']]
            priors['mu_tau']['lambda'] += [prior_params['mu_tau']['lambda'][k]]
            priors['mu_tau']['alpha'] += [kmer_signal['tau']]
            priors['mu_tau']['beta'] += [prior_params['mu_tau']['beta_scale'][k]*1./kmer_signal['tau']]
        
        for k,v in priors['mu_tau'].items():
            priors['mu_tau'][k] = np.array(v)
        
        priors['w']['concentration'] = np.ones([n_groups,K])*1. #GK
        priors['w']['concentration'][:,0] = float(prior_params['w']['concentration'][0])
        priors['w']['concentration'][:,1] = float(prior_params['w']['concentration'][1])

        ### Fit a model.
        if method['prefiltering']:
            pval = StatsTest(data_at_pos).fit(method=method['prefiltering']['method'])
            if np.isnan(pval) | (pval < method['prefiltering']['threshold']):
                models[key] = (GMM(method,
                                  data_at_pos,
                                  priors=priors,
                                  kmer_signal=kmer_signal).fit(),
                                  {method['prefiltering']['method']:pval})
        else:
            models[key] = (GMM(method,
                              data_at_pos,
                              priors=priors,
                              kmer_signal=kmer_signal).fit(),
                              None)
    #END
        
    if save_models & (len(models)>0): #todo: 
        print(out_paths['model_filepath'], idx)
        io.save_models_to_hdf5(models,out_paths['model_filepath'])
        
    if len(models)>0:
        # Generating the result table.
        table = tabulate_results(models, data_info)
        with locks['table'], open(out_paths['table'],'a') as f:
            csv.writer(f,delimiter=',').writerows(table)
        # # Logging
        # log_str = '%s: Saving the result table ... Done.' %(idx)
        # with locks['log'], open(out_paths['log'],'a') as f:
        #     f.write(log_str + '\n')
        
    # Logging
    with locks['log'], open(out_paths['log'],'a') as f:
        f.write(idx + '\n')
                        
def tabulate_results(models, data_info):  # per idx (gene/transcript)
    """
    Generate a table containing learned model parameters and statistic tests.

    Parameters
    ----------
    models
        Learned models for individual genomic positions of a gene.
    group_labels
        Labels of samples.
    data_inf
        Dict

    Returns
    -------
    table
        List of tuples.
    """
    # information from the config file used for modelling.
    condition_names,run_names = get_ordered_condition_run_names(data_info)

    table = []
    for key, (model,prefiltering) in models.items():
        idx, position, kmer = key
        mu = model.nodes['mu_tau'].expected()  # K
        sigma2 = 1./model.nodes['mu_tau'].expected(var='gamma')  # K
        # these were not used anywhere; suppose they were for testing?
        # var_mu = model.nodes['mu_tau'].variance(var='normal')  # K
        # mu = model.nodes['y'].params['mean']
        # sigma2 = model.nodes['y'].params['variance']
        # N = model.nodes['y'].params['N'].round()  # GK
        w = model.nodes['w'].expected()  # GK
        coverage = np.sum(model.nodes['y'].params['N'], axis=-1)  # GK => G # n_reads per group

        p_overlap, list_cdf_at_intersections = stats.calc_prob_overlapping(mu, sigma2)

        model_group_names = model.nodes['x'].params['group_names'] #condition_names if pooling, run_names otherwise.
        
        ### Cluster assignment ###
        conf_mu = [calculate_confidence_cluster_assignment(mu[0],model.kmer_signal),calculate_confidence_cluster_assignment(mu[1],model.kmer_signal)]
    
        cluster_idx = {}
        if conf_mu[0] > conf_mu[1]:
            cluster_idx['unmod'] = 0
            cluster_idx['mod'] = 1
        else:
            cluster_idx['unmod'] = 1
            cluster_idx['mod'] = 0

        mu_assigned = [mu[cluster_idx['unmod']],mu[cluster_idx['mod']]]
        sigma2_assigned = [sigma2[cluster_idx['unmod']],sigma2[cluster_idx['mod']]] 
        conf_mu = [conf_mu[cluster_idx['unmod']],conf_mu[cluster_idx['mod']]]
        w_mod = w[:,cluster_idx['mod']]
        mod_assignment = [['higher','lower'][(mu[0]<mu[1])^cluster_idx['mod']]]
            
        
        ### calculate stats_pairwise
        stats_pairwise = []
        for cond1, cond2 in itertools.combinations(condition_names, 2):
            if model.method['pooling']:
                cond1, cond2 = [cond1], [cond2]
            else:
                cond1, cond2 = list(data_info[cond1].keys()), list(data_info[cond2].keys())
            if any(r in model_group_names for r in cond1) and any(r in model_group_names for r in cond2):
                w_cond1 = w[np.isin(model_group_names, cond1), cluster_idx['mod']].flatten()
                w_cond2 = w[np.isin(model_group_names, cond2), cluster_idx['mod']].flatten()
                n_cond1 = coverage[np.isin(model_group_names, cond1)]
                n_cond2 = coverage[np.isin(model_group_names, cond2)]

                z_score, p_ws = stats.z_test(w_cond1, w_cond2, n_cond1, n_cond2) # two=tailed
                #TODO hook in more fancy test here
                w_mod_mean_diff = np.mean(w_cond1)-np.mean(w_cond2)

                stats_pairwise += [w_mod_mean_diff, p_ws, z_score]
            else:
                stats_pairwise += [None, None, None]

        if len(condition_names) > 2:
            ### calculate stats_one_vs_all
            stats_one_vs_all = []
            for cond in condition_names:
                if model.method['pooling']:
                    cond = [cond]
                else:
                    cond = list(data_info[cond].keys())
                if any(r in model_group_names for r in cond):
                    w_cond1 = w[np.isin(model_group_names, cond), cluster_idx['mod']].flatten()
                    w_cond2 = w[~np.isin(model_group_names, cond), cluster_idx['mod']].flatten()
                    n_cond1 = coverage[np.isin(model_group_names, cond)]
                    n_cond2 = coverage[~np.isin(model_group_names, cond)]

                    z_score, p_ws = stats.z_test(w_cond1, w_cond2, n_cond1, n_cond2)
                    #TODO hook in more fancy test here
                    w_mod_mean_diff = np.mean(w_cond1)-np.mean(w_cond2)

                    stats_one_vs_all += [w_mod_mean_diff, p_ws, z_score]
                else:
                    stats_one_vs_all += [None, None, None]

        ###
        w_mod_ordered, coverage_ordered = [], [] # ordered by conditon_names or run_names based on headers.        
        if model.method['pooling']:
            names = condition_names
        else:
            names = run_names
        for name in names:
            if name in model_group_names:
                w_mod_ordered += list(w_mod[np.isin(model_group_names, name)])
                coverage_ordered += list(coverage[np.isin(model_group_names, name)])
            else:
                w_mod_ordered += [None]
                coverage_ordered += [None]

        ###
        ### prepare values to write
        row = [idx, position, kmer]
        row += stats_pairwise
        if len(condition_names) > 2:
            row += stats_one_vs_all

        # row += [p_overlap]
        # row += list_cdf_at_intersections
        row += list(w_mod_ordered)
        row += list(coverage_ordered)
        row += mu_assigned + sigma2_assigned + conf_mu + mod_assignment


        if prefiltering is not None:
            row += [prefiltering[model.method['prefiltering']['method']]]
        ### Filtering those positions with a nearly single distribution.
        cdf_threshold = 0.1
        x_x1, y_x1, x_x2, y_x2 = list_cdf_at_intersections
        is_not_inside = ((y_x1 < cdf_threshold) & (x_x1 < cdf_threshold)) | ((y_x2 < cdf_threshold) & (x_x2 < cdf_threshold)) | (( (1-y_x1) < cdf_threshold) & ((1-x_x1) < cdf_threshold)) | (( (1-y_x2) < cdf_threshold) & ((1-x_x2) < cdf_threshold))
        if (p_overlap <= 0.5) and (is_not_inside):
            table += [tuple(row)]

    return table


def diffmod(args):
    
    n_processes = args.n_processes       
    config_filepath = args.config
    save_models = args.save_models
    resume = args.resume
    ids = args.ids

    config = Configurator(config_filepath) 
    paths = config.get_paths()
    data_info = config.get_data_info()
    method = config.get_method()
    criteria = config.get_criteria()
    prior_params = config.get_priors()

    print('Using the signal of unmodified RNA from',paths['model_kmer'])
    model_kmer = pandas.read_csv(paths['model_kmer']).set_index('model_kmer')
    ###

    ###
    # Get gene ids for modelling
    # todo
    
    # Create output paths and locks.
    out_paths,locks = dict(),dict()
    for out_filetype in ['model','table','log']:
        out_paths[out_filetype] = os.path.join(paths['out_dir'], f"diffmod.{out_filetype}")
        locks[out_filetype] = multiprocessing.Lock()
        
    # Create communication queues.
    task_queue = multiprocessing.JoinableQueue(maxsize=n_processes * 2)

    # Writing the starting of the files.
    ids_done = []
    if resume and os.path.exists(out_paths['log']):
        ids_done = [line.rstrip('\n') for line in open(out_paths['log'],'r')]  
    else:
        with open(out_paths['table'],'w') as f:
            csv.writer(f,delimiter=',').writerow(io.get_result_table_header(data_info,method))
        with open(out_paths['log'],'w') as f:
            f.write(helper.decor_message('diffmod'))


    # Create and start consumers.
    consumers = [helper.Consumer(task_queue=task_queue,task_function=execute,locks=locks) for i in range(n_processes)]
    for p in consumers:
        p.start()

    ### Load tasks in to task_queue. ###
    f_index,f_data = {},{}
    for _condition_name, run_names in data_info.items():
        for run_name, dirpath in run_names.items():
            # Read index files
            df_index = pandas.read_csv(os.path.join(dirpath,'data.index'),sep=',') 
            f_index[run_name] = dict(zip(df_index['idx'],zip(df_index['start'],df_index['end'])))

            # Read readcount files
            # df_readcount[run_name] = pandas.read_csv(os.path.join(info['dirpath'],'readcount.csv')).groupby('gene_id')['n_reads'].sum() # todo: data.readcount

            # Open data files
            f_data[run_name] = open(os.path.join(dirpath,'data.json'),'r') 
    
    # Load tasks into task_queue.
    # gene_ids = helper.get_gene_ids(config.filepath)
    
    if len(ids) == 0:
        ids = helper.get_ids(f_index,data_info)


    print(len(ids),'ids to be testing ...')
    
    for idx in ids:
        if resume and (idx in ids_done):
            continue
        
        data_dict = dict()
        for condition_name, run_names in data_info.items():
            for run_name, _dirpath in run_names.items():
                try:
                    pos_start,pos_end = f_index[run_name][idx]
                except KeyError:
                    data_dict[(condition_name,run_name)] = None
                else:
                    # print(idx,run_name,pos_start,pos_end,df_readcount[run_name].loc[idx])
                    f_data[run_name].seek(pos_start,0)
                    json_str = f_data[run_name].read(pos_end-pos_start)
                    # print(json_str[:50])
                    # json_str = '{%s}' %json_str # used for old dataprep
                    data_dict[(condition_name,run_name)] = ujson.loads(json_str) # A data dict for each gene.
                
        # tmp
        out_paths['model_filepath'] = os.path.join(paths['models'],'%s.hdf5' %idx)
        #
        # if data_dict[run_name][idx] is not None: # todo: remove this line. Fix in dataprep
        task_queue.put((idx, data_dict, data_info, method, criteria, model_kmer, prior_params, out_paths,save_models)) # Blocked if necessary until a free slot is available.

        
    # Put the stop task into task_queue.
    task_queue = helper.end_queue(task_queue,n_processes)

    # Wait for all of the tasks to finish.
    task_queue.join()

    # Close data files
    for f in f_data.values():
        f.close()   

    with open(out_paths['log'],'a+') as f:
        f.write(helper.decor_message('successfully finished'))
