#%%
import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import gridspec
from matplotlib.patches import ConnectionPatch
import os
#import statsmodels.api as sm
import itertools
from sklearn.mixture import GaussianMixture
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.transforms as mtransforms
import torch
import torch.nn as nn



def bimodality_gmmcriteria(data, separation = 0.5, threshold=0.1):    
    data_ = data.copy()
    data_[data>np.percentile(data,95)] = np.nan
    data_[data<np.percentile(data,5)] = np.nan
    gmm_labels = np.nan*data    
    gmm_probs = np.nan*data    
    gmm = GaussianMixture(n_components=2, random_state=0).fit(data_[~np.isnan(data_)].reshape(-1,1))    
    # Calculate the separation criterion
    mu1, mu2 = gmm.means_.flatten()
    sigma1, sigma2 = np.sqrt(gmm.covariances_.flatten())
    S = abs(mu1 - mu2) / np.sqrt(sigma1**2 + sigma2**2)
    # Compute the weights of the two distributions
    weight1, weight2 = gmm.weights_
    if (S>separation) and (weight1>threshold) and (weight2>threshold):
        bimod_gmmcriteria = True        
        gmm_labels[~np.isnan(data)] = gmm.predict(data[~np.isnan(data)].reshape(-1,1))
        gmm_probs[~np.isnan(data)] = np.max(gmm.predict_proba(data[~np.isnan(data)].reshape(-1,1)), axis=1)
    else:
        bimod_gmmcriteria = False            
    return bimod_gmmcriteria, gmm_labels, gmm_probs


#%%
def select_sessions(df_sessions, N_top = 50, regions_of_interest=['NAc', 'mPFC', 'CeA']):
    df_sessions_all = df_sessions
    df_sessions = df_sessions[df_sessions.task == 'coupled_block_baiting']
    df_sessions_temp = pd.DataFrame()
    for region in regions_of_interest:
        df_sessions_NM = df_sessions[(df_sessions['Region_0']==region) | (df_sessions['Region_1']==region)] 
        criterion = df_sessions_NM['finished_trials'].values + 1000 * df_sessions_NM['foraging_eff'].values
        idxs = criterion.argsort()[-N_top:][::-1]
        df_sessions_NM = df_sessions_NM.iloc[idxs]
        df_sessions_temp = pd.concat([df_sessions_temp, df_sessions_NM], axis=0)

    return df_sessions_temp


#%%
def build_trials_with_modelfit(df_sessions, df_trials, behavior_folder='/data/Foraging_Models_Fit/'):        
    df_trials_fit = pd.DataFrame()
    df_sessions_params = pd.DataFrame()
    N_sessions = len(df_sessions)
    for i_session in range(N_sessions):
        # Select the session
        df_ses = df_sessions.iloc[[i_session]]
        ses_idx = df_ses['ses_idx'].values[0]

        # Select the trials of that session
        df_ses_trials_fip = df_trials[df_trials['ses_idx']==ses_idx]
        df_ses_trials_fip = df_ses_trials_fip[['ses_idx', 'trial', 'reward', 'choice', 'go_cue_absolute_time', 'go_cue', 'choice_time', 'reward_time', 'bins_trial', 'G_0', 'R_0', 'G_1', 'R_1', 'bit_code']]

        # Load and select trials behavior
        trainer = df_ses['h2o'].values[0]
        date = str(df_ses['session_date'].values[0]).replace('-','')
        i_ses = str(df_ses['session'].values[0])
        session_name = trainer +'_'+date+'_'+i_ses
        filename_pipeline = glob.glob(behavior_folder + session_name + '.pkl')
                    
        df_ses_trials_behavior = pd.read_pickle(filename_pipeline[0])    

        filename_pipeline_params = glob.glob(behavior_folder + session_name + '_params.pkl')
        df_ses_params = pd.read_pickle(filename_pipeline_params[0])    
        df_ses_params['ses_idx'] = ses_idx

        # df_ses_trials_behavior = df_ses_trials_behavior[['early_lick', 'outcome', 'right_reward_prob', 'left_reward_prob', 'trial_bit_code', 'left_action_value', 'right_action_value', 'rpe']]
        # df_ses_trials_behavior = df_ses_trials_behavior.rename(columns={'trial_bit_code':'bit_code'})
        if len(df_ses_trials_behavior) == len(df_ses_trials_fip):
            Ntrials_match = True
        else:
            Ntrials_match = False        
            print('#trials not matching between behavior and fiber photometry')
        df_ses_trials = df_ses_trials_fip.merge(df_ses_trials_behavior, how='inner', left_on='bit_code', right_on='bit_code')  
        df_trials_fit = pd.concat([df_ses_trials, df_trials_fit], axis=0)
        
        df_sessions_params = pd.concat([df_sessions_params, df_ses_params], axis=0)

    df_trials_fit = df_trials_fit.rename(columns={'p_reward_left':'left_reward_prob', 'p_reward_right':'right_reward_prob'})
    return df_trials_fit, df_sessions_params

#%%
def reshape_df_sessions(df_sessions):
    df = df_sessions[['ses_idx', 'subject_id', 'foraging_eff', 'reward_rate', 'ignore_rate', 'finished_trials', 'G_0', 'G_1', 'R_0', 'R_1']]
    cols2idx = ['ses_idx', 'subject_id', 'foraging_eff', 'reward_rate', 'ignore_rate', 'finished_trials']
    NMs = ['DA', 'NE', '5HT', 'ACh']

    df = df.set_index(cols2idx).stack().reset_index()
    level_col = [col for col in df.columns.values if 'level' in str(col)]
    df = df.rename(columns={0:'NM', level_col[0]:'loc'})
    df = df.drop_duplicates()
    df['channel'] = df['loc'].apply(lambda x:x[0])
    df['hemisphere'] = df['loc'].apply(lambda x:int(x[2]))
    df_reg = df_sessions[['ses_idx', 'Region_0', 'Region_1']]
    df_reg = df_reg.set_index('ses_idx').stack().reset_index().rename(columns={0:'region', 'level_1':'hemisphere'})
    df_reg['hemisphere'] = df_reg['hemisphere'].apply(lambda x:int(x[-1]))
    df = df.merge(df_reg, on=['ses_idx', 'hemisphere'])

    df['loc'] = pd.Categorical(df['loc'])
    df['region'] = pd.Categorical(df['region'])
    df = df[df.NM.isin(NMs)]
    df['NM'] = pd.Categorical(df['NM'], categories=NMs)

    return df

#%%
def build_trials_NMs(df_sessions_NMs, df_trials_fit, df_trials_han, NMs = ['DA', 'NE', '5HT', 'ACh'], models=['Hattori2019', 'Hattori2019_ignore']):
    cols2keep_all = ['bit_code', 'ses_idx', 'rpe', 'left_action_value', 'right_action_value', 'licks L', 'licks R', 'Lick L (raw)', 'Lick R (raw)']
    # cols2keep = ['trial', 'reward', 'choice', 'bit_code', 'ses_idx', 'rpe', 'go_cue', 'choice_time', 'reward_time', 'left_action_value', 'right_action_value', 'left_reward_prob', 'right_reward_prob', 'onset', 'NM' , 'NM_name']
    if len(models):
        cols_modelsfit = [x+'_'+y for (x,y) in itertools.product(models,['Lprob', 'Rprob', 'Lvalue', 'Rvalue', 'Lkernel', 'Rkernel'])]
        if 'Hattori2019_ignore' in models:
            cols_modelsfit += [x+'_'+y for (x,y) in itertools.product(['Hattori2019_ignore'],['Iprob', 'Ivalue', 'Ikernel'])]
        cols_modelsfit += ['left_reward_prob', 'right_reward_prob']
    else:
        cols_modelsfit = []
    # cols_modelsfit = ['Hattori2019_Lprob', 'Hattori2019_Rprob', 'Hattori2019_Lvalue', 'Hattori2019_Rvalue', 'Hattori2019_Lkernel', 'Hattori2019_Rkernel', 'Bari2019_Lprob', 'Bari2019_Rprob', 'Bari2019_Lvalue', 'Bari2019_Rvalue', 'Bari2019_Lkernel', 'Bari2019_Rkernel', 'Hattori2019_ignore_Ikernel', 'Hattori2019_ignore_Iprob', 'Hattori2019_ignore_Ivalue', 'Hattori2019_ignore_Lkernel', 'Hattori2019_ignore_Lprob', 'Hattori2019_ignore_Lvalue', 'Hattori2019_ignore_Rkernel', 'Hattori2019_ignore_Rprob', 'Hattori2019_ignore_Rvalue']
    cols2keep_fit = ['trial', 'reward', 'choice', 'bit_code', 'ses_idx', 'go_cue_absolute_time', 'go_cue', 'choice_time', 'reward_time',  'onset', 'NM', 'NM_name'] + cols_modelsfit

    df_trials_NMs = pd.DataFrame()
    for NM in NMs:
        df_sessions_NM = df_sessions_NMs[df_sessions_NMs['NM']==NM]
        for i_line in range(len(df_sessions_NM)):        
            df_ses = df_sessions_NM.iloc[[i_line]]
            ses_idx = df_ses['ses_idx'].values[0]
            # Switch this line depending on whether to retain Han's datajoint dataset or Code Ocean pipeline model fits
            df_trials_ses_all = df_trials_han[df_trials_han['ses_idx']==ses_idx]
            df_trials_ses_all = df_trials_ses_all[cols2keep_all]               

            df_trials_ses_fit = df_trials_fit[df_trials_fit['ses_idx']==ses_idx]                
            if len(df_trials_fit)==0:
                continue
            df_trials_ses_fit.loc[:,['onset']] = df_trials_ses_fit.bins_trial.apply(lambda x: x[:-1])        
            df_trials_ses_fit = df_trials_ses_fit.rename(columns={df_ses['loc'].values[0]:'NM'})    
            df_trials_ses_fit.loc[:,['NM_name']] = NM     
            df_trials_ses_fit = df_trials_ses_fit[cols2keep_fit]               
            df_trials_ses = df_trials_ses_all.merge(df_trials_ses_fit, how='inner', on=['bit_code', 'ses_idx'], suffixes=('', ''))  
            df_trials_NMs = pd.concat([df_trials_NMs, df_trials_ses], axis=0)
    return df_trials_NMs

def tryexcept(x, function):
    try:
        return function(x)
    except:
        return np.nan    

def nanmax(x):
    if len(x):
        return np.nanmax(x)
    else:
        return np.nan

def ReLU(x):
    return x * (x > 0)

def enrich_df_trials_NM(df_trials_ses):    
    df_trials_ses.loc[:,['NM_norm']] = df_trials_ses.apply(lambda x: (x.NM-np.mean(x.NM[(x.onset>-.5) & (x.onset<0.)])), axis=1).values
    df_trials_ses.loc[:,['NM_avg_choice2choicep1']] = df_trials_ses.apply(lambda x: np.mean(x.NM[(x.onset>x.choice_time) & (x.onset<(x.choice_time+0.5))]), axis=1)
    df_trials_ses.loc[:,['NM_avg_choice2choicep1_norm']] = df_trials_ses.apply(lambda x: np.mean(x.NM_norm[(x.onset>x.choice_time) & (x.onset<(x.choice_time+1.))]), axis=1)    
    df_trials_ses.loc[:,['NM_avg_gocue2choice']] = df_trials_ses.apply(lambda x: np.mean(x.NM[(x.onset>0) & (x.onset<x.choice_time)]), axis=1)
    df_trials_ses.loc[:,['NM_avg_gocue2choice_norm']] = df_trials_ses.apply(lambda x: np.mean(x.NM_norm[(x.onset>0) & (x.onset<x.choice_time)]), axis=1)    
    df_trials_ses.loc[:,['NM_avg_endoftrial']] = df_trials_ses.apply(lambda x: np.mean(x.NM[x.onset>(x.onset.max()-1.)]), axis=1)
    df_trials_ses.loc[:,['NM_avg_endoftrial_norm']] = df_trials_ses.apply(lambda x: np.mean(x.NM_norm[x.onset>(x.onset.max()-1.)])-np.mean(x.NM[(x.onset>-1.) & (x.onset<-0.5)]), axis=1)        
    df_trials_ses.loc[:,['NM_avg_trialstart']] = df_trials_ses.apply(lambda x: np.mean(x.NM[(x.onset>-.5) & (x.onset<0.)]), axis=1)
    df_trials_ses.loc[:,['NM_std']] = df_trials_ses.apply(lambda x: np.std(x.NM), axis=1)    
    df_trials_ses.loc[:,['NM_AUC']] = df_trials_ses.apply(lambda x: np.sum(x.NM_norm[(x.onset>0) & (x.NM_norm>0)]), axis=1)        
    df_trials_ses.loc[:,['NM_AUC_std']] = df_trials_ses.apply(lambda x: np.sum(x.NM_norm[(x.onset>0) & (x.NM_norm>0)]/np.std(x.NM_norm)), axis=1)        
    df_trials_ses.loc[:,['NM_gocue2choice_AUC']] = df_trials_ses.apply(lambda x: np.sum(x.NM_norm[(x.onset>0) & (x.onset<x.choice_time) & (x.NM_norm>0)]), axis=1)        
    df_trials_ses.loc[:,['NM_gocue2choice_AUC_std']] = df_trials_ses.apply(lambda x: np.sum(x.NM_norm[(x.onset>0) & (x.onset<x.choice_time) & (x.NM_norm>0)]/np.std(x.NM_norm)), axis=1)        
    df_trials_ses.loc[:,['NM_choice_AUC']] = df_trials_ses.apply(lambda x: np.sum(ReLU(x.NM_norm[(x.onset>x.choice_time)])), axis=1)
    df_trials_ses.loc[:,['NM_choice_negAUC']] = df_trials_ses.apply(lambda x: -np.sum(ReLU(-x.NM_norm[(x.onset>x.choice_time)])), axis=1)
    df_trials_ses.loc[:,['NM_choice-gocue_AUC']] = df_trials_ses.apply(lambda x: np.sum(ReLU(x.NM[(x.onset>x.choice_time)]-x.NM_avg_gocue2choice)), axis=1)
    df_trials_ses.loc[:,['NM_choice-gocue_negAUC']] = df_trials_ses.apply(lambda x: -np.sum(ReLU(-(x.NM[(x.onset>x.choice_time)]-x.NM_avg_gocue2choice))), axis=1)
        
    df_trials_ses.loc[:,['NM_gocue2choice_avg']] = df_trials_ses.apply(lambda x: np.mean(ReLU(x.NM_norm[(x.onset>0) & (x.onset<x.choice_time)])), axis=1)            
    df_trials_ses.loc[:,['NM_tot_avg']] = df_trials_ses.apply(lambda x: np.mean(ReLU(x.NM_norm)), axis=1)
    df_trials_ses.loc[:,['NM_choice-gocue_avg']] = df_trials_ses.apply(lambda x: np.mean(ReLU(x.NM[(x.onset>x.choice_time)]-x.NM_avg_gocue2choice)), axis=1)    
    df_trials_ses.loc[:,['NM_choice2choicep1_avg']] = df_trials_ses.apply(lambda x: np.mean(ReLU(x.NM_norm[(x.choice_time>0) & (x.onset<(x.choice_time+1.))])), axis=1)            

    df_trials_ses.loc[:,['prob_ratio']] = df_trials_ses['left_reward_prob']/df_trials_ses['right_reward_prob']
    df_trials_ses.loc[:,['block_change']] = (np.diff(df_trials_ses['left_reward_prob'].values,prepend=0)!=0) | (np.diff(df_trials_ses['right_reward_prob'].values,prepend=0)!=0)
    df_trials_ses.loc[:,['block_num']] = np.cumsum(df_trials_ses['block_change'])
    df_trials_ses.loc[:,['block_trial']] = np.sum(np.vstack([np.cumsum(df_trials_ses['block_num'].values==i)*(df_trials_ses['block_num'].values==i).astype(int) for i in range(np.max(df_trials_ses['block_num']))]), axis=0)
    # There is an error in the way raw licks have been stored and this can't be used at the moment    
    # df_trials_ses['licks anticipatory'] = df_trials_ses.apply(lambda x: np.sum(x['Lick R (raw)'][(x.onset>0) & (x.onset<x.choice_time)]+np.sum(x['Lick L (raw)'][(x.onset>0) & (x.onset<x.choice_time)])), axis=1)
    # df_trials_ses['licks consummatory'] = df_trials_ses.apply(lambda x: np.sum(x['Lick R (raw)'][(x.onset>x.choice_time)]+np.sum(x['Lick L (raw)'][(x.onset>x.choice_time)])), axis=1)
    df_trials_ses.loc[:,['ITI']] = df_trials_ses.apply(lambda x: x.onset[-1]-x.reward_time, axis=1)
    return df_trials_ses

def enrich_df_trials_model(df_trials, models=['Hattori2019_ignore', 'Hattori2019']):
    df_trials_model = pd.DataFrame()
    sessions = np.unique(df_trials['ses_idx'])
    for i_ses, ses in enumerate(sessions[:]):
        df_ses = df_trials[df_trials['ses_idx']==ses]            
        df_ses.loc[:,['choice']] = df_ses['choice'].fillna('Ignore', inplace=False)
        if (np.sum(df_ses[df_ses.choice=='R']['licks R'])<np.sum(df_ses[df_ses.choice=='R']['licks L'])) and (np.sum(df_ses[df_ses.choice=='L']['licks L'])<np.sum(df_ses[df_ses.choice=='L']['licks R'])):
            df_ses.loc[:,['choice']] = df_ses['choice'].map({'L':'R', 'R':'L','Ignore':'Ignore'})
            # print('swapped right and left')
        for mod in models:
            if (np.sum(df_ses[df_ses.choice=='R'][mod+'_Rprob'])<np.sum(df_ses[df_ses.choice=='R'][mod+'_Lprob'])) and (np.sum(df_ses[df_ses.choice=='L'][mod+'_Lprob'])<np.sum(df_ses[df_ses.choice=='L'][mod+'_Lprob'])):                
                df_ses = df_ses.rename(columns={mod+'_Lvalue':mod+'_Rvalue', mod+'_Rvalue':mod+'_Lvalue', mod+'_Lprob':mod+'_Rprob', mod+'_Rprob':mod+'_Lprob', mod+'_Lkernel':mod+'_Rkernel', mod+'_Rkernel':mod+'_Lkernel'})  
                print('swapped right and left in ' + mod)
        df_ses.loc[:,['prob_change']] = np.insert(np.diff(df_ses['right_reward_prob'].values)!=0,0,0)
        df_ses.loc[:,['idx_in_block']] = compute_difference_vectorized(df_ses['prob_change'].values)
        df_ses.loc[:,['idx_of_block']] = np.cumsum(df_ses['prob_change'])
        choices = df_ses.choice.values        
        chosen_values = np.nan * np.zeros((len(df_ses),len(models)))
        unchosen_values = np.nan * np.zeros((len(df_ses),len(models)))
        chosen_kernels = np.nan * np.zeros((len(df_ses),len(models)))
        unchosen_kernels = np.nan * np.zeros((len(df_ses),len(models)))
        chosen_probabilities = np.nan * np.zeros((len(df_ses),len(models)))
        unchosen_probabilities = np.nan * np.zeros((len(df_ses),len(models)))
        chosen_stay_probabilities = np.nan * np.zeros((len(df_ses),len(models)))
        chosen_licks = np.nan * np.zeros((len(df_ses),len(models)))
        for i_idx in range(len(df_ses)):
            choice = choices[i_idx]            
            for i_mod, mod in enumerate(models):                                                              
                if choice == 'Ignore' and 'ignore' not in mod:
                    chosen_values[i_idx, i_mod] = np.nan
                    chosen_kernels[i_idx, i_mod] = np.nan
                    chosen_probabilities[i_idx, i_mod] = np.nan
                    chosen_stay_probabilities[i_idx, i_mod] = np.nan  
                    chosen_licks[i_idx, i_mod] = np.nan                    
                else:
                    chosen_values[i_idx, i_mod] =  df_ses[mod+'_'+choice[0]+'value'].values[i_idx]                    
                    chosen_kernels[i_idx, i_mod] =  df_ses[mod+'_'+choice[0]+'kernel'].values[i_idx]                    
                    chosen_probabilities[i_idx, i_mod] = df_ses[mod+'_'+choice[0]+'prob'].values[i_idx]                    
                    if choice[0]!='I':
                        unchosen_values[i_idx, i_mod] =  df_ses[mod+'_'+{'L':'R','R':'L'}[choice[0]]+'value'].values[i_idx]
                        unchosen_probabilities[i_idx, i_mod] = df_ses[mod+'_'+{'L':'R','R':'L'}[choice[0]]+'prob'].values[i_idx]
                        unchosen_kernels[i_idx, i_mod] = df_ses[mod+'_'+{'L':'R','R':'L'}[choice[0]]+'kernel'].values[i_idx]
                        chosen_licks[i_idx, i_mod] = df_ses['licks '+choice[0]].values[i_idx]
                    if i_idx < len(df_ses)-1:
                        chosen_stay_probabilities[i_idx, i_mod] = df_ses[mod+'_'+choice[0]+'prob'].values[i_idx+1]
        for i_mod, mod in enumerate(models):                               
            df_ses.loc[:,[mod+'_Q_chosen']] = chosen_values[:, i_mod]
            df_ses.loc[:,[mod+'_Q_unchosen']] = unchosen_values[:, i_mod]
            df_ses.loc[:,[mod+'_Q_sum']] = df_ses[mod+'_Lvalue'].values+df_ses[mod+'_Rvalue'].values
            df_ses.loc[:,[mod+'_Q_Delta']] = df_ses[mod+'_Q_chosen'].values-df_ses[mod+'_Q_unchosen'].values
            df_ses.loc[:,[mod+'_Q_change']] = np.concatenate([[0], np.diff(chosen_values[:, i_mod])])                        

            df_ses.loc[:,[mod+'_P_chosen']] = chosen_probabilities[:, i_mod]
            df_ses.loc[:,[mod+'_P_unchosen']] = unchosen_probabilities[:, i_mod]
            df_ses.loc[:,[mod+'_P_sum']] = df_ses[mod+'_Lprob'].values+df_ses[mod+'_Rprob'].values
            df_ses.loc[:,[mod+'_P_Delta']] = df_ses[mod+'_P_chosen'].values-df_ses[mod+'_P_unchosen'].values
            df_ses.loc[:,[mod+'_P_change']] = np.concatenate([[0], np.diff(chosen_probabilities[:, i_mod])])     

            df_ses.loc[:,[mod+'_K_chosen']] = chosen_kernels[:, i_mod]
            df_ses.loc[:,[mod+'_K_unchosen']] = unchosen_kernels[:, i_mod]
            df_ses.loc[:,[mod+'_K_sum']] = df_ses[mod+'_Lkernel'].values+df_ses[mod+'_Rkernel'].values
            df_ses.loc[:,[mod+'_K_Delta']] = df_ses[mod+'_K_chosen'].values-df_ses[mod+'_K_unchosen'].values
            df_ses.loc[:,[mod+'_K_change']] = np.concatenate([[0], np.diff(chosen_kernels[:, i_mod])])            

            df_ses.loc[:,[mod+'_Cprobstay']] = chosen_stay_probabilities[:, i_mod]
            df_ses.loc[:,[mod+'_RPE']] = df_ses['reward'] - chosen_values[:, i_mod]
            df_ses.loc[~df_ses['choice'].isin(['R','L']), mod+'_RPE'] = np.nan

            values, bins = np.histogram(df_ses[mod+'_RPE'].values, np.arange(-1.,1.01,0.05))
            ratio = np.sum(values>0)/len(bins)
            df_ses.loc[:,[mod+'_QCoccupancy']] = ratio
            df_ses.loc[:,['licks_chosen']] = chosen_licks[:, i_mod]
        df_trials_model = pd.concat([df_trials_model, df_ses], axis=0)
    return df_trials_model    

#%%



#%% Summary Plots on correlations
def analysis_correlations_summary(df_trials_all, df_corrs_all, df_pvals_all, model2plot='Hattori2019_ignore', significant_only=True):
    df_corrs_all['ses_idx'] = pd.Categorical(df_corrs_all['ses_idx'])
    df2plot = df_corrs_all.drop(columns=['ses_idx', 'region', 'DA (AUC)', 'DA go cue (AUC)', 'DA go cue vs choice (AUC)', 'DA choice (AUC)'])
    if significant_only:
        df2plot_pvals = df_pvals_all.drop(columns=['ses_idx', 'region', 'DA (AUC)', 'DA go cue (AUC)', 'DA go cue vs choice (AUC)', 'DA choice (AUC)'])
        cols_pval = [col for col in df2plot_pvals.columns if col not in ['reward', 'var']]
        pvals_values = df2plot_pvals.loc[:,cols_pval].values
        pvals_values[pvals_values > 0.05] = np.nan
        df2plot_pvals.loc[:,cols_pval] = pvals_values
        df2plot.loc[:,cols_pval] = df2plot.loc[:,cols_pval].values*(1-0*pvals_values)
    grouped = df2plot.groupby(['reward','var']).mean()

    # fig, axs = plt.subplots(8,3, figsize=(30,20))
    # fig1, ax1 = plt.subplots(1,1,figsize=(30,5))
    fig, axs = plt.subplot_mosaic([[0,0,0,0,0,0,0,0] , [10, 11, 12, 13,14, 15, 16, 17], [20, 21, 22, 23,24, 25, 26, 27], [30, 31, 32, 33,34, 35, 36, 37], [40, 41, 42, 43,44, 45, 46, 47]], layout='constrained', figsize=(13, 7), height_ratios=[2.,1.,1.,1.,1.])

    sns.heatmap(grouped, cmap='vlag', fmt='', vmin=-0.4, vmax=0.4, ax=axs[0])
    # pos, textvals = plt.xticks();
    # plt.xticks(pos, textvals, rotation=330,ha='left');

    def plot_corr_hist(data, var1, var2, hue, show_pv=False, data_pv=None, ax=None):
        df2plot = data[data['var']==var1].reset_index()    
        if show_pv:
            df2plot_pv = df_pvals_all[data_pv['var']==var1].reset_index()
            df2plot[var2].loc[df2plot_pv[var2].values<0.05] = np.nan
        if ax is None:
            fig, ax = plt.subplots(1,1)
        g = sns.histplot(df2plot, x=var2, hue=hue, element='step', ax=ax, legend=False)
        return g
    
    for i_var, var in enumerate(['DA (AUC)', 'DA go cue (AUC)', 'DA go cue vs choice (AUC)', 'DA choice (AUC)']):
            for i_reg, reg in enumerate(['RPE', 'Q_Delta', 'Q_chosen', 'Q_change', 'stay prob', 'ITI', 'choice_time']):
                plot_corr_hist(df_corrs_all, var, reg, 'reward', ax=axs[(i_var+1)*10+i_reg])
                # plot_corr_hist(df_corrs_all, var, reg, 'reward', show_pv=True, data_pv=df_pvals_all, ax=axs[i_var+3, i_reg])
    sns.despine()
    return fig, axs

#%%
def compute_sessionwide_traces(df_trials_ses, time_span=[0,600]):    
    trace_name = df_trials_ses.NM_name.unique().astype(str)
    # trace_region = df_trials_ses.region.unique().astype(str)
    traces_names = [trace_name]
    NM = 'NM'#trace_name    
    N_trials = df_trials_ses.shape[0]

    # df_trials_ses = df_trials_ses.rename(columns={'NM':NM, 'go_cue':'go cue', 'choice_time':'choice time', 'reward_time':'reward time'})        
    df_trials_ses['last_value_'+NM] = np.concatenate([[np.nan], df_trials_ses.apply(lambda x: x[NM][-1], axis=1).values[:-1]])
    df_trials_ses['overlap_index'] = df_trials_ses.apply(lambda x: np.where(x[NM]==x['last_value_'+NM])[0], axis=1)
    df_trials_ses['overlap_index'] = df_trials_ses['overlap_index'].apply(lambda x: x[0] if len(x) else 0)
    df_trials_ses[NM+'_no_overlap'] = df_trials_ses.apply(lambda x: x[NM][x.overlap_index:-1], axis=1)
    df_trials_ses['bins_mids'] = df_trials_ses.onset.apply(lambda x: x[:]+np.mean(np.diff(x))/2)
    df_trials_ses['bins_mids_no_overlap'] = df_trials_ses.apply(lambda x: x.bins_mids[x.overlap_index:-1], axis=1)

    events_names = ['go_cue', 'choice_time', 'reward_time']
    events2plot = {}
    for event_name in events_names:
        df_trials_ses[event_name] = df_trials_ses.apply(lambda x: x[event_name] + x['go_cue_absolute_time'], axis=1)
        event_times = df_trials_ses[event_name].values
        if event_name == 'choice_time':
            for side in ['L', 'R']:            
                event_times_side = df_trials_ses[event_name].values[df_trials_ses['choice'] == side]
                event_name_side = event_name + ' ' + side
                events2plot.update({event_name_side: event_times_side})
        else:
            events2plot.update({event_name: event_times[~np.isnan(event_times)]})


    df_trials_ses['bins_mids_no_overlap'] = df_trials_ses.apply(lambda x: x['bins_mids_no_overlap'] + x['go_cue_absolute_time'], axis=1)
    trace_times = df_trials_ses['bins_mids_no_overlap'].explode().values# - df_trials_ses.go_cue_absolute_time.min()
    traces2plot = {}
    # for trace_name in traces_name:    
    trace = df_trials_ses[NM+'_no_overlap'].explode().values
    traces2plot.update({NM: trace})

    events2plot = {key:events2plot[key][(events2plot[key] > time_span[0]) & (events2plot[key] < time_span[1])] for key in events2plot}
    traces2plot = {key:traces2plot[key][(trace_times > time_span[0]) & (trace_times < time_span[1])] for key in traces2plot}
    trace_times2plot = trace_times[(trace_times > time_span[0]) & (trace_times < time_span[1])]

    return df_trials_ses, events2plot, traces2plot, trace_times2plot

#%%
def compute_sessionwide_traces_multi(df_trials_ses, time_span = None):    
    trace_names_list = df_trials_ses[['NM_name','region']].drop_duplicates().values

    N_trials = df_trials_ses.shape[0]

    # df_trials_ses = df_trials_ses.rename(columns={'go_cue':'go cue', 'choice_time':'choice time', 'reward_time':'reward time'})        
    df_trials_ses['last_value_NM'] = np.concatenate([[np.nan], df_trials_ses.apply(lambda x: x['NM'][-1], axis=1).values[:-1]])
    df_trials_ses['overlap_index'] = df_trials_ses.apply(lambda x: np.where(x['NM']==x['last_value_NM'])[0], axis=1)
    df_trials_ses['overlap_index'] = df_trials_ses['overlap_index'].apply(lambda x: x[0] if len(x) else 0)
    df_trials_ses['NM_no_overlap'] = df_trials_ses.apply(lambda x: x['NM'][x.overlap_index:-1], axis=1)
    df_trials_ses['bins_mids'] = df_trials_ses.onset.apply(lambda x: x[:]+np.mean(np.diff(x))/2)
    df_trials_ses['bins_mids_no_overlap'] = df_trials_ses.apply(lambda x: x.bins_mids[x.overlap_index:-1], axis=1)

    events_names = ['go_cue', 'choice_time', 'reward_time']
    events2plot = {}
    NM_name, region = trace_names_list[0][0], trace_names_list[0][1]    
    for event_name in events_names:
        df_trials_ses[event_name] = df_trials_ses.apply(lambda x: x[event_name] + x['go_cue_absolute_time'], axis=1)
        df_trials_ses_trace = df_trials_ses[(df_trials_ses['NM_name']==NM_name) & (df_trials_ses['region']==region)]
        event_times = df_trials_ses_trace[event_name].values
        if event_name == 'choice_time':
            for side in ['L', 'R']:            
                event_times_side = df_trials_ses[event_name].values[df_trials_ses['choice'] == side]
                event_name_side = event_name + ' ' + side
                events2plot.update({event_name_side: event_times_side})
        else:
            events2plot.update({event_name: event_times[~np.isnan(event_times)]})

    df_trials_ses['bins_mids_no_overlap'] = df_trials_ses.apply(lambda x: x['bins_mids_no_overlap'] + x['go_cue_absolute_time'], axis=1)
    
    for i_trace, trace_name in enumerate(trace_names_list):
        NM_name, region = trace_name[0], trace_name[1]
        df_trials_ses_trace = df_trials_ses[(df_trials_ses['NM_name']==NM_name) & (df_trials_ses['region']==region)]
        if i_trace==0:
            trace_times = df_trials_ses_trace['bins_mids_no_overlap'].explode().values# - df_trials_ses.go_cue_absolute_time.min()
            traces2plot = {}    
        trace = df_trials_ses_trace['NM_no_overlap'].explode().values
        traces2plot.update({NM_name+' '+region: trace})

    if time_span is not None:
        events2plot = {key:events2plot[key][(events2plot[key] > time_span[0]) & (events2plot[key] < time_span[1])] for key in events2plot}
        traces2plot = {key:traces2plot[key][(trace_times > time_span[0]) & (trace_times < time_span[1])] for key in traces2plot}
        trace_times2plot = trace_times[(trace_times > time_span[0]) & (trace_times < time_span[1])]
    else:
        trace_times2plot = trace_times
    return df_trials_ses, events2plot, traces2plot, trace_times2plot


#def lagged_regression(x, y, max_lag=10):
    df = pd.DataFrame({'y': y, 'x': x})
    for i in range(0, max_lag + 1):
        df[f'x_lag_{i}'] = df['x'].shift(i) # shift the x values by i periods

    # Drop the rows that contain missing values due to the lagging
    df = df.dropna()
    y = df['y'] # the dependent variable
    X = df.drop('y', axis=1) # the independent variables
    X = X.drop('x', axis=1) # the independent variables
    X = sm.add_constant(X) # add a constant term to the regression

    # Fit the linear regression model to the data
    model = sm.OLS(y, X)
    result = model.fit()

    # Print the model summary
    # print(result.summary())

    # Extract the regression coefficients and their standard errors from the result object
    coef = result.params[1:][::-1] # the first element is the constant term, which we do not need
    se = result.bse[1:][::-1] # the standard errors of the coefficients
    lower = result.conf_int()[0][1:][::-1] # the first element is the constant term, which we do not need
    upper = result.conf_int()[1][1:][::-1] # the first element is the constant term, which we do not need
    return coef, se, lower, upper


#%%

def rename_labels_by_average(labels, data):
    unique_labels = np.unique(labels)
    avg_values = {label: np.nanmean(data[labels == label]) for label in unique_labels}
    sorted_labels = sorted(avg_values, key=avg_values.get)
    renamed_labels = {label: i for i, label in enumerate(sorted_labels)}
    new_labels = np.array([np.nan if np.isnan(label) else renamed_labels[label] for label in labels])
    return new_labels

#%

def generate_sessionwide_figure(df_trials_ses, model='Hattori2019', plot=False):
    # Enrich dataset
    # df_trials_ses, events2plot, traces2plot, trace_times2plot = compute_sessionwide_traces_multi(df_trials_ses)
    trace_names_list = df_trials_ses[['NM_name','region']].drop_duplicates().values
    trace_name = ' '.join(trace_names_list[0])
    NM = trace_names_list[0][0]
    ses_idx = df_trials_ses.ses_idx.unique()[0]
    foraging_eff = df_trials_ses.foraging_eff.unique()[0]

    df_trials_ses['choice_value'] = df_trials_ses['choice'].map({'R':0, 'L':1, np.nan:0.5, 'Ignore':0.5})
    df_trials_ses['choice_value_roll'] = df_trials_ses['choice_value'].dropna().rolling(5).mean().reindex(df_trials_ses.index)
    df_trials_ses['reaction_time'] = df_trials_ses['choice_time'] - df_trials_ses['go_cue']
    df_trials_ses['reaction_time_roll'] = df_trials_ses['reaction_time'].dropna().rolling(15).mean().reindex(df_trials_ses.index)

    if plot:
        events_colors = [matplotlib.colormaps.get_cmap('Paired')(i)[:3] for i in [0, 2, 3, 6, 4, 5]]
        colset1 = [matplotlib.colormaps['tab20'](i)[:3] for i in [2, 3, 0, 1]]
        colset2 = [matplotlib.colormaps['tab20'](i)[:3] for i in [4, 5, 6, 7]]
        left_color, right_color, ignore_color = 'red', 'blue', 'green'
        # left_color, right_color, ignore_color = events_colors[0], events_colors[1], events_colors[2]
        choice_cols = [right_color, left_color, ignore_color]

        import matplotlib.transforms as mtransforms
        fig, axs = plt.subplot_mosaic([['a', 'a', 'a','a'] , ['b', 'b', 'b', 'b'], ['c', 'c', 'c', 'c'], ['d', 'd', 'd', 'd']], layout='constrained', figsize=(13, 7), height_ratios=[1.,1.,1.,1.])

        title = 'Trial Structure ' + ses_idx + ' ' + trace_name + ' forag. eff. ' + str(foraging_eff) + ' N trials ' + str(len(df_trials_ses)) + ' Left/Right (top/bottom)' + ' (rolling mean 5)'
        label = 'a'
        ax = axs[label]
        x = df_trials_ses.trial.values
        y = df_trials_ses[model+'_Lprob'].values
        ax.plot(x,0.1+0.8*y, c='black', linewidth=2.)

        x = df_trials_ses.trial.values
        y = df_trials_ses.choice_value_roll.values
        ax.plot(x,0.1+0.8*y, '--',c='gray',linewidth=1.)

        yl = 0.5+0.6*df_trials_ses.left_reward_prob.values
        yr = 0.5-0.6*df_trials_ses.right_reward_prob.values
        d = 0.5*np.ones(len(yl))
        x = df_trials_ses.trial.values
        ax.fill_between(x, d, yl, where=yl>=d, interpolate=True, color='red', alpha=0.3)
        ax.fill_between(x, yr, d, where=yr<=d, interpolate=True, color='blue', alpha=0.3)

        for i_choice, choice_val in enumerate([0, 1, 0.5]):
            idxs = df_trials_ses['choice_value'] == choice_val
            for i_reward, reward_val in enumerate([True, False]):
                idxs_rew = df_trials_ses['reward'].values == reward_val
                x = df_trials_ses.trial.values[idxs & idxs_rew]
                y = df_trials_ses.choice_value[idxs & idxs_rew]            
                ax.scatter(x, y, s=60, c=choice_cols[i_choice], marker='|')
                if reward_val == True:
                    ax.scatter(x, y, s=40, c='black', marker='|')
        
        trans = mtransforms.ScaledTranslation(10/72, -5/72, fig.dpi_scale_trans)
        ax.text(0.0, 1.2, title, transform=ax.transAxes + trans, fontsize='medium', verticalalignment='top', fontfamily='serif', bbox=dict(facecolor='1.0', edgecolor='none', pad=3.0))        
        ax.margins(x=0)
        ax.set_axis_off()

        red_line = mlines.Line2D([], [], color='black', label='Model L prob')
        blue_line = mlines.Line2D([], [], color='gray', label='Choices (roll 5)')
        green_line = mlines.Line2D([], [], color='green', label='Ignored trials')
        red_patch = mpatches.Patch(color='red', alpha=0.3, label='Left prob')
        blue_patch = mpatches.Patch(color='blue', alpha=0.3, label='Right prob')
        ax.legend(handles=[red_patch,blue_patch,blue_line,red_line,green_line],loc='center left', bbox_to_anchor=(1, 0.5), borderaxespad=0,frameon=False)

        title = 'Reaction Times over trials (rolling mean 15)'
        label = 'b'
        ax = axs[label]
        choice_colors = [matplotlib.cm.get_cmap('tab20')(i)[:3] for i in [0, 2, 3, 6, 4, 5]]
        for i_choice, choice_val in enumerate([0, 1]):
            idxs = df_trials_ses['choice_value'] == choice_val
            x = df_trials_ses.trial.values[idxs]
            y = df_trials_ses.reaction_time[idxs]    
            ax.scatter(x, y, s=40, c=choice_cols[i_choice], marker='|')
        x_roll = df_trials_ses.trial.values
        y_roll = df_trials_ses.reaction_time_roll
        ax.plot(x_roll, y_roll, c='gray')
        trans = mtransforms.ScaledTranslation(10/72, -5/72, fig.dpi_scale_trans)
        ax.text(0.0, 1.1, title, transform=ax.transAxes + trans, fontsize='medium', verticalalignment='top', fontfamily='serif', bbox=dict(facecolor='1.0', edgecolor='none', pad=3.0))    
        ax.set_ylim([0,np.nanpercentile(df_trials_ses['reaction_time'],100)])
        ax.margins(x=0)
        # ax.set_axis_off()
        ax.get_xaxis().set_visible(False)
        ax.get_xaxis().set_ticks([])
        sns.despine()
        red_line = mlines.Line2D([], [], color='red', label='Left choice t')
        blue_line = mlines.Line2D([], [], color='blue', label='Right choice t')
        gray_line = mlines.Line2D([], [], color='gray', label='Choice t (roll 15')
        ax.legend(handles=[blue_line,red_line,gray_line],loc='center left', bbox_to_anchor=(1, 0.5), borderaxespad=0,frameon=False)

        title = 'Licks over trials (choice side only, rolling mean 15)'
        label = 'c'
        ax = axs[label]
        choice_colors = [matplotlib.cm.get_cmap('tab20')(i)[:3] for i in [0, 2, 3, 6, 4, 5]]
        # df_trials_ses['licks_L_roll'] = df_trials_ses['licks L'].rolling(window=5).mean()
        # df_trials_ses['licks_R_roll'] = df_trials_ses['licks R'].rolling(window=5).mean()
        df_trials_ses['lick_choice_roll'] = np.nan*df_trials_ses.index.values
        choice_val2RL = {1:'R', 0:'L', 0.5:'ignore'}
        x_roll = df_trials_ses.trial.values
        for i_choice, choice_val in enumerate([0, 1]):
            idxs = df_trials_ses['choice_value'] == choice_val
            x = df_trials_ses.trial.values[idxs]
            y = df_trials_ses['licks '+choice_val2RL[choice_val]][idxs]
            df_trials_ses['lick_choice_roll'][idxs] = y
            # yr = df_trials_ses['licks R'][idxs]
            ax.scatter(x, y, s=40, c=choice_cols[i_choice], marker='|')
            # ax.scatter(x, yr, s=40, c=events_colors[i_choice], marker='|')
        # yr_roll = df_trials_ses.licks_R_roll
        # yl_roll = df_trials_ses.licks_L_roll
        y_roll = df_trials_ses['lick_choice_roll'].dropna().rolling(15).mean().reindex(df_trials_ses.index)
        ax.plot(x_roll, y_roll, c='gray', alpha=0.8)
        # ax.plot(x_roll, yl_roll, c='gray', alpha=0.5)
        trans = mtransforms.ScaledTranslation(10/72, -5/72, fig.dpi_scale_trans)
        ax.text(0.0, 1.1, title, transform=ax.transAxes + trans, fontsize='medium', verticalalignment='top', fontfamily='serif', bbox=dict(facecolor='1.0', edgecolor='none', pad=3.0))
        ax.set_ylim([0,np.nanmax(y_roll)+2])
        ax.margins(x=0)
        # ax.set_axis_off()
        ax.get_xaxis().set_visible(False)
        ax.get_xaxis().set_ticks([])
        sns.despine()
        red_line = mlines.Line2D([], [], color='red', label='Left licks')
        blue_line = mlines.Line2D([], [], color='blue', label='Right licks')
        gray_line = mlines.Line2D([], [], color='gray', label='Licks (roll 15)')
        ax.legend(handles=[blue_line,red_line,gray_line],loc='center left', bbox_to_anchor=(1, 0.5), borderaxespad=0,frameon=False)

        
        title = 'DA (AUC)'
        label = 'd'
        ax = axs[label]
        choice_colors = [matplotlib.cm.get_cmap('tab20')(i)[:3] for i in [0, 2, 3, 6, 4, 5]]    
        choice_val2RL = {0:'R', 1:'L', 0.5:'ignore'}
        x_roll = df_trials_ses.trial.values
        for i_choice, choice_val in enumerate([0, 1]):
            idxs = df_trials_ses['choice_value'] == choice_val
            x = df_trials_ses.trial.values[idxs]
            y = df_trials_ses['NM_avg_gocue2choice_norm'][idxs]            
            ax.scatter(x, y, s=40, c=choice_cols[i_choice], marker='|')
            y = df_trials_ses['NM_avg_choice2choicep1_norm'][idxs]                    
            ax.scatter(x, y, s=40, c='black', marker='|')        
        y_roll = df_trials_ses['NM_avg_gocue2choice_norm'].dropna().rolling(15).mean().reindex(df_trials_ses.index)
        ax.plot(x_roll, y_roll, c='gray', alpha=0.8)    
        y_roll = df_trials_ses['NM_avg_choice2choicep1_norm'].dropna().rolling(15).mean().reindex(df_trials_ses.index)
        ax.plot(x_roll, y_roll, c='black', alpha=0.6)    
        trans = mtransforms.ScaledTranslation(10/72, -5/72, fig.dpi_scale_trans)
        ax.text(0.0, 1.1, title, transform=ax.transAxes + trans, fontsize='medium', verticalalignment='top', fontfamily='serif', bbox=dict(facecolor='1.0', edgecolor='none', pad=3.0))
        # ax.set_ylim([0,np.nanmax(y_roll)])
        ax.margins(x=0)
        ax.set_xlabel('Trials')
        # ax.set_axis_off()
        # ax.get_xaxis().set_visible(False)
        # ax.get_xaxis().set_ticks([])
        sns.despine()
        red_line = mlines.Line2D([], [], color='red', label='Left gocue response DA')
        blue_line = mlines.Line2D([], [], color='blue', label='Right gocue response DA')
        gray_line = mlines.Line2D([], [], color='black', label='Reward response DA')
        ax.legend(handles=[blue_line,red_line,gray_line],loc='center left', bbox_to_anchor=(1, 0.5), borderaxespad=0,frameon=False)

    else:
        fig, axs = np.nan, np.nan
    return df_trials_ses, fig, axs

#%% Analysis of reaction times it returns a label
def analysis_choice_times(df_trials_ses, plot=False):
    events_colors = [matplotlib.colormaps.get_cmap('Paired')(i)[:3] for i in [0, 2, 3, 6, 4, 5]]
    colset1 = [matplotlib.colormaps['tab20'](i)[:3] for i in [2, 3, 0, 1]]
    colset2 = [matplotlib.colormaps['tab20'](i)[:3] for i in [4, 5, 6, 7]]
    left_color, right_color, ignore_color = 'red', 'blue', 'green'
    choice_cols = [right_color, left_color, ignore_color]

    idxs = df_trials_ses['choice'].map({'R':0, 'L':1})
    # idxs = df_trials_ses['choice_value']
    axis_max = np.nanpercentile(df_trials_ses['choice_time'].values,95)+0.1
    labels = np.nan*idxs
    choice_times = df_trials_ses['choice_time'].values
    choice_timesR = df_trials_ses['choice_time'].values[idxs==0]
    choice_timesL = df_trials_ses['choice_time'].values[idxs==1]
    bimod_gmmcriteria0, gmm_labels0, gmm_probs0 = bimodality_gmmcriteria(choice_timesR)    
    bimod_gmmcriteria1, gmm_labels1, gmm_probs1 = bimodality_gmmcriteria(choice_timesL)    
    if bimod_gmmcriteria0:
        gmm_labels0 = rename_labels_by_average(gmm_labels0, choice_timesR)    
        labels[idxs==0] = gmm_labels0
    else:
        labels[idxs==0] = 0
    max_label0 = int(np.nanmax(labels[idxs==0]))
    if bimod_gmmcriteria1:
        gmm_labels1 = rename_labels_by_average(gmm_labels1, choice_timesL)
        labels[idxs==1] = gmm_labels1 + max_label0 + 1    
    else:
        labels[idxs==1] = max_label0 + 1     
    df_trials_ses['choice_times_labels'] = labels

    if plot:
        fig, axs = plt.subplot_mosaic([['a', 'a', 'b','b']], layout='constrained', figsize=(12, 5), height_ratios=[2.])
        title = 'Reaction Times'
        label = 'a'
        ax = axs[label]
        trans = mtransforms.ScaledTranslation(10/72, -5/72, fig.dpi_scale_trans)
        ax.text(0.0, 1.0, title, transform=ax.transAxes + trans, fontsize='medium', verticalalignment='top', fontfamily='serif', bbox=dict(facecolor='1.0', edgecolor='none', pad=3.0))
        # idxs = df_trials_ses['choice'].map({'R':0, 'L':1})
        axis_max = np.nanpercentile(df_trials_ses['choice_time'].values,95)+0.1
        ax.hist(df_trials_ses['choice_time'].values[idxs==0], color=choice_cols[0], bins=np.arange(0,axis_max,0.02), alpha=0.4, label='Right')
        ax.hist(df_trials_ses['choice_time'].values[idxs==1], color=choice_cols[1], bins=np.arange(0,axis_max,0.02), alpha=0.4, label='Left')
        ax.legend()
        sns.despine()
        ax.set_xlabel('Choice time')

        title = 'Reaction Times'
        label = 'b'
        ax = axs[label]
        trans = mtransforms.ScaledTranslation(10/72, -5/72, fig.dpi_scale_trans)
        ax.text(0.0, 1.0, title, transform=ax.transAxes + trans, fontsize='medium', verticalalignment='top', fontfamily='serif', bbox=dict(facecolor='1.0', edgecolor='none', pad=3.0))

        colset = colset1[:max_label0+1] + colset2
        for i_label, label in enumerate(np.unique(labels)):
            if label > max_label0:
                label_name = 'Left ' + str(i_label - max_label0 + 2)
            else:
                label_name = 'Right ' + str(i_label + 1)
            if np.sum(labels==label) <10:
                continue
            ax.hist(choice_times[labels==label], bins=np.arange(0,axis_max,0.02), alpha=0.4, color=colset[i_label], label=label_name)
        ax.legend()
        ax.set_xlabel('Choice time')
        sns.despine()    
    else:
        fig, axs = np.nan, np.nan
    return df_trials_ses, fig, axs

#%% Summary Plots on Reaction Times
def analysis_DAauc_vs_choicelabel(df_trials_ses):
    NM_cols = ['NM_choice-gocue_AUC', 'choice_time', 'choice_value', 'reward', 'idx_of_block']
    idxs_cols = ['ses_idx', 'choice_times_labels']

    df_rt_ses = pd.DataFrame()
    df_trials_ses.loc[:,['ses_idx']] = df_trials_ses['ses_idx'].astype(str)
    blocks = np.unique(df_trials_ses['idx_of_block'].values)
    for block in blocks:
        for reward in [True, False]:
            df_sesrew = df_trials_ses[(df_trials_ses['reward']==reward) & (df_trials_ses['idx_of_block']==block)]
            df_sesrew.loc[:,['reward']] = df_sesrew['reward'].astype(float)            
            for choice_value in [0., 1.]:
                df_ses_lab = df_sesrew[df_sesrew['choice_value']==choice_value]    
                num_trials_per_label = pd.value_counts(df_ses_lab['choice_times_labels']).values        
                if any(num_trials_per_label<5):
                    continue
                df_ses_lab = df_ses_lab[idxs_cols+NM_cols].groupby(['ses_idx', 'choice_times_labels']).agg(np.nanmean).reset_index()
                rt_labels = np.unique(df_ses_lab['choice_times_labels'].values)
                rt_labels = rt_labels[~np.isnan(rt_labels)]    
                if len(rt_labels) > 1:
                    df_ses_lab.loc[:,['NM_diff']] = np.diff(df_ses_lab['NM_choice-gocue_AUC'].values, prepend=[np.nan])            
                    df_ses_lab.loc[:,['choice_time_diff']] = np.diff(df_ses_lab['choice_time'].values, prepend=[np.nan])            
                df_rt_ses = pd.concat([df_rt_ses, df_ses_lab])
    return df_rt_ses

#%%
def figure_RPE_vs_reactiontimes(df_trials_ses, model='Hattori2019', plot=False):
    events_colors = [matplotlib.colormaps.get_cmap('Paired')(i)[:3] for i in [0, 2, 3, 6, 4, 5]]
    colset1 = [matplotlib.colormaps['tab20'](i)[:3] for i in [2, 3, 0, 1]]
    colset2 = [matplotlib.colormaps['tab20'](i)[:3] for i in [4, 5, 6, 7]]
    left_color, right_color, ignore_color = 'red', 'blue', 'green'
    choice_cols = [right_color, left_color, ignore_color]

    idxs = df_trials_ses['choice_value']
    labels = df_trials_ses['choice_times_labels']
    max_label0 = int(np.nanmax(labels[idxs==0]))

    colset = colset1[:max_label0+1] + colset2

    if plot:
        fig, axs = plt.subplot_mosaic([['d1', 'd2','d3'], ['d4', 'd5', 'd6']], layout='constrained', figsize=(12, 5), height_ratios=[1.,1.])

        title = 'RPE vs. NM reward response'
        label = 'd1'
        ax = axs[label]
        trans = mtransforms.ScaledTranslation(10/72, -5/72, fig.dpi_scale_trans)
        ax.text(0.0, 1.0, title, transform=ax.transAxes + trans, fontsize='medium', verticalalignment='top', fontfamily='serif', bbox=dict(facecolor='1.0', edgecolor='none', pad=3.0))
        sns.regplot(data=df_trials_ses[(df_trials_ses['reward']==True) & (df_trials_ses[model+'_RPE']>0)], x=model+'_RPE', y='NM_choice-gocue_AUC', color='orange', scatter_kws=dict(s=4, alpha=0.5), line_kws=dict(color="r"), ax=ax)
        sns.regplot(data=df_trials_ses[(df_trials_ses['reward']==False) & (df_trials_ses['choice_value']!=0.5)& (df_trials_ses[model+'_RPE']<0)], x=model+'_RPE', y='NM_choice-gocue_AUC', color='green', scatter_kws=dict(s=4, alpha=0.5), line_kws=dict(color="r"), ax=ax)
        # ax.set_ylim([-1.,5.])
        ax.legend()
        sns.despine()

        title = 'RPE vs. NM reward response'
        label = 'd4'
        ax = axs[label]
        trans = mtransforms.ScaledTranslation(10/72, -5/72, fig.dpi_scale_trans)
        ax.text(0.0, 1.0, title, transform=ax.transAxes + trans, fontsize='medium', verticalalignment='top', fontfamily='serif', bbox=dict(facecolor='1.0', edgecolor='none', pad=3.0))
        labels = np.unique(df_trials_ses['choice_times_labels'])
        for i_label, label in enumerate(labels):
            sns.regplot(data=df_trials_ses[(df_trials_ses['reward']==True) & (df_trials_ses[model+'_RPE']>0) & (df_trials_ses['choice_times_labels']==label)], x=model+'_RPE', y='NM_choice-gocue_AUC', color=colset[i_label], scatter_kws=dict(s=4, alpha=0.5), line_kws=dict(color=colset[i_label]), ax=ax)
            sns.regplot(data=df_trials_ses[(df_trials_ses['reward']==False) & (df_trials_ses['choice_value']!=0.5)& (df_trials_ses[model+'_RPE']<0) & (df_trials_ses['choice_times_labels']==label)], x=model+'_RPE', y='NM_choice-gocue_AUC', color=colset[i_label], scatter_kws=dict(s=4, alpha=0.5), line_kws=dict(color=colset[i_label]), ax=ax)
        # ax.set_ylim([-1.,5.])
        ax.legend()
        sns.despine()

        title = 'Q-value choice vs. NM go-cue response'
        label = 'd2'
        ax = axs[label]
        trans = mtransforms.ScaledTranslation(10/72, -5/72, fig.dpi_scale_trans)
        ax.text(0.0, 1.0, title, transform=ax.transAxes + trans, fontsize='medium', verticalalignment='top', fontfamily='serif', bbox=dict(facecolor='1.0', edgecolor='none', pad=3.0))
        sns.regplot(data=df_trials_ses[(df_trials_ses['reward']==True) & (df_trials_ses[model+'_RPE']>0)], x=model+'_Q_Delta', y='NM_gocue2choice_AUC', color=events_colors[0], scatter_kws=dict(s=4, alpha=0.5), line_kws=dict(color="r"), ax=ax)
        sns.regplot(data=df_trials_ses[(df_trials_ses['reward']==False) & (df_trials_ses['choice_value']!=0.5)& (df_trials_ses[model+'_RPE']<0)], x=model+'_Q_Delta', y='NM_gocue2choice_AUC', color=events_colors[1], scatter_kws=dict(s=4, alpha=0.5), line_kws=dict(color="r"), ax=ax)
        # ax.set_ylim([-1.,5.])
        ax.legend()
        sns.despine()

        title = 'Q-value choice vs. NM go-cue response'
        label = 'd5'
        ax = axs[label]
        trans = mtransforms.ScaledTranslation(10/72, -5/72, fig.dpi_scale_trans)
        ax.text(0.0, 1.0, title, transform=ax.transAxes + trans, fontsize='medium', verticalalignment='top', fontfamily='serif', bbox=dict(facecolor='1.0', edgecolor='none', pad=3.0))
        for i_label,label in enumerate(labels):
            sns.regplot(data=df_trials_ses[(df_trials_ses['reward']==True) & (df_trials_ses[model+'_RPE']>0) & (df_trials_ses['choice_times_labels']==label)], x=model+'_Q_Delta', y='NM_gocue2choice_AUC', color=colset[i_label], scatter_kws=dict(s=4, alpha=0.5), line_kws=dict(color=colset[i_label]), ax=ax)
            sns.regplot(data=df_trials_ses[(df_trials_ses['reward']==False) & (df_trials_ses['choice_value']!=0.5)& (df_trials_ses[model+'_RPE']<0) & (df_trials_ses['choice_times_labels']==label)], x=model+'_Q_Delta', y='NM_gocue2choice_AUC', color=colset[i_label], scatter_kws=dict(s=4, alpha=0.5), line_kws=dict(color=colset[i_label]), ax=ax)
        # ax.set_ylim([-1.,5.])
        ax.legend()
        sns.despine()


        title = 'RPE vs. NM (reward-gocue) response'
        label = 'd3'
        ax = axs[label]
        trans = mtransforms.ScaledTranslation(10/72, -5/72, fig.dpi_scale_trans)
        ax.text(0.0, 1.0, title, transform=ax.transAxes + trans, fontsize='medium', verticalalignment='top', fontfamily='serif', bbox=dict(facecolor='1.0', edgecolor='none', pad=3.0))
        sns.regplot(data=df_trials_ses[(df_trials_ses['reward']==True) & (df_trials_ses[model+'_RPE']>0)], x=model+'_RPE', y='NM_choice-gocue_AUC', color=events_colors[0], scatter_kws=dict(s=4, alpha=0.5), line_kws=dict(color="r"), ax=ax)
        sns.regplot(data=df_trials_ses[(df_trials_ses['reward']==False) & (df_trials_ses['choice_value']!=0.5)& (df_trials_ses[model+'_RPE']<0)], x=model+'_RPE', y='NM_choice-gocue_AUC', color=events_colors[1], scatter_kws=dict(s=4, alpha=0.5), line_kws=dict(color="r"), ax=ax)
        # ax.set_ylim([-1.,5.])
        ax.legend()
        sns.despine()


        title = 'RPE vs. NM (reward-gocue) response'
        label = 'd6'
        ax = axs[label]
        trans = mtransforms.ScaledTranslation(10/72, -5/72, fig.dpi_scale_trans)
        ax.text(0.0, 1.0, title, transform=ax.transAxes + trans, fontsize='medium', verticalalignment='top', fontfamily='serif', bbox=dict(facecolor='1.0', edgecolor='none', pad=3.0))
        labels = np.unique(df_trials_ses['choice_times_labels'])
        for i_label,label in enumerate(labels):
            sns.regplot(data=df_trials_ses[(df_trials_ses['reward']==True) & (df_trials_ses[model+'_RPE']>0) & (df_trials_ses['choice_times_labels']==label)], x=model+'_RPE', y='NM_choice-gocue_AUC', color=colset[i_label], scatter_kws=dict(s=4, alpha=0.5), line_kws=dict(color=colset[i_label]), ax=ax)
            sns.regplot(data=df_trials_ses[(df_trials_ses['reward']==False) & (df_trials_ses['choice_value']!=0.5)& (df_trials_ses[model+'_RPE']<0) & (df_trials_ses['choice_times_labels']==label)], x=model+'_RPE', y='NM_choice-gocue_AUC', color=colset[i_label], scatter_kws=dict(s=4, alpha=0.5), line_kws=dict(color=colset[i_label]), ax=ax)
        # ax.set_ylim([-1.,5.])
        ax.legend()
        sns.despine()
    else:
        fig, axs = np.nan, np.nan
    
    return fig, axs





def generate_session_figure(df_trials_ses, model='Hattori2019'):
    # Enrich dataset
    df_trials_ses, events2plot, traces2plot, trace_times2plot = compute_sessionwide_traces_multi(df_trials_ses)
    trace_names_list = df_trials_ses[['NM_name','region']].drop_duplicates().values
    trace_name = ' '.join(trace_names_list[0])
    # trace_name = df_trials_ses.NM_name.unique()[0]
    NM = trace_names_list[0][0]
    ses_idx = df_trials_ses.ses_idx.unique()[0]
    foraging_eff = df_trials_ses.foraging_eff.unique()[0]

    events_colors = [matplotlib.colormaps.get_cmap('Paired')(i)[:3] for i in [0, 2, 3, 6, 4, 5]]
    colset1 = [matplotlib.colormaps['tab20'](i)[:3] for i in [2, 3, 0, 1]]
    colset2 = [matplotlib.colormaps['tab20'](i)[:3] for i in [4, 5, 6, 7]]
    left_color, right_color, ignore_color = 'red', 'blue', 'green'
    # left_color, right_color, ignore_color = events_colors[0], events_colors[1], events_colors[2]
    choice_cols = [right_color, left_color, ignore_color]

    import matplotlib.transforms as mtransforms
    fig, axs = plt.subplot_mosaic([['a', 'a', 'a','a'] , ['b', 'b', 'b', 'b'], ['b1', 'b1', 'b1', 'b1'], ['b2', 'b2', 'b2', 'b2'], ['c1', 'd1', 'd2', 'd3'],['c1', 'd1', 'd2','d3'], ['c2', 'd4', 'd5', 'd6'],['c2', 'd4', 'd5','d6'], ['e', 'e', 'e','e'], ['f', 'f', 'f','f'], ['g1', 'g2', 'g3','g7'], ['g4', 'g5', 'g6','g8'], ['h1', 'h2', 'h3','h4'], ['h1', 'i2', 'i3','i4'],['j1', 'j2', 'j3', 'j4']], layout='constrained', figsize=(30, 20), height_ratios=[2.,1.,1.,1.,1.,1.,1.,1.,1.,2.,3.,3.,1.,1.,1.])

    title = 'Trial Structure ' + ses_idx + ' ' + trace_name + ' forag. eff. ' + str(foraging_eff) + ' N trials ' + str(len(df_trials_ses)) + ' Left/Right (top/bottom)' + ' (rolling mean 5)'
    label = 'a'
    ax = axs[label]
    x = df_trials_ses.trial.values
    y = df_trials_ses[model+'_Lprob'].values
    ax.plot(x,0.1+0.8*y, c='black', linewidth=2.)

    df_trials_ses['choice_value'] = df_trials_ses['choice'].map({'R':0, 'L':1, np.nan:0.5, 'Ignore':0.5})
    df_trials_ses['choice_value_roll'] = df_trials_ses['choice_value'].dropna().rolling(5).mean().reindex(df_trials_ses.index)
    x = df_trials_ses.trial.values
    y = df_trials_ses.choice_value_roll.values
    ax.plot(x,0.1+0.8*y, '--',c='gray',linewidth=1.)

    yl = 0.5+0.6*df_trials_ses.left_reward_prob.values
    yr = 0.5-0.6*df_trials_ses.right_reward_prob.values
    d = 0.5*np.ones(len(yl))
    x = df_trials_ses.trial.values
    ax.fill_between(x, d, yl, where=yl>=d, interpolate=True, color='red', alpha=0.3)
    ax.fill_between(x, yr, d, where=yr<=d, interpolate=True, color='blue', alpha=0.3)

    for i_choice, choice_val in enumerate([0, 1, 0.5]):
        idxs = df_trials_ses['choice_value'] == choice_val
        for i_reward, reward_val in enumerate([True, False]):
            idxs_rew = df_trials_ses['reward'].values == reward_val
            x = df_trials_ses.trial.values[idxs & idxs_rew]
            y = df_trials_ses.choice_value[idxs & idxs_rew]            
            ax.scatter(x, y, s=60, c=choice_cols[i_choice], marker='|')
            if reward_val == True:
                ax.scatter(x, y, s=40, c='black', marker='|')
    
    
    trans = mtransforms.ScaledTranslation(10/72, -5/72, fig.dpi_scale_trans)
    ax.text(0.0, 0., title, transform=ax.transAxes + trans, fontsize='medium', verticalalignment='top', fontfamily='serif', bbox=dict(facecolor='1.0', edgecolor='none', pad=3.0))        
    ax.margins(x=0)
    ax.set_axis_off()

    title = 'Reaction Times over trials (rolling mean 15)'
    label = 'b'
    ax = axs[label]
    choice_colors = [matplotlib.cm.get_cmap('tab20')(i)[:3] for i in [0, 2, 3, 6, 4, 5]]
    df_trials_ses['reaction_time'] = df_trials_ses['choice_time'] - df_trials_ses['go_cue']
    df_trials_ses['reaction_time_roll'] = df_trials_ses['reaction_time'].dropna().rolling(15).mean().reindex(df_trials_ses.index)
    for i_choice, choice_val in enumerate([0, 1]):
        idxs = df_trials_ses['choice_value'] == choice_val
        x = df_trials_ses.trial.values[idxs]
        y = df_trials_ses.reaction_time[idxs]    
        ax.scatter(x, y, s=40, c=choice_cols[i_choice], marker='|')
    x_roll = df_trials_ses.trial.values
    y_roll = df_trials_ses.reaction_time_roll
    ax.plot(x_roll, y_roll, c='gray')
    trans = mtransforms.ScaledTranslation(10/72, -5/72, fig.dpi_scale_trans)
    ax.text(0.0, 1.0, title, transform=ax.transAxes + trans, fontsize='medium', verticalalignment='top', fontfamily='serif', bbox=dict(facecolor='1.0', edgecolor='none', pad=3.0))    
    ax.set_ylim([0,np.nanpercentile(df_trials_ses['reaction_time'],95)])
    ax.margins(x=0)
    ax.set_axis_off()


    title = 'Licks over trials (choice side only, rolling mean 15)'
    label = 'b1'
    ax = axs[label]
    choice_colors = [matplotlib.cm.get_cmap('tab20')(i)[:3] for i in [0, 2, 3, 6, 4, 5]]
    # df_trials_ses['licks_L_roll'] = df_trials_ses['licks L'].rolling(window=5).mean()
    # df_trials_ses['licks_R_roll'] = df_trials_ses['licks R'].rolling(window=5).mean()
    df_trials_ses['lick_choice_roll'] = np.nan*df_trials_ses.index.values
    choice_val2RL = {0:'R', 1:'L', 0.5:'ignore'}
    x_roll = df_trials_ses.trial.values
    for i_choice, choice_val in enumerate([0, 1]):
        idxs = df_trials_ses['choice_value'] == choice_val
        x = df_trials_ses.trial.values[idxs]
        y = df_trials_ses['licks '+choice_val2RL[choice_val]][idxs]
        df_trials_ses['lick_choice_roll'][idxs] = y
        # yr = df_trials_ses['licks R'][idxs]
        ax.scatter(x, y, s=40, c=choice_cols[i_choice], marker='|')
        # ax.scatter(x, yr, s=40, c=events_colors[i_choice], marker='|')
    # yr_roll = df_trials_ses.licks_R_roll
    # yl_roll = df_trials_ses.licks_L_roll
    y_roll = df_trials_ses['lick_choice_roll'].dropna().rolling(15).mean().reindex(df_trials_ses.index)
    ax.plot(x_roll, y_roll, c='gray', alpha=0.8)
    # ax.plot(x_roll, yl_roll, c='gray', alpha=0.5)
    trans = mtransforms.ScaledTranslation(10/72, -5/72, fig.dpi_scale_trans)
    ax.text(0.0, 1.0, title, transform=ax.transAxes + trans, fontsize='medium', verticalalignment='top', fontfamily='serif', bbox=dict(facecolor='1.0', edgecolor='none', pad=3.0))
    ax.set_ylim([0,np.nanmax(y_roll)+2])
    ax.margins(x=0)
    ax.set_axis_off()

    
    title = 'DA go cue AUC'
    label = 'b2'
    ax = axs[label]
    choice_colors = [matplotlib.cm.get_cmap('tab20')(i)[:3] for i in [0, 2, 3, 6, 4, 5]]    
    choice_val2RL = {0:'R', 1:'L', 0.5:'ignore'}
    x_roll = df_trials_ses.trial.values
    for i_choice, choice_val in enumerate([0, 1]):
        idxs = df_trials_ses['choice_value'] == choice_val
        x = df_trials_ses.trial.values[idxs]
        y = df_trials_ses['NM_gocue2choice_AUC'][idxs]            
        ax.scatter(x, y, s=40, c=choice_cols[i_choice], marker='|')
        y = df_trials_ses['NM_choice2choicep1_AUC'][idxs]                    
        ax.scatter(x, y, s=40, c='black', marker='|')        
    y_roll = df_trials_ses['NM_gocue2choice_AUC'].dropna().rolling(15).mean().reindex(df_trials_ses.index)
    ax.plot(x_roll, y_roll, c='gray', alpha=0.8)    
    y_roll = df_trials_ses['NM_choice2choicep1_AUC'].dropna().rolling(15).mean().reindex(df_trials_ses.index)
    ax.plot(x_roll, y_roll, c='black', alpha=0.6)    
    trans = mtransforms.ScaledTranslation(10/72, -5/72, fig.dpi_scale_trans)
    ax.text(0.0, 1.0, title, transform=ax.transAxes + trans, fontsize='medium', verticalalignment='top', fontfamily='serif', bbox=dict(facecolor='1.0', edgecolor='none', pad=3.0))
    ax.set_ylim([0,np.nanmax(y_roll)+2])
    ax.margins(x=0)
    ax.set_axis_off()


    title = 'Reaction Times'
    label = 'c1'
    ax = axs[label]
    trans = mtransforms.ScaledTranslation(10/72, -5/72, fig.dpi_scale_trans)
    ax.text(0.0, 1.0, title, transform=ax.transAxes + trans, fontsize='medium', verticalalignment='top', fontfamily='serif', bbox=dict(facecolor='1.0', edgecolor='none', pad=3.0))
    idxs = df_trials_ses['choice_value']
    axis_max = np.nanpercentile(df_trials_ses['reaction_time'].values,95)+0.1
    ax.hist(df_trials_ses['reaction_time'].values[idxs==0], color=choice_cols[0], bins=np.arange(0,axis_max,0.02), alpha=0.4, label='Right')
    ax.hist(df_trials_ses['reaction_time'].values[idxs==1], color=choice_cols[1], bins=np.arange(0,axis_max,0.02), alpha=0.4, label='Left')
    ax.legend()
    sns.despine()

    title = 'Reaction Times'
    label = 'c2'
    ax = axs[label]
    trans = mtransforms.ScaledTranslation(10/72, -5/72, fig.dpi_scale_trans)
    ax.text(0.0, 1.0, title, transform=ax.transAxes + trans, fontsize='medium', verticalalignment='top', fontfamily='serif', bbox=dict(facecolor='1.0', edgecolor='none', pad=3.0))
    idxs = df_trials_ses['choice_value']
    axis_max = np.nanpercentile(df_trials_ses['reaction_time'].values,95)+0.1
    labels = np.nan*idxs
    reaction_times = df_trials_ses['reaction_time'].values
    reaction_timesR = df_trials_ses['reaction_time'].values[idxs==0]
    reaction_timesL = df_trials_ses['reaction_time'].values[idxs==1]
    bimod_gmmcriteria0, gmm_labels0, gmm_probs0 = bimodality_gmmcriteria(reaction_timesR)    
    bimod_gmmcriteria1, gmm_labels1, gmm_probs1 = bimodality_gmmcriteria(reaction_timesL)    
    if bimod_gmmcriteria0:
        gmm_labels0 = rename_labels_by_average(gmm_labels0, reaction_timesR)    
        labels[idxs==0] = gmm_labels0
    else:
        labels[idxs==0] = 0
    max_label0 = int(np.nanmax(labels[idxs==0]))
    if bimod_gmmcriteria1:
        gmm_labels1 = rename_labels_by_average(gmm_labels1, reaction_timesL)
        labels[idxs==1] = gmm_labels1 + max_label0 + 1    
    else:
        labels[idxs==1] = max_label0 + 1          
    print(max_label0)
    colset = colset1[:max_label0+1] + colset2
    for i_label, label in enumerate(np.unique(labels)):
        if label > max_label0:
            label_name = 'Left ' + str(i_label - max_label0 + 1)
        else:
            label_name = 'Right ' + str(i_label + 1)
        if np.sum(labels==label) <10:
            continue
        ax.hist(reaction_times[labels==label], bins=np.arange(0,axis_max,0.02), alpha=0.4, color=colset[i_label], label=label_name)
    df_trials_ses['choice_times_labels'] = labels
    ax.legend()
    sns.despine()

    title = 'RPE vs. NM reward response'
    label = 'd1'
    ax = axs[label]
    trans = mtransforms.ScaledTranslation(10/72, -5/72, fig.dpi_scale_trans)
    ax.text(0.0, 1.0, title, transform=ax.transAxes + trans, fontsize='medium', verticalalignment='top', fontfamily='serif', bbox=dict(facecolor='1.0', edgecolor='none', pad=3.0))
    sns.regplot(data=df_trials_ses[(df_trials_ses['reward']==True) & (df_trials_ses[model+'_RPE']>0)], x=model+'_RPE', y='NM_choice2choicep1_AUC', color='orange', scatter_kws=dict(s=4, alpha=0.5), line_kws=dict(color="r"), ax=ax)
    sns.regplot(data=df_trials_ses[(df_trials_ses['reward']==False) & (df_trials_ses['choice_value']!=0.5)& (df_trials_ses[model+'_RPE']<0)], x=model+'_RPE', y='NM_choice2choicep1_AUC', color='green', scatter_kws=dict(s=4, alpha=0.5), line_kws=dict(color="r"), ax=ax)
    ax.set_ylim([-1.,5.])
    ax.legend()
    sns.despine()

    title = 'RPE vs. NM reward response'
    label = 'd4'
    ax = axs[label]
    trans = mtransforms.ScaledTranslation(10/72, -5/72, fig.dpi_scale_trans)
    ax.text(0.0, 1.0, title, transform=ax.transAxes + trans, fontsize='medium', verticalalignment='top', fontfamily='serif', bbox=dict(facecolor='1.0', edgecolor='none', pad=3.0))
    labels = np.unique(df_trials_ses['choice_times_labels'])
    for i_label, label in enumerate(labels):
        sns.regplot(data=df_trials_ses[(df_trials_ses['reward']==True) & (df_trials_ses[model+'_RPE']>0) & (df_trials_ses['choice_times_labels']==label)], x=model+'_RPE', y='NM_choice2choicep1_AUC', color=colset[i_label], scatter_kws=dict(s=4, alpha=0.5), line_kws=dict(color=colset[i_label]), ax=ax)
        sns.regplot(data=df_trials_ses[(df_trials_ses['reward']==False) & (df_trials_ses['choice_value']!=0.5)& (df_trials_ses[model+'_RPE']<0) & (df_trials_ses['choice_times_labels']==label)], x=model+'_RPE', y='NM_choice2choicep1_AUC', color=colset[i_label], scatter_kws=dict(s=4, alpha=0.5), line_kws=dict(color=colset[i_label]), ax=ax)
    ax.set_ylim([-1.,5.])
    ax.legend()
    sns.despine()

    title = 'Q-value choice vs. NM go-cue response'
    label = 'd2'
    ax = axs[label]
    trans = mtransforms.ScaledTranslation(10/72, -5/72, fig.dpi_scale_trans)
    ax.text(0.0, 1.0, title, transform=ax.transAxes + trans, fontsize='medium', verticalalignment='top', fontfamily='serif', bbox=dict(facecolor='1.0', edgecolor='none', pad=3.0))
    sns.regplot(data=df_trials_ses[(df_trials_ses['reward']==True) & (df_trials_ses[model+'_RPE']>0)], x=model+'_Cvalue', y='NM_gocue2choice_AUC', color=events_colors[0], scatter_kws=dict(s=4, alpha=0.5), line_kws=dict(color="r"), ax=ax)
    sns.regplot(data=df_trials_ses[(df_trials_ses['reward']==False) & (df_trials_ses['choice_value']!=0.5)& (df_trials_ses[model+'_RPE']<0)], x=model+'_Cvalue', y='NM_gocue2choice_AUC', color=events_colors[1], scatter_kws=dict(s=4, alpha=0.5), line_kws=dict(color="r"), ax=ax)
    ax.set_ylim([-1.,5.])
    ax.legend()
    sns.despine()

    title = 'Q-value choice vs. NM go-cue response'
    label = 'd5'
    ax = axs[label]
    trans = mtransforms.ScaledTranslation(10/72, -5/72, fig.dpi_scale_trans)
    ax.text(0.0, 1.0, title, transform=ax.transAxes + trans, fontsize='medium', verticalalignment='top', fontfamily='serif', bbox=dict(facecolor='1.0', edgecolor='none', pad=3.0))
    for i_label,label in enumerate(labels):
        sns.regplot(data=df_trials_ses[(df_trials_ses['reward']==True) & (df_trials_ses[model+'_RPE']>0) & (df_trials_ses['choice_times_labels']==label)], x=model+'_Cvalue', y='NM_gocue2choice_AUC', color=colset[i_label], scatter_kws=dict(s=4, alpha=0.5), line_kws=dict(color=colset[i_label]), ax=ax)
        sns.regplot(data=df_trials_ses[(df_trials_ses['reward']==False) & (df_trials_ses['choice_value']!=0.5)& (df_trials_ses[model+'_RPE']<0) & (df_trials_ses['choice_times_labels']==label)], x=model+'_Cvalue', y='NM_gocue2choice_AUC', color=colset[i_label], scatter_kws=dict(s=4, alpha=0.5), line_kws=dict(color=colset[i_label]), ax=ax)
    ax.set_ylim([-1.,5.])
    ax.legend()
    sns.despine()


    title = 'RPE vs. NM (reward-gocue) response'
    label = 'd3'
    ax = axs[label]
    trans = mtransforms.ScaledTranslation(10/72, -5/72, fig.dpi_scale_trans)
    ax.text(0.0, 1.0, title, transform=ax.transAxes + trans, fontsize='medium', verticalalignment='top', fontfamily='serif', bbox=dict(facecolor='1.0', edgecolor='none', pad=3.0))
    sns.regplot(data=df_trials_ses[(df_trials_ses['reward']==True) & (df_trials_ses[model+'_RPE']>0)], x=model+'_RPE', y='NM_choice-gocue_AUC', color=events_colors[0], scatter_kws=dict(s=4, alpha=0.5), line_kws=dict(color="r"), ax=ax)
    sns.regplot(data=df_trials_ses[(df_trials_ses['reward']==False) & (df_trials_ses['choice_value']!=0.5)& (df_trials_ses[model+'_RPE']<0)], x=model+'_RPE', y='NM_choice-gocue_AUC', color=events_colors[1], scatter_kws=dict(s=4, alpha=0.5), line_kws=dict(color="r"), ax=ax)
    ax.set_ylim([-1.,5.])
    ax.legend()
    sns.despine()


    title = 'RPE vs. NM (reward-gocue) response'
    label = 'd6'
    ax = axs[label]
    trans = mtransforms.ScaledTranslation(10/72, -5/72, fig.dpi_scale_trans)
    ax.text(0.0, 1.0, title, transform=ax.transAxes + trans, fontsize='medium', verticalalignment='top', fontfamily='serif', bbox=dict(facecolor='1.0', edgecolor='none', pad=3.0))
    labels = np.unique(df_trials_ses['choice_times_labels'])
    for i_label,label in enumerate(labels):
        sns.regplot(data=df_trials_ses[(df_trials_ses['reward']==True) & (df_trials_ses[model+'_RPE']>0) & (df_trials_ses['choice_times_labels']==label)], x=model+'_RPE', y='NM_choice-gocue_AUC', color=colset[i_label], scatter_kws=dict(s=4, alpha=0.5), line_kws=dict(color=colset[i_label]), ax=ax)
        sns.regplot(data=df_trials_ses[(df_trials_ses['reward']==False) & (df_trials_ses['choice_value']!=0.5)& (df_trials_ses[model+'_RPE']<0) & (df_trials_ses['choice_times_labels']==label)], x=model+'_RPE', y='NM_choice-gocue_AUC', color=colset[i_label], scatter_kws=dict(s=4, alpha=0.5), line_kws=dict(color=colset[i_label]), ax=ax)
    ax.set_ylim([-1.,5.])
    ax.legend()
    sns.despine()


    title = 'Trials Events'
    label = 'e'
    min_time = np.nanmin(trace_times2plot.astype(float))
    time_span = (min_time, min_time+600)
    ax = axs[label]
    min_last = 0.5
    for i_event, event in enumerate(events2plot.keys()):
        for event_time in events2plot[event]:
            if event_time < time_span[1]:
                ax.plot([event_time, event_time+min_last], [i_event+1, i_event+1], color=events_colors[i_event], linewidth=4., zorder=0, clip_on=False)
    # for i_event, event in enumerate(events_names):
        # event_times = events2plot[event]
        # ax.scatter(event_times, np.ones_like(event_times) * i_event * 0.4, s=100, c=events_colors[i_event], marker='|', label=event)#, transform=trans+offset(0))    
    trans = mtransforms.ScaledTranslation(10/72, -5/72, fig.dpi_scale_trans)
    ax.text(0.0, 1.0, title, transform=ax.transAxes + trans, fontsize='medium', verticalalignment='top', fontfamily='serif', bbox=dict(facecolor='1.0', edgecolor='none', pad=3.0))
    # ax.margins(x=min_time)
    ax.set_xlim(time_span)    
    ax.set_axis_off()
    sns.despine()
    # plt.tight_layout()

    title = trace_name+' Trace'
    label = 'f'
    ax = axs[label]
    # ax.margins(x=0)
    ax.set_xlim(time_span)
    ax.plot(trace_times2plot, traces2plot[trace_name] * 100, linewidth=1., color='black')
    ax.plot(trace_times2plot, np.zeros(len(trace_times2plot)), '--', color='gray', linewidth=0.4)
    ax.set_ylabel('dF/F (%)'+ trace_name)
    ax_min, ax_max = ax.get_ylim()
    for i_event, event in enumerate(events2plot.keys()):
        for event_time in events2plot[event]:    
            con = ConnectionPatch(xyA=[event_time, ax_max], xyB=[event_time, ax_min], coordsA="data", coordsB="data", axesA=ax, axesB=ax, color=events_colors[i_event], linewidth=1., zorder=0, clip_on=False)
            ax.add_artist(con)
    # ax.set_axis_off()
    trans = mtransforms.ScaledTranslation(10/72, -5/72, fig.dpi_scale_trans)
    ax.text(0.0, 1.0, title, transform=ax.transAxes + trans, fontsize='medium', verticalalignment='top', fontfamily='serif', bbox=dict(facecolor='1.0', edgecolor='none', pad=3.0))    
    ax.spines['left'].set_visible(False)
    ax.axes.get_yaxis().set_visible(False)


    # def find_constant_sequences(arr):    
    #     change_indices = np.where(arr[:-1] != arr[1:])[0] + 1
    #     constant_sequences = np.split(arr, change_indices)
    #     sequence_lengths = np.array([len(seq) for seq in constant_sequences if len(seq) > 1])    
    #     sequence_values = np.array([seq[0] for seq in constant_sequences if len(seq) > 1])    
    #     return sequence_values, sequence_lengths

    # # Example usage
    # input_array = df_trials_ses.choice.map({'R':0,'L':1}).values
    # sequence_values, sequence_lengths = find_constant_sequences(input_array)

    # title = 'Lengths of sequences'
    # label = 'c3'
    # ax = axs[label]
    # trans = mtransforms.ScaledTranslation(10/72, -5/72, fig.dpi_scale_trans)
    # ax.text(0.0, 1.0, title, transform=ax.transAxes + trans, fontsize='medium', verticalalignment='top', fontfamily='serif', bbox=dict(facecolor='1.0', edgecolor='none', pad=3.0))
    # ax.hist(sequence_lengths[sequence_values==0], color=choice_cols[0], bins=np.arange(2,25,1), alpha=0.4, label='Right')
    # ax.hist(sequence_lengths[sequence_values==1], color=choice_cols[1], bins=np.arange(2,25,1), alpha=0.4, label='Left')
    # ax.legend()
    # sns.despine()


    title = 'Average NM go cue aligned'
    label = 'g1'
    ax = axs[label]
    trans = mtransforms.ScaledTranslation(10/72, -5/72, fig.dpi_scale_trans)
    ax.text(0.0, 1.0, title, transform=ax.transAxes + trans, fontsize='medium', verticalalignment='top', fontfamily='serif', bbox=dict(facecolor='1.0', edgecolor='none', pad=3.0))
    df_trials_ses['go_cue_ts'] = df_trials_ses.apply(lambda x: x['bins_mids'] + x['go_cue_absolute_time']-x['go cue'], axis=1)
    df_trials_ses['NM_go_cue_aligned'] = df_trials_ses.apply(lambda x: x['NM_norm'][(x['go_cue_ts']>-.5) & (x['go_cue_ts']<1.)], axis=1)
    df_trials_ses['go_cue_ts_aligned'] = df_trials_ses.apply(lambda x: x['go_cue_ts'][(x['go_cue_ts']>-.5) & (x['go_cue_ts']<1.)], axis=1)

    df2plot = df_trials_ses
    cols2keep = ['trial', 'reward', 'choice', 'bit_code', 'ses_idx', 'rpe', 'left_action_value', 'right_action_value', 'left_reward_prob', 'right_reward_prob', 'NM_name' , 'go_cue_ts_aligned', 'NM_go_cue_aligned']
    df2plot = df2plot.set_index(cols2keep[:-2]).explode(cols2keep[-2:]).reset_index()
    df2plot['go_cue_ts_aligned'] = pd.cut(df2plot['go_cue_ts_aligned'], bins=np.arange(-1.5,0.5,0.05), labels=np.arange(-1.5,0.45,0.05), right=True)

    sns.lineplot(data=df2plot, x='go_cue_ts_aligned', y='NM_go_cue_aligned', hue='choice', style='reward', palette=[choice_cols[1],choice_cols[0],choice_cols[2]], ax=ax)
    ax.legend([],[], frameon=False)
    ax.axes.get_yaxis().set_visible(False)
    sns.despine()

    
    title = 'Average NM go cue aligned'
    label = 'g4'
    ax = axs[label]
    trans = mtransforms.ScaledTranslation(10/72, -5/72, fig.dpi_scale_trans)
    ax.text(0.0, 1.0, title, transform=ax.transAxes + trans, fontsize='medium', verticalalignment='top', fontfamily='serif', bbox=dict(facecolor='1.0', edgecolor='none', pad=3.0))    
    sns.lineplot(data=df2plot, x='go_cue_ts_aligned', y='NM_go_cue_aligned', hue='choice_times_labels', style='reward', palette=colset, ax=ax)
    ax.legend([],[], frameon=False)
    ax.axes.get_yaxis().set_visible(False)
    sns.despine()


    title = 'Average '+trace_name+' choice aligned'
    label = 'g2'
    ax = axs[label]
    trans = mtransforms.ScaledTranslation(10/72, -5/72, fig.dpi_scale_trans)
    ax.text(0.0, 1.0, title, transform=ax.transAxes + trans, fontsize='medium', verticalalignment='top', fontfamily='serif', bbox=dict(facecolor='1.0', edgecolor='none', pad=3.0))
    df_trials_ses['choice_ts'] = df_trials_ses.apply(lambda x: x['bins_mids'] + x['go_cue_absolute_time']-x['choice time'], axis=1)
    df_trials_ses['NM_choice_aligned'] = df_trials_ses.apply(lambda x: x['NM_norm'][(x['choice_ts']>-0.5) & (x['choice_ts']<0.5)], axis=1)
    df_trials_ses['choice_ts_aligned'] = df_trials_ses.apply(lambda x: x['choice_ts'][(x['choice_ts']>-0.5) & (x['choice_ts']<0.5)], axis=1)

    df2plot = df_trials_ses
    cols2keep = ['trial', 'reward', 'choice', 'bit_code', 'ses_idx', 'rpe', 'left_action_value', 'right_action_value', 'left_reward_prob', 'right_reward_prob', 'NM_name' , 'choice_ts_aligned', 'NM_choice_aligned']
    df2plot = df2plot.set_index(cols2keep[:-2]).explode(cols2keep[-2:]).reset_index()
    df2plot['choice_ts_aligned'] = pd.cut(df2plot['choice_ts_aligned'], bins=np.arange(-0.5,0.5,0.05), labels=np.arange(-0.5,0.45,0.05), right=True)

    sns.lineplot(data=df2plot, x='choice_ts_aligned', y='NM_choice_aligned',  hue='choice', style='reward', palette=[choice_cols[1],choice_cols[0],choice_cols[2]], ax=ax)
    ax.legend([],[], frameon=False)
    ax.axes.get_yaxis().set_visible(False)    
    sns.despine()

    
    title = 'Average '+trace_name+' choice aligned'
    label = 'g5'
    ax = axs[label]
    trans = mtransforms.ScaledTranslation(10/72, -5/72, fig.dpi_scale_trans)
    ax.text(0.0, 1.0, title, transform=ax.transAxes + trans, fontsize='medium', verticalalignment='top', fontfamily='serif', bbox=dict(facecolor='1.0', edgecolor='none', pad=3.0))
    sns.lineplot(data=df2plot, x='choice_ts_aligned', y='NM_choice_aligned',  hue='choice_times_labels', style='reward', palette=colset, ax=ax)
    ax.legend([],[], frameon=False)
    ax.axes.get_yaxis().set_visible(False)    
    sns.despine()

    title = 'Average '+trace_name+' reward aligned'
    label = 'g3'
    ax = axs[label]
    trans = mtransforms.ScaledTranslation(10/72, -5/72, fig.dpi_scale_trans)
    ax.text(0.0, 1.0, title, transform=ax.transAxes + trans, fontsize='medium', verticalalignment='top', fontfamily='serif', bbox=dict(facecolor='1.0', edgecolor='none', pad=3.0))
    df_trials_ses['reward time gen'] = df_trials_ses.apply(lambda x: x['choice time'] if np.isnan(x['reward time']) else x['reward time'], axis=1)
    df_trials_ses['reward_ts'] = df_trials_ses.apply(lambda x: x['bins_mids'] + x['go_cue_absolute_time']-x['reward time gen'], axis=1)
    df_trials_ses['NM_reward_aligned'] = df_trials_ses.apply(lambda x: x['NM_norm'][(x['reward_ts']>-0.5) & (x['reward_ts']<3.)], axis=1)
    df_trials_ses['reward_ts_aligned'] = df_trials_ses.apply(lambda x: x['reward_ts'][(x['reward_ts']>-0.5) & (x['reward_ts']<3.)], axis=1)

    df2plot = df_trials_ses
    cols2keep = ['trial', 'reward', 'choice', 'bit_code', 'ses_idx', 'rpe', 'left_action_value', 'right_action_value', 'left_reward_prob', 'right_reward_prob', 'NM_name' , 'reward_ts_aligned', 'NM_reward_aligned']
    df2plot = df2plot.set_index(cols2keep[:-2]).explode(cols2keep[-2:]).reset_index()
    df2plot['reward_ts_aligned'] = pd.cut(df2plot['reward_ts_aligned'], bins=np.arange(-0.5,3.,0.05), labels=np.arange(-0.5,2.95,0.05), right=True)

    sns.lineplot(data=df2plot, x='reward_ts_aligned', y='NM_reward_aligned',  hue='choice', style='reward', palette=[choice_cols[1],choice_cols[0],choice_cols[2]], ax=ax)
    # sns.move_legend(ax, "upper center", bbox_to_anchor=(0.5, 1.15), ncol=2)
    ax.legend([],[], frameon=False)
    ax.spines['left'].set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    sns.despine()

    
    title = 'Average '+trace_name+' reward aligned'
    label = 'g6'
    ax = axs[label]
    trans = mtransforms.ScaledTranslation(10/72, -5/72, fig.dpi_scale_trans)
    ax.text(0.0, 1.0, title, transform=ax.transAxes + trans, fontsize='medium', verticalalignment='top', fontfamily='serif', bbox=dict(facecolor='1.0', edgecolor='none', pad=3.0))    
    sns.lineplot(data=df2plot, x='reward_ts_aligned', y='NM_reward_aligned',  hue='choice_times_labels', style='reward', palette=colset, ax=ax)
    # sns.move_legend(ax, "upper center", bbox_to_anchor=(0.5, 1.15), ncol=2)
    ax.legend([],[], frameon=False)
    ax.spines['left'].set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    sns.despine()

    title = 'Average '+trace_name+' reward vs past rewards'
    label = 'h1'
    ax = axs[label]
    x = df_trials_ses.NM_avg_choice_norm.values
    y = df_trials_ses.reward.values.astype(float)
    coef, se, lower, upper = lagged_regression(x, y, max_lag=10)
    # Create a figure and an axis for the plot    
    # Plot the coefficients as a line with markers
    ax.plot(coef, marker='o', label='Coefficient')
    # Plot the error bars using the standard errors
    # ax.errorbar(range(len(coef)), coef, yerr=se, fmt='none', capsize=3, label='Error')
    # Plot the lower and upper bounds of the 95% confidence interval as shaded areas
    ax.fill_between(range(len(coef)), lower, upper, color='lightgray', label='Confidence Interval')
    # Set the x-axis labels as the lag names
    ax.set_xticks(range(len(coef)))
    # ax.set_xticklabels(['lag '+x.split('_')[2] for x in X.columns[1:][::-1].values], rotation=-45, ha='center')
    # Add some labels and a title to the plot
    ax.set_xlabel('Lag')
    ax.set_ylabel('Coefficient')
    ax.set_title('Regression coefficients of x as a function of the lag')
    ax.legend()
    sns.despine()

    title = 'P stay analysis'
    label = 'h2'
    ax = axs[label]
    df_trials_ses['choice_value'] = df_trials_ses['choice'].map({'R':0, 'L':1, np.nan:0.5, 'Ignore':0.5})
    df_trials_ses['stay'] = np.nan
    df_trials_ses['stay'].iloc[:-1] = np.diff(df_trials_ses['choice_value'].values)==0
    sns.barplot(data=df_trials_ses, x='reward', y='stay', hue='choice', ax=ax)
    sns.despine()

    title = 'P stay analysis reward block'
    label = 'h3'
    ax = axs[label]    
    df_trials_ses['choice_rew'] = df_trials_ses['choice'].values + df_trials_ses['reward'].map({True:' Rewarded', False:' Unrewarded'}).values
    sns.barplot(data=df_trials_ses, x='left_reward_prob', y='stay', hue='choice_rew', ax=ax)
    sns.despine()
    
    title = 'Matching law plot'
    label = 'i2'
    ax = axs[label]        
    df_trials_ses['tot_L_choice'] = np.cumsum(df_trials_ses['choice']=='L')/len(df_trials_ses)
    df_trials_ses['tot_L_cum_reward'] = np.cumsum((df_trials_ses['choice']=='L')&(df_trials_ses['reward']==True))/len(df_trials_ses)    
    df_trials_ses['tot_R_choice'] = np.cumsum(df_trials_ses['choice']=='R')/len(df_trials_ses)
    df_trials_ses['tot_R_cum_reward'] = np.cumsum((df_trials_ses['choice']=='R')&(df_trials_ses['reward']==True))/len(df_trials_ses)    
    sns.lineplot(data=df_trials_ses, x='tot_L_cum_reward', y='tot_L_choice', color='red', ax=ax)
    sns.lineplot(data=df_trials_ses, x='tot_R_cum_reward', y='tot_R_choice', color='blue', ax=ax)
    sns.despine()


    title = 'Probability matrix'
    label = 'i3'
    ax = axs[label]        
    R_probstay = np.sum((df_trials_ses['choice'].values[:-1]=='R') & (np.diff(df_trials_ses['choice_value'].values)==0)) / np.sum(df_trials_ses['choice'].values[:-1]=='R')
    R_probchange = np.sum((df_trials_ses['choice'].values[:-1]=='R') & (np.diff(df_trials_ses['choice_value'].values)==1)) / np.sum(df_trials_ses['choice'].values[:-1]=='R')
    L_probstay = np.sum((df_trials_ses['choice'].values[:-1]=='L') & (np.diff(df_trials_ses['choice_value'].values)==0)) / np.sum(df_trials_ses['choice'].values[:-1]=='L')
    L_probchange = np.sum((df_trials_ses['choice'].values[:-1]=='L') & (np.diff(df_trials_ses['choice_value'].values)==1)) / np.sum(df_trials_ses['choice'].values[:-1]=='L')

    probmatrix = np.array([[L_probstay, L_probchange], [R_probchange, R_probstay]])
    counts = np.array([np.sum(df_trials_ses['choice'].values[:-1]=='R'), np.sum(df_trials_ses['choice'].values[:-1]=='L')])   
    cax = ax.matshow(probmatrix, cmap='Blues')
    for i in range(probmatrix.shape[0]):
        for j in range(probmatrix.shape[1]):
            ax.text(j, i, str(int(probmatrix[i, j]*counts[i])) +' / ' + str(counts[i]), va='center', ha='center')
    sns.despine()
    

    title = 'Matching law plot'
    label = 'j1'
    ax = axs[label]        

    df_trials_ses['Rchoices_cum'] = np.cumsum(df_trials_ses['choice'].values=='R')
    df_trials_ses['Lchoices_cum'] = np.cumsum(df_trials_ses['choice'].values=='L')
    df_trials_ses['prob_ratio'] = df_trials_ses['left_reward_prob']/df_trials_ses['right_reward_prob']
    df_trials_ses['block_change'] = (np.diff(df_trials_ses['left_reward_prob'].values,prepend=0)!=0) | (np.diff(df_trials_ses['right_reward_prob'].values,prepend=0)!=0)
    df_trials_ses['block_num'] = np.cumsum(df_trials_ses['block_change'])
    df_trials_ses['block_trial'] = np.sum(np.vstack([np.cumsum(df_trials_ses['block_num'].values==i)*(df_trials_ses['block_num'].values==i).astype(int) for i in range(np.max(df_trials_ses['block_num']))]), axis=0)

    plt.plot(df_trials_ses['Rchoices_cum'].values,df_trials_ses['Lchoices_cum'].values)
    for i_block in range(1,np.max(df_trials_ses['block_num'])):
        idxs_block = np.where(df_trials_ses['block_num'].values==i_block)[0]
        y_shift = df_trials_ses['Lchoices_cum'].values[idxs_block[0]]
        x_shift = df_trials_ses['Rchoices_cum'].values[idxs_block[0]]
        x_axis = np.arange(np.max(df_trials_ses['Rchoices_cum'].values[idxs_block])-np.min(df_trials_ses['Rchoices_cum'].values[idxs_block]))
        y_axis = df_trials_ses['prob_ratio'].values[idxs_block[0]] * x_axis
        plt.plot(x_axis+x_shift, y_axis+y_shift, color='black')
    sns.despine()
    # df_trials_ses['tot_L_choice'] = np.cumsum(df_trials_ses['choice']=='L')/len(df_trials_ses)
    # df_trials_ses['tot_L_cum_reward'] = np.cumsum((df_trials_ses['choice']=='L')&(df_trials_ses['reward']==True))/len(df_trials_ses)    
    # df_trials_ses['tot_R_choice'] = np.cumsum(df_trials_ses['choice']=='R')/len(df_trials_ses)
    # df_trials_ses['tot_R_cum_reward'] = np.cumsum((df_trials_ses['choice']=='R')&(df_trials_ses['reward']==True))/len(df_trials_ses)    
    # sns.lineplot(data=df_trials_ses, x='tot_L_cum_reward', y='tot_L_choice', color='red', ax=ax)
    # sns.lineplot(data=df_trials_ses, x='tot_R_cum_reward', y='tot_R_choice', color='blue', ax=ax)
    # sns.despine()
    # plt.show()
    return fig, axs


#%%
# events, traces, trace_times = events2plot, traces2plot, trace_times2plot

def plot_rawexample(events, traces, trace_times, time_span=[0,600], events_colors = None, session_name='', save=True, plot=False, resultsfolder='../results', tag=''):
    time_span = [np.min([time_span[0], trace_times.min()]), np.max([time_span[1],trace_times.min()])]
    traces_names = list(traces.keys())
    events_names = list(events.keys())
    events2plot = {event_name:event_times[(event_times > time_span[0]) & (event_times < time_span[1])] for event_name, event_times in events.items()}
    traces2plot = {trace_name:trace[(trace_times > time_span[0]) & (trace_times < time_span[1])] for trace_name, trace in traces.items()}
    trace_times2plot = trace_times[(trace_times > time_span[0]) & (trace_times < time_span[1])]
    # events2plot['Lick L (raw)'] = list(set(events2plot['Lick L (raw)']).difference(set(events2plot['choice L'])))
    # events2plot['Lick R (raw)'] = list(set(events2plot['Lick R (raw)']).difference(set(events2plot['choice R'])))
    #%
    if events_colors == None:
        events_colors = [matplotlib.cm.get_cmap('Paired')(i)[:3] for i in [0, 2, 3, 6, 4, 5]]
    plt.clf()
    # fig, axs = plt.subplots(5,1, figsize=(20, 6))
    fig = plt.figure( figsize=(60, 6))
    gs = gridspec.GridSpec(6, 1, height_ratios=[.2,.3,1,1,1,1], wspace=0.0, hspace=0.0, top=0.95, bottom=0.05, left=0.01, right=0.99)

    # offset = lambda p: transforms.ScaledTranslation(p/72.,0, plt.gcf().dpi_scale_trans)
    # trans = plt.gca().transData
    # import matplotlib.transforms as transforms

    ax = plt.subplot(gs[0])
    ax.set_axis_off()
    ax.margins(x=0)
    ax.set_xlim(time_span)
    ax.set_ylim([0,5])
    min_last = np.min(np.diff(events2plot['go_cue']))
    for i_event, event in enumerate(events_names[:4]):
        for event_time in events2plot[event]:
            ax.plot([event_time, event_time+min_last], [i_event+1, i_event+1], color=events_colors[i_event], linewidth=4., zorder=0, clip_on=False)
    ax.set_xlim(time_span)

    ax = plt.subplot(gs[1])
    for i_event, event in enumerate(events_names):
        event_times = events2plot[event]
        ax.scatter(event_times, np.ones_like(event_times) * i_event * 0.3, s=20, c=events_colors[i_event], marker='|', label=event)#, transform=trans+offset(0))
        ax.set_axis_off()
        ax.margins(x=0)
    ax.set_xlim(time_span)

    for i_trace, trace_name in enumerate(traces_names):
        ax = plt.subplot(gs[i_trace+2])
        ax.margins(x=0)
        ax.plot(trace_times2plot, traces2plot[trace_name] * 100, linewidth=1., color='black')
        ax.plot(trace_times2plot, np.zeros(len(trace_times2plot)), '--', color='gray', linewidth=0.4)
        ax.set_axis_off()
        ax.set_ylabel('dF/F (%)')
        ax.set_title('dF/F  ROI# ' + trace_name)
        ax_min, ax_max = ax.get_ylim()
        for i_event, event in enumerate(events_names[:4]):
            for event_time in events2plot[event]:
                # ax.plot([event_time, event_time], [ax_min*.5, ax_max*.5], color=events_colors[i_event], linewidth=0.4, zorder=0, clip_on=False)
                if i_trace == 0:
                    con = ConnectionPatch(xyA=[event_time, ax_max*.5], xyB=[event_time, ax_min*.5], coordsA="data", coordsB="data", axesA=plt.subplot(gs[2]), axesB=plt.subplot(gs[-1]), color=events_colors[i_event], linewidth=1., zorder=0, clip_on=False)
                    ax.add_artist(con)
        ax.set_ylim([ax_min*.75, ax_max*.75])
        ax.set_xlim(time_span)

    ax.set_xlabel('Time (seconds)')
    for i_event, event in enumerate(events_names):
        ax.scatter(time_span[0], time_span[0], s=8., c=events_colors[i_event], label=event)        

    plt.subplots_adjust(wspace=-0.0, hspace=-0.0)
    sns.despine()
    plt.legend(loc='center left')
    # plt.tight_layout()
    if save:
        plt.savefig(resultsfolder +'/'+ 'fig_RawOverview_'+session_name+'.pdf')
    if plot:
        plt.show()
        # plt.close()
    return fig


def compute_difference_vectorized(arr):
    # Ensure the array is a numpy array
    arr = np.array(arr)
    # Find the indices of ones
    one_indices = np.where(arr == 1)[0]
    one_indices = np.concatenate([[0],one_indices])
    # Create an array of indices
    indices = np.arange(len(arr))
    # Use broadcasting to compute the differences between each index and each one index
    differences = indices[:, np.newaxis] - one_indices
    # For each index, find the maximum difference that is less than or equal to zero
    difference = np.min(np.where(differences >= 0, differences, np.Inf), axis=1)    
    return difference

#%%

def pad_and_concatenate(arrays):
    max_length = max(arr.shape[0] for arr in arrays)
    padded_arrays = [np.pad(arr, (0, max_length - arr.shape[0]), constant_values=np.nan) for arr in arrays]
    result = np.stack(padded_arrays)
    return result

def idxs_to_binvec(indices, length):    
    binary_vector = torch.zeros(length) 
    indices = np.array(indices)
    indices = indices[~np.isnan(indices)]
    indices = indices[indices<length]
    if len(indices) > 0:
        binary_vector[indices] = 1
    return binary_vector

def time_to_index(values, array):
    if np.isnan(values):
        return np.nan
    else:
        return np.digitize(values, array)    

def mse_loss_with_nans(input, target):
    mask = torch.isnan(target)
    out = (input[~mask]-target[~mask])**2
    loss = out.mean()
    return loss

def kernel_analysis(df_trials_ses, maxlen=200, Nepochs=2000, l2reg=True):    
#%
# for i in range(1):
    y = pad_and_concatenate(df_trials_ses['NM_norm'].values)
    # maxlen = 100# y.shape[1]
    y = torch.tensor(y[:,:maxlen], dtype=float)

    choice_times = torch.tensor(df_trials_ses.choice_time.values - df_trials_ses.go_cue.values, dtype=float)
    choice_times = choice_times.float()
    choice_idxs = df_trials_ses.apply(lambda x: time_to_index(x.choice_time, x.onset), axis=1).values
    choice_rew_idxs, choice_norew_idxs = choice_idxs.copy(), choice_idxs.copy()
    choice_rew_idxs[df_trials_ses.reward==False] = np.nan
    choice_norew_idxs[df_trials_ses.reward==True] = np.nan    
    gocue_idxs = df_trials_ses.apply(lambda x: time_to_index(x.go_cue, x.onset), axis=1).values

    choice_rew_vecs = np.stack([idxs_to_binvec(choice_idx, maxlen) for choice_idx in choice_rew_idxs], axis=0)
    choice_norew_vecs = np.stack([idxs_to_binvec(choice_idx, maxlen) for choice_idx in choice_norew_idxs], axis=0)
    gocue_vecs = np.stack([idxs_to_binvec(gocue_idx, maxlen) for gocue_idx in gocue_idxs], axis=0)

    gocue_vecs = torch.tensor(gocue_vecs, dtype=float)
    choice_rew_vecs = torch.tensor(choice_rew_vecs, dtype=float)
    choice_norew_vecs = torch.tensor(choice_norew_vecs, dtype=float)
    x = {'gocue':gocue_vecs, 'choice_rew':choice_rew_vecs, 'choice_norew':choice_norew_vecs}
    kernels_th = {'gocue':nn.Parameter(torch.randn(maxlen)), 'choice_rew':nn.Parameter(torch.randn(maxlen)), 'choice_norew':nn.Parameter(torch.randn(maxlen))}
    # constants_th = {'exp_decay':nn.Parameter(torch.tensor(10e-9)), 'exp_const':nn.Parameter(torch.tensor(1.)), 'alpha':nn.Parameter(torch.tensor(1e-5))}
    constants_th = {'alpha':nn.Parameter(torch.tensor(1e-5))}
    # kernels_all = nn.Parameter(torch.randn(3,maxlen,dtype=float)).view(1,3,-1)
    # x_all = torch.tensor(torch.stack([gocue_vecs, choice_rew_vecs, choice_norew_vecs], axis=1), dtype=float)
    
    model = nn.ParameterDict(kernels_th)
    model.update(constants_th)
    model.float()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = mse_loss_with_nans
    if l2reg:
        reg_strength = 0.0001
#%
    # Training loop    
    for epoch in range(Nepochs):  # number of epochs
        running_loss = 0.0
        # output = nn.functional.conv1d(x_all, torch.flip(kernels_all,dims=[2]), padding=kernels_all.shape[-1]-1).squeeze().float()
        # output = output[:,:maxlen]

        output = torch.zeros_like(y)
        for event, events_binvec in x.items():
            optimizer.zero_grad()
            events_binvec = events_binvec.float()
            kernel  = torch.flip(model[event], dims=[0]).view(1,1,-1)            
            events = events_binvec.view(events_binvec.shape[0],1,-1)            
            conv = nn.functional.conv1d(events, kernel, padding=kernel.shape[-1]-1).squeeze().float()
            conv = conv[:,:maxlen]
            output = output + conv
        
        output[~torch.isnan(choice_times)] = (1.+torch.tanh(model['alpha'] * choice_times[~torch.isnan(choice_times)])).view(-1,1) * output[~torch.isnan(choice_times)]
        # output = output * model['exp_const'] * torch.exp(-model['exp_decay']*torch.arange(output.shape[0])).view(-1,1)
        loss = criterion(output, y)     

        if l2reg:
            l2reg_loss = None
            for event, kernel in kernels_th.items():
                if l2reg_loss is None:
                    l2reg_loss = kernel.norm(2)
                else:
                    l2reg_loss = l2reg_loss + kernel.norm(2)
            loss = loss + reg_strength * l2reg_loss
        loss.backward()
        optimizer.step()        
        running_loss += loss.item()

        if epoch%100==1:
            print(f'Epoch {epoch + 1}, Loss: {running_loss / len(x)}')
        
    output = output.cpu().detach().numpy()
    kernels, constants = {}, {}
    for event, ker in kernels_th.items():
        kernels[event] = ker.cpu().detach().numpy()            
    for event, const in constants_th.items():
        constants[event] = const.cpu().detach().numpy()       
#%
    return output, kernels, constants, running_loss


#%% Analisys of DA incresease for consecutive rewards
def analysis_baseline_shift(df_trials_ses, plot=False):
#% Analisys
    df_trials_ses['NM_precue'] = df_trials_ses.apply(lambda x:np.mean(x.NM[(x.onset<x.go_cue) & (x.onset>(x.go_cue-1.))]), axis=1)
    df_trials_ses['NM_endtrial'] = df_trials_ses.apply(lambda x:np.mean(x.NM[-20:-1]), axis=1)
    df_trials_ses['NM_choice'] = df_trials_ses.apply(lambda x:np.mean(x.NM[(x.onset>(x.choice_time+0.)) & (x.onset<(x.choice_time+1.))]), axis=1)
    df_trials_ses['NM_std'] = df_trials_ses.apply(lambda x:np.std(x.NM), axis=1)
    df_trials_ses['NM_diff_abs'] = df_trials_ses['NM_endtrial'] - df_trials_ses['NM_precue']
    df_trials_ses['NM_diff_std'] = (df_trials_ses['NM_endtrial'] - df_trials_ses['NM_precue']) / df_trials_ses['NM_std']
    df_trials_ses = df_trials_ses.reset_index(drop=True)

    df_rewdyn_ses = pd.DataFrame()
    for i_Npast, Npast in enumerate(range(1,8)):
        df_rewdyn = df_trials_ses[['trial', 'region', 'ses_idx']].copy()
        df_rewdyn['reward_tot'] = df_trials_ses['reward'].rolling(Npast).apply(lambda x: np.sum(x==True)-np.sum(x==False)).copy()
        df_rewdyn['reward_true'] = df_trials_ses['reward'].rolling(Npast).apply(lambda x: np.sum(x==True)).copy()
        df_rewdyn['reward_false'] = df_trials_ses['reward'].rolling(Npast).apply(lambda x: np.sum(x==False)).copy()
        df_rewdyn['NM_diff_abs'] = df_trials_ses['NM_diff_abs'].rolling(Npast).sum().copy()
        df_rewdyn['NM_diff_std'] = df_trials_ses['NM_diff_std'].rolling(Npast).sum().copy()
        df_rewdyn['N_past'] = Npast
        df_rewdyn_ses = pd.concat([df_rewdyn_ses, df_rewdyn])
    df_rewdyn_ses = df_rewdyn_ses.reset_index(drop=True)

    #% Plot distribution of DA across times in trials
    if plot:
        fig, axs = plt.subplots(3,4, figsize=(20,12))

        df2plot = df_trials_ses
        df2plot = df_trials_ses[['NM_precue', 'NM_endtrial', 'NM_choice','reward', 'trial']]
        df2plot = df2plot.set_index(['trial','reward']).stack().reset_index().rename(columns={0:'DA', 'level_2':'var'})
        # sns.displot(df2plot, x='DA', hue='var', col='reward', kind='hist', element='step')
        sns.histplot(df2plot[df2plot['reward']==True], x='DA', hue='var', element='step', ax=axs[0,0])
        sns.histplot(df2plot[df2plot['reward']==False], x='DA', hue='var', element='step', ax=axs[1,0], legend=False)
        # sns.despine()
        # plt.show()

        #% Plot distribution of DA differences between times in trials
        df2plot = df_trials_ses
        df2plot = df_trials_ses[['NM_diff_abs', 'NM_diff_std','reward', 'trial']]
        df2plot = df2plot.set_index(['trial','reward']).stack().reset_index().rename(columns={0:'DA', 'level_2':'var'})
        # sns.displot(df2plot, x='DA', hue='reward', col='var', kind='hist', element='step', facet_kws=dict(sharex=False), common_bins=False)
        sns.histplot(df2plot[df2plot['var']=='NM_diff_std'], x='DA', hue='reward', element='step', ax=axs[0,1])
        sns.histplot(df2plot[df2plot['var']=='NM_diff_abs'], x='DA', hue='reward', element='step', ax=axs[1,1], legend=False)
        # sns.despine()
        # plt.show()

        #% Consecutive rewards vs DA change histogram
        df2plot = df_rewdyn_ses[(df_rewdyn_ses['N_past'] == df_rewdyn_ses['reward_false']) | (df_rewdyn_ses['N_past'] == df_rewdyn_ses['reward_true'])]
        df2plot = df2plot[['reward_tot', 'trial', 'NM_diff_std', 'NM_diff_abs']]
        df2plot = df2plot.set_index(['trial','reward_tot']).stack().reset_index().rename(columns={0:'DA_change', 'level_2':'var'})
        # sns.displot(df2plot, x='DA_change', col='var', hue='reward_tot', kind='hist', palette='RdYlGn', element='step', facet_kws=dict(sharex=False), common_bins=False)
        sns.histplot(df2plot[df2plot['var']=='NM_diff_std'], x='DA_change', hue='reward_tot', palette='RdYlGn', element='step', ax=axs[0,2])
        sns.histplot(df2plot[df2plot['var']=='NM_diff_abs'], x='DA_change', hue='reward_tot', palette='RdYlGn', element='step', ax=axs[1,2], legend=False)
        # sns.despine()
        # plt.show()

        #% Consecutive rewards vs DA change barplot
        # sns.catplot(df2plot, y='DA_change', col='var', x='reward_tot', kind='bar', palette='RdYlGn', sharey=False)
        sns.barplot(df2plot[df2plot['var']=='NM_diff_abs'], y='DA_change', x='reward_tot',palette='RdYlGn', ax=axs[0,3])
        sns.barplot(df2plot[df2plot['var']=='NM_diff_std'], y='DA_change', x='reward_tot',palette='RdYlGn', ax=axs[1,3])
        sns.despine()
        # plt.show()

        #% Display full table
        df2plot = df_rewdyn_ses
        df2plot = df2plot[['reward_true', 'reward_false', 'NM_diff_abs']]
        df2plot = pd.pivot_table(data=df2plot, index=['reward_false'], columns=['reward_true'], values=['NM_diff_abs'])
        sns.heatmap(df2plot, cmap='RdYlGn',ax=axs[2,0])
        sns.despine()
        # plt.show()
    else:
        fig, axs = np.nan, np.nan
    return df_trials_ses, df_rewdyn_ses, fig, axs


# Summary plots across sessions of baseline shift analysis
def df_analysis_baseline_shift_summary(df_trials_all, df_rewdyn_all):
    fig, axs = plt.subplots(2,3, figsize=(10,6))
    #% Consecutive rewards vs DA change histogram
    df2plot = df_rewdyn_all[(df_rewdyn_all['N_past'] == df_rewdyn_all['reward_false']) | (df_rewdyn_all['N_past'] == df_rewdyn_all['reward_true'])]
    df2plot = df2plot[['reward_tot', 'trial', 'NM_diff_std', 'NM_diff_abs']]
    df2plot = df2plot.set_index(['trial','reward_tot']).stack().reset_index().rename(columns={0:'DA_change', 'level_2':'var'})
    # sns.displot(df2plot, x='DA_change', col='var', hue='reward_tot', kind='hist', palette='RdYlGn', element='step', facet_kws=dict(sharex=False), common_bins=False)
    sns.histplot(df2plot[df2plot['var']=='NM_diff_std'], x='DA_change', hue='reward_tot', palette='RdYlGn', element='step', ax=axs[0,0])
    sns.histplot(df2plot[df2plot['var']=='NM_diff_abs'], x='DA_change', hue='reward_tot', palette='RdYlGn', element='step', ax=axs[1,0], legend=False)
    # sns.despine()
    # plt.show()

    #% Consecutive rewards vs DA change barplot
    # sns.catplot(df2plot, y='DA_change', col='var', x='reward_tot', kind='bar', palette='RdYlGn', sharey=False)
    sns.barplot(df2plot[df2plot['var']=='NM_diff_abs'], y='DA_change', x='reward_tot',palette='RdYlGn', ax=axs[0,1])
    sns.barplot(df2plot[df2plot['var']=='NM_diff_std'], y='DA_change', x='reward_tot',palette='RdYlGn', ax=axs[1,1])
    sns.despine()
    # plt.show()

    #% Display full table
    df2plot = df_rewdyn_all
    df2plot = df2plot[['reward_true', 'reward_false', 'NM_diff_abs']]
    df2plot = pd.pivot_table(data=df2plot, index=['reward_false'], columns=['reward_true'], values=['NM_diff_abs'])
    sns.heatmap(df2plot, cmap='RdYlGn',ax=axs[0,2])
    sns.despine()
    # plt.show()

    return fig, axs
# %%

#--------------------------------------------------------------------------------------------------------------
# function to generate dataframe with info on all trials in one session
def make_sess_df():
    #-----------------------------------------------------------------------------------------
    #Import Datasets of Fiber photometry
    datafolder = '/data/'

    # Session level dataframe
    
    #df_sessions = pd.read_pickle('/data/fiber_photometry_29-06-2023_23-58-51_all_sessions/df_all.pkl')
    
    df_sessions['ses_idx'] = df_sessions['subject_id'].astype(str) + '_' + df_sessions['session_date'].astype(str)
    df_sessions['ses_idx'] = pd.Categorical(df_sessions.ses_idx)

    #Clean session level dataframe
    df_sessions = df_sessions[df_sessions['finished_trials']>0]
    # Removing these variables allows to remove spurious duplicates due to preprocessing
    delete, idxs = np.unique(df_sessions.ses_idx.values, return_index=True)
    df_sessions = df_sessions.iloc[idxs]
    df_sessions['ses_idx'] = pd.Categorical(df_sessions['ses_idx'])
    df_sessions['ses_idx'] = df_sessions['ses_idx'].values.remove_unused_categories()
    df_sessions['subject_id'] = pd.Categorical(df_sessions['subject_id'])
    df_sessions['subject_id'] = df_sessions['subject_id'].values.remove_unused_categories()

    # Replace PL with mPFC
    for i_region in range(4):
        df_sessions.loc[df_sessions["Region_"+str(i_region)] == "PL", "Region_"+str(i_region)] = 'mPFC'

    #  Select sessions
    # This is a preprocessing step that didn't work in some cases depending on how files were stored
    df_sessions = df_sessions[df_sessions['state']=='saving file']
    
    #% Selects N_top sessions for each region of interest based on criterion = (efficiency * 1000 + N_finished_trials)
    df_sessions = select_sessions(df_sessions, N_top = 1000, regions_of_interest=['NAc'])


    # Load Trials level dataframe
    # Trial level dataframe
    df_trials = pd.read_pickle('/scratch/fiber_photometry_29-06-2023_23-58-51_all_files/df_all.pkl')
    df_trials['ses_idx'] = df_trials['subject_id'].astype(str) +'_' + df_trials['session_date'].astype(str)
    df_trials['ses_idx'] = pd.Categorical(df_trials.ses_idx)

    # Intersect sessions and trial level dataframes
    idxs_sessions = df_sessions.ses_idx
    idxs_trials = df_trials.ses_idx
    idxs = set(idxs_sessions.values).intersection(set(idxs_trials))
    df_sessions = df_sessions.loc[df_sessions.ses_idx.isin(idxs), :]
    df_trials = df_trials.loc[df_trials.ses_idx.isin(idxs), :]
    df_sessions['ses_idx'] = df_sessions['ses_idx'].values.remove_unused_categories()
    df_trials['ses_idx'] = df_trials['ses_idx'].values.remove_unused_categories()

    # Add Behavioral information to trial level dataframe from Han's pipeline

    # NECESSARY BUT COMMENTED OUT
    #df_trials_han = build_trials_with_behavior(df_sessions, df_trials, behavior_folder='/data/s3_foraging_all_nwb/')


    # This function outputs a warning when the number of trials in the behavior and fiber photometry pipelines do not match exactly. 
    # Only trials where bit_code is consistent across behavior and fiber photometry are retained. 

    # Add Behavioral information to trial level dataframe from Model Fit Behavior pipeline
    # df_trials_fit, df_sessions_params = build_trials_with_modelfit(df_sessions, df_trials, behavior_folder='/data/Foraging_Models_Fit2/')
    df_trials_fit = df_trials
    #Subselect top 100 sessions 
    df_sessions = select_sessions(df_sessions, N_top = 450, regions_of_interest=['NAc'])

    # Change session level dataset to be based on NM and location of recordings
    # Process session dataset generating a summary dataset for each neuromodulator
    df_sessions_NMs = reshape_df_sessions(df_sessions)
    df_sessions_DA = df_sessions_NMs[df_sessions_NMs['NM']=='DA']
    df_sessions_5HT = df_sessions_NMs[df_sessions_NMs['NM']=='5HT']
    df_sessions_NE = df_sessions_NMs[df_sessions_NMs['NM']=='NE']
    df_sessions_ACh = df_sessions_NMs[df_sessions_NMs['NM']=='ACh']

    # Build trial level Dataset for neuromodulators
    # This shows how to compute some basic metrics such as the average activity of a neuromodulator in between the oneset and the choice time
    df_trials_NMs = build_trials_NMs(df_sessions_NMs, df_trials_fit, df_trials_han, NMs = ['DA', 'NE', '5HT', 'ACh'], models=[])


    return df_trials_NMs