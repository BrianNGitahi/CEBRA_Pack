"""
This is a library of helper functions for the demo note-books
"""
#%%
import sys

import os # my addtion

import numpy as np
import matplotlib.pyplot as plt
#import joblib as jl
import cebra.datasets
from cebra import CEBRA
import torch
import matplotlib.gridspec as gridspec
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import pandas as pd
import logging

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import auc

from matplotlib.collections import LineCollection
import sklearn.linear_model

import cebra_pack.utils as cpu


#--------------------------------------------------------------------
# function to view the ideal embedding from different angles
def view_embedding(embed1, embed2, label, label_class, titles=['time embedding','behaviour_embedding'], main_title="Different Angles", s=0.8, n_angles=2):

    fig1=plt.figure(figsize=(8,4*n_angles))
    gs = gridspec.GridSpec(n_angles, 2, figure=fig1)

    c = ['cool','plasma','pink','winter']

    for i, ii in enumerate(range(60,360,int(300/n_angles))):

        # create the axes
        ax1 = fig1.add_subplot(gs[1*i,0], projection='3d')
        ax1.view_init(elev=10., azim=ii) 

        ax2 = fig1.add_subplot(gs[1*i,1], projection='3d')
        ax2.view_init(elev=10., azim=ii)

        # loop over the number of labels
        for j,value in enumerate(label_class):
            
            # plot time embedding
            cebra.plot_embedding(embedding=embed1[value,:], embedding_labels=label[value], ax=ax1, markersize=s,title=titles[0],cmap=c[j])

            # plot behaviour embedding
            cebra.plot_embedding(embedding=embed2[value,:], embedding_labels=label[value], ax=ax2, markersize=s,title=titles[1],cmap=c[j])

            plt.tight_layout()

        plt.suptitle(main_title)

#-------------------------------------------------------------------

# function to build, train and compute an embedding
def build_train_compute(neural_data, b_label, max_iterations=2000, d=3, arch='offset10-model',metric='cosine'):


    # build time and behaviour models
    cebra_time_model = CEBRA(model_architecture=arch,
                        batch_size=512,
                        learning_rate=3e-4,
                        temperature=1,
                        output_dimension=d,
                        max_iterations=max_iterations,
                        distance=metric,
                        conditional='time',
                        device='cuda_if_available',
                        verbose=True,
                        time_offsets=10) 

    cebra_behaviour_model = CEBRA(model_architecture=arch,
                        batch_size=512,
                        learning_rate=3e-4,
                        temperature=1,
                        output_dimension=d,
                        max_iterations=max_iterations,
                        distance=metric,
                        conditional='time_delta',
                        device='cuda_if_available',
                        verbose=True,
                        time_offsets=10)

    # train them both
    cebra_time_model.fit(neural_data)
    cebra_behaviour_model.fit(neural_data, b_label)

    # compute the embeddings
    time_embedding = cebra_time_model.transform(neural_data)
    behaviour_embedding = cebra_behaviour_model.transform(neural_data)

    # return the embeddings 

    return time_embedding, behaviour_embedding

#--------------------------------------------------------------------

# divide the labels into positive and negative
def define_label_classes(trial_labels):

    positive = trial_labels==1
    negative = trial_labels==0

    positive = positive.flatten()
    negative = negative.flatten()

    return positive, negative

#--------------------------------------------------------------------

# define a function to view the embeddings
def view(time_embedding, behaviour_embedding, labels, label_classes, scores=None, titles=["Time embedding", "Behaviour embedding"],main_title="Reward Label Embeddings", size=5):
 
    # create a figure and make the plots
    fig = plt.figure(figsize=(17,8))
    gs = gridspec.GridSpec(1, 2, figure=fig)


    ax81 = fig.add_subplot(gs[0,0], projection='3d')
    ax82 = fig.add_subplot(gs[0,1], projection='3d')
    
    for ax in [ax81,ax82]:
        ax.set_xlabel("latent 1", labelpad=0.001, fontsize=13)
        ax.set_ylabel("latent 2", labelpad=0.001, fontsize=13)
        ax.set_zlabel("latent 3", labelpad=0.001, fontsize=13)

        # Hide X and Y axes label marks
        ax.xaxis.set_tick_params(labelbottom=False)
        ax.yaxis.set_tick_params(labelleft=False)
        ax.zaxis.set_tick_params(labelright=False)

        # Hide X and Y axes tick marks
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])


    # colour maps
    colours = ['cool', 'plasma', 'spring']

    embeddings = [time_embedding, behaviour_embedding]

    if scores == None:
        scores, errors = np.round(get_auc(embeddings, labels),3)

    # plot the time embedding 
    cebra.plot_embedding(embedding=time_embedding[label_classes[0],:], embedding_labels=labels[label_classes[0]],ax=ax81, markersize=size, title=titles[0], cmap=colours[0])
    cebra.plot_embedding(embedding=time_embedding[label_classes[1],:], embedding_labels=labels[label_classes[1]],ax=ax81, markersize=size, title= f'{titles[0]}, Score:{scores[0]}', cmap=colours[1])


    # plot the behaviour embedding 
    cebra.plot_embedding(embedding=behaviour_embedding[label_classes[0],:], embedding_labels=labels[label_classes[0]],ax=ax82, markersize=size, title=titles[1], cmap=colours[0],)
    cebra.plot_embedding(embedding=behaviour_embedding[label_classes[1],:], embedding_labels=labels[label_classes[1]],ax=ax82,markersize=size, title= f'{titles[1]}, Score: {scores[1]}',  cmap=colours[1])

    plt.suptitle(main_title)
    gs.tight_layout(figure=fig)

#--------------------------------------------------------------------
# Make a function to format the NM data into a 1s window around the choice

def format_data(neural_data, df, trace_times_, choice_times_ , window=None , window_size=10):

    # define the number of trials where the mouse made a choice
    n_choice_trials = np.unique(np.isnan(choice_times_),return_counts=True)[1][0]

    # define total number of trials
    n_total_trials = np.sum(np.unique(np.isnan(choice_times_),return_counts=True)[1])

    # list to hold all the 1s windows
    n_data_window = []

    # new labels
    reward_labels = []
    choice_labels = []
    rpe_labels = []
    n_licks = []


    # loop over all trials
    for i in range(0,n_total_trials):

        # skip trials where the animal didn't make a choice (null choice time)
        if np.isnan(choice_times_[i]):
            continue

        # find the index of the closest time to the choice time in the trace_times array 
        idx = np.abs(trace_times_ - choice_times_[i]).argmin()

        # take the previous 10 and/or the next 10 values of the NM data at these indices - 1s window
        if window =='before':
            n_data_window.append(neural_data[idx-10:idx])

        if window == 'after':
            n_data_window.append(neural_data[idx:idx+10])

        if window == None:
            n_data_window.append(neural_data[idx-10:idx+10])

        # label the timepoints as rewarded or unrewarded
        if df['reward'].iloc[i]:
            # new trial label
            reward_labels.append(1)

        elif df['reward'].iloc[i]==False:
            # new trial label
            reward_labels.append(0)
        
        # label the timepoints as left or right choice
        if df['licks L'].iloc[i] >= df['licks R'].iloc[i]:
            # new trial label
            choice_labels.append(1)
            n_licks.append(df['licks L'].iloc[i])

        elif df['licks R'].iloc[i] > df['licks L'].iloc[i]:
            # new trial label
            choice_labels.append(0)
            n_licks.append(df['licks R'].iloc[i])

        # get the rpe values at each trial
        rpe_labels.append(df['rpe'].iloc[i])

    # stack the nm data for each trial
    nms_HD = np.stack(n_data_window).reshape((n_choice_trials,-1))
    # format it into a tensor
    nms_HD = torch.from_numpy(nms_HD.astype(np.float64))
    print("neural tensor shape: ", nms_HD.shape)

    # convert trial labels into an array
    reward_labels = np.array(reward_labels)
    print("reward labels shape: ",reward_labels.shape)

    choice_labels = np.array(choice_labels)
    print("choice labels shape: ",choice_labels.shape)

    # convert rpe labels to arrays
    rpe_labels = np.array(rpe_labels)
    print("rpe labels shape:", rpe_labels.shape)


    return nms_HD, reward_labels, choice_labels, n_licks, rpe_labels

#--------------------------------------------------------------------

# for each NM combination
def nm_analysis(data, df_, t_times_, c_times_,arch_ ='offset10-model', metric_ ='cosine',labels='reward',other_label_=None, window_=None,dimension=3,missing_nm=""):

    # Define the list of valid behaviour labels
    valid_behaviour_labels = ['reward', 'choice', 'rpe']
    
    # Use assert to check if the provided behaviour_label is in the list of valid labels
    assert labels in valid_behaviour_labels, f"Invalid behaviour_label: {labels}. Must be one of {valid_behaviour_labels}."


    # format the data into 1s window around the choice and create the labels
    nms_HD, reward_labels, choice_labels, n_licks, rpe_labels = format_data(data, df_, t_times_,c_times_, window=window_)


    # choose the labels and define label classes (p=rewarded/left n= unrewarded/right)
    if labels=='reward':
        positive, negative = define_label_classes(reward_labels)
        t_labels = reward_labels

    elif labels=='choice':
        positive, negative = define_label_classes(choice_labels)
        t_labels = choice_labels 

    # use reward labels to define classes for rpe
    elif labels=='rpe':
        positive, negative = define_label_classes(reward_labels)
        t_labels = rpe_labels

    # if another label was provided, use that one instead    
    if other_label_ is not None:
        t_labels = other_label_


    # Build and train the model then compute embeddings
    t_embed, b_embed = build_train_compute(nms_HD, t_labels,d=dimension, arch=arch_, metric=metric_)

    # view the embeddings
    #view_embedding(t_embed, b_embed, t_labels,label_class=[rewarded, unrewarded],title=missing_nm)

    return t_embed, b_embed, t_labels, [positive,negative]

#--------------------------------------------------------------------

# first make function to make the plots given a list of embeddings
def plot4_embeddings(embeddings, labels , l_class, titles=['DA only', 'NE only', '5HT only', 'ACh only'], t=""):

    # number of plots
    n_plots = len(embeddings)

    n_columns = 2
    n_rows = n_plots//n_columns

    # create axis
    fig = plt.figure(figsize=(8,4*n_plots))
    gs = gridspec.GridSpec(n_rows, n_columns, figure=fig)

    # colour 
    c = ['cool','plasma','pink','winter']

    for i, embed in enumerate(embeddings):

        # create the axes
        ax = fig.add_subplot(gs[i // n_columns, i%n_columns], projection='3d')

        ax.set_xlabel("latent 1", labelpad=0.001, fontsize=13)
        ax.set_ylabel("latent 2", labelpad=0.001, fontsize=13)
        ax.set_zlabel("latent 3", labelpad=0.001, fontsize=13)

        # Hide X and Y axes label marks
        ax.xaxis.set_tick_params(labelbottom=False)
        ax.yaxis.set_tick_params(labelleft=False)
        ax.zaxis.set_tick_params(labelright=False)

        # Hide X and Y axes tick marks
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

        # plot the embedding
        cebra.plot_embedding(embedding=embed[l_class[0],:], embedding_labels=labels[l_class[0]], ax=ax, markersize=2,title=titles[i], cmap=c[0])
        cebra.plot_embedding(embedding=embed[l_class[1],:], embedding_labels=labels[l_class[1]], ax=ax, markersize=2,title=titles[i], cmap=c[1])

    plt.suptitle(t, fontsize=15)
    plt.tight_layout()

#--------------------------------------------------------------------

# run nm analysis on mutliple nm datasets 
def nm_analysis_2(data, df, trace_times, choice_times, title,arch='offset10-model', metric ='cosine', label='reward', other_label = None, window=None):

    # collect embeddings, and the labels in lists
    behaviour_embeddings = []
    time_embedings =[]

    # run the nm analysis on the individual nms
    for i, dataset in enumerate(data):

        t_embed, b_embed, t_labels, [positive,negative] = nm_analysis(dataset, df, trace_times, choice_times,arch_ = arch, metric_ = metric, labels=label, other_label_=other_label, window_=window)

        behaviour_embeddings.append(b_embed)
        time_embedings.append(t_embed)

        # collect the labels and label classes for use in the plotting
        # note that we assume they're the same for all datasets

        print("COMPLETED ANALYSIS OF NM {}: ".format(i))

    # plot them
    #plot4_embeddings(behaviour_embeddings,labels=t_labels,l_class=[rewarded,unrewarded],titles=title)

    return behaviour_embeddings, time_embedings, t_labels, [positive,negative]
#--------------------------------------------------------------------

# function to make datasets of combinations of 3 NMs
# format the arrays
def create_datasets(traces_):

    # create a list to hold the different combinations of NM data
    datasets = []

    # iterate through the keys in the dictionary holding the NM data
    for key in traces_:

        # at each iteration make an array of NM data and exclude the current NM from the array
        array = np.array([traces_[trace] for trace in traces_.keys() if trace !=key ])

        # format the array 
        f_array = np.transpose(array)
        f_array = f_array.astype(np.float64)
        print("shape of formatted array:", f_array.shape)
        datasets.append(f_array)


    return datasets

#--------------------------------------------------------------------

# get the data as individual datasets of each nm
def individual_datasets(traces_):

    # create a list to hold the different NMs data
    datasets = []

    # loop through the traces
    for trace in traces_.keys():

        # select the trace of the current NM
        array = np.array([traces_[trace]])

        # format the array 
        f_array = np.transpose(array)
        f_array = f_array.astype(np.float64)
        print("shape of formatted array:", f_array.shape)
        datasets.append(f_array)

    return datasets

#--------------------------------------------------------------------

# define function to get the auc scores
def get_auc(set_of_embeddings,trial_labels, n_iterations=1):   

     # list to store mean auc scores at each of these embedding dimensions
    mean_scores = []
    errors = []

    for j, embedding in enumerate(set_of_embeddings):

        # quantify with AUC score
        scores = []

        # for each NM make a couple of runs of the log regression model to get error bars
        for i in range(n_iterations):

            # make logistic function, fit it and use it to predict the initial labels from the embedding
            logreg = LogisticRegression(random_state=42)
            logreg.fit(embedding, trial_labels)
            prediction = logreg.predict(embedding)

            # quantify how well the embedding mirrors the labels using the auc score

            # make a precision recall curve and get the threshold
            precision, recall, threshold = precision_recall_curve(trial_labels, prediction)
            threshold = np.concatenate([np.array([0]), threshold])

            # calculate the fpr and tpr for all thresholds of the classification
            fpr, tpr, threshold = roc_curve(trial_labels, prediction)

            # get the auc score and append it to the list
            roc_auc = auc(fpr, tpr)
            scores.append(roc_auc)

        # store the mean and the standard deviation 
        mean_scores.append(np.mean(scores))
        errors.append(np.std(scores))

    return mean_scores, errors

#--------------------------------------------------------------------

# gets number of cpus available
def get_code_ocean_cpu_limit():
    """
    Gets the Code Ocean capsule CPU limit

    Returns
    -------
    int:
        number of cores available for compute
    """
    # Checks for environmental variables
    co_cpus = os.environ.get("CO_CPUS")
    aws_batch_job_id = os.environ.get("AWS_BATCH_JOB_ID")

    if co_cpus:
        return co_cpus
    if aws_batch_job_id:
        return 1
    with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us") as fp:
        cfs_quota_us = int(fp.read())
    with open("/sys/fs/cgroup/cpu/cpu.cfs_period_us") as fp:
        cfs_period_us = int(fp.read())
    container_cpus = cfs_quota_us // cfs_period_us
    # For physical machine, the `cfs_quota_us` could be '-1'
    return psutil.cpu_count(logical=False) if container_cpus < 1 else container_cpus

#-----------------------------------------------------------------------------------------------------------------

# Set up logging configuration
logging.basicConfig(
    filename='error_log_.log',  # Log file name
    level=logging.ERROR,       # Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format='%(asctime)s - %(levelname)s - %(message)s'  # Log message format
)


# analyse one session and generate a dataframe with the stats
def sess_analysis(df_trials_sess, label_='reward'):

    try:
        df_trials_sess['region'] = 'NAc'
        # df_trials_ses, events2plot, traces2plot, trace_times2plot = compute_sessionwide_traces(df_trials_ses)
        df_trials_ses, events, traces, trace_times = cpu.compute_sessionwide_traces_multi(df_trials_sess)

        # if the session doesn't have all 4 NMs then skip that session
        if len(traces) != 4:
            print("SESSION HAS ONLY {} NEUROMODULATOR(S)".format(len(traces)))
            return None
           

        # for each session
        # 1. LOAD THE DATA
        #----------------------------------------------------------------------------------------------

        # Combine the traces for all NMs into one 2D array
        all_nms = np.array([traces[trace] for trace in traces.keys()])
        all_nms = np.transpose(all_nms)


        # number of times the rows are repeated
        n_repeats = df_trials_ses['NM_no_overlap'][0].shape[0]

        # number of trials in the session
        n_trials = int(df_trials_ses.shape[0]/n_repeats)

        # get the choice times 
        choice_times = df_trials_ses['choice_time'][0:n_trials].to_numpy()

        print("Loaded session data")
        # 2. RECORD THE SESSION DETAILS: animal, session index and signal average for each of the NMs
        #-------------------------------------------------------------------------------------------------

        # make sure that the df has only data from one session then record it as the session index
        n_sessions = np.size(np.unique(df_trials_ses['ses_idx'].values, return_counts=True)[0])

        if  n_sessions==1:
            # get the subject ID and the session ID
            subject_ID = df_trials_ses['ses_idx'].iloc[0].split("_")[0]
            session_index = df_trials_ses['ses_idx'].iloc[0]

        else:
            print("THIS DATAFRAME HAS MORE THAN 1 SESSION: {}".format(df_trials_ses['ses_idx'].iloc[0]))
            return None
           


        # 3. GET AUC SCORES: individual and all together + before and after choice
        #----------------------------------------------------------------------------------------------------

        # Individual NMs AUC scores
        ind_nm_data =  individual_datasets(traces_=traces)
        b_embeds, t_embeds, labels, [rewarded, unrewarded] =  nm_analysis_2(ind_nm_data, df=df_trials_ses,trace_times=trace_times, choice_times=choice_times, label=label_, title='Individual NMs')
        
        if label_ == 'rpe':
            r2_scores, sds = get_r2(b_embeds, labels)
            print("completed ind NMs r2s")
        else:
            auc_scores, sds =   get_auc(b_embeds, labels)
            print("completed ind NMS aucs")


        # AUC Score for all of them
        ball_embeds, tall_embeds, labels_all, [rewardeda, unrewardeda] =  nm_analysis_2([all_nms], df=df_trials_ses,trace_times=trace_times, choice_times=choice_times, label=label_, title='ALL NMs')

        if label_ =='rpe':
            r2a_scores, sds_a = get_r2(ball_embeds, labels_all)
            print("completed all NMS r2s")
        else:
            auca_scores, sds_a =  get_auc(ball_embeds, labels_all)
            print("completed all NMS aucs")

    
        # Before and after choice AUC scores + (bonus) best two embedding pairs 
        b4b_embeds, b4t_embeds, labels_b4, [r, unr] =  nm_analysis_2([all_nms],window='before', df=df_trials_ses,trace_times=trace_times, choice_times=choice_times, label=label_, title='ALL NMs')
        afb_embeds, aft_embeds, labels_af, [r_af, unr_af] =  nm_analysis_2([all_nms], window='after', df=df_trials_ses,trace_times=trace_times, choice_times=choice_times, label=label_, title='ALL NMs')

        if label_=='rpe':
            r2_b4_scores, sds_b4 = get_r2(b4b_embeds, labels_b4)
            r2_af_scores, sds_af = get_r2(afb_embeds, labels_af)
            print("completed b4/af r2s")
        else:
            auc_b4_scores, sds_b4 =  get_auc(b4b_embeds, labels_b4)
            auc_af_scores, sds_af =  get_auc(afb_embeds, labels_af)
            print("completed b4/af aucs")

        # Signal Average in the 1 sec window (BONUS: add later)

        if label_ == 'rpe':
            # add new row with session details to the DF
            new_row = {
                "subject_ID":subject_ID, 
                "ses_idx": session_index, 
                f"all4_R2_{label_}":r2a_scores[0], 
                f"DA_R2_{label_}":r2_scores[0], 
                f"NE_R2_{label_}":r2_scores[1], 
                f"5HT_R2_{label_}":r2_scores[2], 
                f"ACh_R2_{label_}": r2_scores[3], 
                f"b4_R2_{label_}": r2_b4_scores[0], 
                f"af_R2_{label_}": r2_af_scores[0]
            }

        else:
            # add new row with session details to the DF
            new_row = {
                "subject_ID":subject_ID, 
                "ses_idx": session_index, 
                f"all4_AUC_{label_}": auca_scores[0], 
                f"DA_AUC_{label_}": auc_scores[0], 
                f"NE_AUC_{label_}":auc_scores[1], 
                f"5HT_AUC_{label_}":auc_scores[2], 
                f"ACh_AUC_{label_}": auc_scores[3], 
                f"b4_AUC_{label_}": auc_b4_scores[0], 
                f"af_AUC_{label_}": auc_af_scores[0]
                }


        print("COMPLETED ANALYSIS OF SESSION: {}, label used: {}".format(session_index, label_))
        return new_row

    except Exception as e:
        logging.error(f"Error occurred in session {df_trials_ses['ses_idx'].iloc[0]}", exc_info=True)
        return None


#------------------------------------------------------------------------------------------------------

# get r2 score (x==embedding, y=target/label)
def reconstruction_score(x, y):

    def _linear_fitting(x, y):
        lin_model = sklearn.linear_model.LinearRegression()
        lin_model.fit(x, y)
        return lin_model.score(x, y), lin_model.predict(x)

    return _linear_fitting(x, y)

#------------------------------------------------------------------------------------------------------
# get the r2 score for a set of embeddings
def get_r2(set_of_embeddings, labels, n_iterations=1):

    # list to store mean auc scores at each of these embedding dimensions
    mean_scores = []
    errors = []

    for embedding in set_of_embeddings:

        scores=[]

        for i in range(n_iterations):
            
            # get score
            r2_score, prediction = reconstruction_score(embedding, labels)
            scores.append(r2_score)

        # append score
        mean_scores.append(np.mean(scores))
        errors.append(np.std(scores))


    return np.array(mean_scores), np.array(errors)

#------------------------------------------------------------------------------------------------------

# Function to analyze a list of sessions
def analyze_sessions(sessions, behaviour_label='reward'):

    # Define the list of valid behaviour labels -- can add more later
    valid_behaviour_labels = ['reward', 'choice', 'rpe']
    
    # Check if the provided behaviour_label is in the list of valid labels
    if behaviour_label not in valid_behaviour_labels:
        raise ValueError(f"Invalid behaviour_label: {behaviour_label}. Must be one of {valid_behaviour_labels}.")

    # define dataframe
    if behaviour_label == 'rpe':
        session_stats_df = pd.DataFrame(columns=["subject_ID", "ses_idx", f"all4_R2_{behaviour_label}", f"DA_R2_{behaviour_label}", f"NE_R2_{behaviour_label}", f"5HT_R2_{behaviour_label}", f"ACh_R2_{behaviour_label}", f"b4_R2_{behaviour_label}", f"af_R2_{behaviour_label}"])
    else:
        session_stats_df = pd.DataFrame(columns=["subject_ID", "ses_idx", f"all4_AUC_{behaviour_label}", f"DA_AUC_{behaviour_label}", f"NE_AUC_{behaviour_label}", f"5HT_AUC_{behaviour_label}", f"ACh_AUC_{behaviour_label}", f"b4_AUC_{behaviour_label}", f"af_AUC_{behaviour_label}"])

    # analyse the sessions
    for i, session in enumerate(sessions):

        session_df = pd.read_pickle(session)

        # make sure there's only one session's data in the input df then get the session index
        assert(np.unique(session_df['ses_idx'].values).shape[0]==1)

        session_stats = sess_analysis(session_df, behaviour_label)

        if session_stats:
            session_stats_df.loc[len(session_stats_df)] = session_stats

    return session_stats_df

#------------------------------------------------------------------------------------------------------

# function to define results folder
def define_resultsDir(folder='session_stats', save_dir = '../results'):
    results_folder = os.path.join(save_dir,folder)
    os.makedirs(results_folder, exist_ok=True)
    return results_folder
#------------------------------------------------------------------------------------------------------
# %%
