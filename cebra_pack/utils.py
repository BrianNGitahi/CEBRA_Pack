"""
This is a library of helper functions for the demo note-books
"""

import sys

import os # my addtion

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
#import joblib as jl
import cebra.datasets
from cebra import CEBRA
import torch
import matplotlib.gridspec as gridspec
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import auc

from matplotlib.collections import LineCollection
import sklearn.linear_model

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
def build_train_compute(neural_data, b_label, max_iterations=2000, d=3):


    # build time and behaviour models
    cebra_time_model = CEBRA(model_architecture='offset10-model-mse',
                        batch_size=512,
                        learning_rate=3e-4,
                        temperature=1,
                        output_dimension=d,
                        max_iterations=max_iterations,
                        distance='euclidean',
                        conditional='time',
                        device='cuda_if_available',
                        verbose=True,
                        time_offsets=10) 

    cebra_behaviour_model = CEBRA(model_architecture='offset10-model-mse',
                        batch_size=512,
                        learning_rate=3e-4,
                        temperature=1,
                        output_dimension=d,
                        max_iterations=max_iterations,
                        distance='euclidean',
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

def view(time_embedding, behaviour_embedding, labels, label_classes, title ="Different Angles", size=0.8):
 
    # create a figure and make the plots
    fig = plt.figure(figsize=(14,8))
    gs = gridspec.GridSpec(1, 2, figure=fig)


    ax81 = fig.add_subplot(gs[0,0], projection='3d')
    ax82 = fig.add_subplot(gs[0,1], projection='3d')
    ax81.axis('off')
    ax82.axis('off')


    # colour maps
    colours = ['cool', 'plasma', 'spring']

    # plot the time embedding 
    cebra.plot_embedding(embedding=time_embedding[label_classes[0],:], embedding_labels=labels[label_classes[0]],ax=ax81, markersize=0.7, title='Time embedding', cmap=colours[0])
    cebra.plot_embedding(embedding=time_embedding[label_classes[1],:], embedding_labels=labels[label_classes[1]],ax=ax81, markersize=0.7, title='Time embedding', cmap=colours[1])


    # plot the behaviour embedding 
    cebra.plot_embedding(embedding=behaviour_embedding[label_classes[0],:], embedding_labels=labels[label_classes[0]],ax=ax82, markersize=0.7, title='Behaviour embedding', cmap=colours[0],)
    cebra.plot_embedding(embedding=behaviour_embedding[label_classes[1],:], embedding_labels=labels[label_classes[1]],ax=ax82,markersize=0.7, title='Behaviour embedding',  cmap=colours[1])

    gs.tight_layout(figure=fig)

    #print("preparing figure at multiple angles")

    # then view it at multiple angles
    #view_embedding(time_embedding, behaviour_embedding,s=size,label=labels,label_class=label_classes, titles=['time embedding','behaviour_embedding'], main_title=title)

#--------------------------------------------------------------------
# Make a function to format the NM data into a 1s window around the choice

def format_data(neural_data, df, trace_times_, choice_times_ , window=None , window_size=10, n_trials=1765):

    # define the number of trials where the mouse made a choice
    n_choice_trials = np.unique(np.isnan(choice_times_),return_counts=True)[1][0]

    # list to hold all the 1s windows
    n_data_window = []

    # new labels
    reward_labels = []
    choice_labels = []
    rpe_labels = []
    n_licks = []


    # loop over all trials
    for i in range(0,n_trials):

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

# define function to take the choice labels and make a 'Switch' label

def make_switch_label(choice_label):

    # make sure input is in array form
    assert type(choice_label)==np.ndarray

    switch_labels = []

    for i in range(0,choice_label.shape[0]):

        # should I just skip this first one?
        if i==0:
            switch_labels.append(0)
            continue

        # make switch label based on previous trial
        if choice_label[i]!=choice_label[i-1]:
            switch_labels.append(1)        
        
        elif choice_label[i]==choice_label[i-1]:
            switch_labels.append(0)

    switch_labels = np.array(switch_labels)
    print('Switch labels shape:', switch_labels.shape)

    return switch_labels

#--------------------------------------------------------------------

# for each NM combination
def nm_analysis(data, df_, t_times_, c_times_,labels='reward',window_=None,dimension=3,missing_nm=""):

    # format the data into 1s window around the choice and create the labels
    nms_HD, reward_labels, choice_labels, n_licks, rpe_labels = format_data(data, df_, t_times_,c_times_, window=window_)

    # choose the labels and define label classes (p=rewarded/left n= unrewarded/right)
    if labels=='reward':
        t_labels = reward_labels
        positive, negative = define_label_classes(t_labels)

    if labels=='choice':
        t_labels = choice_labels
        positive, negative = define_label_classes(t_labels)

    if labels == 'switch':
        t_labels = make_switch_label(choice_labels)
        positive, negative = define_label_classes(t_labels)
 
    # use reward labels for rpe
    if labels=='rpe':
        positive, negative = define_label_classes(reward_labels)
        t_labels = rpe_labels

    # Build and train the model then compute embeddings
    t_embed, b_embed = build_train_compute(nms_HD, t_labels,d=dimension)


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
def nm_analysis_2(data, df, trace_times, choice_times, title, label='reward', window=None):

    # collect embeddings, and the labels in lists
    behaviour_embeddings = []
    time_embedings =[]

    # run the nm analysis on the individual nms
    for i, dataset in enumerate(data):

        t_embed, b_embed, t_labels, [positive,negative] = nm_analysis(dataset, df, trace_times, choice_times,labels=label, window_=window)

        behaviour_embeddings.append(b_embed)
        time_embedings.append(t_embed)

        # collect the labels and label classes for use in the plotting
        # note that we assume they're the same for all datasets

        print("COMPLETED ANALYSIS OF NM {}".format(i))

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
# get r2 score (x==embedding, y=target/label)
def reconstruction_score(x, y):

    def _linear_fitting(x, y):
        lin_model = sklearn.linear_model.LinearRegression()
        lin_model.fit(x, y)
        return lin_model.score(x, y), lin_model.predict(x)

    return _linear_fitting(x, y)