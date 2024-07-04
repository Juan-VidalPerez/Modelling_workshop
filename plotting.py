import numpy as np
import matplotlib.pyplot as plt
import scipy
import pandas as pd
import seaborn as sns

def plot_q_values(data, block=0):
    
    """
    Plots the Q-values of a block of the ytwo-armed bandit task.

    Input:
    data: numpy array with the dataset, with the following structure
        - Dimension 0: block number
        - Dimension 1: trial number
        - Dimension 2: [choices,outcomes,accuracy,q_values]
        
    Optional inputs:
    block: index for the block that will be plotted (default=0)
    """
    
    if (data.shape[2]>5):
        raise TypeError('Simulation with more than 2 bandits. This function only works for simulations with 2 bandits.')
    
    q_values=data[block,:,3:]
    choose_high = data[block,:,0] == 1
    choose_low = data[block,:,0] == 0
    rewarded = data[block,:,1] == 1
    
    y_high = np.max(q_values) + 0.1
    y_low = np.min(q_values) - 0.1
    fig, ax = plt.subplots(1, figsize=(10, 3))
    
    # Plot Q-values
    ax.plot(q_values[:,0], label='Bandit 1', color='blue')
    ax.plot(q_values[:,1], label='Bandit 2', color='orange')
    ax.set_ylabel('Q-values')
    ax.set_xlabel('Trial')
    ax.legend(bbox_to_anchor=(1, 1))
    ax.grid(True)

    # Plot rewards for Bandit 1
    ax.scatter(
        np.argwhere(choose_high & rewarded),
        y_high * np.ones(np.sum(choose_high & rewarded)),
        color='green',
        marker=3)
    ax.scatter(
        np.argwhere(choose_high & rewarded),
        y_high * np.ones(np.sum(choose_high & rewarded)),
        color='green',
        marker='|')
    # Omission high
    ax.scatter(
        np.argwhere(choose_high & 1 - rewarded),
        y_high * np.ones(np.sum(choose_high & 1 - rewarded)),
        color='red',
        marker='|')

    # Rewarded low
    ax.scatter(
        np.argwhere(choose_low & rewarded),
        y_low * np.ones(np.sum(choose_low & rewarded)),
        color='green',
        marker='|')
    ax.scatter(
        np.argwhere(choose_low & rewarded),
        y_low * np.ones(np.sum(choose_low & rewarded)),
        color='green',
        marker=2)
    # Omission Low
    ax.scatter(
        np.argwhere(choose_low & 1 - rewarded),
        y_low * np.ones(np.sum(choose_low & 1 - rewarded)),
        color='red',
        marker='|')

    plt.show()

    
def plot_accuracy(data, ax=None, color='k', label=None):
    """
    Plots the accuracy in the two-armed bandit task.

    Input:
    data: numpy array with the dataset, with the following structure
        - Dimension 0: block number
        - Dimension 1: trial number
        - Dimension 2: [choices,outcomes,accuracy,q_values]
    """
    
    if ax is None:
        fig,ax = plt.subplots(figsize=(3, 4))
        
    if isinstance(data,list):
        means=np.full([len(data),data[0].shape[1]],np.nan)

        for ss in range(len(data)):
            means[ss,:]=np.mean(data[ss][:,:,2], axis=0)
            
        means= np.mean(means, axis=0)
        sems = np.std(means, axis=0, ddof=1) / np.sqrt(means.shape[0])
        x = np.arange(1, data[0].shape[1] + 1)
            
    else:
        means= np.mean(data[:,:,2], axis=0)
        sems = np.std(data[:,:,2], axis=0, ddof=1) / np.sqrt(data.shape[0])
        x = np.arange(1, data.shape[1] + 1)

    

    # Plot the mean
    ax.plot(x, means, color=color, label=label)

    # Plot the shaded area for the standard error
    ax.fill_between(x, means - sems, means + sems, color=color, alpha=0.1)

    # Add labels and title
    plt.xlabel('Trial')
    plt.ylabel('Accuracy')
    plt.ylim(0.4,1)


def plot_prepeat(data, ax=None, color='k', label=None):
    """
    Plots the probability of repeat after receiving a reward/non-reward in our two-armed bandit task.

    Input:
    data: numpy array with the dataset, with the following structure
        - Dimension 0: block number
        - Dimension 1: trial number
        - Dimension 2: [choices,outcomes,accuracy,q_values]
        
    Optional inputs:
    ax: axis of the figure in which to plot the results.
    color: color of the plot.
    label: label for the plot
    """
    if isinstance(data,list):
        n_subjects=len(data)
        
        p_rep=np.full([n_subjects,2], np.nan)
        for ss in range(n_subjects):

            choice=data[ss][:,:,0]
            outcome=data[ss][:,:,1]
            repeat= choice[:,0:-1]==choice[:,1:]

            p_rep[ss,0]= np.mean(repeat[outcome[:,0:-1].astype(bool)])
            p_rep[ss,1]= np.mean(repeat[(1-outcome[:,0:-1]).astype(bool)])
            means = np.mean(p_rep, axis=0)
            standard_errors = np.std(p_rep, axis=0, ddof=1) / np.sqrt(p_rep.shape[0])
    else:
        n_subjects=1
    
        p_rep=np.full(2, np.nan)
        for ss in range(n_subjects):

            choice=data[:,:,0]
            outcome=data[:,:,1]
            repeat= choice[:,0:-1]==choice[:,1:]

            p_rep[0]= np.mean(repeat[outcome[:,0:-1].astype(bool)])
            p_rep[1]= np.mean(repeat[(1-outcome[:,0:-1]).astype(bool)])
            means = p_rep
            standard_errors = np.full(2, np.nan)

    

    # Create an array for x positions
    x = np.arange(1, len(means) + 1)

    # Plotting
    if ax is None:
        fig,ax = plt.subplots(figsize=(3, 4))
    
    if not np.isnan(standard_errors).any():
        ax.errorbar(x, means, yerr=standard_errors, fmt='none', color=color, capsize=10)
        
    # Plot the means with a line connecting them
    ax.plot(x, means, marker='o', linestyle='-', color=color, label=label, ms=10)

    
    
    plt.xlim(0.5,2.5)
    plt.xticks([1,2], ['Rewarded', 'Non-rewarded'])
    plt.xlabel('Previous outcome')
    plt.ylabel('P(repeat)')
    plt.ylim(0,1)
    
def plot_parhm (mll,alpha,beta):

    fig, ax = plt.subplots(1, figsize=(4, 4))

    max_val = np.max(mll)
    max_coords = np.unravel_index(np.argmax(mll), mll.shape)
    print(f'Maximum loglikelihood= {max_val:.3f}')
    print('ML parameters')
    print(f'  -Learning rate= {alpha[max_coords[0]]:.2f}')
    print(f'  -Inverse temperature parameter= {beta[max_coords[1]]:.2f}')
    # Plot the heatmap
    plt.imshow(mll, cmap='viridis', interpolation='nearest', extent=[np.min(alpha),np.max(alpha),np.min(beta),np.max(beta)], aspect='auto')
    plt.colorbar(label='LogLikelihood')

    # Mark the minimum value with a cross
    plt.scatter(alpha[max_coords[0]], beta[max_coords[1]], color='red', marker='x', s=100, label='Maximum LogL')



    # Add labels and title
    plt.xlabel(r'Learning rate ($\alpha$)')
    plt.ylabel(r'Inverse temperature ($\beta$)')
    plt.legend()

def plot_recovery(parameters: float, recovery: float, labels: str=None):
    print('Spearman''s correlation between simulation and fit parameters:')
    for pp in range(parameters.shape[0]):
        fig, ax = plt.subplots(1, figsize=(4, 4))
        
        plt.plot(np.arange(np.min(parameters[pp,:]),np.max(parameters[pp,:]),0.01),
                          np.arange(np.min(parameters[pp,:]),np.max(parameters[pp,:]),0.01),
                                  linestyle='--', c='k')
        plt.scatter(x=parameters[pp,:],y=recovery[pp,:], marker='.')
        plt.xlabel('Simulation parameter')
        plt.ylabel('Fitted parameter')
        
        r=scipy.stats.spearmanr(parameters[pp,:],recovery[pp,:])
        if labels==None:
            plt.title(f'Parameter {pp}')
            print(f'   -Parameter {pp}:')
        else:
            plt.title(labels[pp])
            print(f'   -{labels[pp]}')
            
        print(f'   r= {r.statistic: .2f}')
        
        
def my_barplot(data, x_tick_labels: list[str]=None, ylabel: str=None):
    fig, ax = plt.subplots(1, figsize=(10, 4))
        
    means = np.mean(data, axis=0)
    
    # Calculate the standard error of each column
    # Standard error is the standard deviation divided by the square root of the number of samples
    std_errors = np.std(data, axis=0) / np.sqrt(data.shape[0])
    
    print(means.max)
    # Create a bar plot with error bars
    x = np.arange(len(means))  # the label locations
    plt.bar(x, means, yerr=std_errors, capsize=5, alpha=0.7, color='thistle', ecolor='black', edgecolor='black')
    
    
    # Add labels and title
    plt.ylim(np.min(means) - np.max(std_errors) - 0.005 * np.min(np.abs(means)), np.max(means) + np.max(std_errors) + 0.005 * np.max(np.abs(means)))
    plt.xlabel('Model')
    if ylabel is not None:
        plt.ylabel(ylabel)
        
    if x_tick_labels is not None:
        plt.xticks(x, x_tick_labels)
        
    plt.xticks(x)  # Set the x-ticks to be at the center of the bars
    
    # Show the plot
    plt.show()
    
    
def plot_parameters(data, labels, colors='gray', ax=None):
    if ax is None:
        fig,ax = plt.subplots(figsize=(3, 4))
    
    df = pd.DataFrame(data, columns=labels)
    df_melted = df.melt(var_name='Parameter', value_name='Fitted value')
    
    if isinstance(colors,list):
        sns.boxplot(x='Parameter', y='Fitted value', data=df_melted, width=0.3, palette=colors, ax=ax)
    else:
        sns.boxplot(x='Parameter', y='Fitted value', data=df_melted, width=0.3, color=colors, ax=ax)