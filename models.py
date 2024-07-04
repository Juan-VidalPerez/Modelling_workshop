import numpy as np
import scipy

def softmax(x):
    # Subtract the max value from each element for numerical stability
    x_max = np.max(x)
    e_x = np.exp(x - x_max)
    return e_x / e_x.sum(axis=0)

def logl_RW (parameters: list[float],
                       data):
    """
    Calculates the loglikelihood of the dataset based on the Rescorla-Wagner model.

    Inputs:
    parameters: list with the following structure:
        parameters[0]: learning rate
        parameters[1]:inverse temperature
    data: dataset structured as the output of bandit_simulations
      

    Returns:
    mll: negative loglikelihood of choices given parameters
    """
    # Unpack parameters
    alpha=parameters[0]
    beta=parameters[1]
    
    #Read number of bandits
    n_bandits = (np.max(data[:,:,0])+1).astype(int)
    
    #Extract data
    choice = data[:,:,0].astype(int)
    outcome = data[:,:,1]
    
    #Initialize numpy array to store likelihoods
    likelihood = np.full(data.shape[0:2], np.nan)
    
    for bb in range(data.shape[0]):
        Q_values=0.5*np.ones(n_bandits) # Initialize Q-values at the beginning of each block
        for tt in range(data.shape[1]):
            # Calculate probability of chosen bandit (based on Q-values)
            p_choice = softmax(beta*Q_values)
            
            # Update Q-values based on outcome
            Q_values[choice[bb,tt]] += alpha*(outcome[bb,tt]-Q_values[choice[bb,tt]])
            
            # Store liklihood of choice    
            likelihood[bb,tt] = p_choice[choice[bb,tt]]
            
    return -np.nansum(np.log(likelihood))

def fit_RWmodel(dataset, n_attempts=10, ub_beta=10):
    """
    Finds the maximum-likelihood parameters based on the Rescorla-Wagner model.

    Inputs:
    dataset: dataset structured as the output of bandit_simulations
    
    Optional inputs:
    n_attempts: number of attempts to find the minimum (to find the global minimum) (default=10)
    ub_beta: upper bound of the beta parameter
      

    Returns:
    parameters: maxmum likelihood parameters.
    mll: loglikelihood of choices given the maximum likelihood parameters
    """
    tmp_param=[]
    tmp_mll=[]
    bounds=scipy.optimize.Bounds(ub=np.array([1,ub_beta]),
                                          lb=np.zeros(2))
    init_par=np.array([np.random.rand(n_attempts), ub_beta*np.random.rand(n_attempts)])
    for aa in range(n_attempts):
        
        mle=scipy.optimize.minimize(logl_RW, init_par[:,aa], args=dataset,
                     method='L-BFGS-B', bounds=bounds)
        tmp_param.append(mle.x)
        tmp_mll.append(mle.fun)

    parameters=tmp_param[np.argmin(tmp_mll)]
    mll=np.min(tmp_mll)

    return parameters,-mll


def RW_simulation (alpha: float, beta: float,
                       p_reward: list[float]=[0.25, 0.75], n_blocks: int=20, n_trials: int=30):
    """
    Simulates an n-armed bandit task.

    Input:
    alpha: learning rate for the simulation.
    beta: inverse temperature parameter for softmax.
    
    Optional inputs:
    p_reward: list specifying reward probabilities of the bandits (default: [0.25, 0.75])
    n_blocks: number of blocks in the task (default: 10)
    n_trials: number of trials per block in the task (default: 30)
      

    Returns:
    output: numpy array with the results, with the following structure
        - Dimension 0: block number
        - Dimension 1: trial number
        - Dimension 2: [choices,outcomes,accuracy,q_values]
    """
    n_bandits = len(p_reward) #number of bandits

    output = np.full([n_blocks,n_trials, 3+n_bandits], np.nan) #initialize numpy array to store outputs
    
    for bb in range(n_blocks):
        Q_values=0.5*np.ones(n_bandits) # Initialize Q-values at the beginning of each block
        for tt in range(n_trials):
            # Calculate the probability of choosing each bandit (based on softmax)
            p_choice = softmax(beta*Q_values)
            # Choose one of the bandits given the calculated probabilities
            choice = np.random.choice(n_bandits, p=p_choice)
            
            # Generate an outcome from the bandit
            outcome = np.random.choice(2, p=[1-p_reward[choice], p_reward[choice]])
            
            # Update the Q-value of the chosen bandit based on the outcome
            Q_values[choice]+= alpha*(outcome-Q_values[choice])
            
            # Check accuracy (i.e., whether the chosen bandit is the most rewarding one)
            accuracy = choice==p_reward.index(max(p_reward))
            
            # Store information in numpy array
            output[bb,tt,0]=choice
            output[bb,tt,1]=outcome
            output[bb,tt,2]=accuracy
            output[bb,tt,3:]=Q_values
            
    return output
 


def softmax(x):
    # Subtract the max value from each element for numerical stability
    x_max = np.max(x)
    e_x = np.exp(x - x_max)
    return e_x / e_x.sum(axis=0)
def fit_CKmodel(dataset, n_attempts=10, ub_beta=10):
    
    
    tmp_param=[]
    tmp_mll=[]
    bounds=scipy.optimize.Bounds(ub=np.array([1,ub_beta]),
                                          lb=np.zeros(2))
    init_par=np.array([np.random.rand(n_attempts), ub_beta*np.random.rand(n_attempts)])
    for aa in range(n_attempts):
        
        mle=scipy.optimize.minimize(logl_CK, init_par[:,aa], args=dataset,
                     method='L-BFGS-B', bounds=bounds)
        tmp_param.append(mle.x)
        tmp_mll.append(mle.fun)

    parameters=tmp_param[np.argmin(tmp_mll)]
    mll=np.min(tmp_mll)

    return parameters,-mll

def logl_CK (parameters: list[float],
                       data):
    """
    Calculates the loglikelihood of the dataset based on the choice kernel model.

    Inputs:
    parameters: list with the following structure:
        parameters[0]: learning rate
        parameters[1]:inverse temperature
    data: dataset structured as the output of bandit_simulations
      

    Returns:
    mll: negative loglikelihood of choices given parameters
    """
    # Unpack parameters
    alpha=parameters[0]
    beta=parameters[1]
    
    #Read number of bandits
    n_bandits = (np.max(data[:,:,0])+1).astype(int)    
    #Extract data
    choice = data[:,:,0].astype(int)
    
    #Initialize numpy array to store likelihoods
    mll = np.full(data.shape[0:2], np.nan)
    
    for bb in range(data.shape[0]):
        CK_values=0.5*np.ones(n_bandits) # Initialize choice kernel at the beginning of each block
        for tt in range(data.shape[1]):
            # Calculate probability of chosen bandit (based on CK values)
            p_choice = softmax(beta*CK_values)
            
            # Store log-likelihood of choice 
            mll[bb,tt] = np.log(p_choice[choice[bb,tt]])
            
            # Create a one-hot array with a 1 in the position of the chosen bandit
            a=np.zeros(np.max(choice)+1)
            a[choice[bb,tt]]=1
            
            # Update CK values based on choice
            CK_values += alpha*(a-CK_values)
            

            
    return -np.nansum(mll)

def CK_simulation (alpha: float, beta: float,
                       p_reward: list[float]=[0.25, 0.75], n_blocks: int=20, n_trials: int=30):
    """
    Simulates an n-armed bandit task.

    Input:
    alpha: learning rate for the simulation.
    beta: inverse temperature parameter for softmax.
    
    Optional inputs:
    p_reward: list specifying reward probabilities of the bandits (default: [0.25, 0.75])
    n_blocks: number of blocks in the task (default: 10)
    n_trials: number of trials per block in the task (default: 30)
      

    Returns:
    output: numpy array with the results, with the following structure
        - Dimension 0: block number
        - Dimension 1: trial number
        - Dimension 2: [choices,outcomes,accuracy,q_values]
    """
    n_bandits = len(p_reward) #number of bandits

    output = np.full([n_blocks,n_trials, 3+n_bandits], np.nan) #initialize numpy array to store outputs
    
    for bb in range(n_blocks):
        CK=0.5*np.ones(n_bandits) # Initialize Q-values at the beginning of each block
        for tt in range(n_trials):
            # Calculate the probability of choosing each bandit (based on softmax)
            p_choice = softmax(beta*CK)
            # Choose one of the bandits given the calculated probabilities
            choice = np.random.choice(n_bandits, p=p_choice)
            
            # Generate an outcome from the bandit
            outcome = np.random.choice(2, p=[1-p_reward[choice], p_reward[choice]])
            
            # Create a one-hot array with a 1 in the position of the chosen bandit
            a=np.zeros(np.max(n_bandits))
            a[choice]=1
            
            # Update CK values based on choice
            CK += alpha*(a-CK)
            
            # Check accuracy (i.e., whether the chosen bandit is the most rewarding one)
            accuracy = choice==p_reward.index(max(p_reward))
            
            # Store information in numpy array
            output[bb,tt,0]=choice
            output[bb,tt,1]=outcome
            output[bb,tt,2]=accuracy
            output[bb,tt,3:]=CK
            
    return output


def fit_RWCKmodel(dataset, n_attempts=10, ub_beta=10):
    
    
    tmp_param=[]
    tmp_mll=[]
    bounds=scipy.optimize.Bounds(ub=np.array([1,ub_beta,1,ub_beta]),
                                          lb=np.zeros(4))
    init_par=np.array([np.random.rand(n_attempts), ub_beta*np.random.rand(n_attempts), np.random.rand(n_attempts), ub_beta*np.random.rand(n_attempts)])
    for aa in range(n_attempts):
        
        mle=scipy.optimize.minimize(logl_RWCK, init_par[:,aa], args=dataset,
                     method='L-BFGS-B', bounds=bounds)
        tmp_param.append(mle.x)
        tmp_mll.append(mle.fun)

    parameters=tmp_param[np.argmin(tmp_mll)]
    mll=np.min(tmp_mll)

    return parameters,-mll

def logl_RWCK (parameters: list[float],
                       data):
    """
    Calculates the loglikelihood of the dataset based on the choice kernel model.

    Inputs:
    parameters: list with the following structure:
        parameters[0]: learning rate
        parameters[1]:inverse temperature
    data: dataset structured as the output of bandit_simulations
      

    Returns:
    mll: negative loglikelihood of choices given parameters
    """
    # Unpack parameters
    alpha=parameters[0]
    beta=parameters[1]
    alphac=parameters[0]
    betac=parameters[1]
    
    #Read number of bandits
    n_bandits = (np.max(data[:,:,0])+1).astype(int)
    
    #Extract data
    choice = data[:,:,0].astype(int)
    outcome = data[:,:,1]
    
    #Initialize numpy array to store likelihoods
    mll = np.full(data.shape[0:2], np.nan)
    
    for bb in range(data.shape[0]):
        CK=0.5*np.ones(n_bandits) # Initialize choice kernel at the beginning of each block
        Q_values=0.5*np.ones(n_bandits) # Initialize Q-values at the beginning of each block

        for tt in range(data.shape[1]):
            # Calculate probability of chosen bandit (based on CK values)
            p_choice = softmax(beta*Q_values + betac*CK)
            
            # Store log-likelihood of choice 
            mll[bb,tt] = np.log(p_choice[choice[bb,tt]])
            
            # Create a one-hot array with a 1 in the position of the chosen bandit
            a=np.zeros(np.max(choice)+1)
            a[choice[bb,tt]]=1
            
            # Update CK values based on choice
            CK += alphac*(a-CK)
            
                    
            # Update Q-values based on outcome
            Q_values[choice[bb,tt]] += alpha*(outcome[bb,tt]-Q_values[choice[bb,tt]])
            
    return -np.nansum(mll)

def RWCK_simulation (alpha: float, beta: float,alphac: float, betac: float,
                       p_reward: list[float]=[0.25, 0.75], n_blocks: int=20, n_trials: int=30):
    """
    Simulates an n-armed bandit task.

    Input:
    alpha: learning rate for the simulation.
    beta: inverse temperature parameter for softmax.
    
    Optional inputs:
    p_reward: list specifying reward probabilities of the bandits (default: [0.25, 0.75])
    n_blocks: number of blocks in the task (default: 10)
    n_trials: number of trials per block in the task (default: 30)
      

    Returns:
    output: numpy array with the results, with the following structure
        - Dimension 0: block number
        - Dimension 1: trial number
        - Dimension 2: [choices,outcomes,accuracy,q_values]
    """
    n_bandits = len(p_reward) #number of bandits

    output = np.full([n_blocks,n_trials, 3+n_bandits], np.nan) #initialize numpy array to store outputs
    
    for bb in range(n_blocks):
        CK=0.5*np.ones(n_bandits)
        Q_values=0.5*np.ones(n_bandits) # Initialize Q-values at the beginning of each block
        for tt in range(n_trials):
            # Calculate the probability of choosing each bandit (based on softmax)
            p_choice = softmax(beta*Q_values + betac*CK)
            # Choose one of the bandits given the calculated probabilities
            choice = np.random.choice(n_bandits, p=p_choice)
            
            # Generate an outcome from the bandit
            outcome = np.random.choice(2, p=[1-p_reward[choice], p_reward[choice]])
            
            # Update the Q-value of the chosen bandit based on the outcome
            Q_values[choice]+= alpha*(outcome-Q_values[choice])
            
            # Create a one-hot array with a 1 in the position of the chosen bandit
            a=np.zeros(np.max(n_bandits))
            a[choice]=1
            
            # Update CK values based on choice
            CK += alphac*(a-CK)
            
            # Check accuracy (i.e., whether the chosen bandit is the most rewarding one)
            accuracy = choice==p_reward.index(max(p_reward))
            
            # Store information in numpy array
            output[bb,tt,0]=choice
            output[bb,tt,1]=outcome
            output[bb,tt,2]=accuracy
            output[bb,tt,3:]=Q_values
            
    return output


def fit_RWvalmodel(dataset, n_attempts=10, ub_beta=10):
    tmp_param=[]
    tmp_mll=[]
    bounds=scipy.optimize.Bounds(ub=np.array([1,1,ub_beta]),
                                          lb=np.zeros(3))
    init_par=np.array([np.random.rand(n_attempts), np.random.rand(n_attempts), ub_beta*np.random.rand(n_attempts)])
    for aa in range(n_attempts):
        
        mle=scipy.optimize.minimize(logl_RWval, init_par[:,aa], args=dataset,
                     method='L-BFGS-B', bounds=bounds)
        tmp_param.append(mle.x)
        tmp_mll.append(mle.fun)

    parameters=tmp_param[np.argmin(tmp_mll)]
    mll=np.min(tmp_mll)

    return parameters,-mll

def logl_RWval (parameters: list[float],
                       data):
    """
    Calculates the loglikelihood based on the Rescorla-Wagner with valence bias model.

    Input:
    parameters: list with the following structure:
        parameters[0]: learning rate for PE<0
        parameters[1]: learning rate for PE>0
        parameters[2]:inverse temperature
    data: dataset structured as the output of bandit_simulations
      

    Output:
    mll: negative loglikelihood of choices given parameters
    """
    
    alpha_neg=parameters[0]
    alpha_pos=parameters[1]
    beta=parameters[2]
    n_bandits = (np.max(data[:,:,0])+1).astype(int) #number of bandits
    choice = data[:,:,0].astype(int)
    outcome = data[:,:,1]
    mll = np.full(data.shape[0:2], np.nan)
    
    for bb in range(data.shape[0]):
        Q_values=0.5*np.ones(n_bandits) # Initialize Q-values at the beginning of each block
        for tt in range(data.shape[1]):

            p_choice = softmax(beta*Q_values)
    
            mll[bb,tt] = np.log(p_choice[choice[bb,tt]])
            
            PE=outcome[bb,tt]-Q_values[choice[bb,tt]]
            Q_values[choice[bb,tt]] += alpha_neg*PE*(PE<0) + alpha_pos*PE*(PE>0)
            

            
    return -np.nansum(mll)

def RWval_simulation (alpha_neg: float, alpha_pos: float, beta: float,
                       p_reward: list[float]=[0.25, 0.75], n_blocks: int=20, n_trials: int=30):
    """
    Simulates an n-armed bandit task.

    Input:
    alpha: learning rate for the simulation.
    beta: inverse temperature parameter for softmax.
    
    Optional inputs:
    p_reward: list specifying reward probabilities of the bandits (default: [0.25, 0.75])
    n_blocks: number of blocks in the task (default: 10)
    n_trials: number of trials per block in the task (default: 30)
      

    Returns:
    output: numpy array with the results, with the following structure
        - Dimension 0: block number
        - Dimension 1: trial number
        - Dimension 2: [choices,outcomes,accuracy,q_values]
    """
    n_bandits = len(p_reward) #number of bandits

    output = np.full([n_blocks,n_trials, 3+n_bandits], np.nan) #initialize numpy array to store outputs
    
    for bb in range(n_blocks):
        Q_values=0.5*np.ones(n_bandits) # Initialize Q-values at the beginning of each block
        for tt in range(n_trials):
            # Calculate the probability of choosing each bandit (based on softmax)
            p_choice = softmax(beta*Q_values)
            # Choose one of the bandits given the calculated probabilities
            choice = np.random.choice(n_bandits, p=p_choice)
            
            # Generate an outcome from the bandit
            outcome = np.random.choice(2, p=[1-p_reward[choice], p_reward[choice]])
            
            # Update the Q-value of the chosen bandit based on the outcome
            PE=outcome-Q_values[choice]
            Q_values[choice] += alpha_neg*PE*(PE<0) + alpha_pos*PE*(PE>0)
            
            # Check accuracy (i.e., whether the chosen bandit is the most rewarding one)
            accuracy = choice==p_reward.index(max(p_reward))
            
            # Store information in numpy array
            output[bb,tt,0]=choice
            output[bb,tt,1]=outcome
            output[bb,tt,2]=accuracy
            output[bb,tt,3:]=Q_values
            
    return output

def fit_REmodel(dataset, n_attempts=10):
    tmp_param=[]
    tmp_mll=[]
    bounds=scipy.optimize.Bounds(ub=1,lb=0)
    init_par=np.random.rand(n_attempts)
    for aa in range(n_attempts):
        
        mle=scipy.optimize.minimize(logl_RE, init_par[aa], args=dataset,
                     method='L-BFGS-B', bounds=bounds)
        tmp_param.append(mle.x)
        tmp_mll.append(mle.fun)

    parameters=tmp_param[np.argmin(tmp_mll)]
    mll=np.min(tmp_mll)

    return parameters,-mll

def logl_RE (parameters: list[float],
                       data):
    """
    Simulates an n-armed bandit task.

    Parameters:
    parameters: list with the following structure:
        parameters[0]: probability of choosing the bandit[0]
    data: dataset structured as the output of bandit_simulations
      

    Returns:
    mll: negative loglikelihood of choices given parameters
    """
    
    p=[parameters[0], 1-parameters[0]]
    n_bandits = (np.max(data[:,:,0])+1).astype(int) #number of bandits
    choice = data[:,:,0].astype(int)
    mll = np.full(data.shape[0:2], np.nan)
    
    if (parameters==0) | (parameters==1):
        mll=-np.inf
    else:
    
        for bb in range(data.shape[0]):
            for tt in range(data.shape[1]):
                mll[bb,tt] = np.log(p[choice[bb,tt]])
            

            
    return -np.nansum(mll)

def RE_simulation (b: float,
                       p_reward: list[float]=[0.25, 0.75], n_blocks: int=20, n_trials: int=30):
    """
    Simulates an n-armed bandit task.

    Input:
    alpha: learning rate for the simulation.
    beta: inverse temperature parameter for softmax.
    
    Optional inputs:
    p_reward: list specifying reward probabilities of the bandits (default: [0.25, 0.75])
    n_blocks: number of blocks in the task (default: 10)
    n_trials: number of trials per block in the task (default: 30)
      

    Returns:
    output: numpy array with the results, with the following structure
        - Dimension 0: block number
        - Dimension 1: trial number
        - Dimension 2: [choices,outcomes,accuracy,q_values]
    """
    n_bandits = len(p_reward) #number of bandits

    output = np.full([n_blocks,n_trials, 3+n_bandits], np.nan) #initialize numpy array to store outputs
    
    for bb in range(n_blocks):
        for tt in range(n_trials):
            # Calculate the probability of choosing each bandit (based on softmax)
            p_choice = np.array([b, 1-b])
            # Choose one of the bandits given the calculated probabilities
            choice = np.random.choice(n_bandits, p=p_choice)
            
            # Generate an outcome from the bandit
            outcome = np.random.choice(2, p=[1-p_reward[choice], p_reward[choice]])
            
            
            # Check accuracy (i.e., whether the chosen bandit is the most rewarding one)
            accuracy = choice==p_reward.index(max(p_reward))
            
            # Store information in numpy array
            output[bb,tt,0]=choice
            output[bb,tt,1]=outcome
            output[bb,tt,2]=accuracy
            
    return output
 
def fit_WSLSmodel(dataset, n_attempts=10):
    tmp_param=[]
    tmp_mll=[]
    bounds=scipy.optimize.Bounds(ub=1,lb=0)
    init_par=np.random.rand(n_attempts)
    for aa in range(n_attempts):
        
        mle=scipy.optimize.minimize(logl_WSLS, init_par[aa], args=dataset,
                     method='L-BFGS-B', bounds=bounds)
        tmp_param.append(mle.x)
        tmp_mll.append(mle.fun)

    parameters=tmp_param[np.argmin(tmp_mll)]
    mll=np.min(tmp_mll)

    return parameters,-mll

def logl_WSLS (parameters: list[float],
                       data):
    """
    Simulates an n-armed bandit task.

    Parameters:
    parameters: list with the following structure:
        parameters[0]: probability of choosing the bandit[0]
    data: dataset structured as the output of bandit_simulations
      

    Returns:
    mll: negative loglikelihood of choices given parameters
    """
    
    eps=parameters[0]
    
    n_bandits = (np.max(data[:,:,0])+1).astype(int) #number of bandits
    choice = data[:,:,0].astype(int)
    outcome = data[:,:,1]
    mll = np.full(data.shape[0:2], np.nan)
    
    if (parameters==0) | (parameters==1):
        mll=-np.inf
    else:
    
        for bb in range(data.shape[0]):
            p=np.array([0.5,0.5])
            for tt in range(data.shape[1]):
                mll[bb,tt] = np.log(p[choice[bb,tt]])
                
                if outcome[bb,tt]==1:
                    p[choice[bb,tt]]=1-eps
                    p[1-choice[bb,tt]]=eps
                    
                elif outcome[bb,tt]==0:
                    p[choice[bb,tt]]=eps
                    p[1-choice[bb,tt]]=1-eps
                
            
    return -np.nansum(mll)

def WSLS_simulation (eps: float,
                       p_reward: list[float]=[0.25, 0.75], n_blocks: int=20, n_trials: int=30):
    """
    Simulates an n-armed bandit task.

    Input:
    alpha: learning rate for the simulation.
    beta: inverse temperature parameter for softmax.
    
    Optional inputs:
    p_reward: list specifying reward probabilities of the bandits (default: [0.25, 0.75])
    n_blocks: number of blocks in the task (default: 10)
    n_trials: number of trials per block in the task (default: 30)
      

    Returns:
    output: numpy array with the results, with the following structure
        - Dimension 0: block number
        - Dimension 1: trial number
        - Dimension 2: [choices,outcomes,accuracy,q_values]
    """
    n_bandits = len(p_reward) #number of bandits

    output = np.full([n_blocks,n_trials, 3+n_bandits], np.nan) #initialize numpy array to store outputs
    
    for bb in range(n_blocks):
        p_choice=np.array([0.5,0.5])
        for tt in range(n_trials):
            
            # Choose one of the bandits given the calculated probabilities
            choice = np.random.choice(n_bandits, p=p_choice)
            
            # Generate an outcome from the bandit
            outcome = np.random.choice(2, p=[1-p_reward[choice], p_reward[choice]])
            
            
            if outcome==1:
                p_choice[choice]=1-eps/2
                p_choice[1-choice]=eps/2
                    
            elif outcome==0:
                p_choice[choice]=eps/2
                p_choice[1-choice]=1-eps/2
            # Check accuracy (i.e., whether the chosen bandit is the most rewarding one)
            accuracy = choice==p_reward.index(max(p_reward))
            
            # Store information in numpy array
            output[bb,tt,0]=choice
            output[bb,tt,1]=outcome
            output[bb,tt,2]=accuracy
            
    return output
