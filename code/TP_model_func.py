# Helper functions for applying the HMM sequential dependency models to the DfE dataset
# model adapted from Meyniel et al. (2016) https://doi.org/10.1371/journal.pcbi.1005260 from Matlab

import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
import os
working_dir = "C:/Users/jiani/Documents/ARC-project"
os.chdir(working_dir)

# generate test sequence

length = 10
p_2given1 = 0.25  # Probability of transitioning from 1 to 2
p_1given2 = 0.5   # Probability of transitioning from 2 to 1

# Initialize the sequence
sequence = [1]  # Start with 1

# Generate the sequence
for _ in range(1, length):
    current_state = sequence[-1]
    if current_state == 1:
        next_state = np.random.choice([1, 2], p=[1 - p_2given1, p_2given1])
    else:  # current_state == 2
        next_state = np.random.choice([1, 2], p=[p_1given2, 1 - p_1given2])
    sequence.append(next_state)

sequence = np.array(sequence)

print(sequence)

######################################

######################################
# MAIN FUNCTION

def TransProbInferencePredict(s, MemParam=[np.inf, np.inf], default_prior=True):
    ###
    # 0: 'Input' variables
    #n = 40 # grid resolution
    #t = np.linspace(0,1,n) # univariate probability grid to return full distributions
    #nt = len(t) 
    L  = len(s)

    # default flat priors
    if default_prior:
        priorpAgB = np.array([1, 1])
        priorpBgA = np.array([1, 1])
    else:
        priorpAgB = np.array([0.3, 0.7])
        priorpBgA = np.array([0.3, 0.7])
    # A conjugate prior distribution is used for p(A|B) and p(B|A). 
    # This distribution is a Beta distributions.
    # We rename the beta parameters for brevity.
    pNAgB = priorpAgB[0]
    pNBgB = priorpAgB[1]
    pNBgA = priorpBgA[0]
    pNAgA = priorpBgA[1]

    # memory parameter
    #if MemParam[0] == np.inf:
    #    print('No memory window specified, use all observed events for inference.')
    #if MemParam[1] == np.inf:
    #    print('No memory decay specified, use all observed events for inference.')

    
    ###
    # 1: compute posterior transition probabilities estimates
    # initialize variables
    MAP = np.zeros((2, L)) # max. a posterior transition probabilities; 'mode/peak'; tp value that maximizes P(theta|data)
    m_hat = np.zeros((2, L)) # mean/expected value of theta under post. distribution; 'center of mass' of post. distrib
    s_hat = np.zeros((2, L)) # sd of ....
    predA = np.zeros((L)) # likelihood of next item (k+1) being 1
    predA_sd = np.zeros((L)) # sd of ...
    pmY = np.zeros((L, 1)) #  log model evidence (marginal likelihood of observations received so far)

    # initial TP estimate and prediction (for unobserved + 1st observed outcome)
    # based purely on priors
    idx_1st = np.where(~np.isnan(s))[0][0] #index of the first trial with observed outcome

    for i in range(idx_1st+1):
        MAP[:,i], m_hat[:,i], s_hat[:,i] = ComputeMAPandPrediction(0, 0, 0, 0, pNBgA, pNAgB, pNAgA, pNBgB)

        if s[0] == 1:
            predA[0] = 1-m_hat[1,0]; # = 1-p(B|A)
            predA_sd[0] = s_hat[1,0]
        elif s[0] == 2:
            predA[0] = m_hat[0,0]; # = p(A|B)
            predA_sd[0] = s_hat[0,0]
        else:
            predA[0] = 0.5
            predA_sd[0] = s_hat[1,0] # double check this


    # Later observations TP and prediction
    for k in range(idx_1st+1,L):

        if np.isnan(s[k]): #if current trial doesn't output observation, copy last value
            MAP[:,k] = MAP[:,k-1]
            m_hat[:,k] = m_hat[:,k-1]
            s_hat[:,k] = s_hat[:,k-1]
            predA[k] = predA[k-1]
            predA_sd[k] = predA_sd[k-1]
            #print('skip current trial')
            continue

        # Count transitions including the current one
        NBgA, NAgB, NAgA, NBgB, s1 = CountEventInMemory_partial(s[:k+1], window=MemParam[0], decay=MemParam[1]) # rn without decay/window

        # Likelihood of the 1st event - assumed to be equiprobable
        p1 = 1/2
        # placeholder for full posterior distribution
        post = []

        # Compute MAP
        MAP[:,k], m_hat[:,k], s_hat[:,k] = ComputeMAPandPrediction(NBgA, NAgB, NAgA, NBgB, pNBgA, pNAgB, pNAgA, pNBgB)
        
        # compute likelihood of next event
        # = post. expectancy of the transition rate
        if s[-1] == 1:  # use posterior marginal exectancy of B|A
            predA[k] = 1 - m_hat[1,k] 
            predA_sd[k] = s_hat[1,k]
        else:           # use posterior marginal exectancy of A|B
            predA[k] = m_hat[0,k]
            predA_sd[k] = s_hat[0,k]
    
    # (at this point, we have MAP, m_hat, s_hat, predA, predA_sd)

    ###
    # 2: Get prediction and confidence given observation at each time point
    p1g2_mean = m_hat[0,:]
    p1g2_sd   = s_hat[0,:]
    p2g1_mean = m_hat[1,:]
    p2g1_sd   = s_hat[1,:]

    p1_mean = GetConditionalValue(p1g2_mean, 1-p2g1_mean, s) #p(1|2), p(1|1)
    p1_sd = GetConditionalValue(p1g2_sd, p2g1_sd, s)
    
    ###
    # 3: get Shannon's surprise given these predictions and the actual outcomes
    surprise   = ComputeSurprise(p1_mean, s)
    surprise_w = ComputeSurprise_weighted(p1_mean, p1_sd, s)

    ###
    # return results as a dictionary
    return {
        'p1_mean': p1_mean,
        'p1_sd': p1_sd,
        'surprise': surprise,
        'surprise_w': surprise_w,
        'p1g2_mean': p1g2_mean,
        'p2g1_mean': p2g1_mean
    }



######################################
############# FUNCTIONS #############
######################################
def ComputeMAPandPrediction(NBgA, NAgB, NAgA, NBgB, pNBgA, pNAgB, pNAgA, pNBgB):
    
    # Compute posterior MAP, mean and standard deviation, using the analytical formula for beta distributions.
    # The advantage of using conjugate distribution is that posterior estimates
    # (mean, variance, MAP) have analytical solutions that depend purely on the
    # event counts NXgY (X given Y) augmented by the prior event counts pNygY.

    # Beta parameters are equals to the event counts + 1.
    # We convert event counts into beta parameters:
    NAgA = NAgA + 1
    NAgB = NAgB + 1
    NBgA = NBgA + 1
    NBgB = NBgB + 1

    # The prior and the likelihood are both beta functions; their product is another
    # beta function, whose parameters are the sum of paramaters of each beta
    # distribution - 1
    # We thus compute the parameters of the posterior beta distributions as:
    NAgA = NAgA + pNAgA - 1
    NAgB = NAgB + pNAgB - 1
    NBgA = NBgA + pNBgA - 1
    NBgB = NBgB + pNBgB - 1

    # ANALYTICAL SOLUTION
    try:
        MAP = [(NAgB-1)/(NAgB+NBgB-2), 
           (NBgA-1)/(NBgA+NAgA-2)]
    except ZeroDivisionError:
        MAP = [np.nan, np.nan]
    
    m_h = [NAgB/(NAgB+NBgB), 
           NBgA/(NBgA+NAgA)]
    s_h = [np.sqrt((NAgB * NBgB) / ((NAgB + NBgB)**2 * (NAgB + NBgB + 1))),
           np.sqrt((NBgA * NAgA) / ((NBgA + NAgA)**2 * (NBgA + NAgA + 1)))]
    
    return MAP, m_h, s_h


def CountEventInMemory(s, window=np.inf, decay=np.inf):
    Ls = len(s)
    # windowed memory
    if window < np.inf and window <= Ls:
        # count only within window
        trn = np.diff(s[-window:]) #1: A -> B; -1: B -> A
        subs4trn = s[-window:-1]
        subs = s[-window:]
        s1 = s[-window] #1st event in the windowed sequence
    else:
        # count all events if they fit in the memory window
        trn = np.diff(s)
        subs4trn = s[:-1]
        subs = s
        s1 = s[0]
    
    # exponential decay
    if decay < np.inf:
        trnDecay = np.exp(-(1 / decay) * (np.arange(len(subs4trn), 0, -1)))
        NBgA = np.sum((trn[(subs4trn==1)]==1).astype(int) * trnDecay[(subs4trn==1)])
        NAgB = np.sum((trn[(subs4trn==2)]==-1).astype(int) * trnDecay[(subs4trn==2)])
        NAgA = np.sum((trn[(subs4trn==1)]==0).astype(int) * trnDecay[(subs4trn==1)])
        NBgB = np.sum((trn[(subs4trn==2)]==0).astype(int) * trnDecay[(subs4trn==2)])

    else:
        NBgA = np.sum((trn[(subs4trn==1)]==1).astype(int))
        NAgB = np.sum((trn[(subs4trn==2)]==-1).astype(int))
        NAgA = np.sum((trn[(subs4trn==1)]==0).astype(int))
        NBgB = np.sum((trn[(subs4trn==2)]==0).astype(int))
    
    return NBgA, NAgB, NAgA, NBgB, s1

def CountEventInMemory_partial(s, window=np.inf, decay=np.inf, countOutcome=False):
    
    #
    # to be completed
    #try
    s = s[~np.isnan(s)]
    Ls = len(s)
    # windowed memory
    if window < np.inf and window <= Ls:
        # count only within window
        s_mem = s[-window:]
        s_mem = s_mem[~np.isnan(s_mem)]

        trn = np.diff(s_mem[-window:]) #1: A -> B; -1: B -> A
        subs4trn = s_mem[:-1]
        subs = s_mem
        s1 = s_mem[0] #1st event in the windowed sequence

        #trn = np.diff(s[-window:]) #1: A -> B; -1: B -> A
        #subs4trn = s[-window:-1]
        #subs = s[-window:]
        #s1 = s[-window] #1st event in the windowed sequence
    else:
        # count all events if they fit in the memory window
        #s = s[~np.isnan(s)]
        trn = np.diff(s)
        subs4trn = s[:-1]
        subs = s
        s1 = s[0]
    
    # exponential decay
    if decay < np.inf:
        trnDecay = np.exp(-(1 / decay) * (np.arange(len(subs4trn), 0, -1)))
        NBgA = np.sum((trn[(subs4trn==1)]==1).astype(int) * trnDecay[(subs4trn==1)])
        NAgB = np.sum((trn[(subs4trn==2)]==-1).astype(int) * trnDecay[(subs4trn==2)])
        NAgA = np.sum((trn[(subs4trn==1)]==0).astype(int) * trnDecay[(subs4trn==1)])
        NBgB = np.sum((trn[(subs4trn==2)]==0).astype(int) * trnDecay[(subs4trn==2)])

    else:
        NBgA = np.sum((trn[(subs4trn==1)]==1).astype(int))
        NAgB = np.sum((trn[(subs4trn==2)]==-1).astype(int))
        NAgA = np.sum((trn[(subs4trn==1)]==0).astype(int))
        NBgB = np.sum((trn[(subs4trn==2)]==0).astype(int))
      
    return NBgA, NAgB, NAgA, NBgB, s1

def GetConditionalValue(val_g2, val_g1, s):
    seqL = s.size
    val = np.full((seqL,), np.nan)
    for k in range(seqL):
        if s[k] == 1:
            val[k] = val_g1[k]
        elif s[k] == 2:
            val[k] = val_g2[k]
        else: #if np.nan
            try:
                val[k] = val[k-1]
            except IndexError:
                val[k] = np.nan
    return val


def ComputeSurprise(p1_mean, s):
    # Shannon's surprise
    # 'how unexpected an event is given its probability of occurence'
    # for binary events, quantified as surprise = -log2(p1)
    seqL = s.size
    surp = np.full((seqL,), np.nan)
    surp[0] = -math.log2(0.5)
    for k in range(1,seqL):
        if np.isnan(s[k]):
            surp[k] = np.nan
        elif s[k] == 1:
            surp[k] = -math.log2(p1_mean[k-1]);      # likelihood of s(k)=1 given s(1:k-1)
        else:
            surp[k] = -math.log2(1-p1_mean[k-1]);    # likelihood of s(k)=2 given s(1:k-1)
    return surp


def ComputeSurprise_weighted(p1_mean, p1_sd, s):
    # surprise weighted by prediction uncertainty
    # high uncertainty/sd, low w, lower surprise
    seqL = s.size
    surp = np.full((seqL,), np.nan)
    surp[0] = -math.log2(0.5)

    C = 0.5 #arbitrarily chosen for now
    w = 1 - (p1_sd / np.max(p1_sd))*C #also arbitrarily chosen for now

    for k in range(1,seqL):
        if np.isnan(s[k]):
            surp[k] = np.nan
        elif s[k] == 1:
            surp[k] = -math.log2(p1_mean[k-1]) * w[k];      # likelihood of s(k)=1 given s(1:k-1)
        else:
            surp[k] = -math.log2(1-p1_mean[k-1]) * w[k];    # likelihood of s(k)=2 given s(1:k-1)
    return surp


######################################
######### APPLY TO TEST DATA #########
######################################

def ExtractSequence(sub_df, output):
    L = len(sub_df)
    seq1 = np.full((L,), np.nan)
    seq2 = np.full((L,), np.nan)
    opt1, opt2 = sub_df['options'][0].split("_")

    #if sub_df['group'][0]=='lb_full':

    for i in range(L):
        fb1, *fb2 = sub_df['outcome'][i].split("_") #only fb1 gets assigned value if ony one outcome gets displayed
    
        fb1 = fb1.split(':')
        if fb2 != []:
            fb2 = fb2[0].split(':')
        
        if fb1[0] == opt1:
            seq1[i] = float(fb1[1])
            try:
                seq2[i] = float(fb2[1])
            except IndexError: # if no second feedback, leave the value in the extracted sequence as np.nan
                None
        else:
            seq2[i] = float(fb1[1])
            try:
                seq1[i] = float(fb2[1])
            except IndexError:
                None

    #update output dictionary
    output.update({"seq1_name": opt1,
                   "seq2_name": opt2,
                   "seq1": seq1,
                   "seq2": seq2
                   })
    

def ToFactor(seq, output):
    rare_type = output['rare_type']
    unique_vals = np.unique(seq)

    #full feedback / seq without missing observations
    if ~np.any(np.isnan(unique_vals)):
        if len(unique_vals) == 1: # safe option
            out_seq = np.full((len(seq),), 1)
        elif len(unique_vals) == 2: # option with two possible outcomes
            out_seq = np.where(seq == unique_vals[0], 1, 2)
        else:
            print("sequence not supported")
    
    #partial feedback
    else:
        if len(unique_vals) == 2:
            out_seq = np.where(np.isnan(seq), np.nan,  #keep NaN as it is
                        np.where(seq==unique_vals[0], 1, seq))
        elif len(unique_vals) == 3:
            out_seq = np.where(np.isnan(seq), np.nan,
                        np.where(seq==unique_vals[0], 1,
                        np.where(seq==unique_vals[1], 2, seq)))
        elif len(unique_vals) < 2:
            print("sequence not supported: no observed event")
        else:
            print("sequence not supported: too many values")

    return unique_vals, out_seq

def get_more_extreme(unique_vals, rare_type):
    if rare_type == 'favorable':
        if unique_vals[0] > unique_vals[1]:
            seq_rare = [1, unique_vals[0]]
        elif unique_vals[0] < unique_vals[1]:
            seq_rare = [2, unique_vals[1]]
        else:
            seq_rare = None
    else:
        if unique_vals[0] > unique_vals[1]:
            seq_rare = [2, unique_vals[1]]
        elif unique_vals[0] < unique_vals[1]:
            seq_rare = [1, unique_vals[0]]
        else:
            seq_rare = None
    return seq_rare



def FeedbackToFactor(output):
    seq1 = output['seq1']
    seq2 = output['seq2']
    seq1_val, s1 = ToFactor(seq1, output)
    seq2_val, s2 = ToFactor(seq2, output)

    # identify rare outcome
    if output['rare_option'] == output['seq1_name']:
        seq1_rare_value = output['rare_out']
        seq1_rare_code = np.where(seq1_val == seq1_rare_value)[0][0] + 1
        seq1_rare = [seq1_rare_code, seq1_rare_value]
        seq2_rare = None
    else:
        seq2_rare_value = output['rare_out']
        seq2_rare_code = np.where(seq2_val == seq2_rare_value)[0][0] + 1
        seq2_rare = [seq2_rare_code, seq2_rare_value]
        seq1_rare = None

    output.update({"seq1_val": seq1_val,
                   "seq2_val": seq2_val,
                   "s1": s1,
                   "s2": s2,
                   "seq1_rare": seq1_rare,
                   "seq2_rare": seq2_rare})
    
def ExtractChoices(sub_df, output):
    L = len(sub_df)
    choice1 = np.full((L,), np.nan)
    choice2 = np.full((L,), np.nan)
    for i in range(L):
        if sub_df['choice'][i] == output['seq1_name']:
            choice1[i] = 1
        elif sub_df['choice'][i] == output['seq2_name']:
            choice2[i] = 1
    output.update({"choice1": choice1,
                   "choice2": choice2})


# predict rare event likelihood
def PredictRareFromTP(output):
    #seq1
    seqL = len(output['s1'])
    pred_rare_by_tp = np.full((seqL,), np.nan)

    if output['seq1_rare'] != None:
        if output['seq1_rare'][0] == 1:
            pred_rare_by_tp = output['s1_result']['p1_mean']
        elif output['seq1_rare'][0] == 2:
            pred_rare_by_tp = 1 - output['s1_result']['p1_mean']
    if output['seq2_rare'] != None:
        if output['seq2_rare'][0] == 1:
            pred_rare_by_tp = output['s2_result']['p1_mean']
        elif output['seq2_rare'][0] == 2:
            pred_rare_by_tp = 1 - output['s2_result']['p1_mean']
    return pred_rare_by_tp
    
    
def SplitPredByChoice(output):
    #pred_rare_chose_rare = []
    #pred_rare_chose_nonrare = []
    pred_rare_chose_rare_stay = []
    pred_rare_chose_nonrare_stay = []
    pred_rare_chose_rare_shift = []
    pred_rare_chose_nonrare_shift = []

    #seq1
    s1_next_choice = np.roll(output['choice1'],-1)
    s1_next_choice[-1] = np.nan
    #seq2
    s2_next_choice = np.roll(output['choice2'],-1)
    s2_next_choice[-1] = np.nan

    #predicted rare likelihood before choosing rare option
    pred = output['s2_result']['pred_rare_tp']
    for i in range(len(s2_next_choice)-1):
        if s2_next_choice[i]==1 and output['choice2'][i]==1:
            pred_rare_chose_rare_stay.append(pred[i])
        elif s2_next_choice[i]==1 and output['choice2'][i]!=1:
            pred_rare_chose_rare_shift.append(pred[i])
        
        elif s2_next_choice[i] != 1 and output['choice2'][i]!=1:
            pred_rare_chose_nonrare_stay.append(pred[i])
        elif s2_next_choice[i] != 1 and output['choice2'][i] ==1:
            pred_rare_chose_nonrare_shift.append(pred[i])

    pred = output['s1_result']['pred_rare_tp']
    for i in range(len(s1_next_choice)-1):
        if s1_next_choice[i]==1 and output['choice1'][i]==1:
            pred_rare_chose_rare_stay.append(pred[i])
        elif s1_next_choice[i]==1 and output['choice1'][i]!=1:
            pred_rare_chose_rare_shift.append(pred[i])
        
        elif s1_next_choice[i] != 1 and output['choice1'][i]!=1:
            pred_rare_chose_nonrare_stay.append(pred[i])
        elif s1_next_choice[i] != 1 and output['choice1'][i] ==1:
            pred_rare_chose_nonrare_shift.append(pred[i])
    
    pred_rare_chose_rare_stay = np.array(pred_rare_chose_rare_stay)
    pred_rare_chose_nonrare_stay = np.array(pred_rare_chose_nonrare_stay)
    pred_rare_chose_rare_shift = np.array(pred_rare_chose_rare_shift)
    pred_rare_chose_nonrare_shift = np.array(pred_rare_chose_nonrare_shift)
    pred_rare_chose_rare_stay = pred_rare_chose_rare_stay[~np.isnan(pred_rare_chose_rare_stay)]
    pred_rare_chose_rare_shift = pred_rare_chose_rare_shift[~np.isnan(pred_rare_chose_rare_shift)]
    pred_rare_chose_nonrare_stay = pred_rare_chose_nonrare_stay[~np.isnan(pred_rare_chose_nonrare_stay)]
    pred_rare_chose_nonrare_shift = pred_rare_chose_nonrare_shift[~np.isnan(pred_rare_chose_nonrare_shift)]
    
        
    # chose_rare_s2 = pred[np.where(~np.isnan(s2_next_choice))]
    # chose_rare_s2 = chose_rare_s2[~np.isnan(chose_rare_s2)]
    
    # chose_nonrare_s2 = pred[np.where(np.isnan(s2_next_choice))][:-1]
    # chose_nonrare_s2 = chose_nonrare_s2[~np.isnan(chose_nonrare_s2)]

    # pred = output['s1_result']['pred_rare_tp']
    # chose_rare_s1 = pred[np.where(~np.isnan(s1_next_choice))]
    # chose_rare_s1 = chose_rare_s1[~np.isnan(chose_rare_s1)]
    # #predicted rare likelihood before choosing rare option
    # chose_nonrare_s1 = pred[np.where(np.isnan(s1_next_choice))][:-1]
    # chose_nonrare_s1 = chose_nonrare_s1[~np.isnan(chose_nonrare_s1)]

    # pred_rare_chose_rare.extend(chose_rare_s2)
    # pred_rare_chose_rare.extend(chose_rare_s1)
    # pred_rare_chose_nonrare.extend(chose_nonrare_s2)
    # pred_rare_chose_nonrare.extend(chose_nonrare_s1)

    return pred_rare_chose_rare_stay, pred_rare_chose_rare_shift, pred_rare_chose_nonrare_stay, pred_rare_chose_nonrare_shift
        


#############################################
################ Plotting ###################
#############################################

def plot_1d(data, data2, ylabel="Inferred likelihood of outcome1"):
   
   data2 = np.append(data2[1:], np.nan)
   
   x, y = zip(*enumerate(data))
   x = np.array(x)
   y = np.array(y)
   plt.figure(figsize=(20, 6))
   #plt.plot(x, y, marker='.', linestyle='', color='b', linewidth=1)
   plt.plot(x[data2==1], data[data2==1], color='r',marker='.', linestyle='', linewidth=1, label='chose option')
   plt.plot(x[data2!=1], data[data2!=1], color='b',marker='.', linestyle='', linewidth=1, label = 'chose other', alpha=0.4)
   plt.plot(x, data2, linestyle='', marker='o', color='r', linewidth=4, label='')
   #plt.ylim([0,1])
   plt.title("Plot series")
   plt.xlabel("trial index")
   plt.ylabel(ylabel)
   plt.legend()
   plt.grid(True)
   plt.show()

def plot_hist(data, data2):
    data2 = data2[1:]
    data = data[:-1]
    arr1 = data[data2==1]
    arr2 = data[data2!=1]

    bins = np.linspace(0,1,10)
    plt.hist(arr1, bins=bins, alpha=0.5, label='chose rare', color='blue', edgecolor='black')
    plt.hist(arr2, bins=bins, alpha=0.5, label='chose non-rare', color='orange', edgecolor='black')

    plt.xlabel('predicted non-rare likelihood')
    plt.ylabel('Frequency')
    plt.title('chosen option ~ predicted non-rare likelihood')
    plt.legend()

    plt.show()

def plot_p1_mean(output, title="Plot pred. P(val1)", xlabel="trials", ylabel="Inferred likelihood of 0s (good rare event)"):
   data = output['s2_result']['p1_mean']
   data2 = output['choice2'][1:]
   x, y = zip(*enumerate(data))
   x = np.array(x)
   y = np.array(y)
   plt.figure(figsize=(20, 6))
   #plt.plot(x, y, marker='.', linestyle='', color='b', linewidth=1)
   plt.plot(x[data2==1], data[data2==1], color='r',marker='.', linestyle='', linewidth=1, label='chose option')
   plt.plot(x[data2!=1], data[data2!=1], color='b',marker='.', linestyle='', linewidth=1, label = 'chose other')
   plt.plot(x, data2, linestyle='', marker='o', color='r', linewidth=4, label='')
   #plt.ylim([0,1])
   plt.title(title)
   plt.xlabel(xlabel)
   plt.ylabel(ylabel)
   plt.legend()
   plt.grid(True)
   plt.show()



#############################################
############### Load Data ###################
#############################################
import ast
def load_data_lb():
    data = pd.read_csv('temp/combined_df_LB_jan26.csv')

    #label full vs partial feedback
    for i in data.index:
        if len(data['outcome'][i].split("_")) > 1:
            group = 'lb_full'
        else:
            group = 'lb_partial'
        data.loc[i, 'group'] = group
    
    return data

def load_data_lb_partitioned():
    data = pd.read_csv('temp/combined_df_LB_jan26.csv')

    data['subject'] = data['subject'].astype(str)

    full_pos = data[(data['group']=='lb_full') & (data['rare_type']=='favorable')]
    full_neg = data[(data['group']=='lb_full') & (data['rare_type']=='unfavorable')]
    part_pos = data[(data['group']=='lb_partial') & (data['rare_type']=='favorable')]
    part_neg = data[(data['group']=='lb_partial') & (data['rare_type']=='unfavorable')]
    
    del data
    return full_pos, full_neg, part_pos, part_neg


#%%
# count proportion of alternations (negative autocorrelation)
def CountAlternation(s, window=np.inf, decay=np.inf):
    
    NBgA, NAgB, NAgA, NBgB, _ = CountEventInMemory_partial(s,window,decay)

    N_rep = NAgA + NBgB
    N_alt = NAgB + NBgA

    return N_alt / (N_alt + N_rep)


def P_Alternation(s, MemParam=[np.inf, np.inf]):

    L  = len(s)

    # memory parameter
    #if MemParam[0] == np.inf:
        #print('No memory window specified, use all observed events for inference.')
    #if MemParam[1] == np.inf:
        #print('No memory decay specified, use all observed events for inference.')

    # initialize variables
    p_alt = np.zeros((L))
    
    idx_1st = np.where(~np.isnan(s))[0][0] #index of the first trial with observed outcome

    # first observation (no alt/repeat info yet)
    for i in range(idx_1st+1):
        p_alt[i] = np.nan

    # Later observations TP and prediction
    for k in range(idx_1st+1,L):

        if np.isnan(s[k]): #if current trial doesn't output observation, copy last value
            p_alt[k] = p_alt[k-1]
            continue

        # Count proportion of alternations
        p_alt[k] = CountAlternation(s[:k+1], window=MemParam[0], decay=MemParam[1]) # rn without decay/window

    return p_alt

def P_RareOutcome(output, MemParam=[np.inf, np.inf]):
    #rare_option = get_rare_option(output)
    if output['seq1_rare'] != None:
        rare_outcome = output['seq1_rare'][0]
        p_one = P_Outcome1(output['s1'], MemParam=MemParam)
        if rare_outcome == 1:
            p_rare = p_one
        else:
            p_rare = 1 - p_one
    elif output['seq2_rare'] != None:
        rare_outcome = output['seq2_rare'][0]
        p_one = P_Outcome1(output['s2'], MemParam=MemParam)
        if rare_outcome == 1:
            p_rare = p_one
        else:
            p_rare = 1 - p_one
    return p_rare



def P_Outcome1(s, MemParam=[np.inf, np.inf]):

    L  = len(s)

    # memory parameter
    #if MemParam[0] == np.inf:
        #print('No memory window specified, use all observed events for inference.')
    #if MemParam[1] == np.inf:
        #print('No memory decay specified, use all observed events for inference.')

    # initialize variables
    pA = np.zeros((L))
    
    idx_1st = np.where(~np.isnan(s))[0][0] #index of the first trial with observed outcome

    # before first observation
    for i in range(idx_1st):
        pA[i] = np.nan

    # Later observations TP and prediction
    for k in range(idx_1st,L):

        if np.isnan(s[k]): #if current trial doesn't output observation, copy last value
            pA[k] = pA[k-1]
            continue

        # Count proportion of 1 outcomes
        pA[k] = CountEventInMemory_outcome(s[:k+1], window=MemParam[0], decay=MemParam[1]) # rn without decay/window

    return pA

def CountEventInMemory_outcome(s, window=np.inf, decay=np.inf):
    
    s = s[~np.isnan(s)]
    Ls = len(s)
    # windowed memory
    if window < np.inf and window <= Ls:
        # count only within window
        s_mem = s[-window:]
        s_mem = s_mem[~np.isnan(s_mem)]

    else:
        # count all events if they fit in the memory window
        #s = s[~np.isnan(s)]
        s_mem = s
        
    # exponential decay
    if decay < np.inf:
        length_subs = len(s_mem)
        SDecay = np.exp(-(1 / decay) * (length_subs + 1 - np.arange(1, length_subs + 1)))[::-1]
        NA = np.sum(SDecay[s_mem == 1])
        NB = np.sum(SDecay[s_mem == 2])
        
    else:
        NA = np.sum(s_mem==1)
        NB = np.sum(s_mem==2)
        
    return NA / (NA+NB)

def cum_rare_choice(output):
    rare_option = get_rare_option(output)
    if rare_option == 1:
        p_rare_choice = np.sum(output['choice1']==1)/len(output['choice1'])
       
    elif rare_option == 2:

        p_rare_choice = np.sum(output['choice2']==1)/len(output['choice2'])

    else:
        p_rare_choice = np.nan
    return p_rare_choice

def get_rare_option(output):
    rare_option = 0
    if (output['seq2_rare'] == None) & (output['seq1_rare'] == None):
        rare_option = 0
    elif output['seq2_rare'] == None and output['seq1_rare'] != None:
        rare_option = 1
    elif output['seq2_rare'] != None and output['seq1_rare'] == None:
        rare_option = 2 
    elif output['seq1_rare'][1]>output['seq2_rare'][1] and output['rare_type']=='favorable':
        rare_option = 1
    elif output['seq1_rare'][1]<output['seq2_rare'][1] and output['rare_type']=='unfavorable':
        rare_option = 1
    elif output['seq2_rare'][1]>output['seq1_rare'][1] and output['rare_type']=='favorable':
        rare_option = 2
    elif output['seq2_rare'][1]<output['seq1_rare'][1] and output['rare_type']=='unfavorable':
        rare_option = 2
    
    return rare_option
    

def idx_RareEvent(output,skip=0):

    if output['rare_type']=='favorable':
        
        # s1
        idx_pos_recent_stay = []
        idx_pos_recent_shift = []
        idx_neg_recent_stay = []
        idx_neg_recent_shift = []

        if output['seq1_rare'] != None:
            rare_outcome = output['seq1_rare'][0]
            rare_idx = np.where(output['s1']==rare_outcome)[0]

            #skip early rare events
            #rare_idx = [x for x in rare_idx if x >= skip]
            rare_idx = rare_idx[skip:]

            for i in rare_idx:
                if i < len(output['choice1'])-1 and i>0:
                    if output['choice1'][i+1] == 1:
                        #if the choice after pos. rare event is the rare option
                        #count shift and stay choice separatly
                        if (output['choice1'][i+1] == output['choice1'][i]) or (np.isnan(output['choice1'][i+1]) and np.isnan(output['choice1'][i])):
                            idx_pos_recent_stay.append(i) #this is the p_alt prior to the stay choice
                        else:
                            idx_pos_recent_shift.append(i)
                    else:
                        #similarly, for choosing non-rare option after seeing good rare event
                        if (output['choice1'][i+1] == output['choice1'][i]) or (np.isnan(output['choice1'][i+1]) and np.isnan(output['choice1'][i])):
                            idx_neg_recent_stay.append(i)
                        else:
                            idx_neg_recent_shift.append(i)

        s1_rare_index = {"pos_recent_stay": idx_pos_recent_stay,
                        "pos_recent_shift": idx_pos_recent_shift,
                        "neg_recent_stay": idx_neg_recent_stay,
                        "neg_recent_shift": idx_neg_recent_shift
                         }
    
        ## s2
        idx_pos_recent_stay = []
        idx_pos_recent_shift = []
        idx_neg_recent_stay = []
        idx_neg_recent_shift = []

        if output['seq2_rare'] != None:
            rare_outcome = output['seq2_rare'][0]
            rare_idx = np.where(output['s2']==rare_outcome)[0]
            rare_idx = rare_idx[skip:]

            for i in rare_idx:
                if i < len(output['choice2'])-1 and i>0:
                    if output['choice2'][i+1] == 1:
                        #if the choice after pos. rare event is the rare option
                        #count shift and stay choice separatly
                        if (output['choice2'][i+1] == output['choice2'][i]) or (np.isnan(output['choice2'][i+1]) and np.isnan(output['choice2'][i])):
                            idx_pos_recent_stay.append(i) #this is the p_alt prior to the stay choice
                        else:
                            idx_pos_recent_shift.append(i)
                    else:
                        #similarly, for choosing non-rare option after seeing good rare event
                        if (output['choice2'][i+1] == output['choice2'][i]) or (np.isnan(output['choice2'][i+1]) and np.isnan(output['choice2'][i])):
                            idx_neg_recent_stay.append(i)
                        else:
                            idx_neg_recent_shift.append(i)
            
        s2_rare_index = {"pos_recent_stay": idx_pos_recent_stay,
                        "pos_recent_shift": idx_pos_recent_shift,
                        "neg_recent_stay": idx_neg_recent_stay,
                        "neg_recent_shift": idx_neg_recent_shift
                        }
    #########################################################################
    elif output['rare_type']=='unfavorable':
        # s1
        idx_pos_recent_stay = []
        idx_pos_recent_shift = []
        idx_neg_recent_stay = []
        idx_neg_recent_shift = []

        if output['seq1_rare'] != None:
            rare_outcome = output['seq1_rare'][0]
            rare_idx = np.where(output['s1']==rare_outcome)[0]
            rare_idx = rare_idx[skip:]

            for i in rare_idx:
                if i < len(output['choice1'])-1 and i>0:
                    if output['choice1'][i+1] == 1:
                        #for unfavorable rare event,
                        # if the choice after neg. rare event is the rare option
                        # then it is negative recency
                        if (output['choice1'][i+1] == output['choice1'][i]) or (np.isnan(output['choice1'][i+1]) and np.isnan(output['choice1'][i])):
                            idx_neg_recent_stay.append(i)
                        else:
                            idx_neg_recent_shift.append(i)
                    else:
                        #similarly, for choosing non-rare option after seeing bad rare event
                        if (output['choice1'][i+1] == output['choice1'][i]) or (np.isnan(output['choice1'][i+1]) and np.isnan(output['choice1'][i])):
                            idx_pos_recent_stay.append(i)
                        else:
                            idx_pos_recent_shift.append(i)
            
        s1_rare_index = {"pos_recent_stay": idx_pos_recent_stay,
                        "pos_recent_shift": idx_pos_recent_shift,
                        "neg_recent_stay": idx_neg_recent_stay,
                        "neg_recent_shift": idx_neg_recent_shift
                         }
        
        ## s2
        idx_pos_recent_stay = []
        idx_pos_recent_shift = []
        idx_neg_recent_stay = []
        idx_neg_recent_shift = []

        if output['seq2_rare'] != None:
            rare_outcome = output['seq2_rare'][0]
            rare_idx = np.where(output['s2']==rare_outcome)[0]
            rare_idx = rare_idx[skip:]

            for i in rare_idx:
                if i < len(output['choice2'])-1 and i>0:
                    if output['choice2'][i+1] == 1:
                        #if the choice after neg. rare event is the rare option
                        #negative recency/lose-stay
                        if (output['choice2'][i+1] == output['choice2'][i]) or (np.isnan(output['choice2'][i+1]) and np.isnan(output['choice2'][i])):
                            idx_neg_recent_stay.append(i) #this is the p_alt prior to the stay choice
                        else:
                            idx_neg_recent_shift.append(i)
                    else:
                        #similarly, for choosing non-rare option after seeing bad rare event
                        if (output['choice2'][i+1] == output['choice2'][i]) or (np.isnan(output['choice2'][i+1]) and np.isnan(output['choice2'][i])):
                            idx_pos_recent_stay.append(i)
                        else:
                            idx_pos_recent_shift.append(i)
            
        s2_rare_index = {"pos_recent_stay": idx_pos_recent_stay,
                        "pos_recent_shift": idx_pos_recent_shift,
                        "neg_recent_stay": idx_neg_recent_stay,
                        "neg_recent_shift": idx_neg_recent_shift
                        }

    else:
        print('experimental condition not supported.')      
    return s1_rare_index, s2_rare_index


# get quartile
def get_quartile(arr):
    q1 = np.percentile(arr, 25)  # 25th percentile (Q1)
    median = np.percentile(arr, 50)  # 50th percentile (median)
    q3 = np.percentile(arr, 75)  # 75th percentile (Q3)
    return q1, median, q3

def get_choice_alt_quartile(output):
    # convert choice array into 1s and 2s
    choice_arr = np.where(np.isnan(output['choice1']), 1, np.where(output['choice1'] == 1, 2, output['choice1']))
    choice_alt = P_Alternation(choice_arr, [np.inf, np.inf])
    #mean = choice_alt[-1]
    q1, median, q3 = get_quartile(choice_alt[~np.isnan(choice_alt)])
    return q1, median, q3

def get_cumulative_choice_alt(output):
    choice_arr = np.where(np.isnan(output['choice1']), 1, np.where(output['choice1'] == 1, 2, output['choice1']))
    choice_alt = P_Alternation(choice_arr, [np.inf, np.inf])
    mean = choice_alt[-1]
    return mean

def sort_quartile_index(list, sort_idx=None):

    q1_values = np.array([triple[0] for triple in list])
    median_values = np.array([triple[1] for triple in list])
    q3_values = np.array([triple[2] for triple in list])

    mask = ~np.isnan(median_values)

    q1_values = q1_values[mask]
    median_values = median_values[mask]
    q3_values = q3_values[mask]

    if sort_idx ==None:
        sorted_indices = sorted(range(len(median_values)), key=lambda i: median_values[i])
    else:
        sorted_indices = sort_idx #use given sort index

    q1_sorted = [q1_values[i] for i in sorted_indices]
    median_sorted = [median_values[i] for i in sorted_indices]
    q3_sorted = [q3_values[i] for i in sorted_indices]

    return q1_sorted, median_sorted, q3_sorted, sorted_indices, mask




#%%
def count_recency_sub(output):
    s1_rare_idx, s2_rare_idx = idx_RareEvent(output)

    if all(len(value)==0 for value in s1_rare_idx.values()): #if opt 1 is not rare event option
        total_rare = 0
        for value in s2_rare_idx.values():
            total_rare += len(value)
        
        PR_stay = len(s2_rare_idx['pos_recent_stay']) / total_rare
        PR_shift = len(s2_rare_idx['pos_recent_shift']) / total_rare
        NR_stay = len(s2_rare_idx['neg_recent_stay']) / total_rare
        NR_shift = len(s2_rare_idx['neg_recent_shift']) / total_rare

    elif all(len(value)==0 for value in s2_rare_idx.values()): #if opt 2 is not rare event option
        total_rare = 0
        for value in s1_rare_idx.values():
            total_rare += len(value)
        
        PR_stay = len(s1_rare_idx['pos_recent_stay']) / total_rare
        PR_shift = len(s1_rare_idx['pos_recent_shift']) / total_rare
        NR_stay = len(s1_rare_idx['neg_recent_stay']) / total_rare
        NR_shift = len(s1_rare_idx['neg_recent_shift']) / total_rare
    
    return PR_stay, PR_shift, NR_stay, NR_shift, total_rare



#%%
def get_list_mean(list):
    list = np.array(list)
    list = list[~np.isnan(list)]
    return np.mean(list)
