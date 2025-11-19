## Code to 
# 1) filter and combined lottery bandit data with two possible outcomes and enough observed rare events (>=2 before the last trial)
# 2) annotate rare option name, rare event type (favorable/unfavorable)


import pandas as pd
import os
import numpy as np
working_dir = "C:/Users/jiani/Documents/ARC-project"
os.chdir(working_dir)

# List of lottery studies taken from the summary table
study_list = ['AR2016_1', 'AKH2014_2', 'BE2003_2', 'BE2003_3', 'BE2003_4', 'BE2003_5', 'BY2009_2',\
             'CN2011b_2_2', 'CN2011b_2_3', 'CN2013_1', 'DEM2015_1',\
            'DEM2015_2', 'DEM2015_3', 'DME2012_1', 'EERH2010_3', 'EEY2008_1', 'FJM2011_1', 'HG2015_1',\
            'HG2015_2', 'LG2011_1', 'LS2011_1', 'MEL2006_1', 'MEL2006_2', 'NE2012_1', 'NE2012_2', 'NEDO2012_1',\
            'RH2016_1', 'RH2016_2', 'S2011_1', 'SP2011_1', 'SPAF2013_1', 'SRE2014_1', 'TAE2013_2',\
            'TAE2013_3', 'YE2007_1', 'YH2013_1', 'YH2013_4', 'YZA2015_1', 'YZA2015_2']


#%%
data_folder = "Data_withpaper/RepeatedChoice_Yujia"

combined_df = pd.DataFrame()

for study_index in study_list:
    try:
        # get study paper/study/condition
        paper, study, *condition = study_index.split('_')

        data_path = os.path.join(data_folder, paper, 'processed', paper+'_'+study+'_data.csv')
        options_path = os.path.join(data_folder, paper, 'processed', paper+'_'+study+'_options.csv')

        # read study data
        study_data = pd.read_csv(data_path)
        if condition != []:
            study_data = study_data[study_data['condition']==int(condition[0])]
        study_data = study_data.dropna(subset=['choice', 'outcome', 'options']) #drop incomplete data rows
        study_data = study_data.reset_index(drop=True)
        study_options = pd.read_csv(options_path)

        # 
        problem_list = study_data['problem'].unique()
        for problem in problem_list:
            try:
                problem_df = pd.DataFrame()

                problem_options = study_data[study_data['problem']==problem]['options'].unique()[0].split('_')

                # skip problems that do not contain rare event by design or have more than two options/displayed outcomes
                problem_excluded = ExcludeProblem(problem_options, study_options)
                if problem_excluded:
                    print(f'{paper}, {study}, {problem} excluded: no rare event by design')
                    continue
                
                rare_option = get_rare_option(problem_options, study_options)
                problem_type = get_problem_type(study_data, problem)

                study_data['subject'] = study_data['subject'].astype(str)
                sub_list = study_data[study_data['problem']==problem]['subject'].unique()
                for subject in sub_list:
                    sub_df = study_data[(study_data['problem']==problem) & (study_data['subject']==subject)]
                    sub_df = sub_df.reset_index(drop=True)

                    sub_excluded = ExcludeSubject(sub_df, rare_option)
                    if sub_excluded:
                        #print(f'{paper}, {study}, {problem}, {subject} excluded; not enough rare events')
                        continue

                    for key, value in rare_option.items():
                        sub_df[key] = value
                    for key, value in problem_type.items():
                        sub_df[key] = value
                    
                    problem_df = pd.concat([problem_df, sub_df], axis=0, ignore_index=True)
            except Exception as e:
                print(f"Error while processing {paper}, {study}, {problem}: {e}")
                continue


            # if more than 5 subject seuqences for problem, add to combined data; otherwise skip problem
            if len(problem_df)==0 or problem_df['subject'].nunique() >= 5:
                combined_df = pd.concat([combined_df, problem_df], axis=0, ignore_index=True)        
            else:
                print(f'{paper}, {study}, {problem} excluded: not enough participants ({problem_df['subject'].nunique()})')
    
    except Exception as e:
        print(f"Error while processing {study_index}: {e}")
        continue 



#%%
print(len(combined_df)) #376204
combined_df.to_csv('temp/combined_df_LB_jan26.csv', index=False)

#%% Functions
def ExcludeProblem(options, study_options):
    exclude = False

    # if more than two options to choose from
    if len(options) > 2:
        exclude = True

    else:
        # if the two options are safe and equiprobable, respectively
        # or if any option has more than 2 possible outcomes
        opt_no_rare = 0

        for option in options:
            opt_data = study_options[study_options['option']==option]

            if len(opt_data.columns) > 5: # if more than 2 possible outcomes
                exclude = True
        
            elif (opt_data['pr_1'].sum()==1) or (opt_data['pr_1'].sum() == 0.5):
                opt_no_rare += 1

        if opt_no_rare == 2:
            exclude = True
    
    return exclude


def get_rare_option(options, study_options):

    option_data = study_options[study_options['option'].isin(options)]
    option_data = option_data.sort_values(by='pr_1', ascending=False).reset_index(drop=True)
    
    # if first option safe, then second option is rare option
    if option_data['pr_1'][0] == 1:
        rare_alt = 'safe'
        rare_opt_idx = 1
    #else, rare option is the one with rarer outcome
    else: 
        rare_alt = 'risky'
        min_rare_prob_1 = np.minimum(option_data['pr_1'][0], option_data['pr_2'][0])
        min_rare_prob_2 = np.minimum(option_data['pr_1'][1], option_data['pr_2'][1])

        option_probs = [min_rare_prob_1, min_rare_prob_2]

        # if there is a rarer outcome
        if option_probs.count(min(option_probs)) == 1:
            rare_opt_idx = option_probs.index(min(option_probs))
        # if two rare outcomes are equally probably, choose the one with more extreme value
        elif option_probs.count(min(option_probs)) == 2:
            outcome_range_1 = abs(option_data['out_1'][0] - option_data['out_2'][0])
            outcome_range_2 = abs(option_data['out_1'][1] - option_data['out_2'][1])
            option_ranges = [outcome_range_1, outcome_range_2]
            rare_opt_idx = option_ranges.index(max(option_ranges))

    rare_opt = option_data['option'][rare_opt_idx]

    if option_data['pr_1'][rare_opt_idx] < option_data['pr_2'][rare_opt_idx]:
        rare_prob = option_data['pr_1'][rare_opt_idx]
        rare_out = option_data['out_1'][rare_opt_idx]
    else:
        rare_prob = option_data['pr_2'][rare_opt_idx]
        rare_out = option_data['out_2'][rare_opt_idx]
    
    if (option_data['out_1'][rare_opt_idx] < option_data['out_2'][rare_opt_idx]) & (rare_out == option_data['out_1'][rare_opt_idx]):
        rare_type = 'unfavorable'
    elif (option_data['out_1'][rare_opt_idx] > option_data['out_2'][rare_opt_idx]) & (rare_out == option_data['out_1'][rare_opt_idx]):
        rare_type = 'favorable'
    elif (option_data['out_1'][rare_opt_idx] < option_data['out_2'][rare_opt_idx]):
        rare_type = 'favorable'
    elif (option_data['out_1'][rare_opt_idx] > option_data['out_2'][rare_opt_idx]):
        rare_type = 'unfavorable'

    rare_option = {"rare_opt": rare_opt,
                   "rare_prob": rare_prob,
                   "rare_out": rare_out,
                   "rare_alt": rare_alt,
                   "rare_type": rare_type
                   }

    return rare_option


def ExcludeSubject(sub_df, rare_option, min_rare=3):
    exclude = False

    #get rare option sequence
    rare_seq, exclude = ExtractRareSequence(sub_df, rare_option)

    rare_event_idx = np.where(rare_seq == rare_option['rare_out'])[0]
    rare_event_idx = rare_event_idx[rare_event_idx<len(rare_seq)-1]

    if len(rare_event_idx) < min_rare:
        exclude=True

    # old code
    # # exclude subject if less than 2 rare events have been encountered from the 2nd and the second to last trial
    # if rare_option['rare_out'].is_integer():
    #     rare_feedback = str(int(rare_option['rare_out']))
    # else:
    #     rare_feedback = str(rare_option['rare_out'])
    # got_rare_feedback = sub_df[sub_df['outcome'].str.contains(rare_feedback, case=False, na=False)].index

    # if len(got_rare_feedback[got_rare_feedback < (len(sub_df)-1)]) < 3:
    #     exclude = True
    #     #print(len(got_rare_feedback[got_rare_feedback < (len(sub_df)-1)]))
    return exclude


def get_problem_type(study_data, problem):
    row = study_data[study_data['problem']==problem].iloc[0]
    if len(row['outcome'].split('_')) == 1:
        group = 'lb_partial'
    else:
        group = 'lb_full'

    return {'group': group}


def ExtractRareSequence(sub_df, rare_option):
    exclude = False
    L = len(sub_df)
    seq1 = np.full((L,), np.nan)
    seq2 = np.full((L,), np.nan)
    opt1, opt2 = sub_df['options'][0].split("_")

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

    if np.all(np.isnan(seq1)) or np.all(np.isnan(seq2)):
        exclude = True

    if rare_option['rare_opt'] == opt1:
        return seq1, exclude
    else:
        return seq2, exclude
