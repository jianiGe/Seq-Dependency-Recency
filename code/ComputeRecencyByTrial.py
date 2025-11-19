from TP_model_func import *

#load partitioned dataframe
full_pos, full_neg, part_pos, part_neg = load_data_lb_partitioned()

# test df
sub_df = full_pos[(full_pos['paper']=='AR2016') & 
                  (full_pos['study']==1) & 
                  (full_pos['problem']==4) & 
                  (full_pos['subject']=='1') ]


columns = ['paper', 'study', 'problem', 'subject', 'trial',
           'recency_type',
           'group', 'rare_type']
sum_recency_trial = pd.DataFrame(columns=columns)


#%%
df = part_neg
df = df.reset_index(drop=True)
group = df['group'][0]
rare_type = df['rare_type'][0]

paper_list = df['paper'].unique()
for paper in paper_list:
   study_list = df[df['paper']==paper]['study'].unique()
   for study in study_list:
      problem_list = df[(df['paper']==paper) & (df['study']==study)]['problem'].unique()
      for problem in problem_list:
         sub_list = df[(df['paper']==paper) & (df['study']==study) & (df['problem']==problem)]['subject'].unique()
         for subject in sub_list:
            sub_df = df[(df['paper']==paper) & (df['study']==study) & (df['problem']==problem) & (df['subject']==subject)]
            sub_df = sub_df.reset_index(drop=True)

            try:
            # initialize dictionary to store subject outputs
               output = {}
               output.update({"group": group,
                           "rare_type": rare_type,
                           "rare_option": sub_df.loc[0,'rare_opt'],
                           "rare_out": sub_df.loc[0,'rare_out']})
            # extract option sequence
               ExtractSequence(sub_df, output)

            # convert sequence value to 1s and 2s
               FeedbackToFactor(output)
               ExtractChoices(sub_df, output)
 
            # get choice index after rare event (chosen or observed, from 2nd rare event onwards)
               #PR_stay, PR_shift, NR_stay, NR_shift, n_rare = count_recency_sub(output)
               s1_rare_idx, s2_rare_idx = idx_RareEvent(output,skip=0)

               for key in s1_rare_idx.keys():
                  for item in s1_rare_idx[key]:
                     if item != None:
                        sum_recency_trial.loc[len(sum_recency_trial)] = [paper,
                                                         study,
                                                         problem,
                                                         subject,
                                                         item, # rare event trialNo
                                                         key, # recency type
                                                         group,
                                                         rare_type]
               for key in s2_rare_idx.keys():
                  for item in s2_rare_idx[key]:
                     if item != None:
                        sum_recency_trial.loc[len(sum_recency_trial)] = [paper,
                                                         study,
                                                         problem,
                                                         subject,
                                                         item, # rare event trialNo
                                                         key, # recency type
                                                         group,
                                                         rare_type]
            
            except Exception as e:
               print(f"Error while processing {paper}, {study}, {problem}, {subject}: {e}")
               continue 

           

# %%
sum_recency_sub.to_csv('temp/summary_recency_by_subject.csv',index=False)

data = sum_recency_trial
# %%



sum_recency_trial = data[data['paper']=='CN2011']

block_size = 25
sum_recency_trial['block'] = (sum_recency_trial['trial'] - 1) // block_size + 1

recency_type = 'neg_recent_stay'

# Step 2: Calculate the proportion of A's in each block
block_proportions = sum_recency_trial.groupby('block')['recency_type'].apply(lambda x: (x == f'{recency_type}').mean()).reset_index()
block_counts = sum_recency_trial.groupby('block')['recency_type'].apply(lambda x: (x == f'{recency_type}').sum()).reset_index()

# Step 3: Plot the count of A's for each block
plt.figure(figsize=(10, 6))
#plt.plot(block_counts['block'], block_counts['recency_type'], marker='o', linestyle='-', color='grey')

plt.plot(block_proportions['block'], block_proportions['recency_type'], marker='o', linestyle='-', color='b')
plt.title(f'Proportion of {recency_type} by {block_size}-Trial Blocks')
plt.xlabel('Trial Block')
plt.ylabel('Proportion of As')
plt.xticks(block_proportions['block'])
plt.grid(True)
plt.tight_layout()
plt.show()
# %%

recency_type = 'neg_recent_stay'
block_size = 20

data['problem_id'] = data['paper'] + '-' + data['study'].astype(str) + '-' + data['problem'].astype(str)


problems = data['problem_id'].unique()

plt.figure(figsize=(15, 6))

colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']  # Add more colors as needed
color_cycler = cycle(colors)
for problem in problems:

   sum_recency_trial = data[data['problem_id']==problem]

   sum_recency_trial['block'] = (sum_recency_trial['trial'] - 1) // block_size + 1

   # Step 2: Calculate the proportion of A's in each block
   block_proportions = sum_recency_trial.groupby('block')['recency_type'].apply(lambda x: (x == f'{recency_type}').mean()).reset_index()
   block_counts = sum_recency_trial.groupby('block')['recency_type'].apply(lambda x: (x == f'{recency_type}').sum()).reset_index()


   #plt.plot(block_counts['block'], block_counts['recency_type'], marker='o', linestyle='-', color='grey')

   plt.plot(block_proportions['block'], block_proportions['recency_type'], marker='.', linestyle=next(line_cycler), color=next(color_cycler), alpha=.5, linewidth=1, label=f'{problem}')


plt.title(f'Proportion of {recency_type} by {block_size}-Trial Blocks')
plt.xlabel('Trial Block')
plt.ylabel(f'Proportion of {recency_type}s')
plt.xticks(block_proportions['block'])
plt.grid(True)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=8)
plt.tight_layout()
plt.show()
# %%
from itertools import cycle
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']  # Add more colors as needed
color_cyler = cycle(colors)

lines = ['-', '--', ':', '-.']
line_cycler = cycle(lines)