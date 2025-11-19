from TP_model_func import *

#load partitioned dataframe
full_pos, full_neg, part_pos, part_neg = load_data_lb_partitioned()

# test df
sub_df = full_pos[(full_pos['paper']=='AR2016') & 
                  (full_pos['study']==1) & 
                  (full_pos['problem']==4) & 
                  (full_pos['subject']=='1') ]


columns = ['paper', 'study', 'problem', 'subject', 
           'PR_stay', 'PR_shift', 'NR_stay', 'NR_shift', 'n_rare',
           'group', 'rare_type']
sum_recency_sub = pd.DataFrame(columns=columns)


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
               PR_stay, PR_shift, NR_stay, NR_shift, n_rare = count_recency_sub(output)


            
            except Exception as e:
               print(f"Error while processing {paper}, {study}, {problem}, {subject}: {e}")
               continue 

            sum_recency_sub.loc[len(sum_recency_sub)] = [paper,
                                                         study,
                                                         problem,
                                                         subject,
                                                         PR_stay,
                                                         PR_shift,
                                                         NR_stay,
                                                         NR_shift,
                                                         n_rare,
                                                         group,
                                                         rare_type]

# %%
sum_recency_sub.to_csv('temp/summary_recency_by_subject.csv',index=False)


# %%

columns = ['paper', 'study', 'problem', 
           'PR_stay', 'PR_shift', 'NR_stay', 'NR_shift', 'n_rare', 'n_sub',
           'group', 'rare_type']
sum_recency_problem = pd.DataFrame(columns=columns)


df = sum_recency_sub
paper_list = df['paper'].unique()
for paper in paper_list:
   study_list = df[df['paper']==paper]['study'].unique()
   for study in study_list:
      problem_list = df[(df['paper']==paper) & (df['study']==study)]['problem'].unique()
      for problem in problem_list:
        problem_df = df[(df['paper']==paper) & (df['study']==study) & (df['problem']==problem)]
        problem_df = problem_df.reset_index(drop=True)

        sum_recency_problem.loc[len(sum_recency_problem)] = [paper,
                                                         study,
                                                         problem,
                                                         problem_df['PR_stay'].mean(),
                                                         problem_df['PR_shift'].mean(),
                                                         problem_df['NR_stay'].mean(),
                                                         problem_df['NR_shift'].mean(),
                                                         problem_df['n_rare'].sum(),
                                                         len(problem_df),
                                                         problem_df['group'][0],
                                                         problem_df['rare_type'][0]]


# %%
sum_recency_problem.to_csv('temp/summary_recency_by_problem.csv',index=False)

# %%
# visualize rececny, pick out problems with high NR rate

#check k & t's explanation of gamblaer's fallacy (NStay after bad rare) and hot hand

# plan/compute alt/p_rare/pred_rare_from_tp ~ NR for each study separately
# check value, maybe random effects?


#%%
rare_type = 'unfavorable'
group = 'lb_full'
df_subset = sum_recency_problem[(sum_recency_problem['group']==f'{group}') & (sum_recency_problem['rare_type']==f'{rare_type}')]


y = [df_subset['PR_stay'], df_subset['NR_shift']]

for idx, array in enumerate(y, start=1):
    # Jitter x positions slightly for better visibility
    x_jitter = np.random.uniform(-0.08, 0.08, size=len(array))
    plt.scatter(idx + x_jitter, array, alpha=0.25, color='black', s=1, label='Data Points' if idx == 1 else "")

means = [np.nanmean(array) for array in y]
plt.plot(range(1, len(y)+1), means, color='red', marker='+', linestyle='--', linewidth=1, label='mean')

plt.violinplot(y, widths=0.2)
plt.xticks(range(1, 3), ['PR stay', 'NR shift']) 
plt.ylabel('Proportion of recency behavior')  # Set y-axis label
plt.ylim([0,1.05])
plt.title(f'{group}, {rare_type} rare event')

plt.show()
# %%
#'PR shift', 'NR stay', 