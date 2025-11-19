from TP_model_func import *

#load partitioned dataframe
full_pos, full_neg, part_pos, part_neg = load_data_lb_partitioned()

# test df
test_df = part_pos[(part_pos['paper']=='EERH2010') & 
                  (part_pos['study']==3) & 
                  (part_pos['problem']==59)]

#%%
# group observed P(alt) based on recency effects
output_pos_recent_stay = []
output_pos_recent_shift = []
output_neg_recent_stay = []
output_neg_recent_shift = []

#s1_p_alt_qs = []
#s2_p_alt_qs = []
# choice_p_alt_qs = []
# choice_alt_cums = []

df = full_pos
df = df.reset_index(drop=True)
group = df['group'][0]
rare_type = df['rare_type'][0]
problem_id = df['paper'][0] + '-' + str(df['study'][0]) + '-' + str(df['problem'][0])

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

            ###############################
            ##### calculate dependency statistics #####
            # memory parameter
               MemParam = [np.inf, np.inf]

            ###############################
            ## Predict likelihood of rare outcome based on
            #  Transition Probability between outcomes in observed sequences 
               # # predict likelihood of outcome 1 based on observed transition probability
               s1_result = TransProbInferencePredict(output['s1'], MemParam, False)
               s2_result = TransProbInferencePredict(output['s2'], MemParam, False)
               output.update({"s1_result": s1_result,
                              "s2_result": s2_result})
               
               # # get predicted likelihood of getting rare event
               pred_rare_tp = PredictRareFromTP(output)
               output.update({"pred_rare_tp": pred_rare_tp})
 
            # get choice index after rare event (chosen or observed)
               s1_rare_idx, s2_rare_idx = idx_RareEvent(output,skip=0)
               

            # get p_rare that informed the choice after rare event
               output_pos_recent_stay.extend(output['pred_rare_tp'][s1_rare_idx['pos_recent_stay']].tolist())
               output_pos_recent_shift.extend(output['pred_rare_tp'][s1_rare_idx['pos_recent_shift']].tolist())
               output_neg_recent_stay.extend(output['pred_rare_tp'][s1_rare_idx['neg_recent_stay']].tolist())
               output_neg_recent_shift.extend(output['pred_rare_tp'][s1_rare_idx['neg_recent_shift']].tolist())

               output_pos_recent_stay.extend(output['pred_rare_tp'][s2_rare_idx['pos_recent_stay']].tolist())
               output_pos_recent_shift.extend(output['pred_rare_tp'][s2_rare_idx['pos_recent_shift']].tolist())
               output_neg_recent_stay.extend(output['pred_rare_tp'][s2_rare_idx['neg_recent_stay']].tolist())
               output_neg_recent_shift.extend(output['pred_rare_tp'][s2_rare_idx['neg_recent_shift']].tolist())
            

            except Exception as e:
               print(f"Error while processing {paper}, {study}, {problem}, {subject}: {e}")
               continue 

output_pos_recent_stay = np.array(output_pos_recent_stay)
output_neg_recent_shift = np.array(output_neg_recent_shift)
output_pos_recent_shift = np.array(output_pos_recent_shift)
output_neg_recent_stay = np.array(output_neg_recent_stay)

#p_alt_pos_recent_stay = p_alt_pos_recent_stay[~np.isnan(p_alt_pos_recent_stay)]
#p_alt_neg_recent_shift = p_alt_neg_recent_shift[~np.isnan(p_alt_neg_recent_shift)]
#p_alt_pos_recent_shift = p_alt_pos_recent_shift[~np.isnan(p_alt_pos_recent_shift)]
#p_alt_neg_recent_stay = p_alt_neg_recent_stay[~np.isnan(p_alt_neg_recent_stay)]

#%%
# full feedback

y = [output_pos_recent_stay, output_pos_recent_shift, 
     output_neg_recent_stay, output_neg_recent_shift]

for idx, array in enumerate(y, start=1):
    # Jitter x positions slightly for better visibility
    x_jitter = np.random.uniform(-0.15, 0.15, size=len(array))
    plt.scatter(idx + x_jitter, array, alpha=0.4, color='black', s=2, label='Data Points' if idx == 1 else "")

means = [np.nanmean(array) for array in y]
plt.plot(range(1, len(y)+1), means, color='red', marker='+', linestyle='--', linewidth=1, label='mean')

plt.violinplot(y)
plt.xticks(range(1, 5), ['PR stay', 'PR shift', 'NR stay', 'NR shift']) 
plt.ylabel('Predicted P(rare) from observed TP')  # Set y-axis label
plt.title(f'full feedback, {rare_type} rare event')

plt.show()


# %%
# partial feedback

if rare_type == 'favorable':
   y = [output_pos_recent_stay, output_neg_recent_shift]
   x_labels = ['PR stay', 'NR shift']
else:
   y = [output_pos_recent_shift, output_neg_recent_stay]
   x_labels = ['PR shift', 'NR stay']


for idx, array in enumerate(y, start=1):
    # Jitter x positions slightly for better visibility
    x_jitter = np.random.uniform(-0.2, 0.2, size=len(array))
    plt.scatter(idx + x_jitter, array, alpha=0.4, color='black', s=2, label='Data Points' if idx == 1 else "")

means = [np.nanmean(array) for array in y]
plt.plot(range(1, len(y)+1), means, color='red', marker='+', linestyle='--', linewidth=1, label='mean')

plt.violinplot(y)
plt.xticks(range(1, 3), labels=x_labels) 
plt.ylabel('Predicted P(rare) from observed TP')  # Set y-axis label
plt.title(f'partial feedback, {rare_type} rare event, {problem_id}')

plt.show()

# %%
