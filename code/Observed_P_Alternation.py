# Compute trial-by-trial expected outcome probabilities based on observed outcome alternation rate in past trials

from TP_model_func import *

#load partitioned dataframe
full_pos, full_neg, part_pos, part_neg = load_data_lb_partitioned()

# test df
sub_df = part_pos[(part_pos['paper']=='CN2013') & 
                  (part_pos['study']==1) & 
                  (part_pos['problem']==4) & 
                  (part_pos['subject']==110) ]

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

            ###############################
            ##### calculate dependency statistics #####
            # memory parameter
               MemParam = [np.inf, np.inf]

            ###############################
            ## PROPORTION OF ALTERNATION (negative autocorrelation) in observed sequences
               s1_p_alt = P_Alternation(output['s1'],MemParam)
               s2_p_alt = P_Alternation(output['s2'],MemParam)
               output.update({"s1_p_alt": s1_p_alt,
                           "s2_p_alt": s2_p_alt})
 
            # get choice index after rare event (chosen or observed)
               s1_rare_idx, s2_rare_idx = idx_RareEvent(output)
               

            # get p_alt that informed the choice after rare event
               output_pos_recent_stay.extend(output['s1_p_alt'][s1_rare_idx['pos_recent_stay']].tolist())
               output_pos_recent_shift.extend(output['s1_p_alt'][s1_rare_idx['pos_recent_shift']].tolist())
               output_neg_recent_stay.extend(output['s1_p_alt'][s1_rare_idx['neg_recent_stay']].tolist())
               output_neg_recent_shift.extend(output['s1_p_alt'][s1_rare_idx['neg_recent_shift']].tolist())
            
               output_pos_recent_stay.extend(output['s2_p_alt'][s2_rare_idx['pos_recent_stay']].tolist())
               output_pos_recent_shift.extend(output['s2_p_alt'][s2_rare_idx['pos_recent_shift']].tolist())
               output_neg_recent_stay.extend(output['s2_p_alt'][s2_rare_idx['neg_recent_stay']].tolist())
               output_neg_recent_shift.extend(output['s2_p_alt'][s2_rare_idx['neg_recent_shift']].tolist())
       
            # #
            # # get quartile range of p_alt
            #    s1_p_alt_q = get_quartile(output['s1_p_alt'][~np.isnan(output['s1_p_alt'])])
            #    s2_p_alt_q = get_quartile(output['s2_p_alt'][~np.isnan(output['s2_p_alt'])])

            #    s1_p_alt_qs.append(s1_p_alt_q)
            #    s2_p_alt_qs.append(s2_p_alt_q)

            # # get choice alternation rate
            #    choice_alt_q = get_choice_alt_quartile(output)
            #    choice_alt_cum = get_cumulative_choice_alt(output)

            #    choice_p_alt_qs.append(choice_alt_q)
            #    choice_alt_cums.append(choice_alt_cum)


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
    plt.scatter(idx + x_jitter, array, alpha=0.1, color='black', s=1, label='Data Points' if idx == 1 else "")

means = [np.nanmean(array) for array in y]
plt.plot(range(1, len(y)+1), means, color='red', marker='+', linestyle='--', linewidth=1, label='mean')

plt.violinplot(y)
plt.xticks(range(1, 5), ['PR stay', 'PR shift', 'NR stay', 'NR shift']) 
plt.ylabel('Observed P(alt)')  # Set y-axis label
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
    plt.scatter(idx + x_jitter, array, alpha=0.1, color='black', s=1, label='Data Points' if idx == 1 else "")

means = [np.nanmean(array) for array in y]
plt.plot(range(1, len(y)+1), means, color='red', marker='+', linestyle='--', linewidth=1, label='mean')

plt.violinplot(y)
plt.xticks(range(1, 3), labels=x_labels) 
plt.ylabel('Observed P(alt)')  # Set y-axis label
plt.title(f'partial feedback, {rare_type} rare event')

plt.show()










# %%
# observed P_ALT quartiles and choice p_alt quartiles

s1_q1s, s1_medians, s1_q3s, sort_idx = sort_quartile_index(s2_p_alt_qs)
s2_q1s, s2_medians, s2_q3s, _ = sort_quartile_index(s1_p_alt_qs, sort_idx)
choice_q1s, choice_medians, choice_q3s, _ = sort_quartile_index(choice_p_alt_qs, sort_idx)
choice_alt_cums = [choice_alt_cums[i] for i in sort_idx]

# X-axis: Indices
x = range(len(s2_p_alt_qs))

# Plot Q1, median, Q3
plt.figure(figsize=(12, 6))
plt.plot(x, s2_q1s, label='Option 1 P(alt) Q1', marker=None, linestyle='-', linewidth=1, color='grey', alpha=.7)
plt.plot(x, s2_medians, label='Option 1 P(alt) Median', marker='.', linestyle='', linewidth=1, color='orange', alpha=.5)
plt.plot(x, s2_q3s, label='Option 1 P(alt) Q3', marker=None, linestyle='-', linewidth=1, color='grey', alpha=.7)

plt.plot(x, s1_q1s, label='Option 2 P(alt) Q1', marker=None, linestyle='-', linewidth=1, color='b', alpha=.2)
plt.plot(x, s1_medians, label='Option 2 P(alt) Median', marker='.', linestyle='', linewidth=1, color='orange', alpha=.5)
plt.plot(x, s1_q3s, label='Option 2 P(alt) Q3', marker=None, linestyle='-', linewidth=1, color='b', alpha=.2)

plt.plot(x, choice_alt_cums, label='Choice P(alt) Median', marker='.', linestyle='', linewidth=1, color='midnightblue', alpha=.7)

# Add labels, legend, and grid
plt.xlabel("subject-problem index")
plt.ylabel("P(alt)")
plt.title("Q1, Median, Q3s of observed alternation rate (overlayed with choice P(alt) median)")
#plt.legend()
plt.grid(True)
plt.show()


# %%
# X-axis: Indices
x = range(len(s2_p_alt_qs))
plt.figure(figsize=(12, 6))
plt.plot(x, choice_q1s, label='Choice P(alt) Q1', marker=None, linestyle='-', linewidth=1, color='dodgerblue', alpha=.3)
plt.plot(x, choice_medians, label='Choice P(alt) Median', marker='.', linestyle='', linewidth=1, color='midnightblue', alpha=.5)
plt.plot(x, choice_q3s, label='Choice P(alt) Q3', marker=None, linestyle='-', linewidth=1, color='dodgerblue', alpha=.3)

# Add labels, legend, and grid
plt.xlabel("subject-problem index")
plt.ylabel("P (alt)")
plt.title("Q1, Median, Q3s of choice alternation rate")
#plt.legend()
plt.grid(True)
plt.show()
# %%


s1_s2_median_sum = (np.array(s1_medians) + np.array(s2_medians))
s1_s2_max_median = np.maximum(np.array(s1_medians), np.array(s2_medians))


# X-axis: Indices
x = range(len(s2_p_alt_qs))
plt.figure(figsize=(12, 6))
plt.plot(x, choice_q1s, label='Choice P(alt) Q1', marker=None, linestyle='-', linewidth=1, color='dodgerblue', alpha=.3)
plt.plot(x, choice_medians, label='Choice P(alt) Median', marker='.', linestyle='', linewidth=1, color='midnightblue', alpha=.5)
plt.plot(x, choice_q3s, label='Choice P(alt) Q3', marker=None, linestyle='-', linewidth=1, color='dodgerblue', alpha=.3)

plt.plot(x, s2_medians, label='Choice P(alt) Median', marker='.', linestyle='', linewidth=1, color='orange', alpha=.5)

# Add labels, legend, and grid
plt.xlabel("subject-problem index")
plt.ylabel("P (alt)")
plt.title("Q1, Median, Q3s of choice alternation rate")
#plt.legend()
plt.grid(True)
plt.show()
# %%
x = s1_s2_median_sum
y = choice_medians
plt.figure(figsize=(8, 6))
plt.scatter(x, y, color='black', marker='.', label='P (alt)', alpha=.7)

# Add labels, title, and legend
plt.xlabel("Observed outcome P (alt)")
plt.ylabel("Choice P (alt)")
plt.title("Partial feedback, unfavorable rare event")
plt.legend()
plt.grid(True)

# Display the plot
plt.show()