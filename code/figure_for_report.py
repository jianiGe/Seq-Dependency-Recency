# Code for generating figures for report

import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
working_dir = "C:/Users/jiani/Documents/ARC-project"
os.chdir(working_dir)

df = pd.read_csv('temp/summary_recency_by_problem.csv')

# %%
df['PR_total'] = df['PR_shift'] + df['PR_stay']
df['NR_total'] = df['NR_shift'] + df['NR_stay']


#df['prob_id'] = df['paper']+df['study']+df['problem']
for i in range(len(df)):
    df['prob_id'][i] = df['paper'][i] + '-' + str(df['study'][i]) + '-' + str(df['problem'][i])

#
df = df.sort_values(by='NR_shift', ascending=False)

#
plt.figure(figsize=(20, 6))
plt.plot(df['prob_id'], df['NR_stay'], label='Neg. recency (stay)')
plt.plot(df['prob_id'], df['NR_shift'], label='Neg. recency (shift)')
plt.xticks(rotation=90, fontsize=8)
plt.xlabel('Problem ID')
plt.ylabel('Neg. recency proportion')
plt.title('Proportion of negative recency by problem')
plt.legend()
plt.grid(False)
plt.show()


#
df['group2'] = df['group']+'_'+df['rare_type']

plt.figure(figsize=(8, 6))
sns.boxplot(data=df, x='group2', y='NR_shift')
sns.boxplot(data=df, x='group2', y='NR_stay')
plt.title('Distribution of Measures by Category')
plt.ylabel('Value')
plt.xlabel('Category')
plt.grid(False)
plt.legend(title='Measure')
plt.show()

#
df_melted = df.melt(id_vars='group2',
                    value_vars=['NR_shift', 'NR_stay'],
                    var_name='NR_type', value_name='NR_rate')

# Plot box plots
plt.figure(figsize=(8, 6))
sns.boxplot(data=df_melted, x='group2', y='NR_rate', width=0.4,hue='NR_type')
plt.title('By feedback and rare outcome type')
plt.xlabel('Problem category')
plt.ylabel('Proportion of neg. recency')
plt.legend(title='Neg. recency type')
plt.show()

#
###import summary df
df2 = pd.read_csv('temp/behav_summary.csv')
df2 = df2.drop_duplicates(subset=['paper', 'study', 'problem'])

# convert values to lists
import ast
df2['choice_set_outputs'] = df2['choice_set_outputs'].apply(ast.literal_eval)
df2['choice_set_probs'] = df2['choice_set_probs'].apply(ast.literal_eval)
df2['rare_prob'] = df2['rare_prob'].apply(ast.literal_eval)
df2['rare_output'] = df2['rare_output'].apply(ast.literal_eval)
df2['rare_type'] = df2['rare_type'].apply(ast.literal_eval)
df2['rare_choice'] = df2['rare_choice'].apply(ast.literal_eval)

for i in range(len(df2)):
    opts = df2['options'][i].split('_')
    rare_opt = df2['rare_choice'][i][0]
    index = opts.index(rare_opt)

    rare_opts = df2['choice_set_outputs'][i][index]
    rare_outcome = df2['rare_output'][i][0]
    #rare_outcome_idx = rare_opts.index(rare_outcome)
    non_rare_outcome = [x for x in rare_opts if x != rare_outcome][0]

    rare_nonrare_diff = rare_outcome - non_rare_outcome

    df2['rare_minus_nonrare'][i] = rare_nonrare_diff
    df2['rare_prob'][i] = df2['rare_prob'][i][0]
    df2['rare_output'][i] = df2['rare_output'][i][0]


df3 = df2[['paper', 'study', 'problem', 'rare_output', 'rare_prob', 'rare_minus_nonrare']]

df4 = pd.merge(df, df3, on=['paper', 'study', 'problem'], how='inner')

sub_df = df4[df4['group2']=='lb_partial_favorable']

g = sns.relplot(
    data=df4,
    x='rare_output', y='NR_stay',
    col='group2',  # Facet by col3
    kind='scatter',
    col_wrap=4,   # Wrap columns if there are many categories
    height=4,     # Size of each facet
    aspect=1      # Aspect ratio
)
g.set_titles(col_template="{col_name}", size=20)
for ax in g.axes.flat:
    ax.set_ylabel("NR_stay", fontsize=20)  # Change fontsize as needed
    ax.set_xlabel("Rare outcome (raw value)", fontsize=16)  # Change fontsize as needed
plt.tight_layout()
plt.show()

for i in range(len(df4)):
    df4['transformed'][i] = np.arcsinh(df4['rare_minus_nonrare'][i])