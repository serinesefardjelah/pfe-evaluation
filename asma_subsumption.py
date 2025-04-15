import os.path
from collections import defaultdict, Counter
from itertools import takewhile
import pandas as pd
from matplotlib import pyplot as plt
from codegeex.data.data_utils import write_jsonl
from workshop_evaluation.utils import get_tasks_groups, check_tasks
import matplotlib.patches as mpatches
import seaborn as sns

#merge mutants from each method (by task)
#for each task get the set of subsuming mutants
#for each task find the percentage contributed by each method
#get the porportion of tasks where intent method gets the majority of subsuming mutants


def calculate_minimal_mutants(killed_mutants):

    #convert tests list to tests set
    for mutant in killed_mutants :
        mutant['failing_tests'] = set(mutant['failing_tests'])

    minimal_mutants = set(
        tuple(
            (mutant.get('method'),
             mutant['task_id'],
             mutant['mutant_id'],
             mutant['completion_id'],
             mutant['total_tests'],
             tuple(mutant['failing_tests'])
             )
        ) for mutant in killed_mutants
    )

    for mutant in killed_mutants:
        minimal_mutants = minimal_mutants - set([m for m in minimal_mutants if mutant['failing_tests'].issubset(
            set(m[5])) and mutant['failing_tests'] != set(m[5])])


    #turn back to original format
    minimal_mutants_dict = []
    for mutant in killed_mutants :
        if tuple(
            (mutant.get('method'),
             mutant['task_id'],
             mutant['mutant_id'],
             mutant['completion_id'],
             mutant['total_tests'],
             tuple(mutant['failing_tests'])
             )) in minimal_mutants :
            minimal_mutants_dict.append(mutant)

    for mutant in minimal_mutants_dict :
        mutant['failing_tests'] = list(mutant['failing_tests'])
    return minimal_mutants_dict

def get_subsuming_mutants_file(killed_mutants_file_path) :
    #group mutants by task
    subsuming = []
    task_groups = get_tasks_groups(killed_mutants_file_path)

    for task_id, mutants in task_groups.items() :
         subsuming.extend(calculate_minimal_mutants(mutants))
    write_jsonl(killed_mutants_file_path.replace("killed_mutations_failing_tests.jsonl", "subsuming_mutations_failing_tests.jsonl"), subsuming)

def merge_mutants_by_task(killable_mutants_files_list, min_mutants = 1, task_selection = "all") :
    #what tasks to keep :
    tasks = None
    if task_selection == "mubert" :
        approaches_names = {
            file.split("/")[-2]: file for file in killable_mutants_files_list
        }
        print("Selection based on :  ", approaches_names["muBERT-mutation"])
        tasks = check_tasks(approaches_names["muBERT-mutation"], min_mutants=min_mutants)
    elif task_selection == "all" :
        tasks = check_tasks(killable_mutants_files_list, min_mutants=min_mutants)



    merged_mutants = defaultdict(list)
    for file in killable_mutants_files_list:

        task_groups = get_tasks_groups(file)

        for task_id, mutants in task_groups.items():
                if tasks is None or (tasks is not None and task_id in tasks) :
                    #add mutants to the merged_mutants dict, but also add the name of the method
                    method_name = file.split("/")[-2]
                    if method_name == 'intent-mutation':
                        method_name = "µIntMut"
                    elif method_name == 'muBERT-mutation':
                        method_name = "μBERT"
                    elif method_name == "gpt-alternative-mutation":
                        method_name = "oIntMut"
                    labeled_mutants = [{**mutant, 'method': method_name} for mutant in mutants]

                    merged_mutants[task_id].extend(labeled_mutants)

    if task_selection == "post-merging" :
        #selection
        selected_merged_mutants = defaultdict(list)
        for task_id, mutants in merged_mutants.items() :
            if len(mutants) > min_mutants :
                selected_merged_mutants[task_id] = merged_mutants[task_id]
        return selected_merged_mutants

    return merged_mutants

def get_methods_percentages(subsuming_per_task) :
    contribution_per_task = defaultdict(dict)
    for task_id, subsuming_mutants in subsuming_per_task.items() :

        method_counts = Counter(mutant['method'] for mutant in subsuming_mutants)
        percentages = {method: (count / len(subsuming_mutants)) * 100 for method, count in method_counts.items()}
        contribution_per_task[task_id].update(percentages)
    return contribution_per_task

def study_semantic_diversity(killable_mutants_files, approach_index = 1, min_mutants = 1) :
    tasks_consider = check_tasks(killable_mutants_files, min_mutants=min_mutants)
    task_groups = get_tasks_groups(killable_mutants_files[approach_index])

    print(killable_mutants_files[approach_index])

    subsuming_per_task = defaultdict(dict)


    for task_id, mutants in task_groups.items() :
        if task_id in tasks_consider :

            subsuming = calculate_minimal_mutants(mutants)
            tests = []
            subsuming_minimal = []
            for s in subsuming :

                if s['failing_tests'] not in tests :
                    tests.append(s['failing_tests'])
                    subsuming_minimal.append(subsuming)

            subsuming_per_task[task_id] = {
                "killed": len(mutants),
                "subsuming": len(subsuming),
                "percentage_subsuming": (len(subsuming) / len(mutants))*100,
                "percentage_subsumed": ((len(mutants)-len(subsuming)) / len(mutants))*100,
                "percentage" : (1 - (len(subsuming_minimal) /len(subsuming)))*100

            }
    df = pd.DataFrame.from_dict(subsuming_per_task, orient='index')

    df.reset_index(inplace=True)
    df.rename(columns={'index': 'task_id'}, inplace=True)
    print(df['percentage_subsuming'].describe())
    print("hello")
    print(df['percentage'].describe())
    plt.figure(figsize=(8, 6))
    plt.boxplot(df['percentage_subsuming'], vert=True, patch_artist=True, boxprops=dict(facecolor="#5a8b64", color="black"))
    plt.title(f"Percentage of subsuming mutants per task, min_mutants = {min_mutants}", fontsize=14)
    plt.ylabel("Percentage", fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(f'subsumption-results/boxplot_diversity_min_{min_mutants}.png', format='png')
    df.to_csv("hello.csv")
    return df

def study_subsumption_between_approaches(killable_mutants_files_list, min_mutants = 1, task_selection="all") :
    merged_mutants = merge_mutants_by_task(killable_mutants_files_list, min_mutants = min_mutants, task_selection=task_selection)
    subsuming_per_task = defaultdict(list)
    for task_id, mutants in merged_mutants.items():
        subsuming = calculate_minimal_mutants(mutants)
        subsuming_per_task[task_id].extend(subsuming)

    #percentage of each method's contribution to the list of subsuming mutants per task
    contributions = get_methods_percentages(subsuming_per_task)
    df = pd.DataFrame.from_dict(contributions, orient='index').fillna(0)

    #get the method that contributes the most for each row and count
    max_per_row = df.max(axis=1)
    max_mask = df.eq(max_per_row, axis=0)
    unique_max_mask = max_mask.sum(axis=1) == 1

    final_max_mask = max_mask[unique_max_mask]

    method_counts = final_max_mask.sum(axis=0).astype(int)

    result_df = pd.DataFrame([method_counts])
    df.to_csv(f'subsumption-results/df_methods_min_{min_mutants}.csv', index_label='task_id')
    result_df.to_csv(f'subsumption-results/df_summary_methods_min_{min_mutants}.csv')

    return df


def draw_boxplots_with_hue(dfs) :
    order = ['intent-mutation', 'muBERT-mutation', 'gpt-muBERT-mutation', 'gpt-alternative-mutation']
    mins = [2, 3, 5, 8]
    long_dfs = []
    for i, df in enumerate(dfs):
        long_df = df.melt(var_name='method', value_name='value')
        long_df['source'] = f'min = {mins[i]}'
        long_dfs.append(long_df)

    combined_df = pd.concat(long_dfs, ignore_index=True)
    combined_df['method'] = pd.Categorical(combined_df['method'], categories=order, ordered=True)

    plt.figure(figsize=(12, 8))
    sns.boxplot(
        data=combined_df,
        x='source',
        y='value',
        hue='method',
        palette='Set2'
    )

    plt.xlabel('Approaches')
    plt.ylabel('Contribution to Subsuming Mutants')
    legend = plt.legend(title='Approach', frameon=True)
    legend.get_frame().set_alpha(0.3)
    plt.savefig('subsumption-results/boxplot_methods_grouped_by_source.png', format='png', bbox_inches='tight')
    plt.close()


def draw_boxplots(df):
    long_df = df.melt(var_name='method', value_name='value')
    plt.rcParams.update({'font.size': 18})
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    # Plot the boxplot
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=long_df, color='#8FAADC', x='method', y='value')

    plt.xlabel('Mutation Approach')
    plt.ylabel('Contribution to Subsuming Mutants')

    # Save and show the plot
    plt.savefig('subsumption-results/boxplot_between_approaches_corrected_font.pdf', format="pdf", bbox_inches="tight")

def histogram(df) :

    plt.hist(df['subsuming'], bins=30, alpha=0.5, color='#76453b', label='subsuming')
    plt.hist(df['killed'], bins=30, alpha=0.5, color='#43766c', label='killed')
    subsuming_patch = mpatches.Patch(color='#76453b', label='subsuming')
    killed_patch = mpatches.Patch(color='#43766c', label='killed')
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.title('Distribution of Number of Killable and Subsuming Mutants per Task')
    plt.xlabel('Number of Mutants')
    plt.ylabel('Number of Tasks')
    plt.legend(handles=[subsuming_patch, killed_patch],
               loc='upper right', bbox_to_anchor=(1, 1))

    plt.savefig(f'subsumption-results/barplot_corrected_font.png', format='png')
    plt.close()


def barplot(df) :
    df_melted = df.melt(id_vars='task_id', value_vars=['percentage_subsuming', 'percentage_subsumed'],
                        var_name='Type', value_name='Count')

    df_melted['task_number'] = df_melted['task_id'].apply(lambda x: int(x.split('/')[1]))

    df_melted = df_melted.sort_values(by='task_number')

    df_stacked = df_melted.pivot_table(index='task_id', columns='Type', values='Count', aggfunc='sum', fill_value=0)

    sns.set_style("whitegrid")

    plt.rcParams.update({'font.size': 18})
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42

    plt.figure(figsize=(12, 8))

    plt.bar(df_stacked.index, df_stacked['percentage_subsuming'], color='#505050', label='Subsuming')

    plt.bar(df_stacked.index, df_stacked['percentage_subsumed'],
            bottom=df_stacked['percentage_subsuming'], color='#D0D0D0', label='Subsumed')

    plt.xticks(rotation=90)

    plt.xlabel('Problem ID')
    plt.ylabel('Percentage of Mutants')

    subsuming_patch = mpatches.Patch(color='#505050', label='Subsuming')
    killed_patch = mpatches.Patch(color='#B0B0B0', label='Subsumed')
    plt.legend(handles=[killed_patch, subsuming_patch], loc='upper right', bbox_to_anchor=(1, 1))

    # Save the plot
    plt.savefig('subsumption-results/semantic_diversity_corrected_font.pdf', format='pdf',  dpi=300, bbox_inches="tight")
    plt.close()

def barplot_subsuming(df) :
    df_melted = df.melt(id_vars='task_id', value_vars=['percentage_subsuming'],
                        var_name='Type', value_name='Count')

    df_melted['task_number'] = df_melted['task_id'].apply(lambda x: int(x.split('/')[1]))

    df_melted = df_melted.sort_values(by='task_number')

    df_stacked = df_melted.pivot_table(index='task_id', columns='Type', values='Count', aggfunc='sum', fill_value=0)

    sns.set_style("whitegrid")

    plt.figure(figsize=(12, 8))

    plt.bar(df_stacked.index, df_stacked['percentage_subsuming'], color='#505050', label='Subsuming')


    plt.xticks(rotation=90)

    plt.xlabel('Problem ID')
    plt.ylabel('Percentage of Subsuming Mutants')

    subsuming_patch = mpatches.Patch(color='#505050', label='Subsuming')
    plt.legend(handles=[subsuming_patch], loc='upper right', bbox_to_anchor=(1, 1))

    # Save the plot
    plt.savefig('subsumption-results/barplot.png', format='png')
    plt.close()

if __name__ == "__main__":
    files_list = [
        '/home/asma/PycharmProjects/intent-based-mutation-testing/generated_mutations/gpt-alternative-mutation/killed_mutations_failing_tests.jsonl',
        '/home/asma/PycharmProjects/intent-based-mutation-testing/generated_mutations/intent-mutation/killed_mutations_failing_tests.jsonl',
        '/home/asma/PycharmProjects/intent-based-mutation-testing/generated_mutations/muBERT-mutation/killed_mutations_failing_tests.jsonl',

    ]


    '''df = study_subsumption_between_approaches(files_list, min_mutants=5, task_selection="all")
    print(len(df))
    print(df.mean())
    draw_boxplots(df)'''

    df = study_semantic_diversity(files_list, 1, min_mutants=5)
    barplot(df)



