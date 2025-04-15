import pandas as pd
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob

def calculate_minimal_mutants(killed_mutants_df):
    killed_mutants_df['failing_tests'] = killed_mutants_df['Failed Tests'].apply(
        lambda x: set(eval(x)) if pd.notna(x) and x != "[]" else set()
    )
    grouped = killed_mutants_df.groupby('Function ID')
    minimal_mutants_list = []

    for func_id, group in grouped:
        mutants = group.to_dict(orient='records')
        minimal_mutants = set(
            tuple((mutant['Function Name'], mutant['Function ID'], mutant['Mutant Index'], tuple(mutant['failing_tests'])))
            for mutant in mutants
        )
        for mutant in mutants:
            minimal_mutants -= set([
                m for m in minimal_mutants
                if mutant['failing_tests'].issubset(set(m[3])) and mutant['failing_tests'] != set(m[3])
            ])

        for mutant in mutants:
            key = (mutant['Method'], mutant['Function ID'], mutant['Mutant Index'], tuple(mutant['failing_tests']))
            if key in minimal_mutants:
                minimal_mutants_list.append(mutant)

    return pd.DataFrame(minimal_mutants_list)


def detect_method(repo_name):
    repo_lower = repo_name.lower()
    if "intent" in repo_lower:
        return "μIntMut"
    elif "alternative" in repo_lower:
        return "oIntMut"
    elif "mu" in repo_lower:
        return "μBERT"
    else:
        return "Unknown"


def process_all_csvs(directory="repo_stats_with_code"):
    all_files = glob.glob(os.path.join(directory, "*_mutation_results.csv"))
    all_data = []

    for file_path in all_files:
        df = pd.read_csv(file_path)
        df['Failed Count'] = pd.to_numeric(df['Failed Count'], errors='coerce').fillna(0).astype(int)

        df['Repo Name'] = os.path.basename(file_path).replace("_mutation_results.csv", "")
        all_data.append(df)

    combined_df = pd.concat(all_data, ignore_index=True)

    combined_df['Function ID'] = combined_df['Repo Name'] + "/" + combined_df['Function Name']
    #combined_df['Method'] = combined_df['Repo Name'].apply(detect_method)

    return combined_df


def analyze_subsumption():
    os.makedirs("subsumption-results", exist_ok=True)

    full_df = process_all_csvs()
    killed_df = full_df[full_df['Failed Count'] > 0]

    minimal_df = calculate_minimal_mutants(killed_df)

    # Calculate contributions
    contribution_per_function = defaultdict(dict)
    for func_id, group in minimal_df.groupby('Function ID'):
        method_counts = Counter(group['Method'])
        total = len(group)
        for method, count in method_counts.items():
            contribution_per_function[func_id][method] = (count / total) * 100

    contrib_df = pd.DataFrame.from_dict(contribution_per_function, orient='index').fillna(0)
    contrib_df.index.name = "Function ID"

    contrib_df.to_csv("subsumption-results/function_method_contributions.csv")
    minimal_df.to_csv("subsumption-results/minimal_subsuming_mutants.csv", index=False)

    return contrib_df, minimal_df


def plot_method_contributions(contrib_df):
    long_df = contrib_df.reset_index().melt(id_vars='Function ID', var_name='Method', value_name='Contribution')

    plt.figure(figsize=(12, 6))
    sns.boxplot(data=long_df, x='Method', y='Contribution', palette='Set2')
    plt.ylabel("Contribution to Subsuming Mutants (%)")
    plt.title("Subsuming Mutant Contributions per Method")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("subsumption-results/method_contribution_boxplot.png")
    plt.close()


if __name__ == "__main__":
    contrib_df, minimal_df = analyze_subsumption()
    plot_method_contributions(contrib_df)
