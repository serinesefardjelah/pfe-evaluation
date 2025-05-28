import os
import glob
import pandas as pd
from itertools import combinations
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def detect_method(repo_name):
    return "LLM"

def process_all_csvs(directory="repo_stats"):
    all_files = glob.glob(os.path.join(directory, "*_mutation_results.csv"))
    all_data = []

    for file_path in all_files:
        try:
            df = pd.read_csv(file_path)
            repo_name = os.path.basename(file_path).replace("_mutation_results.csv", "")
            df['Repo Name'] = repo_name
            df['Failed Count'] = pd.to_numeric(df['Failed Count'], errors='coerce').fillna(0).astype(int)
            all_data.append(df)
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")

    if not all_data:
        raise ValueError("No valid CSV files found in the repo_stats directory")

    return pd.concat(all_data, ignore_index=True)

def get_test_failures(row):
    if pd.isna(row['Failed Tests']):
        return set()
    failed_tests = row['Failed Tests'].strip("[]'")
    return set([test.strip().strip("'\"") for test in failed_tests.split(',') if test.strip()])

def analyze_mutants(full_df):
    full_df['Mutant ID'] = full_df['Repo Name'] + "/" + full_df['Function Name'] + "/" + full_df['Mutant Index'].astype(str)
    full_df['Killed'] = full_df['Failed Count'] > 0

    results = {
        'repo': [],
        'total_mutants': [],
        'killed_mutants': [],
        'duplicate_mutants': [],
        'subsuming_mutants': [],
        'subsumed_mutants': [],
        'unique_mutants': []
    }

    for repo, group in full_df.groupby('Repo Name'):
        killed_group = group[group['Killed']]
        total_killed = len(killed_group)

        # Duplicate calculation across all mutants
        mutant_failures = {}
        for _, row in group.iterrows():
            failures = frozenset(get_test_failures(row))
            mutant_failures.setdefault(failures, []).append(row['Mutant ID'])

        duplicate_count = sum(len(v) - 1 for v in mutant_failures.values() if len(v) > 1)

        # Subsumption based on killed mutants only
        killed_mutant_failures = {}
        for _, row in killed_group.iterrows():
            failures = frozenset(get_test_failures(row))
            killed_mutant_failures.setdefault(failures, []).append(row['Mutant ID'])

        unique_failures = list(killed_mutant_failures.keys())
        subsuming = set()
        subsumed = set()

        for i, j in combinations(range(len(unique_failures)), 2):
            if unique_failures[i].issubset(unique_failures[j]):
                subsumed.update(killed_mutant_failures[unique_failures[i]])
                subsuming.add(next(iter(killed_mutant_failures[unique_failures[j]])))
            elif unique_failures[j].issubset(unique_failures[i]):
                subsumed.update(killed_mutant_failures[unique_failures[j]])
                subsuming.add(next(iter(killed_mutant_failures[unique_failures[i]])))

        unique_count = total_killed - len(subsumed) - len(subsuming)

        results['repo'].append(repo)
        results['total_mutants'].append(len(group))
        results['killed_mutants'].append(total_killed)
        results['duplicate_mutants'].append(duplicate_count)
        results['subsuming_mutants'].append(len(subsuming))
        results['subsumed_mutants'].append(len(subsumed))
        results['unique_mutants'].append(unique_count)

    return pd.DataFrame(results)

def create_plots(results_df):
    os.makedirs("mutation_analysis_results", exist_ok=True)

   # Calculate percentages
    results_df['% Duplicates'] = results_df.apply(
    lambda row: (row['duplicate_mutants'] / row['killed_mutants'] * 100) if row['killed_mutants'] > 0 else 0,
    axis=1
)
    results_df['% Subsuming'] = results_df.apply(
    lambda row: (row['subsuming_mutants'] / row['killed_mutants'] * 100) if row['killed_mutants'] > 0 else 0,
    axis=1
)
    results_df['% Subsumed'] = results_df.apply(
    lambda row: (row['subsumed_mutants'] / row['killed_mutants'] * 100) if row['killed_mutants'] > 0 else 0,
    axis=1
)
    results_df['% Unique'] = results_df.apply(
    lambda row: (row['unique_mutants'] / row['killed_mutants'] * 100) if row['killed_mutants'] > 0 else 0,
    axis=1
)

    results_df.fillna(0, inplace=True)
    results_df = results_df.sort_values('total_mutants', ascending=False)

    results_df.to_csv("mutation_analysis_results/mutation_analysis_results.csv", index=False)

    # Plot 1: Composition of killed mutants
    plt.figure(figsize=(14, 8))
    categories = ['% Unique', '% Duplicates', '% Subsumed', '% Subsuming']
    colors = ['#4CAF50', '#2196F3', '#FF9800', '#F44336']
    bottom = np.zeros(len(results_df))

    for i, category in enumerate(categories):
        plt.bar(results_df['repo'], results_df[category], bottom=bottom, label=category.replace('% ', ''), color=colors[i])
        bottom += results_df[category]

    plt.xlabel('Repositories')
    plt.ylabel('Percentage of Killed Mutants')
    plt.title('Composition of Killed Mutants')
    plt.xticks(rotation=90)
    plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.savefig("mutation_analysis_results/mutant_composition.png")
    plt.close()

    # Plot 2: Subsumption relationships
    plt.figure(figsize=(14, 6))
    x = np.arange(len(results_df))
    width = 0.35

    plt.bar(x - width/2, results_df['% Subsuming'], width, label='Subsuming Mutants', color='#FF5722')
    plt.bar(x + width/2, results_df['% Subsumed'], width, label='Subsumed Mutants', color='#607D8B')

    plt.xlabel('Repositories')
    plt.ylabel('Percentage of Killed Mutants')
    plt.title('Subsumption Relationships Among Killed Mutants')
    plt.xticks(x, results_df['repo'], rotation=90)
    plt.legend()
    plt.tight_layout()
    plt.savefig("mutation_analysis_results/subsumption_relationships.png")
    plt.close()

    # Plot 3: Duplicate mutants count
    plt.figure(figsize=(14, 6))
    plt.bar(results_df['repo'], results_df['duplicate_mutants'], color='#03A9F4')
    plt.xlabel('Repositories')
    plt.ylabel('Number of Duplicate Mutants')
    plt.title('Duplicate Mutants per Repository')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig("mutation_analysis_results/duplicate_mutants_per_repo.png")
    plt.close()

if __name__ == "__main__":
    print("Starting mutation analysis...")
    print("Looking for CSV files in 'repo_stats' folder...")

    try:
        full_df = process_all_csvs()
        print(f"Found data for {len(full_df['Repo Name'].unique())} repositories")

        results_df = analyze_mutants(full_df)
        create_plots(results_df)

        print("\nDuplicate Mutants Per Repository:")
        for _, row in results_df.iterrows():
            print(f"{row['repo']}: {row['duplicate_mutants']} duplicate mutants")

        print("\nAnalysis completed successfully!")
        print("Results saved in 'mutation_analysis_results' folder:")
        print("- mutation_analysis_results.csv")
        print("- mutant_composition.png")
        print("- subsumption_relationships.png")
        print("- duplicate_mutants_per_repo.png")

        print("\nSummary Statistics:")
        print(f"Total repositories analyzed: {len(results_df)}")
        print(f"Average duplicate mutants: {results_df['% Duplicates'].mean():.1f}%")
        print(f"Average subsuming mutants: {results_df['% Subsuming'].mean():.1f}%")
        print(f"Average subsumed mutants: {results_df['% Subsumed'].mean():.1f}%")

    except Exception as e:
        print(f"\nError during analysis: {str(e)}")
