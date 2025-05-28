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

def process_all_csvs(directory="repos_stats2"):
    # Find all CSV files with the specific naming pattern in the repo_stats folder
    all_files = glob.glob(os.path.join(directory, "*_mutation_results.csv"))
    all_data = []

    for file_path in all_files:
        try:
            df = pd.read_csv(file_path)
            # Extract repo name from filename
            repo_name = os.path.basename(file_path).replace("_mutation_results.csv", "")
            df['Repo Name'] = repo_name
            all_data.append(df)
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
    
    if not all_data:
        raise ValueError("No valid CSV files found in the repo_stats directory")
    
    return pd.concat(all_data, ignore_index=True)

def get_test_failures(row):
    if pd.isna(row['Failed Tests']):
        return set()
    # Clean and split the failed tests string
    failed_tests = row['Failed Tests'].strip("[]'")
    return set([test.strip().strip("'\"") for test in failed_tests.split(',') if test.strip()])

def analyze_mutants(full_df):
    # Calculate basic statistics
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
        
        # Find duplicate mutants (same failure pattern)
        mutant_failures = {}
        for _, row in killed_group.iterrows():
            failures = frozenset(get_test_failures(row))
            mutant_failures.setdefault(failures, []).append(row['Mutant ID'])
        
        duplicate_count = sum(len(v) - 1 for v in mutant_failures.values() if len(v) > 1)
        
        # Find subsuming/subsumed mutants
        unique_failures = list(mutant_failures.keys())
        subsuming = set()
        subsumed = set()
        
        for i, j in combinations(range(len(unique_failures)), 2):
            if unique_failures[i].issubset(unique_failures[j]):
                subsumed.update(mutant_failures[unique_failures[i]])
                subsuming.add(next(iter(mutant_failures[unique_failures[j]])))
            elif unique_failures[j].issubset(unique_failures[i]):
                subsumed.update(mutant_failures[unique_failures[j]])
                subsuming.add(next(iter(mutant_failures[unique_failures[i]])))
        
        # Calculate unique mutants (not duplicate or part of subsumption)
        unique_count = total_killed - duplicate_count - len(subsumed) - len(subsuming)
        
        # Store results
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
    results_df['% Duplicates'] = results_df['duplicate_mutants'] / results_df['killed_mutants'] * 100
    results_df['% Subsuming'] = results_df['subsuming_mutants'] / results_df['killed_mutants'] * 100
    results_df['% Subsumed'] = results_df['subsumed_mutants'] / results_df['killed_mutants'] * 100
    results_df['% Unique'] = results_df['unique_mutants'] / results_df['killed_mutants'] * 100
    
    # Replace NaN with 0 (for repos with no killed mutants)
    results_df.fillna(0, inplace=True)
    
    # Sort by total mutants
    results_df = results_df.sort_values('total_mutants', ascending=False)
    
    # Save results to CSV
    results_df.to_csv("mutation_analysis_results/mutation_analysis_results.csv", index=False)
    
    # Plot 1: Composition of killed mutants
    plt.figure(figsize=(14, 8))
    
    categories = ['% Unique', '% Duplicates', '% Subsumed', '% Subsuming']
    colors = ['#4CAF50', '#2196F3', '#FF9800', '#F44336']
    
    bottom = np.zeros(len(results_df))
    for i, category in enumerate(categories):
        plt.bar(results_df['repo'], results_df[category], bottom=bottom, 
               label=category.replace('% ', ''), color=colors[i])
        bottom += results_df[category]
    
    plt.xlabel('Repositories')
    plt.ylabel('Percentage of Killed Mutants')
    plt.title('Composition of Killed Mutants')
    plt.xticks(rotation=90)
    plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.savefig("mutation_analysis_results/mutant_composition.png", bbox_inches='tight')
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

if __name__ == "__main__":
    print("Starting mutation analysis...")
    print("Looking for CSV files in 'repo_stats' folder...")
    
    try:
        full_df = process_all_csvs()
        print(f"Found data for {len(full_df['Repo Name'].unique())} repositories")
        
        results_df = analyze_mutants(full_df)
        create_plots(results_df)
        
        print("\nAnalysis completed successfully!")
        print("Results saved in 'mutation_analysis_results' folder:")
        print("- mutation_analysis_results.csv")
        print("- mutant_composition.png")
        print("- subsumption_relationships.png")
        
        # Print summary statistics
        print("\nSummary Statistics:")
        print(f"Total repositories analyzed: {len(results_df)}")
        print(f"Average duplicate mutants: {results_df['% Duplicates'].mean():.1f}%")
        print(f"Average subsuming mutants: {results_df['% Subsuming'].mean():.1f}%")
        print(f"Average subsumed mutants: {results_df['% Subsumed'].mean():.1f}%")
        
    except Exception as e:
        print(f"\nError during analysis: {str(e)}")