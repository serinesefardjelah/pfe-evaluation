import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

# Create a folder for results if it doesn't exist
output_folder = 'mutation_results_plotted'
os.makedirs(output_folder, exist_ok=True)

# Load data from CSV file
df = pd.read_csv('all_repos_stats.csv')

# Rename columns for consistency
df = df.rename(columns={
    'Repo Name': 'repo',
    'Number of Functions': 'num_functions',
    'Alive Mutants': 'alive_mutants',
    'Mutants with Execution Errors': 'execution_errors',
    'Mutants with syntax errors': 'syntax_errors'
})

# Calculate total mutants (num_functions × 3)
df['total_mutants'] = df['num_functions'] * 3

# Calculate valid mutants (excluding execution and syntax errors)
df['valid_mutants'] = df['total_mutants'] - df['execution_errors'] - df['syntax_errors']

# Calculate killed mutants (valid mutants that were killed)
df['killed_mutants'] = df['valid_mutants'] - df['alive_mutants']

# Calculate mutation score (percentage of killed mutants)
df['mutation_score'] = (df['killed_mutants'] / df['valid_mutants']) * 100

# Handle division by zero (if no valid mutants)
df['mutation_score'] = df['mutation_score'].fillna(0)

# Sort by mutation score (descending)
df = df.sort_values(by='mutation_score', ascending=False)

# Print mutation scores
print("\nMutation Scores for All Repositories:")
print(df[['repo', 'num_functions', 'total_mutants', 'valid_mutants', 
          'killed_mutants', 'alive_mutants', 'mutation_score']].to_string())

# Save mutation scores to CSV in the folder
csv_path = os.path.join(output_folder, 'mutation_scores.csv')
df.to_csv(csv_path, index=False)
print(f"\n✅ Mutation scores saved to '{csv_path}'")

# Plot 1: Mutation scores
plt.figure(figsize=(14, 6))
bars = plt.bar(df['repo'], df['mutation_score'], color='green')

# Add value labels on top of bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.1f}%',
             ha='center', va='bottom')

plt.xticks(rotation=90)
plt.ylim(0, 100)
plt.ylabel("Mutation Score (%)")
plt.title("Mutation Score per Repository (Higher is Better)")
plt.tight_layout()

# Save plot 1 in the folder
plot1_path = os.path.join(output_folder, 'mutation_scores_plot.png')
plt.savefig(plot1_path)
plt.close()  # Close the figure to free memory
print(f"\n✅ Plot 1 saved as '{plot1_path}'")

# Plot 2: Killed vs Total mutants
plt.figure(figsize=(14, 6))
bar_width = 0.35
index = np.arange(len(df['repo']))

bars1 = plt.bar(index, df['total_mutants'], bar_width, color='blue', label='Total Mutants')
bars2 = plt.bar(index + bar_width, df['killed_mutants'], bar_width, color='red', label='Killed Mutants')

plt.xlabel('Repositories')
plt.ylabel('Number of Mutants')
plt.title('Killed Mutants vs Total Mutants per Repository')
plt.xticks(index + bar_width/2, df['repo'], rotation=90)
plt.legend()

# Add value labels on top of bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{int(height)}',
                 ha='center', va='bottom')

plt.tight_layout()

# Save plot 2 in the folder
plot2_path = os.path.join(output_folder, 'killed_vs_total_mutants.png')
plt.savefig(plot2_path)
plt.close()  # Close the figure to free memory
print(f"\n✅ Plot 2 saved as '{plot2_path}'")