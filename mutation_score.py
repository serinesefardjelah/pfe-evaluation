import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for matplotlib

import matplotlib.pyplot as plt
import pandas as pd

# Load data from CSV file
df = pd.read_csv('all_repos_stats.csv')

# Rename columns to match your script's expectations
df = df.rename(columns={
    'Repo Name': 'repo',
    'Number of Functions': 'num_functions',
    'Alive Mutants': 'alive_mutants',
    'Mutants with Execution Errors': 'execution_errors',
    'Mutants with syntax errors': 'syntax_errors'
})

# Calculate total mutants (num_functions * 3)
df['total_mutants'] = df['num_functions'] * 3

# Sort by total mutants for better comparison
df = df.sort_values(by='total_mutants', ascending=False)

# Plot setup
plt.figure(figsize=(16, 6))
bar_width = 0.4
x = range(len(df))

# Bar plots
plt.bar(x, df['total_mutants'], width=bar_width, label='Total Mutants (num_functions × 3)', color='lightgray')
plt.bar([i + bar_width for i in x], df['alive_mutants'], width=bar_width, label='Alive Mutants', color='skyblue')

# Labeling
plt.xticks([i + bar_width / 2 for i in x], df['repo'], rotation=90)
plt.xlabel("Repository")
plt.ylabel("Number of Mutants")
plt.title("Alive Mutants vs Total Mutants per Repository")
plt.legend()
plt.tight_layout()

# Save plot
plt.savefig("alive_vs_total_mutants.png")
print("✅ Plot saved as 'alive_vs_total_mutants.png'")

# Calculate mutation score
df['mutation_score'] = df.apply(
    lambda row: 1 - (row['alive_mutants'] / row['total_mutants']) if row['total_mutants'] > 0 else 0,
    axis=1
)

# Mutation score plot
plt.figure(figsize=(14, 6))
plt.bar(df['repo'], df['mutation_score'], color='green')
plt.xticks(rotation=90)
plt.ylim(0, 1)
plt.ylabel("Mutation Score")
plt.title("Mutation Score per Repository")
plt.tight_layout()
plt.savefig("mutation_score_plot.png")
print("✅ Mutation score plot saved as 'mutation_score_plot.png'")

# Print the data
print(df[['repo', 'num_functions', 'alive_mutants', 'total_mutants', 'mutation_score']])