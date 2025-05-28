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
df['total_mutants'] = df['num_functions'] * 5*3

# Calculate percentage of syntax errors
df['syntax_error_percentage'] = df.apply(
    lambda row: (row['syntax_errors'] / row['total_mutants'] * 100) if row['total_mutants'] > 0 else 0,
    axis=1
)

# Sort by total mutants for better comparison
df = df.sort_values(by='total_mutants', ascending=False)

# Plot: Total vs Alive Mutants
plt.figure(figsize=(16, 6))
bar_width = 0.4
x = range(len(df))

plt.bar(x, df['total_mutants'], width=bar_width, label='Total Mutants (num_functions × 3)', color='lightgray')
plt.bar([i + bar_width for i in x], df['alive_mutants'], width=bar_width, label='Alive Mutants', color='skyblue')

plt.xticks([i + bar_width / 2 for i in x], df['repo'], rotation=90)
plt.xlabel("Repository")
plt.ylabel("Number of Mutants")
plt.title("Alive Mutants vs Total Mutants per Repository")
plt.legend()
plt.tight_layout()
plt.savefig("alive_vs_total_mutants.png")
print("✅ Plot saved as 'alive_vs_total_mutants.png'")

# Calculate mutation score
# Correct mutation score calculation
df['mutation_score'] = df.apply(
    lambda row: ((row['total_mutants'] - row['alive_mutants']) / row['total_mutants']) * 100
    if row['total_mutants'] > 0 else 0,
    axis=1
)


# Plot: Mutation Score
plt.figure(figsize=(14, 6))
plt.bar(df['repo'], df['mutation_score'], color='green')
plt.xticks(rotation=90)
plt.ylim(0, 1)
plt.ylabel("Mutation Score")
plt.title("Mutation Score per Repository")
plt.tight_layout()
plt.savefig("mutation_score_plot.png")
print("✅ Mutation score plot saved as 'mutation_score_plot.png'")

# Plot: Syntax Error Percentage
plt.figure(figsize=(14, 6))
plt.bar(df['repo'], df['syntax_error_percentage'], color='red')
plt.xticks(rotation=90)
plt.ylabel("Syntax Error Percentage (%)")
plt.title("Percentage of Syntax Errors per Repository")
plt.tight_layout()
plt.savefig("syntax_error_percentage_plot.png")
print("✅ Syntax error percentage plot saved as 'syntax_error_percentage_plot.png'")

# Print data summary
print(df[['repo', 'num_functions', 'alive_mutants', 'total_mutants', 'syntax_errors', 'syntax_error_percentage', 'mutation_score']])

# Print average syntax error percentage across all repositories
average_syntax_error = df['syntax_error_percentage'].mean()
print(f"\n📊 Average Syntax Error Percentage Across All Repositories: {average_syntax_error:.2f}%")

# Print average mutation score across all repositories
average_mutation_score = df['mutation_score'].mean()
print(f"📈 Average Mutation Score Across All Repositories: {average_mutation_score:.2f}")
