import os
import glob
import pandas as pd
from itertools import combinations
import matplotlib
matplotlib.use('Agg')  # Non-GUI backend
import matplotlib.pyplot as plt
    
def detect_method(repo_name):
    #khalina menha
    return "LLM"

def process_all_csvs(directory="repo_stats_with_code"):
    all_files = glob.glob(os.path.join(directory, "*_mutation_results.csv"))
    all_data = []

    for file_path in all_files:
        df = pd.read_csv(file_path)

        # Ensure numeric columns are treated correctly
        df['Failed Count'] = pd.to_numeric(df['Failed Count'], errors='coerce').fillna(0).astype(int)
        df['Passed Count'] = pd.to_numeric(df['Passed Count'], errors='coerce').fillna(0).astype(int)

        df['Repo Name'] = os.path.basename(file_path).replace("_mutation_results.csv", "")
        all_data.append(df)

    combined_df = pd.concat(all_data, ignore_index=True)

    # Create unique identifier for each function
    combined_df['Function ID'] = combined_df['Repo Name'] + "/" + combined_df['Function Name']
    combined_df['Method'] = combined_df['Repo Name'].apply(detect_method)

    return combined_df

def get_test_failures(row):
    failed_tests = row['Failed Tests']
    if pd.isna(failed_tests):
        return set()
    return set(failed_tests.strip().split(';'))

def is_subset(row1, row2):
    return get_test_failures(row1).issubset(get_test_failures(row2)) and row1.name != row2.name

def analyze_subsumption():
    full_df = process_all_csvs()

    # Filter only mutants that were killed
    killed_df = full_df[full_df['Failed Count'] > 0]

    contrib_records = []
    minimal_records = []

    for function_id, group in killed_df.groupby("Function ID"):
        mutants = group.reset_index(drop=True)
        test_sets = [get_test_failures(row) for _, row in mutants.iterrows()]
        minimal_set = []

        for i, current_set in enumerate(test_sets):
            is_subsumed = False
            for j, other_set in enumerate(test_sets):
                if i != j and current_set.issubset(other_set):
                    is_subsumed = True
                    break
            if not is_subsumed:
                minimal_set.append(i)

        for i in minimal_set:
            minimal_records.append(mutants.iloc[i])

        for i in range(len(mutants)):
            contrib_records.append(mutants.iloc[i])

    contrib_df = pd.DataFrame(contrib_records)
    minimal_df = pd.DataFrame(minimal_records)

    return contrib_df, minimal_df



if __name__ == "__main__":
    contrib_df, minimal_df = analyze_subsumption()

    contrib_df.to_csv("subsumption_contributing_set.csv", index=False)
    minimal_df.to_csv("subsumption_minimal_set.csv", index=False)

    print(f"Total killed mutants: {len(contrib_df)}")
    print(f"Minimal test set size: {len(minimal_df)}")

    # 🟩 Plotting section (move it here!)
   

    # Count killed and minimal per repo
    killed_counts = contrib_df.groupby("Repo Name").size()
    minimal_counts = minimal_df.groupby("Repo Name").size()

    # Combine into DataFrame
    subsumption_df = pd.DataFrame({
        "Total Killed": killed_counts,
        "Subsuming (Minimal)": minimal_counts
    })
    subsumption_df["Subsumed"] = subsumption_df["Total Killed"] - subsumption_df["Subsuming (Minimal)"]

    # Percentages
    subsumption_df["% Subsuming"] = subsumption_df["Subsuming (Minimal)"] / subsumption_df["Total Killed"]
    subsumption_df["% Subsumed"] = subsumption_df["Subsumed"] / subsumption_df["Total Killed"]

    # Sort
    subsumption_df = subsumption_df.sort_values(by="Total Killed", ascending=False)

    # Plot
    plt.figure(figsize=(14, 6))
    plt.bar(subsumption_df.index, subsumption_df["% Subsumed"], color="lightgrey", label="Subsumed")
    plt.bar(subsumption_df.index, subsumption_df["% Subsuming"], bottom=subsumption_df["% Subsumed"], color="dimgray", label="Subsuming (Minimal)")

    plt.xticks(rotation=90)
    plt.ylim(0, 1)
    plt.ylabel("Proportion of Killed Mutants")
    plt.title("Proportion of Subsumed and Subsuming Mutants per Repository")
    plt.legend()
    plt.tight_layout()
    plt.savefig("subsumption_plot.png")
    print("✅ Subsumption plot saved as 'subsumption_plot.png'")
