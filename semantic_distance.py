import os
import pandas as pd
import json


repos = [
       "https://github.com/Ousret/charset_normalizer",
        "https://github.com/PantherPy/pantherdb",
      #  "https://github.com/SimonGreenhill/treemaker",
        "https://github.com/aio-libs/async-lru",
        "https://github.com/akoumjian/datefinder",
        "https://github.com/alexmojaki/executing",
        "https://github.com/andialbrecht/sqlparse",
        "https://github.com/antonagestam/phantom-types",
       
        "https://github.com/cpburnz/python-pathspec",
        "https://github.com/dateutil/dateutil",
        "https://github.com/eigenein/protobuf",
        "https://github.com/facelessuser/soupsieve",
        "https://github.com/foutaise/texttable",
        "https://github.com/hukkin/tomli",
        "https://github.com/jd/tenacity",
        "https://github.com/jsh9/pydoclint",
        "https://github.com/kjd/idna",
        "https://github.com/lidatong/dataclasses-json",
       "https://github.com/magmax/python-readchar",
       
       "https://github.com/marcusbuffett/command-line-chess",
        "https://github.com/martinblech/xmltodict",
        "https://github.com/mbr/asciitree",
        "https://github.com/mgedmin/objgraph", #the repo has an issue with the tests
        #"https://github.com/microsoft/lsprotocol",
       "https://github.com/mkorpela/overrides",
     
      "https://github.com/msiemens/tinydb",
       "https://github.com/pydata/patsy",
       "https://github.com/pygments/pygments",
       "https://github.com/pytoolz/toolz",
        "https://github.com/sybrenstuvel/python-rsa", 
        "https://git.launchpad.net/beautifulsoup",
        "https://github.com/serge-sans-paille/beniget",
       
        "https://github.com/dgasmith/opt_einsum",
        "https://github.com/TheAlgorithms/Python",
        "https://github.com/casbin/pycasbin",
        "https://github.com/graphql-python/graphql-core",
        "https://github.com/mahmoud/boltons",
        "https://github.com/more-itertools/more-itertools",
       # "https://github.com/mozilla/bleach",

        #No test directory found 
    ]



def write_jsonl(file_path, data):
    """Write data to a JSONL file"""
    with open(file_path, 'w') as f:
        for line in data:
            f.write(json.dumps(line) + '\n')


def read_jsonl_lines(file_path):
    """Read JSONL file and return the lines as a list of dictionaries"""
    with open(file_path, 'r') as f:
        return [json.loads(line) for line in f.readlines()]


def calculate_mutant_distance(mutants):
    """
    Calculate the distance of mutants based on their failing tests.
    Formula: distance = 1 - (number of failing tests / total tests)
    """
    distance_data = []
    for mutant in mutants:
        total_tests = mutant['Passed Count'] + mutant['Failed Count']
        if total_tests == 0: 
            distance = 0
        else:
            distance = 1 - mutant['Failed Count'] / total_tests
        distance_data.append({
            "Repo Name": mutant['Repo Name'],
            "Function Name": mutant['Function Name'],
            "Mutant Index": mutant['Mutant Index'],
            "distance": distance
        })
    return distance_data


def report_distance_from_csv(file_path):
    """Process CSV file and report the distance of mutants."""
    df = pd.read_csv(file_path)

    df['Passed Count'] = pd.to_numeric(df['Passed Count'], errors='coerce')
    df['Failed Count'] = pd.to_numeric(df['Failed Count'], errors='coerce')
    
    # Ensure there are no NaN values 
    df['Passed Count'] = df['Passed Count'].fillna(0)
    df['Failed Count'] = df['Failed Count'].fillna(0)
    # Add a distance column
    df['total_tests'] = df['Passed Count'] + df['Failed Count']
    df['distance'] = 1 - df['Failed Count'] / df['total_tests']
    df['distance'] = df['distance'].fillna(0)

    # Optional: Save individual distances to JSONL per repo
    os.makedirs("distance_results", exist_ok=True)
    for repo_name, group in df.groupby("Repo Name"):
        output = group[["Repo Name", "Function Name", "Mutant Index", "distance"]].to_dict(orient="records")
        write_jsonl(f"distance_results/{repo_name}_distances.jsonl", output)

    # Compute mean distance per repo
    mean_distances = df.groupby("Repo Name")['distance'].mean().reset_index()
    mean_distances.columns = ['Repo Name', 'Mean Distance']
    return mean_distances


def report_distance(files_list):
    """Calculate the distance for mutants across multiple files."""
    mean_distance = []
    for file in files_list:
        overall_distances = []

        # Read mutant data from the file
        mutants = read_jsonl_lines(file)

        for mutant in mutants:
            # Calculate distance for each mutant
            total_tests = mutant['Passed Count'] + mutant['Failed Count']
            if total_tests == 0:
                distance = 0
            else:
                distance = 1 - mutant['Failed Count'] / total_tests

            overall_distances.append({
                "Repo Name": mutant['Repo Name'],
                "Function Name": mutant['Function Name'],
                "Mutant Index": mutant['Mutant Index'],
                "distance": distance
            })

        # Write the calculated distances to JSONL
        write_jsonl(f"distance_results/{file.split('/')[-2]}.jsonl", overall_distances)

        df_overall = pd.DataFrame(overall_distances)
        mean_distance.append({"file": file.split('/')[-2], "mean_distance : ": df_overall['distance'].mean()})

    return mean_distance


def process_repositories(repos):
    """Loop through the list of repositories and process their mutation results."""
    for repo_url in repos:
        # Extract the repo name from the URL (e.g., "charset_normalizer" from "https://github.com/Ousret/charset_normalizer")
        repo_name = repo_url.split('/')[-1]

        # Set the path to the mutation results CSV file
        mutation_results_path = f"./repo_stats/{repo_name}mutation_results.csv"

        # Compute the distances from CSV for the current repository
        mean_df = report_distance_from_csv(mutation_results_path)
        print(f"Mean Distances for {repo_name}:")
        print(mean_df)

        # Optional: Merge with `all_repos_stats.csv`
        stats_df = pd.read_csv("all_repos_stats.csv")
        merged = stats_df.merge(mean_df, on="Repo Name", how="left")
        merged.to_csv(f"enhanced_repo_stats_{repo_name}.csv", index=False)


if __name__ == "__main__":
    

    # Process each repository in the repos list
    process_repositories(repos)
