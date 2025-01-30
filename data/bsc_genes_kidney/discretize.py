import numpy as np
import pandas as pd


def z_normalize(data):
    """Z-normalize the input data to have zero mean and unit variance."""
    return (data - np.mean(data, axis=0)) / np.std(data, axis=0)


def paa_transform(data, segments):
    """Reduce the dimensionality of the data using Piecewise Aggregate Approximation (PAA)."""
    n = len(data)
    reshaped = np.array_split(data, segments)
    paa = np.array([np.mean(segment, axis=0) for segment in reshaped])
    return paa


def calculate_breakpoints(n_bins):
    """Calculate breakpoints for SAX symbols assuming a standard normal distribution."""
    return np.percentile(np.random.normal(0, 1, 100000), np.linspace(0, 100, n_bins + 1)[1:-1])


def assign_symbols(data, breakpoints):
    """Assign symbols to data based on breakpoints."""
    symbols = np.digitize(data, breakpoints, right=True)
    return symbols


def convert_to_logical_format(dataset, output_path, label_column="stage"):
    """Convert the dataset into a logical format as specified."""
    logical_lines = []
    trajectory_groups = dataset.groupby(["source", "target"])

    # Generate a unique trajectory ID
    traj_id = 0

    for (_, trajectory) in trajectory_groups:
        # Get label for the trajectory (1 for positive, 0 for negative)
        label = 1 if trajectory[label_column].iloc[0] == "positive" else 0

        # Iterate through features
        for gene in genes_of_interest_lower:
            gene_lines = []
            for i, value in enumerate(trajectory[gene].to_numpy(), start=1):  # time points start from 1
                gene_lines.append(f"seq({traj_id},{gene}({int(value)}),{i}).")
            gene_line = " ".join(gene_lines)
            # Append class label fact at the end of the feature facts
            gene_line += f" class({traj_id},{label})."
            logical_lines.append(gene_line)

        # Separate each trajectory by a blank line
        logical_lines.append("")
        traj_id += 1

    # Save the logical format to a file
    with open(output_path, "w") as file:
        file.write("\n".join(logical_lines).strip())


def discretize_dataset(n_segments, n_bins, in_path, out_path, asp_path):
    data = pd.read_csv(in_path)

    # Extract relevant columns and lowercase gene names
    columns_to_extract = ['source', 'target', 'index', 'stage'] + genes_of_interest
    data_subset = data[columns_to_extract]
    data_subset.columns = ['source', 'target', 'index', 'stage'] + genes_of_interest_lower

    # Separate trajectories based on the "source" and "target"
    trajectories = data_subset.groupby(['source', 'target'])
    breakpoints = calculate_breakpoints(n_bins)
    sax_trajectories = []

    # Process each trajectory
    for (_, trajectory) in trajectories:
        # Extract gene data and Z-normalize
        gene_data = trajectory[genes_of_interest_lower].to_numpy()
        normalized_data = z_normalize(gene_data)

        # Apply PAA transformation
        paa_data = paa_transform(normalized_data, n_segments)

        # Assign symbols using SAX breakpoints
        discretized_data = assign_symbols(paa_data, breakpoints)

        # Combine with metadata
        metadata = trajectory[['source', 'target', 'index', 'stage']].iloc[:n_segments].reset_index(drop=True)
        discretized_df = pd.DataFrame(discretized_data, columns=genes_of_interest_lower)
        combined = pd.concat([metadata, discretized_df], axis=1)
        sax_trajectories.append(combined)

    # Combine all trajectories into a single DataFrame
    final_sax_data = pd.concat(sax_trajectories).reset_index(drop=True)

    # Generate bin ranges corresponding to each symbol
    bin_ranges = {
        f"symbol_{i}": (
            breakpoints[i - 1] if i > 0 else -np.inf,
            breakpoints[i] if i < len(breakpoints) else np.inf
        )
        for i in range(n_bins)
    }

    print(f'Bin ranges for {in_path}:\n{bin_ranges}')

    output_path_sax = out_path
    final_sax_data.to_csv(output_path_sax, index=False)

    print(f'Converting {out_path} to ASP format...')
    convert_to_logical_format(final_sax_data, asp_path)


if __name__ == '__main__':
    train_set_path = 'lstm_train_set_only_partial_negative_trajectories.csv'
    test_set_path = 'lstm_test_set_only_partial_negative_trajectories.csv'
    train_set_out_path = 'train_discrete.csv'
    test_set_out_path = 'test_discrete.csv'
    train_asp_path = 'folds/fold_0/train.csv'
    test_asp_path = 'folds/fold_0/test.csv'

    # List of genes to retain, converted to lowercase
    genes_of_interest = [
        "HUS1B", "SLC22A1", "LOC100132354", "SLC22A16", "ABCA12", "C10orf41",
        "C4orf6", "C9orf129", "CD1C", "TRIM36", "AFAP1L1", "C6orf176", "CABYR",
        "CCDC146", "CES8"
    ]
    genes_of_interest_lower = [gene.lower() for gene in genes_of_interest]

    # Parameters for SAX
    n_segments = 10  # Number of segments for PAA
    n_bins = 10  # Alphabet size for SAX symbols

    discretize_dataset(n_segments, n_bins, train_set_path, train_set_out_path, train_asp_path)
    discretize_dataset(n_segments, n_bins, test_set_path, test_set_out_path, test_asp_path)

    # Save the bin ranges
    # bin_ranges_path = 'sax_bin_ranges.csv'
    # pd.DataFrame.from_dict(bin_ranges, orient='index', columns=['lower_bound', 'upper_bound']).to_csv(bin_ranges_path)
