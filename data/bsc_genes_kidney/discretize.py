import pandas as pd
import numpy as np

file_path = 'lstm_train_set_only_partial_negative_trajectories.csv'
data = pd.read_csv(file_path)

# List of genes to retain, converted to lowercase for ASP
genes_of_interest = [
    "HUS1B", "SLC22A1", "LOC100132354", "SLC22A16", "ABCA12", "C10orf41",
    "C4orf6", "C9orf129", "CD1C", "TRIM36", "AFAP1L1", "C6orf176", "CABYR",
    "CCDC146", "CES8"
]

genes_of_interest_lower = [gene.lower() for gene in genes_of_interest]

# Extract relevant columns and lowercase gene names
columns_to_extract = ['source', 'target', 'index', 'stage'] + genes_of_interest
data_subset = data[columns_to_extract]
data_subset.columns = ['source', 'target', 'index', 'stage'] + genes_of_interest_lower

# Separate trajectories based on the "source" and "target"
trajectories = data_subset.groupby(['source', 'target'])


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


# Parameters for SAX
n_segments = 10  # Number of segments for PAA
n_bins = 10  # Alphabet size for SAX symbols

# Calculate breakpoints globally
global_breakpoints = calculate_breakpoints(n_bins)

# Container to store discretized trajectories
sax_trajectories = []

# Process each trajectory
for (_, trajectory) in trajectories:
    # Extract gene data and Z-normalize
    gene_data = trajectory[genes_of_interest_lower].to_numpy()
    normalized_data = z_normalize(gene_data)

    # Apply PAA transformation
    paa_data = paa_transform(normalized_data, n_segments)

    # Assign symbols using SAX breakpoints
    discretized_data = assign_symbols(paa_data, global_breakpoints)

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
        global_breakpoints[i - 1] if i > 0 else -np.inf,
        global_breakpoints[i] if i < len(global_breakpoints) else np.inf
    )
    for i in range(n_bins)
}

# Save the discretized dataset
output_path_sax = 'sax_discretized_gene_trajectories.csv'
final_sax_data.to_csv(output_path_sax, index=False)

# Save the bin ranges
bin_ranges_path = 'sax_bin_ranges.csv'
pd.DataFrame.from_dict(bin_ranges, orient='index', columns=['lower_bound', 'upper_bound']).to_csv(bin_ranges_path)

