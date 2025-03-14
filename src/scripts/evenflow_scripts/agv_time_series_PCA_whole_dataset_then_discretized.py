import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import KBinsDiscretizer

data_path = '../../../data'
df = pd.read_csv(data_path + '/avg_robot/DemoDataset_1Robot.csv')

# Separate features and labels
X = df.drop(columns=['goal_status'])
y = df['goal_status']

# Apply PCA
pca = PCA(n_components=5)
X_pca = pca.fit_transform(X)

# Convert PCA-transformed data into discrete values using binning
n_bins = 10
discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform')
X_discrete = discretizer.fit_transform(X_pca)

# Create a DataFrame for the discretized data
discrete_df = pd.DataFrame(X_discrete.astype(int), columns=[f'f{i}' for i in range(1, X_pca.shape[1] + 1)])
discrete_df['goal_status'] = y

# Save the discretized DataFrame to a CSV files
output_file_path = "../../../data/avg_robot/PCA_Discretized_Dataset.csv"
discrete_df.to_csv(output_file_path, index=False)

print(f"The discretized dataset has been saved to {output_file_path}")
