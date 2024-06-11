import pandas as pd
import numpy as np

# Set a random seed for reproducibility
np.random.seed(42)

# Generate synthetic data
data = {
    'Age': np.random.randint(18, 70, size=100),  # Random ages between 18 and 70
    'Salary': np.random.randint(30000, 100000, size=100),  # Random salaries between 30k and 100k
    'Education': np.random.randint(1, 4, size=100),  # Education levels 1 to 3
    'TargetColumn': np.random.randint(0, 2, size=100)  # Binary target
}

df = pd.DataFrame(data)

# Save to CSV
csv_file_path = '/mnt/data/SampleMLData.csv'
df.to_csv(csv_file_path, index=False)

csv_file_path
