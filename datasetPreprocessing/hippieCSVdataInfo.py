import pandas as pd
import numpy as np

# Load the generated CSV file
file_path = '/Users/rajan/github/hippie_protein_interactions.csv'
df = pd.read_csv(file_path)

print("="*70)
print("HIPPIE PROTEIN-PROTEIN INTERACTIONS - DATA ANALYTICS")
print("="*70)

# 1. Basic Information
print("\n1. BASIC DATASET INFORMATION")
print("-"*70)
print(f"Dataset shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"Total interactions: {len(df):,}")
print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# 2. Column Information
print("\n2. COLUMN INFORMATION")
print("-"*70)
print(df.info())

# 3. First few rows
print("\n3. FIRST 10 ROWS (HEAD)")
print("-"*70)
print(df.head(10))

# 4. Last few rows
print("\n4. LAST 10 ROWS (TAIL)")
print("-"*70)
print(df.tail(10))

# 5. Random sample
print("\n5. RANDOM SAMPLE (10 rows)")
print("-"*70)
print(df.sample(10, random_state=42))

# 6. Data types
print("\n6. DATA TYPES")
print("-"*70)
print(df.dtypes)

# 7. Missing values
print("\n7. MISSING VALUES")
print("-"*70)
missing = df.isnull().sum()
missing_percent = (df.isnull().sum() / len(df)) * 100
missing_df = pd.DataFrame({
    'Missing Count': missing,
    'Percentage': missing_percent
})
print(missing_df)

# 8. Unique values
print("\n8. UNIQUE VALUES")
print("-"*70)
print(f"Unique proteins in column 0: {df['0'].nunique():,}")
print(f"Unique proteins in column 1: {df['1'].nunique():,}")
total_unique = pd.concat([df['0'], df['1']]).nunique()
print(f"Total unique proteins (combined): {total_unique:,}")

# 9. Value counts for top proteins
print("\n9. TOP 20 MOST CONNECTED PROTEINS (Column 0)")
print("-"*70)
print(df['0'].value_counts().head(20))

print("\n10. TOP 20 MOST CONNECTED PROTEINS (Column 1)")
print("-"*70)
print(df['1'].value_counts().head(20))

# 10. Self-interactions (homo-dimers)
print("\n11. SELF-INTERACTIONS (HOMO-DIMERS)")
print("-"*70)
self_interactions = df[df['0'] == df['1']]
print(f"Number of self-interactions: {len(self_interactions):,}")
print(f"Percentage of total: {(len(self_interactions)/len(df)*100):.2f}%")
print(f"\nSample self-interactions:")
print(self_interactions.head(10))

# 11. Duplicate interactions
print("\n12. DUPLICATE INTERACTIONS")
print("-"*70)
duplicates = df.duplicated().sum()
print(f"Number of duplicate rows: {duplicates:,}")

# Check for reverse duplicates (A-B and B-A)
df_sorted = pd.DataFrame(np.sort(df.values, axis=1), columns=df.columns)
reverse_duplicates = df_sorted.duplicated().sum()
print(f"Number of reverse duplicates (A-B vs B-A): {reverse_duplicates:,}")

# 12. Statistical summary
print("\n13. INTERACTION DEGREE STATISTICS")
print("-"*70)
all_proteins = pd.concat([df['0'], df['1']])
degree_counts = all_proteins.value_counts()
print(f"Mean interactions per protein: {degree_counts.mean():.2f}")
print(f"Median interactions per protein: {degree_counts.median():.2f}")
print(f"Min interactions per protein: {degree_counts.min()}")
print(f"Max interactions per protein: {degree_counts.max()}")
print(f"Std deviation: {degree_counts.std():.2f}")

# 13. Distribution of interactions
print("\n14. INTERACTION DEGREE DISTRIBUTION")
print("-"*70)
bins = [1, 5, 10, 20, 50, 100, 500, float('inf')]
labels = ['1-4', '5-9', '10-19', '20-49', '50-99', '100-499', '500+']
degree_distribution = pd.cut(degree_counts, bins=bins, labels=labels, right=False)
print(degree_distribution.value_counts().sort_index())

# 14. Sample of highly connected proteins
print("\n15. TOP 10 HUB PROTEINS (Most Interactions)")
print("-"*70)
top_hubs = degree_counts.head(10)
for i, (protein, count) in enumerate(top_hubs.items(), 1):
    print(f"{i:2d}. {protein}: {count:,} interactions")

# 15. Summary statistics
print("\n16. SUMMARY STATISTICS")
print("-"*70)
print(f"Total rows: {len(df):,}")
print(f"Total columns: {len(df.columns)}")
print(f"Total cells: {df.size:,}")
print(f"Unique proteins: {total_unique:,}")
print(f"Self-interactions: {len(self_interactions):,}")
print(f"Duplicate rows: {duplicates:,}")
print(f"Most connected protein: {degree_counts.index[0]} ({degree_counts.iloc[0]:,} interactions)")

print("\n" + "="*70)
print("ANALYSIS COMPLETE!")
print("="*70)