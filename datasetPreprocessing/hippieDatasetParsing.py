import pandas as pd
import gzip

# 1. Load the UniProt ID mapping file
print("Loading UniProt mapping file...")
mapping_file = '/Users/rajan/github/HUMAN_9606_idmapping.dat.gz'

# The mapping file format is: UniProtKB-AC | ID_type | ID_value
# We need to create a mapping from UniProt names to accessions
name_to_accession = {}

with gzip.open(mapping_file, 'rt') as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) >= 3:
            accession = parts[0]  # e.g., P04114
            id_type = parts[1]    # e.g., Gene_Name, UniProtKB-ID, etc.
            id_value = parts[2]   # e.g., AL1A1_HUMAN
            
            # Map UniProt entry names (ending in _HUMAN) to accessions
            if id_type == 'UniProtKB-ID':
                name_to_accession[id_value] = accession

print(f"Loaded {len(name_to_accession)} UniProt name-to-accession mappings")
print("\nSample mappings:")
for i, (name, acc) in enumerate(list(name_to_accession.items())[:5]):
    print(f"  {name} -> {acc}")

# 2. Load the MITAB file
print("\nLoading HIPPIE MITAB file...")
file_path = '/Users/rajan/github/hippie_current_mitab.txt'
df_raw = pd.read_csv(file_path, sep='\t', header=None, skiprows=1, low_memory=False)

# 3. Extract columns 2 and 3 (UniProt names)
df_hippie_protein = df_raw[[2, 3]].copy()

# 4. Clean UniProt names (remove 'uniprotkb:' prefix)
def clean_uniprot_name(column):
    return column.str.replace('uniprotkb:', '', regex=False)

df_hippie_protein[2] = clean_uniprot_name(df_hippie_protein[2])
df_hippie_protein[3] = clean_uniprot_name(df_hippie_protein[3])

# 5. Map to accession IDs
print("\nMapping UniProt names to accession IDs...")
df_hippie_protein['0'] = df_hippie_protein[2].map(name_to_accession)
df_hippie_protein['1'] = df_hippie_protein[3].map(name_to_accession)

# 6. Keep only successfully mapped rows
df_result = df_hippie_protein[['0', '1']].copy()

# Count unmapped entries
unmapped_count = df_result.isna().sum().sum()
if unmapped_count > 0:
    print(f"Warning: {unmapped_count} entries could not be mapped")
    print("Removing rows with unmapped proteins...")
    df_result = df_result.dropna()

# 7. Display results
print(f"\n=== FINAL RESULTS ===")
print(f"Total rows loaded: {len(df_result)}")
print(df_result.head(10))

# 8. Statistics
print("\n--- Data Quality Checks ---")
print(f"Unique proteins in column 0: {df_result['0'].nunique():,}")
print(f"Unique proteins in column 1: {df_result['1'].nunique():,}")

print("\nSample of unique accession IDs:")
all_proteins = pd.concat([df_result['0'], df_result['1']]).unique()
print(all_proteins[:20])

# 9. Save results
# output_path = '/Users/rajan/github/hippie_protein_interactions.csv'
# df_result.to_csv(output_path, index=False)
# print(f"\n✓ Saved to: {output_path}")