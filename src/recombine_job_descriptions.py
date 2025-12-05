import pandas as pd
import os
from config import BASE_DIR

def recombine_job_descriptions(classified_csv_path: str, original_csv_path: str, output_csv_path: str):
    """
    Recombine job description sentences that are labeled as 'Description' or 'Requirements'
    back into single job descriptions per RequisitionID.
    
    Args:
        classified_csv_path: Path to the classified sentences CSV (e.g., gpt_classified_job_descriptions.csv)
        original_csv_path: Path to the original Engineer CSV
        output_csv_path: Path to save the recombined output
    """
    print(f"Loading classified data from {classified_csv_path}...")
    df_classified = pd.read_csv(classified_csv_path)
    
    print(f"Loading original data from {original_csv_path}...")
    df_original = pd.read_csv(original_csv_path)
    
    print(f"Total classified sentences: {len(df_classified)}")
    print(f"Label distribution:\n{df_classified['category'].value_counts()}\n")
    
    # Filter for only Description and Requirements
    df_filtered = df_classified[df_classified['category'].isin(['Description', 'Requirements'])].copy()
    print(f"Sentences after filtering for Description/Requirements: {len(df_filtered)}")
    
    # Group by RequisitionID and combine sentences
    print("\nRecombining sentences by RequisitionID...")
    recombined = df_filtered.groupby('RequisitionID').agg({
        'sentence': lambda x: ' '.join(str(s) for s in x if pd.notna(s))  # Convert to str and filter out NaN
    }).reset_index()
    
    recombined.columns = ['RequisitionID', 'description']
    
    # Merge with original data to get OrigJobTitle
    print("Merging with original data to get OrigJobTitle...")
    # Create a mapping from original data
    original_mapping = df_original[['RequisitionID', 'OrigJobTitle']].drop_duplicates()
    original_mapping.columns = ['RequisitionID', 'OrigJobTitle']
    
    # Merge
    final_df = recombined.merge(original_mapping, on='RequisitionID', how='left')
    
    # Reorder columns
    final_df = final_df[['description', 'RequisitionID', 'OrigJobTitle']]
    
    # Save to CSV
    print(f"\nSaving recombined data to {output_csv_path}...")
    final_df.to_csv(output_csv_path, index=False)
    
    print(f"Done! Recombined {len(final_df)} job descriptions")
    print(f"\nSample of output:")
    print(final_df.head(2))
    
    return final_df

if __name__ == "__main__":
    # File paths    
    original_csv_path = os.path.join(BASE_DIR, "data", "Engineer_20230826.csv")
    output_csv_path = os.path.join(BASE_DIR, "data", "all_prepared_job_postings.csv")
    gpt_classified_csv_path = os.path.join(BASE_DIR, "data", "gpt_classified_job_descriptions.csv")
    model_classified_csv_path = os.path.join(BASE_DIR, "data", "model_classified_job_descriptions.csv")

    df_gpt = recombine_job_descriptions(
        classified_csv_path=gpt_classified_csv_path,
        original_csv_path=original_csv_path,
        output_csv_path=None  # Don't save individual file
    )

    df_model = recombine_job_descriptions(
        classified_csv_path=model_classified_csv_path,
        original_csv_path=original_csv_path,
        output_csv_path=None  # Don't save individual file
    )

    df_combined = pd.concat([df_gpt, df_model], ignore_index=True)
    df_combined = df_combined.drop_duplicates(subset=['RequisitionID'], keep='first')
    
    # Save combined output
    print(f"\nSaving combined data to {output_csv_path}...")
    df_combined.to_csv(output_csv_path, index=False)
    
    print(f"\nDone! Combined file contains {len(df_combined)} job descriptions")
    print(f"  - From GPT classified: {len(df_gpt)}")
    print(f"  - From Model classified: {len(df_model)}")