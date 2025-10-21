import os
from dotenv import load_dotenv
from openai import OpenAI
import pandas as pd
import json
from typing import Optional
from tqdm import tqdm
import time


class GPTJobDescriptionCategorizer():
    def __init__(self, job_postings_data: pd.DataFrame, n_samples: int = 500):
        self.job_postings_data = job_postings_data
        self.labeled_df = None
        self.n_samples = n_samples
        self.client = self.get_API_client()

    def get_API_client(self) -> Optional[OpenAI]:
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        load_dotenv(os.path.join(base_dir, ".env"))
        openai_api_key = os.getenv("OPENAI_API_KEY")

        if not openai_api_key:
            print("Warning: OPENAI_API_KEY not found in .env file")
            return None

        return OpenAI(api_key=openai_api_key)

    def load_data(self) -> pd.DataFrame:
        return self.job_postings_data.sample(n=self.n_samples, random_state=42)

    def build_prompt(self, job_description: str) -> str:
        prompt = f"""Analyze the following job posting and categorize each sentence into one of these four categories:
- Marketing: Sentences that sell the company, highlight benefits, or promote the organization
- Description: Sentences describing job duties, responsibilities, and day-to-day work
- Requirements: Sentences listing qualifications, skills, experience, or education needed
- Legal: Sentences about EEO statements, compliance, disclaimers, or legal notices

Job Posting:
{job_description}

Return ONLY a valid JSON array where each element has:
{{"sentence": "the sentence text", "category": "Marketing|Description|Requirements|Legal"}}

Do not include any explanatory text, only the JSON array."""
        return prompt

    def categorize_posting(self, posting, max_retries: int = 3) -> Optional[pd.DataFrame]:
        """Categorizes one job description with retry logic"""
        prompt = self.build_prompt(posting["JobDescription"])
        
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4o-mini",  # Changed from gpt-5-nano
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that categorizes job posting sentences. Return only valid JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0,  # More deterministic
                    response_format={"type": "json_object"}  # Force JSON response
                )

                content = response.choices[0].message.content
                
                # Check if content is None or empty
                if not content or content.strip() == "":
                    print(f"\nEmpty response for {posting['RequisitionID']}")
                    if attempt < max_retries - 1:
                        time.sleep(2)
                        continue
                    return None
                
                # Handle wrapped JSON
                if content.startswith("```json"):
                    content = content.replace("```json", "").replace("```", "").strip()
                
                result = json.loads(content)
                
                # ----- Error handling for result begins -----
                # Handle if result is wrapped in a key
                if isinstance(result, dict):
                    if len(result) == 0:
                        print(f"\nEmpty dict result for {posting['RequisitionID']}")
                        if attempt < max_retries - 1:
                            time.sleep(2)
                            continue
                        return None
                    if len(result) == 1:
                        result = list(result.values())[0]
                
                # Ensure result is a list
                if not isinstance(result, list):
                    print(f"\nUnexpected result format for {posting['RequisitionID']}: {type(result)}")
                    if attempt < max_retries - 1:
                        time.sleep(2)
                        continue
                    return None
                
                # Check if result is empty
                if len(result) == 0:
                    print(f"\nEmpty result for {posting['RequisitionID']}")
                    if attempt < max_retries - 1:
                        time.sleep(2)
                        continue
                    return None
                
                # Ensure all items in list are dicts
                if not all(isinstance(item, dict) for item in result):
                    print(f"\nInvalid list items for {posting['RequisitionID']}")
                    if attempt < max_retries - 1:
                        time.sleep(2)
                        continue
                    return None
                
                # ----- Error handling for result ends -----

                df = pd.DataFrame(result)
                
                # Verify required columns exist
                if 'sentence' not in df.columns or 'category' not in df.columns:
                    print(f"\nMissing required columns for {posting['RequisitionID']}: {df.columns.tolist()}")
                    if attempt < max_retries - 1:
                        time.sleep(2)
                        continue
                    return None
                
                df["RequisitionID"] = posting["RequisitionID"]
                df["JobTitle"] = posting["JobTitle"]
                return df

            except json.JSONDecodeError as e:
                print(f"\nJSON decode error for {posting['RequisitionID']} (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                return None
                
            except Exception as e:
                print(f"\nError for {posting['RequisitionID']} (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                return None
        
        return None

    def process_all_postings(self, checkpoint_file: str = "checkpoint.csv") -> pd.DataFrame:
        """Process postings sequentially with a checkpoint system to save results periodically """
        random_postings = self.load_data()
        all_dfs = []
        
        # Load checkpoint if exists
        processed_ids = set()
        if os.path.exists(checkpoint_file):
            checkpoint_df = pd.read_csv(checkpoint_file)
            processed_ids = set(checkpoint_df["RequisitionID"].unique())
            all_dfs.append(checkpoint_df)
            print(f"Loaded checkpoint with {len(processed_ids)} processed postings")
        
        # Filter out already processed postings
        postings_to_process = random_postings[~random_postings["RequisitionID"].isin(processed_ids)]
        
        if len(postings_to_process) == 0:
            print("All postings already processed!")
            self.labeled_df = pd.concat(all_dfs, ignore_index=True)
            return self.labeled_df
        
        print(f"Processing {len(postings_to_process)} postings sequentially...")
        
        # Process one by one with progress bar
        failed_count = 0
        for idx, (_, posting) in enumerate(tqdm(postings_to_process.iterrows(), 
                                                  total=len(postings_to_process),
                                                  desc="Processing postings")):
            df = self.categorize_posting(posting)
            
            if df is not None:
                all_dfs.append(df)
            else:
                failed_count += 1
            
            # Save checkpoint every 50 postings
            if (idx + 1) % 50 == 0:
                temp_df = pd.concat(all_dfs, ignore_index=True)
                temp_df.to_csv(checkpoint_file, index=False)
                print(f"\nCheckpoint saved at {idx + 1} postings")
        
        # Final save
        if all_dfs:
            self.labeled_df = pd.concat(all_dfs, ignore_index=True)
            self.labeled_df.to_csv(checkpoint_file, index=False)
            
            print(f"\nProcessing complete!")
            print(f"Successfully processed: {len(all_dfs) - (1 if processed_ids else 0)}")
            print(f"Failed: {failed_count}")
            print(f"Total sentences categorized: {len(self.labeled_df)}")
        else:
            print("\nNo postings were successfully processed.")
            self.labeled_df = pd.DataFrame()
        
        return self.labeled_df