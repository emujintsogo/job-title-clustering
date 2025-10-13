import os
from dotenv import load_dotenv
from openai import OpenAI
import pandas as pd
import json
from typing import Optional


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
        return self.job_postings_data.sample(n=self.n_samples)

    def build_prompt(self, job_description:str) -> str:
        prompt = f"""
Analyze the following job posting and categorize each sentence into one of these four categories:
- Marketing: Sentences that sell the company, highlight benefits, or promote the organization
- Description: Sentences describing job duties, responsibilities, and day-to-day work
- Requirements: Sentences listing qualifications, skills, experience, or education needed
- Legal: Sentences about EEO statements, compliance, disclaimers, or legal notices

Job Posting:
{job_description}

Return ONLY a JSON array where each element has:
{{"sentence": "the sentence text", "category": "Marketing|Description|Requirements|Legal"}}
"""
        return prompt

    # categorizes one job description's sentences into 4 categories
    def categorize_posting(self, posting) -> pd.DataFrame:
        prompt = self.build_prompt(posting["JobDescription"])

        try:
            response = self.client.chat.completions.create(
                model="gpt-5-nano",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that categorizes job posting sentences."},
                    {"role": "user", "content": prompt}
                ]
            )

            result = json.loads(response.choices[0].message.content)
            
            df = pd.DataFrame(result)
            df["RequisitionID"] = posting["RequisitionID"]
            df["JobTitle"] = posting["JobTitle"]
            return df

        except Exception as e:
            print(f"Error calling GPT: {e}")
            return

    # runs categorize_posting on every random posting
    # returns a dataframe where every row is a sentence and its category
    def process_random_postings(self) -> pd.DataFrame:
        random_postings = self.load_data()

        all_dfs = []
        
        for idx, posting in random_postings.iterrows():            
            if idx % 50 == 0:
                print(f"Processing posting {idx+1}/{len(random_postings)}")

            df = self.categorize_posting(posting)
            all_dfs.append(df)

        self.labeled_df = pd.concat(all_dfs, ignore_index=True)
        return self.labeled_df