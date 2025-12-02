import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import pickle
import re
from preprocessing import clean_text



def prepare_features(df: pd.DataFrame, text_column: str, tfidf=None, fit=True):
    """Prepare TF-IDF feature matrix from text"""
    # Clean text
    df['clean_text'] = df[text_column].apply(clean_text)
    
    # TF-IDF features
    if fit:
        tfidf = TfidfVectorizer(max_features=200, ngram_range=(1, 2))
        X = tfidf.fit_transform(df['clean_text']).toarray()
        return X, tfidf
    else:
        X = tfidf.transform(df['clean_text']).toarray()
        return X

def train_and_evaluate_models(X_train, X_val, y_train, y_val, label_encoder):
    """Train multiple models and compare their performance"""
    
    models = {
        'Logistic Regression': LogisticRegression(
            max_iter=1000,
            random_state=42,
            multi_class='multinomial'
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            random_state=42,
            n_jobs=-1
        ),
        'XGBoost': XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            eval_metric='mlogloss',
            objective='multi:softmax',
            num_class=len(label_encoder.classes_)
        )
    }
    
    results = {}
    
    print("\n" + "="*70)
    print("TRAINING AND EVALUATING MODELS")
    print("="*70)
    
    for name, model in models.items():
        print(f"\n{name}:")
        print("-" * 50)
        
        # Train
        print("Training...")
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_val)
        
        # Decode predictions
        y_val_decoded = label_encoder.inverse_transform(y_val)
        y_pred_decoded = label_encoder.inverse_transform(y_pred)
        
        # Calculate metrics
        accuracy = accuracy_score(y_val, y_pred)
        f1_weighted = f1_score(y_val, y_pred, average='weighted')
        f1_macro = f1_score(y_val, y_pred, average='macro')
        
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'f1_weighted': f1_weighted,
            'f1_macro': f1_macro,
            'y_pred_decoded': y_pred_decoded,
            'y_val_decoded': y_val_decoded
        }
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1 Score (Weighted): {f1_weighted:.4f}")
        print(f"F1 Score (Macro): {f1_macro:.4f}")
        
        print("\nClassification Report:")
        print(classification_report(y_val_decoded, y_pred_decoded))
    
    return results

def main():
    print("Loading labeled data...")
    # Load the GPT-classified dataset (our labeled training data)
    df_labeled = pd.read_csv('../data/gpt_classified_job_descriptions.csv')
    
    print(f"Total labeled records: {len(df_labeled)}")
    print(f"\nLabel distribution:")
    print(df_labeled['category'].value_counts())
    
    # Prepare features from labeled data (using the 'sentence' column)
    print("\nPreparing features...")
    X, tfidf = prepare_features(df_labeled, text_column='sentence', fit=True)
    
    # Encode labels as numbers for models
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(df_labeled['category'])
    
    print(f"\nLabel mapping:")
    for i, label in enumerate(label_encoder.classes_):
        print(f"  {i}: {label}")
    
    # Split labeled data into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    print(f"\nTraining set size: {len(X_train)}")
    print(f"Validation set size: {len(X_val)}")
    
    # Train and evaluate all models
    results = train_and_evaluate_models(X_train, X_val, y_train, y_val, label_encoder)
    
    # Find best model
    print("\n" + "="*70)
    print("MODEL COMPARISON SUMMARY")
    print("="*70)
    print(f"\n{'Model':<25} {'Accuracy':<12} {'F1 (Weighted)':<15} {'F1 (Macro)':<12}")
    print("-" * 70)
    
    best_model_name = None
    best_f1_weighted = 0
    
    for name, result in results.items():
        print(f"{name:<25} {result['accuracy']:<12.4f} {result['f1_weighted']:<15.4f} {result['f1_macro']:<12.4f}")
        if result['f1_weighted'] > best_f1_weighted:
            best_f1_weighted = result['f1_weighted']
            best_model_name = name
    
    print("\n" + "="*70)
    print(f"BEST MODEL: {best_model_name} (F1 Weighted: {best_f1_weighted:.4f})")
    print("="*70)
    
    # Save the best model
    best_model = results[best_model_name]['model']
    
    print(f"\nSaving best model ({best_model_name}), vectorizer, and label encoder...")
    with open('../models/best_classifier.pkl', 'wb') as f:
        pickle.dump(best_model, f)
    with open('../models/tfidf_vectorizer.pkl', 'wb') as f:
        pickle.dump(tfidf, f)
    with open('../models/label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
    
    # Save model metadata
    with open('../models/model_info.txt', 'w') as f:
        f.write(f"Best Model: {best_model_name}\n")
        f.write(f"Accuracy: {results[best_model_name]['accuracy']:.4f}\n")
        f.write(f"F1 Score (Weighted): {results[best_model_name]['f1_weighted']:.4f}\n")
        f.write(f"F1 Score (Macro): {results[best_model_name]['f1_macro']:.4f}\n")
    
    print("Best model saved to ../models/best_classifier.pkl")
    print("Vectorizer saved to ../models/tfidf_vectorizer.pkl")
    print("Label encoder saved to ../models/label_encoder.pkl")
    print("Model info saved to ../models/model_info.txt")
    
    # Show confusion matrix for best model
    print(f"\nConfusion Matrix for {best_model_name}:")
    print(confusion_matrix(
        results[best_model_name]['y_val_decoded'], 
        results[best_model_name]['y_pred_decoded']
    ))
    
    # Now predict on the full unlabeled dataset
    print("\n" + "="*70)
    print("PREDICTING ON FULL DATASET")
    print("="*70)
    df_full = pd.read_csv('../Engineer_20230826.csv')
    print(f"Full dataset size: {len(df_full)} job postings")
    
    # Exclude RequisitionIDs that were already classified by GPT
    gpt_requisition_ids = set(df_labeled['RequisitionID'].unique())
    print(f"RequisitionIDs already classified by GPT: {len(gpt_requisition_ids)}")
    
    df_full = df_full[~df_full['RequisitionID'].isin(gpt_requisition_ids)]
    print(f"Remaining job postings to classify: {len(df_full)}")
    
    if len(df_full) == 0:
        print("\nNo new job postings to classify. All have been processed by GPT.")
        return
    
    # Preprocess job descriptions
    print("\nPreprocessing job descriptions...")
    df_full['JobDescription'] = df_full['JobDescription'].apply(clean_text)
    
    # Split each job description into sentences
    print("Splitting job descriptions into sentences...")
    all_sentences = []
    
    for _, row in df_full.iterrows():
        job_desc = str(row['JobDescription'])
        # Simple sentence splitting (split by period, question mark, exclamation)
        sentences = re.split(r'[.!?]+', job_desc)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        for sentence in sentences:
            all_sentences.append({
                'RequisitionID': row['RequisitionID'],
                'JobTitle': row['JobTitle'],
                'sentence': sentence
            })
    
    sentences_df = pd.DataFrame(all_sentences)
    print(f"Total sentences to classify: {len(sentences_df)}")
    
    # Predict category for each sentence
    print("\nClassifying sentences...")
    X_sentences = prepare_features(sentences_df, text_column='sentence', tfidf=tfidf, fit=False)
    predictions = best_model.predict(X_sentences)
    
    # Decode predictions back to original category names
    predictions_decoded = label_encoder.inverse_transform(predictions)
    
    # Create output dataframe matching gpt_classified format
    predictions_df = pd.DataFrame({
        'sentence': sentences_df['sentence'],
        'category': predictions_decoded,
        'RequisitionID': sentences_df['RequisitionID'],
        'JobTitle': sentences_df['JobTitle']
    })
    
    # Save predictions to new file
    output_path = '../data/model_classified_job_descriptions.csv'
    predictions_df.to_csv(output_path, index=False)
    
    print(f"\nPredictions saved to {output_path}")
    print(f"Format: sentence | category | RequisitionID | JobTitle")
    print("\nPrediction distribution:")
    print(predictions_df['category'].value_counts())
    print(f"\nPercentages:")
    print(predictions_df['category'].value_counts(normalize=True) * 100)

if __name__ == "__main__":
    main()