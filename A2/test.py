import torch
import torch.nn as nn
import torch.optim as optim
import gensim.downloader as api
from helper import *
import argparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def vectorize_row(rows):
    return " ".join(rows)


def preprocess_text2(text):
    # Tokenize text and remove stop words
    tokens = word_tokenize(str(text))
    tokens = [word.lower() for word in tokens]
    return " ".join(tokens)


def find_most_similar_row(question, rows):
    try:
        # Preprocess the question
        preprocessed_question = preprocess_text2(question)
        
        # Preprocess and vectorize the rows
        preprocessed_rows = [preprocess_text2(row) for row in rows]
        
        # Initialize TF-IDF vectorizer
        vectorizer = TfidfVectorizer()
        
        # Fit and transform the preprocessed rows
        tfidf_matrix = vectorizer.fit_transform(preprocessed_rows)
        
        # Transform the preprocessed question
        question_vector = vectorizer.transform([preprocessed_question])
        
        # Calculate cosine similarity between question and rows
        similarity_scores = cosine_similarity(question_vector, tfidf_matrix)
        
        # Find index of row with highest similarity
        most_similar_row_index = similarity_scores.argmax()
        
        return most_similar_row_index
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return 0


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train a language classification model.")
    parser.add_argument("--test", required=True, help="Path to the test data JSON file.")
    parser.add_argument("--pred", required=True, help="Path to the save pred data JSON file.")
    args = parser.parse_args()
    # Load test data
    input_val_data = process_column(load_data(args.test))

    # Load word vectors
    word_vectors = api.load("glove-wiki-gigaword-100")

    # Convert data to embeddings
    X_test = create_test_batches(input_val_data, 1, word_vectors)

    # Initialize model
    model = LSTMModel(input_dim=word_vectors.vector_size, hidden_dim=256, output_dim=word_vectors.vector_size)

    # Load trained model state dictionary
    model.load_state_dict(torch.load('lstm_model.pth'))
    model.eval()

    # Make predictions
    predictions = []
    with torch.no_grad():
        for X_val, y_val in X_test:
            logits = model(X_val, y_val)
            predicted_index = torch.argmax(logits, dim=1)
            predictions.extend(predicted_index.tolist())
    
    row_prediction = []

    for i in range(len(input_val_data)):
        input_sample = input_val_data[i]
        question = input_sample['question']
        row = [vectorize_row(r) for r in input_sample['rows']]
        row_i = find_most_similar_row(question, row)
        row_prediction.append(row_i)
    

    # Save predictions to a JSONL file
    with open(args.pred, 'w', encoding='utf-8') as file:
        for idx, entry in enumerate(input_val_data):
            cv = entry['columns'][int(predictions[idx])]  # Convert int64 to int
            rv = int(row_prediction[idx])  # Assuming `row_prediction` is defined somewhere
            json.dump({'label_col':[cv],'label_cell':[[rv,cv]],'label_row':[rv],'qid': entry['qid']}, file)
            file.write('\n')
