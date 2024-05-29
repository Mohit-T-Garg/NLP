import json
import numpy as np
import torch
import torch.nn as nn
from nltk.tokenize import word_tokenize

# Function to load data from JSONL file
def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = [json.loads(line) for line in file]
    return data

def separate_input_output(data):
    input_data = []
    output_data = []
    for entry in data:
        input_data.append({'question': entry['question'], 'rows' : entry['table']['rows'], 'columns': entry['table']['cols'], 'qid': entry['qid']})
        output_data.append({'label_col': entry['label_col'], 'label_row' : entry['label_row'], 'qid': entry['qid']})
    return input_data, output_data

def process_column(data):
    input_data = []
    for entry in data:
        input_data.append({'question': entry['question'], 'rows' : entry['table']['rows'], 'columns': entry['table']['cols'], 'qid': entry['qid']})
    return input_data


def process_column2(data):
    output_data = []
    for entry in data:
        output_data.append({'label_col': entry['label_col'], 'label_row' : entry['label_row'], 'qid': entry['qid']})
    return output_data


# Function to preprocess text
def preprocess_text(text):
    text = str(text)
    tokens = word_tokenize(text)
    tokens = [word.lower() for word in tokens]
    return tokens

# Function to compute character-level embedding for a word
def char_level_embedding(word, word_vectors):
    emb_size = word_vectors.vector_size
    emb = np.zeros(emb_size)
    for char in word:
        if char in word_vectors:
            emb += word_vectors[char]
    return emb

# Function to compute averaged embedding for a list of tokens
def averaged_embedding(tokens, word_vectors):
    embeddings = []
    for word in tokens:
        if word in word_vectors.key_to_index:
            embeddings.append(word_vectors.get_vector(word))
        else:
            embeddings.append(char_level_embedding(word, word_vectors))
    return np.mean(embeddings, axis=0)

# Function to convert data to embeddings
# Function to convert data to embeddings
def convert_data_to_embeddings(input_train_data, output_train_data, word_vectors):
    input_embeddings = []
    col_embeddings = []
    labels_value = []
    data_size = len(input_train_data)
    max_question_len = max(len(preprocess_text(entry['question'])) for entry in input_train_data)
    max_col_len = max(len(entry['columns']) for entry in input_train_data)

    for i in range(data_size):
        input_sample = input_train_data[i]
        output_sample = output_train_data[i]
        # Preprocess and convert text to embeddings
        question = preprocess_text(input_sample['question'])
        question_emb = []


        for word in question:
            if word in word_vectors.key_to_index:
                # If word is in vocabulary, use its embedding
                question_emb.append(word_vectors.get_vector(word))
            else:
                # If word is unknown, use the learnable unknown embedding
                question_emb.append(char_level_embedding(word, word_vectors))
        
        pad_length = max_question_len - len(question)
        
        for idx in range(pad_length):
            question_emb.append(np.zeros(word_vectors.vector_size))
            
        col_set = []
        labels = 0
        j = 0
        for col in input_sample['columns']:
            label_col = preprocess_text(col)
            label_emb = averaged_embedding(label_col, word_vectors)  # Average of all
            col_set.append(label_emb)
            if col == output_sample['label_col'][0]:
                labels = j
            else:
                j = j + 1

        for idx in range(max_col_len - len(col_set)):
            col_set.append(np.zeros(word_vectors.vector_size))

        input_embeddings.append(question_emb)
        col_embeddings.append(col_set)
        labels_value.append(labels)

    X_train = torch.tensor(np.array(input_embeddings))
    y_train = torch.tensor(np.array(col_embeddings))
    gt_train = torch.tensor(np.array(labels_value))

    
    return X_train, y_train, gt_train

def convert_test_embeddings(input_train_data, word_vectors):
    input_embeddings = []
    col_embeddings = []

    data_size = len(input_train_data)
    max_question_len = max(len(preprocess_text(entry['question'])) for entry in input_train_data)
    max_col_len = max(len(entry['columns']) for entry in input_train_data)

    for i in range(data_size):
        input_sample = input_train_data[i]
        # Preprocess and convert text to embeddings
        question = preprocess_text(input_sample['question'])
        question_emb = []

        for word in question:
            if word in word_vectors.key_to_index:
                # If word is in vocabulary, use its embedding
                question_emb.append(word_vectors.get_vector(word))
            else:
                # If word is unknown, use the learnable unknown embedding
                question_emb.append(char_level_embedding(word, word_vectors))
        
        pad_length = max_question_len - len(question)
        
        for idx in range(pad_length):
            question_emb.append(np.zeros(word_vectors.vector_size))
            
        col_set = []
  
        for col in input_sample['columns']:
            label_col = preprocess_text(col)
            label_emb = averaged_embedding(label_col, word_vectors)  # Average of all
            col_set.append(label_emb)

        for idx in range(max_col_len - len(col_set)):
            col_set.append(np.zeros(word_vectors.vector_size))

        input_embeddings.append(question_emb)
        col_embeddings.append(col_set)


    X_train = torch.tensor(np.array(input_embeddings))
    y_train = torch.tensor(np.array(col_embeddings))
 
    return X_train, y_train

def create_test_batches(input_train_data, batch_size, word_vectors):
    num_samples = len(input_train_data)
    num_batches = (num_samples + batch_size - 1) // batch_size  # Calculate number of batches needed
    batches = []

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, num_samples)
        batch_input_data = input_train_data[start_idx:end_idx]

        # Convert data to embeddings
        X_batch, y_batch= convert_test_embeddings(batch_input_data, word_vectors)

        batches.append((X_batch.float(), y_batch.float()))

    return batches

def create_batches(input_train_data, output_train_data, batch_size, word_vectors):
    num_samples = len(input_train_data)
    num_batches = (num_samples + batch_size - 1) // batch_size  # Calculate number of batches needed
    batches = []

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, num_samples)
        batch_input_data = input_train_data[start_idx:end_idx]
        batch_output_data = output_train_data[start_idx:end_idx]

        # Convert data to embeddings
        X_batch, y_batch, gt_batch = convert_data_to_embeddings(batch_input_data, batch_output_data, word_vectors)
        X_batch, y_batch, gt_batch = X_batch, y_batch, gt_batch
        batches.append((X_batch.float(), y_batch.float(), gt_batch))

    return batches

def evaluate_model(val_batches, model):
    accuracy = 0
    
    for X_val, y_val, gt_val in val_batches:
        logits = model(X_val, y_val)
        predictions = torch.argmax(logits, dim=1)
        # Compute accuracy
        correct_predictions = torch.eq(predictions, gt_val).sum().item()
        accuracy += correct_predictions
    accuracy = accuracy / (32 * len(val_batches))
    
    return accuracy

# Define your LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(hidden_dim * 2, output_dim)  # Output dimension adjusted for bidirectional LSTM
        self.cos = nn.CosineSimilarity(dim=2)

    def forward(self, inputs, label):
        # inputs: (batch_size, seq_len, input_dim)
        _, (last_hidden_state, _) = self.lstm(inputs)
        # last_hidden_state: (2*num_layers, batch_size, hidden_dim)
        # We need to concatenate the forward and backward hidden states
        last_hidden_state = torch.cat((last_hidden_state[-2, :, :], last_hidden_state[-1, :, :]), dim=1)
        # last_hidden_state: (batch_size, hidden_dim*2)
        output = self.linear(last_hidden_state).unsqueeze(1)
        # print(output.shape, label.shape)
        similarity = self.cos(output, label)
     
        return similarity
    