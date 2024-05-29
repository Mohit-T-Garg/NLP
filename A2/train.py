import torch
import torch.nn as nn
import torch.optim as optim
import gensim.downloader as api
import argparse
from helper import *
    
if __name__ == "__main__":
    # Load training data
    parser = argparse.ArgumentParser(description="Train a language classification model.")
    parser.add_argument("--train", required=True, help="Path to the train data JSON file.")
    parser.add_argument("--val", required=True, help="Path to the validation data JSON file.")
    args = parser.parse_args()

    print("Model Started.")
    train_data = load_data(args.train)
    
    # Separate input and output data
    input_train_data, output_train_data = separate_input_output(train_data)

    input_val_data, output_val_data = separate_input_output(load_data(args.val))

    input_train_data = input_train_data + input_val_data
    output_train_data = output_train_data + output_val_data

    # Load word vectors
    word_vectors = api.load("glove-wiki-gigaword-100")

    batch_size = 32

    training_data_batches = create_batches(input_train_data, output_train_data, batch_size, word_vectors)
    val_batches = create_batches(input_val_data, output_val_data, batch_size, word_vectors)

    # Define parameters
    input_dim = word_vectors.vector_size
    hidden_dim = 256
    output_dim = input_dim

    # Initialize model
    model = LSTMModel(input_dim, hidden_dim, output_dim)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Sample training loop
    for epoch in range(15):
        epoch_loss = 0.0
        
        for X_batch, y_batch, gt_batch in training_data_batches:
            optimizer.zero_grad()
            outputs = model(X_batch, y_batch)
            loss = criterion(outputs, gt_batch)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
        
        avg_epoch_loss = epoch_loss / len(training_data_batches)

        print(f"Epoch {epoch+1}, Average Loss: {avg_epoch_loss}")
        print(f"Training Accuracy: {evaluate_model(training_data_batches, model)}")
        print(f"Validation Accuracy: {evaluate_model(val_batches, model)}")

    # Save trained model
    torch.save(model.state_dict(), 'lstm_model.pth')