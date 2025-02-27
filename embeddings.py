import argparse
import torch
from transformers import T5Tokenizer, T5EncoderModel

def generate_embeddings(input_file, sequences_output_file, embeddings_output_file, model_name="t5-small", max_length=128):
    # Load T5 tokenizer and encoder model
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5EncoderModel.from_pretrained(model_name)
    model.eval()

    # Read input sentences from file (one sentence per line)
    with open(input_file, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    all_sequences = []
    all_embeddings = []

    for line in lines:
        # Tokenize input, forcing a max_length (truncation) and padding up to max_length
        encoded = tokenizer(line, return_tensors="pt", max_length=max_length, truncation=True, padding="max_length")
        input_ids = encoded.input_ids  # shape: (1, max_length)
        # Forward pass through T5 encoder
        with torch.no_grad():
            outputs = model(input_ids)
        # Get the last hidden state (shape: (1, max_length, hidden_dim))
        hidden_states = outputs.last_hidden_state

        # Append token ids and embeddings (remove batch dimension)
        all_sequences.append(input_ids.squeeze(0).tolist())
        all_embeddings.append(hidden_states.squeeze(0))

    # Stack embeddings into a tensor of shape (num_sequences, max_length, hidden_dim)
    embeddings_tensor = torch.stack(all_embeddings, dim=0)

    # Save sequences and embeddings to files
    torch.save(all_sequences, sequences_output_file)
    torch.save(embeddings_tensor, embeddings_output_file)
    print(f"Saved {len(all_sequences)} sequences to {sequences_output_file}")
    print(f"Saved embeddings tensor with shape {embeddings_tensor.shape} to {embeddings_output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True,
                        help="Path to a text file with one sentence per line.")
    parser.add_argument("--sequences_output_file", type=str, required=True,
                        help="Output file path to save the token sequences.")
    parser.add_argument("--embeddings_output_file", type=str, required=True,
                        help="Output file path to save the embeddings tensor.")
    parser.add_argument("--model_name", type=str, default="t5-small",
                        help="Name of the T5 model to use (default: t5-small).")
    parser.add_argument("--max_length", type=int, default=128,
                        help="Maximum sequence length (will pad or truncate accordingly).")
    args = parser.parse_args()

    generate_embeddings(args.input_file, args.sequences_output_file, args.embeddings_output_file,
                        args.model_name, args.max_length)
