# Import necessary libraries
import torch
from transformers import BertModel, BertTokenizer


class SentenceTransformerModel(torch.nn.Module):
    def __init__(self, model_name='bert-base-uncased'):
        super(SentenceTransformerModel, self).__init__()
        # Load a pre-trained BERT model as the transformer backbone
        self.bert = BertModel.from_pretrained(model_name)
        
        # Choice: Decided to use mean pooling to obtain fixed-length sentence embeddings
        # Rationale: Mean pooling over token embeddings provides a simple way to
        # aggregate information from all tokens in a sentence, capturing overall semantics.
        # Alternative options could include using the CLS token embedding or max pooling.
        self.pooling = 'mean'

    def forward(self, input_ids, attention_mask):
        # Obtain hidden states from BERT
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # Hidden states of all tokens in the last layer
        last_hidden_state = outputs.last_hidden_state
        
        # Perform pooling
        if self.pooling == 'mean':
            # Compute mean of the embeddings, weighted by the attention mask to ignore padding tokens
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, dim=1)
            sum_mask = input_mask_expanded.sum(dim=1)
            # Avoid division by zero
            sum_mask = torch.clamp(sum_mask, min=1e-9)
            sentence_embeddings = sum_embeddings / sum_mask
        else:
            # Default to using the CLS token embedding
            # Choice: Using the CLS token is a common practice, but may not capture full sentence semantics
            sentence_embeddings = last_hidden_state[:, 0, :]
        return sentence_embeddings

# Sample test with a few sentences
if __name__ == "__main__":
    # Initialize tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = SentenceTransformerModel()
    
    # Sample sentences
    sentences = [
        "The quick brown fox jumps over the lazy dog.",
        "Artificial intelligence is transforming the world.",
        "I love programming in Python."
    ]
    
    # Tokenize sentences
    encoded_input = tokenizer(
        sentences,
        padding=True,
        truncation=True,
        return_tensors='pt'
    )
    
    # Get sentence embeddings
    with torch.no_grad():
        embeddings = model(
            input_ids=encoded_input['input_ids'],
            attention_mask=encoded_input['attention_mask']
        )
    
    # Print the embeddings
    print("Sentence Embeddings:")
    for idx, embedding in enumerate(embeddings):
        print(f"Sentence {idx+1}: {sentences[idx]}")
        print(f"Embedding shape: {embedding.shape}")
        print(f"Embedding vector:\n{embedding}\n")
