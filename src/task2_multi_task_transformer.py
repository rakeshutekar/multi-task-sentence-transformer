# Import necessary libraries
import torch
from transformers import BertModel, BertTokenizer


class MultiTaskSentenceTransformerModel(torch.nn.Module):
    def __init__(self, model_name='bert-base-uncased', num_classes_task_a=3, num_classes_task_b=2):
        super(MultiTaskSentenceTransformerModel, self).__init__()
        # Load a pre-trained BERT model as the transformer backbone
        self.bert = BertModel.from_pretrained(model_name)
        
        # Choice: Using mean pooling to obtain fixed-length sentence embeddings
        self.pooling = 'mean'
        
        # Task A: Sentence Classification
        # Linear layer mapping embeddings to class scores for Task A
        self.classifier_task_a = torch.nn.Linear(self.bert.config.hidden_size, num_classes_task_a)
        
        # Task B: Sentiment Analysis
        # Linear layer mapping embeddings to class scores for Task B
        self.classifier_task_b = torch.nn.Linear(self.bert.config.hidden_size, num_classes_task_b)

    def forward(self, input_ids, attention_mask):
        # Obtain hidden states from BERT
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        
        # Perform pooling to get sentence embeddings
        if self.pooling == 'mean':
            # Compute mean of the embeddings, weighted by the attention mask to ignore padding tokens
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, dim=1)
            sum_mask = input_mask_expanded.sum(dim=1)
            # Avoid division by zero
            sum_mask = torch.clamp(sum_mask, min=1e-9)
            sentence_embeddings = sum_embeddings / sum_mask
        else:
            # Use the CLS token embedding
            sentence_embeddings = last_hidden_state[:, 0, :]
        
        # Task A output: Class scores for sentence classification
        logits_task_a = self.classifier_task_a(sentence_embeddings)
        
        # Task B output: Class scores for sentiment analysis
        logits_task_b = self.classifier_task_b(sentence_embeddings)
        
        return logits_task_a, logits_task_b

# Sample test with a few sentences
if __name__ == "__main__":
    # Initialize tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Define number of classes for each task
    num_classes_task_a = 3  # e.g., Classes: 'News', 'Opinion', 'Entertainment'
    num_classes_task_b = 2  # e.g., Sentiments: 'Positive', 'Negative'
    
    model = MultiTaskSentenceTransformerModel(
        num_classes_task_a=num_classes_task_a,
        num_classes_task_b=num_classes_task_b
    )
    
    # Sample sentences
    sentences = [
        "The government announced new policies today.",
        "I absolutely love the new design of your website!",
        "The movie was a waste of time."
    ]
    
    # Tokenize sentences
    encoded_input = tokenizer(
        sentences,
        padding=True,
        truncation=True,
        return_tensors='pt'
    )
    
    # Get outputs for both tasks
    with torch.no_grad():
        logits_task_a, logits_task_b = model(
            input_ids=encoded_input['input_ids'],
            attention_mask=encoded_input['attention_mask']
        )
    
    # Apply softmax to get probabilities (optional)
    probabilities_task_a = torch.nn.functional.softmax(logits_task_a, dim=1)
    probabilities_task_b = torch.nn.functional.softmax(logits_task_b, dim=1)
    
    # Print the results
    print("Task A: Sentence Classification")
    for idx, probs in enumerate(probabilities_task_a):
        print(f"Sentence {idx+1}: {sentences[idx]}")
        print(f"Class Probabilities: {probs.numpy()}\n")
    
    print("Task B: Sentiment Analysis")
    for idx, probs in enumerate(probabilities_task_b):
        print(f"Sentence {idx+1}: {sentences[idx]}")
        print(f"Sentiment Probabilities: {probs.numpy()}\n")
