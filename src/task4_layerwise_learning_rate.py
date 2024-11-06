# Import necessary libraries
import torch
from torch.optim import AdamW
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

# Implementing layer-wise learning rates
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
    
    # Define base learning rate
    lr = 1e-5

    # Create parameter groups with different learning rates
    optimizer_grouped_parameters = []

    # Assign a lower learning rate to the embeddings
    optimizer_grouped_parameters.append({
        'params': model.bert.embeddings.parameters(),
        'lr': lr * 0.1  # Lower LR for embeddings
    })

    # Assign decreasing learning rates to each encoder layer
    num_layers = len(model.bert.encoder.layer)  # Typically 12 for BERT-base
    # Start from the bottom layer (layer 0) to the top layer (layer 11)
    for i in range(num_layers):
        # Calculate learning rate for this layer
        layer_lr = lr * (0.95 ** (num_layers - i - 1))  # Decreasing LR for lower layers
        optimizer_grouped_parameters.append({
            'params': model.bert.encoder.layer[i].parameters(),
            'lr': layer_lr
        })

    # Add task-specific parameters with a higher learning rate
    task_learning_rate = 1e-4  # Higher LR for task-specific heads
    optimizer_grouped_parameters.append({
        'params': model.classifier_task_a.parameters(),
        'lr': task_learning_rate
    })
    optimizer_grouped_parameters.append({
        'params': model.classifier_task_b.parameters(),
        'lr': task_learning_rate
    })

    # Create the optimizer with layer-wise learning rates
    optimizer = AdamW(optimizer_grouped_parameters)

    # Sample sentences
    sentences = [
        "The government announced new policies today.",
        "I absolutely love the new design of your website!",
        "The movie was a waste of time."
    ]

    # Dummy labels for the tasks
    labels_task_a = torch.tensor([0, 1, 2])  # Classes for Task A
    labels_task_b = torch.tensor([1, 0, 1])  # Classes for Task B

    # Tokenize sentences
    encoded_input = tokenizer(
        sentences,
        padding=True,
        truncation=True,
        return_tensors='pt'
    )

    # Define loss functions
    criterion_task_a = torch.nn.CrossEntropyLoss()
    criterion_task_b = torch.nn.CrossEntropyLoss()

    # Training step
    model.train()
    optimizer.zero_grad()

    # Forward pass
    logits_task_a, logits_task_b = model(
        input_ids=encoded_input['input_ids'],
        attention_mask=encoded_input['attention_mask']
    )

    # Compute losses
    loss_task_a = criterion_task_a(logits_task_a, labels_task_a)
    loss_task_b = criterion_task_b(logits_task_b, labels_task_b)

    # Total loss
    total_loss = loss_task_a + loss_task_b

    # Backward pass
    total_loss.backward()

    # Optimization step
    optimizer.step()

    print(f"Total loss: {total_loss.item()}")


