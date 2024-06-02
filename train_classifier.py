import os
import torch
import numpy as np
import transformers
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
from dataset import BertDataset
from model import BERT, LongformerClassifier
from sklearn.metrics import accuracy_score
from sklearn.utils.class_weight import compute_class_weight


def evaluate_model(model, dataloader, device):
    model.eval()

    all_preds = []
    all_score_preds = []
    all_labels = []
    all_score_labels = []

    with torch.no_grad():
        for batch in dataloader:
            ids = batch['ids'].to(device)
            mask = batch['mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            scores = batch['target'].to(device)     # Shape: (batch_size, 8), with each element from 0 to 3
            binary_labels = batch['binary_target'].unsqueeze(1).to(device)     # Shape: (batch_size, 1)

            optimizer.zero_grad()

            outputs = model(ids, mask, token_type_ids)  # Shape: (batch_size, 33)

            output_preds = outputs[:, 0].unsqueeze(1)   # Shape: (batch_size, 1)
            output_scores = outputs[:, 1:]              # Shape: (batch_size, 32)
            # Reshape outputs and labels to calculate loss for each task
            output_scores = output_scores.reshape(-1, 4)  # Shape: (batch_size * 8, 4)
            scores = scores.view(-1)  # Shape: (batch_size * 8)

            # Score accuracy
            _, score_preds = torch.max(output_scores, dim=1)  # Get the index of the max log-probability
            all_score_preds.extend(score_preds.view(-1).cpu().numpy())
            all_score_labels.extend(scores.view(-1).cpu().numpy())

            # Label accuracy
            output_preds = output_preds > 0

            all_labels.extend(binary_labels.view(-1).cpu().numpy())
            all_preds.extend(output_preds.view(-1).float().cpu().numpy())

    # Compute metrics
    accuracy = accuracy_score(all_labels, all_preds)
    score_accuracy = accuracy_score(all_score_labels, all_score_preds)
    return accuracy, score_accuracy

def finetune(epochs, train_dataloader, val_dataloader, model, criterion, score_criterion, optimizer, device, model_name, score_loss_weight=1.):
    best_accuracy = 0.0
    best_epoch = -1
    for epoch in range(epochs):
        model.train()

        all_preds = []
        all_score_preds = []
        all_labels = []
        all_score_labels = []
        all_losses = []
        loop = tqdm(enumerate(train_dataloader), leave=False, total=len(train_dataloader))
        for batch, dl in loop:
            ids = dl['ids'].to(device)
            token_type_ids = dl['token_type_ids'].to(device)
            mask = dl['mask'].to(device)
            scores = dl['target'].to(device)     # Shape: (batch_size, 8), with each element from 0 to 3
            binary_labels = dl['binary_target'].unsqueeze(1).to(device)     # Shape: (batch_size, 1)

            optimizer.zero_grad()

            outputs = model(ids, mask, token_type_ids)  # Shape: (batch_size, 33)

            output_preds = outputs[:, 0].unsqueeze(1)   # Shape: (batch_size, 1)
            output_scores = outputs[:, 1:]              # Shape: (batch_size, 32)
            # Reshape outputs and labels to calculate loss for each task
            output_scores = output_scores.reshape(-1, 4)  # Shape: (batch_size * 8, 4)
            scores = scores.view(-1)  # Shape: (batch_size * 8)

            loss = criterion(output_preds, binary_labels.float()) + score_loss_weight * score_criterion(output_scores, scores)
            all_losses.append(loss.item())

            loss.backward()
            optimizer.step()

            # Score accuracy
            _, score_preds = torch.max(output_scores, dim=1)  # Get the index of the max log-probability
            all_score_preds.extend(score_preds.view(-1).cpu().numpy())
            all_score_labels.extend(scores.view(-1).cpu().numpy())
            score_accuracy = accuracy_score(all_score_labels, all_score_preds)

            # Label accuracy
            output_preds = output_preds > 0

            all_labels.extend(binary_labels.view(-1).cpu().numpy())
            all_preds.extend(output_preds.view(-1).float().cpu().numpy())
            accuracy = accuracy_score(all_labels, all_preds)

            # Show progress while training
            loop.set_description(f'Epoch={epoch+1}/{epochs}')
            loop.set_postfix(loss=loss.item(), acc=accuracy, score_acc=score_accuracy)

        accuracy = accuracy_score(all_labels, all_preds)
        score_accuracy = accuracy_score(all_score_labels, all_score_preds)
        print(f'Epoch {epoch+1}/{epochs} Accuracy: {accuracy * 100:.2f}; Score Accuracy: {score_accuracy * 100:.2f}; Loss: {np.mean(all_losses):.2f}')

        # Evaluate the model
        accuracy, score_accuracy = evaluate_model(model, val_dataloader, device)
        print(f"Eval Epoch {epoch+1}/{epochs} || Accuracy: {accuracy * 100:.2f}; Score Accuracy: {score_accuracy * 100:.2f}")

        # Save the model if the current accuracy is the best
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_epoch = epoch + 1
            torch.save(model.state_dict(), f'outputs/{model_name}_best_model.pth')

    torch.save(model.state_dict(), f'outputs/{model_name}_last_model.pth')
    print(f"Best accuracy: {best_accuracy:.4f} on Epoch {best_epoch}")

    return model


if __name__ == '__main__':
    n_epoch = 10
    batch_size = 32
    model_name = 'bert'
    text_tag = 'Synopsis'       # 'Transcript', 'Synopsis', 'Sentiment'
    score_loss_weight = 0.0
    max_length = 4096 if model_name == 'longformer' else 512
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs('outputs', exist_ok=True)

    train_file = '/data/DAIC/train.json'
    val_file = '/data/DAIC/val.json'
    tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased")
    train_dataset = BertDataset(train_file, tokenizer, max_length=max_length, text_tag=text_tag, augmentation=True)
    val_dataset = BertDataset(val_file, tokenizer, max_length=max_length, text_tag=text_tag)

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size)

    if model_name == 'longformer':
        model = LongformerClassifier(num_classes=1).to(device)
    else:
        model = BERT(num_classes=1, version='other').to(device)
        for param in model.bert_model.parameters():
            param.requires_grad = False

    # Compute class weights
    labels = train_dataset.all_scores
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(labels), y=labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    print(f"Score class weights: {class_weights}")

    # Define the loss function with class weights
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2]).to(device))
    score_criterion = nn.CrossEntropyLoss(weight=class_weights)
    # Initialize Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)

    model = finetune(n_epoch, train_dataloader, val_dataloader, model, criterion, score_criterion, optimizer, device, model_name + '_' + text_tag, score_loss_weight=score_loss_weight)