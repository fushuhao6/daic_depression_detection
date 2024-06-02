import os
import torch
import argparse
import numpy as np
import transformers
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
import torch.optim as optim
from tqdm import tqdm
from dataset import BertDataset
from model import BERT, LongformerClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error


def finetune_regression(epochs, train_dataloader, val_dataloader, model, optimizer, device, model_name, patience=100):
    best_mse = float('inf')
    best_epoch = -1
    patience_counter = 0

    mse_loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        model.train()

        all_preds = []
        all_labels = []
        all_losses = []
        loop = tqdm(enumerate(train_dataloader), leave=False, total=len(train_dataloader))

        for batch, dl in loop:
            ids = dl['ids'].to(device)
            token_type_ids = dl['token_type_ids'].to(device)
            mask = dl['mask'].to(device)
            scores = dl['total_target'].unsqueeze(1).to(device)  # Shape: (batch_size, 1), with each element from 0 to 24

            optimizer.zero_grad()

            outputs = model(ids, mask, token_type_ids)  # Shape: (batch_size, 1)

            # Calculate regression loss
            loss = mse_loss_fn(outputs, scores.float())
            all_losses.append(loss.item())

            loss.backward()
            optimizer.step()

            all_labels.extend(scores.view(-1).cpu().numpy())
            all_preds.extend(outputs.detach().float().cpu().numpy())

            # Show progress while training
            loop.set_description(f'Epoch={epoch + 1}/{epochs}')
            loop.set_postfix(loss=loss.item())

        mse = mean_squared_error(all_labels, all_preds)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(all_labels, all_preds)

        print(
            f'Epoch {epoch + 1}/{epochs}: RMSE: {rmse:.2f}; MAE: {mae:.2f}; Loss: {np.mean(all_losses):.2f}')

        # Evaluate the model
        mse, rmse, mae = evaluate_model_regression(model, val_dataloader, device)
        print(f"Eval Epoch {epoch + 1}/{epochs} || RMSE: {rmse:.2f}; MAE: {mae:.2f}")
        print('----------------------------------------------------------------------------')

        # Save the model if the current MSE is the best
        if mse < best_mse:
            best_mse = mse
            best_epoch = epoch + 1
            torch.save(model.state_dict(), f'outputs/{model_name}_best_model.pth')
            patience_counter = 0
        else:
            patience_counter += 1

        # Check if we need to stop early
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1} due to no improvement in validation MSE for {patience} consecutive epochs.")
            break

    torch.save(model.state_dict(), f'outputs/{model_name}_last_model.pth')
    print(f"Best MSE: {best_mse:.4f} on Epoch {best_epoch}")

    return model

def evaluate_model_regression(model, dataloader, device):
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            ids = batch['ids'].to(device)
            mask = batch['mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            scores = batch['total_target'].unsqueeze(1).to(device)  # Shape: (batch_size, 1), with each element from 0 to 24

            outputs = model(ids, mask, token_type_ids)  # Shape: (batch_size, 1)

            all_labels.extend(scores.view(-1).cpu().numpy())
            all_preds.extend(outputs.detach().float().cpu().numpy())

    # Compute metrics
    mse = mean_squared_error(all_labels, all_preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(all_labels, all_preds)

    return mse, rmse, mae

def predict(model, dataloader, device):
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            ids = batch['ids'].to(device)
            mask = batch['mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            scores = batch['total_target'].unsqueeze(1).to(device)  # Shape: (batch_size, 1), with each element from 0 to 24

            outputs = model(ids, mask, token_type_ids)  # Shape: (batch_size, 1)

            print(f"Predicted Score: {torch.round(outputs).int().item()}; Ground Truth Score: {scores.item()}")

            all_labels.extend(scores.view(-1).cpu().numpy())
            all_preds.extend(outputs.detach().float().cpu().numpy())

    # Compute metrics

    mse = mean_squared_error(all_labels, all_preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(all_labels, all_preds)

    print(f"Test || RMSE: {rmse:.2f}; MAE: {mae:.2f}")


def parse_args():
    parser = argparse.ArgumentParser(description="Script to train a model with various configurations.")

    parser.add_argument('--use_synthetic', action='store_true', default=False, help='Whether to use synthetic data.')
    parser.add_argument('--n_epoch', type=int, default=1000, help='Number of epochs for training.')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate.')
    parser.add_argument('--weight_decay', type=float, default=1e-2, help='Weight decay.')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate in the last layer.')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training.')
    parser.add_argument('--model_name', type=str, default='bert', choices=['bert', 'longformer'],
                        help='Model name to use for training.')
    parser.add_argument('--text_tag', nargs='+', default=['Synopsis', 'Sentiment'], help='List of text tags to use.')
    parser.add_argument('--max_length', type=int, default=512, help='Maximum sequence length for the model.')
    parser.add_argument('--gpu', type=str, default='0', help='Choose which gpu to use.')
    parser.add_argument('--eval', action='store_true', default=False, help='Evaluation mode on test dataset')

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # Set max_length based on model_name
    if args.model_name == 'longformer':
        args.max_length = 4096

    return args


if __name__ == '__main__':
    args = parse_args()
    model_save_name = args.model_name + '_' + '_'.join(args.text_tag) + f'_regression_synthetic_{args.use_synthetic}'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs('outputs', exist_ok=True)
    tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased")

    if args.model_name == 'longformer':
        model = LongformerClassifier(num_classes=1, dropout_prob=args.dropout).to(device)
    else:
        model = BERT(num_classes=1, dropout_prob=args.dropout).to(device)

    if args.eval:
        # Load the state dictionary
        model.load_state_dict(torch.load(f'outputs/{model_save_name}_best_model.pth'))

        test_file = '/data/DAIC/test.json'
        test_dataset = BertDataset(test_file, tokenizer, max_length=args.max_length, text_tag=args.text_tag)
        test_dataloader = DataLoader(dataset=test_dataset, batch_size=1)

        predict(model, test_dataloader, device)

    else:
        train_file = '/data/DAIC/train.json'
        val_file = '/data/DAIC/val.json'
        synthetic_file = '/data/synthetic_DAIC/synthetic.json'
        train_dataset = BertDataset(train_file, tokenizer, max_length=args.max_length, text_tag=args.text_tag, augmentation=True)
        val_dataset = BertDataset(val_file, tokenizer, max_length=args.max_length, text_tag=args.text_tag)

        # Concatenate the train and synthetic datasets
        if args.use_synthetic:
            synthetic_dataset = BertDataset(synthetic_file, tokenizer, max_length=args.max_length, text_tag=args.text_tag,
                                            augmentation=True)
            train_dataset = ConcatDataset([train_dataset, synthetic_dataset])

        train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
        val_dataloader = DataLoader(dataset=val_dataset, batch_size=args.batch_size)

        # Initialize Optimizer
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        model = finetune_regression(args.n_epoch, train_dataloader, val_dataloader, model, optimizer, device, model_save_name)