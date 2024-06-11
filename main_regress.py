import os
import torch
import random
import argparse
import numpy as np
import transformers
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
import torch.optim as optim
from tqdm import tqdm
import pandas as pd
from dataset import BertDataset
from model import BERT, LongformerClassifier, RoBERTa
from sklearn.metrics import mean_squared_error, mean_absolute_error


def finetune_regression(epochs, train_dataloader, val_dataloader, model, optimizer, scheduler, criterion, device, model_name, patience=50):
    best_mse = float('inf')
    best_epoch = -1
    patience_counter = 0

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
            loss = criterion(outputs, scores.float())
            all_losses.append(loss.item())

            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            all_labels.extend(scores.view(-1).cpu().numpy())
            all_preds.extend(outputs.detach().float().cpu().numpy())

            # Show progress while training
            loop.set_description(f'Epoch={epoch + 1}/{epochs}')
            loop.set_postfix(loss=loss.item())

        mse = mean_squared_error(all_labels, all_preds)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(all_labels, all_preds)

        current_lr = optimizer.param_groups[0]['lr']
        print(
            f'Epoch {epoch + 1}/{epochs} LR: {current_lr}: RMSE: {rmse:.2f}; MAE: {mae:.2f}; Loss: {np.mean(all_losses):.2f}')

        # Evaluate the model
        mse, rmse, mae, eval_loss = evaluate_model_regression(model, val_dataloader, criterion, device)
        print(f"Eval Epoch {epoch + 1}/{epochs} || RMSE: {rmse:.2f}; MAE: {mae:.2f}; Loss: {eval_loss:.2f}")
        print('----------------------------------------------------------------------------')

        scheduler.step(eval_loss)

        # Save the model if the current MSE is the best
        if mse < best_mse:
            best_mse = mse
            best_epoch = epoch + 1
            torch.save(model.state_dict(), f'outputs/{model_name}_best_model.pth')
            patience_counter = 0
        else:
            patience_counter += 1

        # Check if we need to stop early
        if patience_counter >= patience or current_lr < 1e-7:
            print(f"Early stopping at epoch {epoch + 1} due to no improvement in validation MSE for {patience} consecutive epochs.")
            break

    torch.save(model.state_dict(), f'outputs/{model_name}_last_model.pth')
    print(f"Best MSE: {best_mse:.4f} on Epoch {best_epoch}")

    return model

def evaluate_model_regression(model, dataloader, criterion, device):
    model.eval()

    all_preds = []
    all_labels = []
    eval_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            ids = batch['ids'].to(device)
            mask = batch['mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            scores = batch['total_target'].unsqueeze(1).to(device)  # Shape: (batch_size, 1), with each element from 0 to 24

            outputs = model(ids, mask, token_type_ids)  # Shape: (batch_size, 1)

            loss = criterion(outputs, scores.float())
            eval_loss += loss.item()

            all_labels.extend(scores.view(-1).cpu().numpy())
            all_preds.extend(outputs.detach().float().cpu().numpy())

    # Compute metrics
    mse = mean_squared_error(all_labels, all_preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(all_labels, all_preds)

    eval_loss /= len(dataloader)

    return mse, rmse, mae, eval_loss

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

            all_labels.append(scores.cpu().item())
            all_preds.append(outputs.item())

    # Compute metrics

    mse = mean_squared_error(all_labels, all_preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(all_labels, all_preds)

    print(f"Test || RMSE: {rmse:.2f}; MAE: {mae:.2f}")
    return all_labels, all_preds


def parse_args():
    parser = argparse.ArgumentParser(description="Script to train a model with various configurations.")

    parser.add_argument('--datasets', nargs='+', default=['train'], help='Use training or synthetic data.')
    parser.add_argument('--data_root', type=str, default='/data/DAIC', help='Set up data root.')
    parser.add_argument('--n_epoch', type=int, default=200, help='Number of epochs for training.')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate.')
    parser.add_argument('--weight_decay', type=float, default=1e-2, help='Weight decay.')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate in the last layer.')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training.')
    parser.add_argument('--model_name', type=str, default='bert', choices=['bert', 'longformer', 'roberta'],
                        help='Model name to use for training.')
    parser.add_argument('--text_tag', nargs='+', default=['Synopsis', 'Sentiment'], help='List of text tags to use.')
    parser.add_argument('--max_length', type=int, default=512, help='Maximum sequence length for the model.')
    parser.add_argument('--gpu', type=str, default='0', help='Choose which gpu to use.')
    parser.add_argument('--eval', action='store_true', default=False, help='Evaluation mode on test dataset')
    parser.add_argument('--last', action='store_true', default=False, help='Use last epoch instead of best model')
    parser.add_argument('--seed', type=int, default=1234, help='Random Seed.')

    args = parser.parse_args()

    # os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    # Set max_length based on model_name
    if args.model_name == 'longformer':
        args.max_length = 4096

    return args


if __name__ == '__main__':
    args = parse_args()
    model_save_name = args.model_name + '_' + '_'.join(args.text_tag) + f'_regression_data_{"_".join(args.datasets)}'
    # Set the device to the specified GPU
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    os.makedirs('outputs', exist_ok=True)

    if args.model_name == 'longformer':
        model = LongformerClassifier(num_classes=1, dropout_prob=args.dropout).to(device)
        tokenizer = transformers.LongformerTokenizer.from_pretrained("allenai/longformer-base-4096")
    elif args.model_name == 'bert':
        model = BERT(num_classes=1, dropout_prob=args.dropout).to(device)
        tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased")
    elif args.model_name == 'roberta':
        model = RoBERTa(num_classes=1, dropout_prob=args.dropout).to(device)
        tokenizer = transformers.RobertaTokenizer.from_pretrained('roberta-base')
    else:
        raise NotImplementedError

    # Count the number of parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters in the {args.model_name} model: {num_params}")

    if args.eval:
        # Load the state dictionary
        model_version = 'last' if args.last else 'best'
        model.load_state_dict(torch.load(f'outputs/{model_save_name}_{model_version}_model.pth'))

        test_file = os.path.join(args.data_root, 'test.json')
        test_dataset = BertDataset(test_file, tokenizer, max_length=args.max_length, text_tag=args.text_tag)
        test_dataloader = DataLoader(dataset=test_dataset, batch_size=1)

        labels, preds = predict(model, test_dataloader, device)
        df = pd.DataFrame({'label': labels, 'pred': preds})
        df.to_csv(f'results/{model_save_name}.csv', index=False)
    else:
        file_mapping = {
            'train': os.path.join(args.data_root, 'train.json'),
            'synthetic': os.path.join(args.data_root.replace('DAIC', 'synthetic_DAIC'), 'synthetic_train.json')
        }
        val_file = os.path.join(args.data_root, 'val.json')
        val_dataset = BertDataset(val_file, tokenizer, max_length=args.max_length, text_tag=args.text_tag)

        train_dataset = []
        for dataset in args.datasets:
            filename = file_mapping[dataset]
            train_dataset.append(BertDataset(filename, tokenizer, max_length=args.max_length, text_tag=args.text_tag, augmentation=True))

        if len(train_dataset) == 1:
            train_dataset = train_dataset[0]
        else:
            train_dataset = ConcatDataset(train_dataset)

        train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
        val_dataloader = DataLoader(dataset=val_dataset, batch_size=args.batch_size)

        # Initialize Optimizer
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10,
                                                         verbose=True)  # Learning rate scheduler

        criterion = nn.MSELoss()
        model = finetune_regression(args.n_epoch, train_dataloader, val_dataloader, model, optimizer, scheduler, criterion, device, model_save_name)
