import argparse
import random

import pandas as pd

from tqdm.auto import tqdm

from transformers import TrainingArguments, Trainer
import torch
import pytorch_lightning as pl

import optuna
from dotenv import load_dotenv

from .dataloader import Dataloader
from .model import Model

torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
random.seed(0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='klue/roberta-small', type=str)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--max_epoch', default=1, type=int)
    parser.add_argument('--shuffle', default=True)
    parser.add_argument('--learning_rate', default=1e-5, type=float)
    parser.add_argument('--train_path', default='../data/train.csv')
    parser.add_argument('--dev_path', default='../data/dev.csv')
    parser.add_argument('--test_path', default='../data/dev.csv')
    parser.add_argument('--predict_path', default='../data/test.csv')
    parser.add_argument('--num_workers', default=1, type=int)
    parser.add_argument('--do_optimize', default=False)
    parser.add_argument('--n_trials', default=1, type=int)

    args = parser.parse_args()
    
    wandb_logger = pl.WandbLogger(
        project="wandb project",
        name="wandb run name",
        log_model="all",
        save_dir="wandb"
    )

    study = optuna.create_study(direction='minimize')

    dataloader = Dataloader(
        args.model_name, 
        args.batch_size,
        args.shuffle,
        args.train_path,
        args.dev_path,
        args.test_path,
        args.predict_path,
        args.num_workers
    )
    if args.do_optimize:
        def objective(trial, args):
            learning_rate = trial.suggest_float('learning_rate', 1e-6, 1e-4, log=True)
            
            model = Model(args.model_name, lr=learning_rate)
            trainer = pl.Trainer(
                accelerator="gpu", devices=1, 
                max_epochs=args.max_epoch,
                log_every_n_steps=1,
                logger=False,
                enable_checkpointing=False
            )
            
            trainer.fit(model=model, datamodule=dataloader)
            val_metrics = trainer.validate(model=model, datamodule=dataloader, verbose=False)
            val_loss = val_metrics[0]['val_loss']
            return val_loss

        study.optimize(lambda trial: objective(trial, args), n_trials=args.n_trials)

        best_params = study.best_params
    learning_rate = best_params['learning_rate'] or args.learning_rate


    model = Model(args.model_name, learning_rate)

    trainer = pl.Trainer(
        accelerator="gpu", 
        devices=1, 
        max_epochs=args.max_epoch, 
        log_every_n_steps=1, 
        logger=wandb_logger
    )

    # Train part
    trainer.fit(model=model, datamodule=dataloader)
    trainer.test(model=model, datamodule=dataloader)

    torch.save(model, 'model.pt')
