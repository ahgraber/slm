# %%
import collections
import functools
import itertools
import json
import logging
import math
import os
from pathlib import Path
import pickle
import random
import sys
from typing import Any, Callable, Iterable, Literal, Optional, Union

import tqdm

import numpy as np
import pandas as pd

import datasets
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

# may have to include `.env` file at project root containing `PYTHONPATH="./../src"`
sys.path.insert(0, str(Path(__file__ + "/../../").resolve()))
from slm.word2vec import loaders, models, trainer, vocab  # NOQA: E402

# %%
logger = logging.getLogger(__name__)


# %%
# NOTE:
# For the experiments reported ..., we used
# three training epochs with stochastic gradient descent and backpropagation.
# We chose starting learning rate 0.025 and decreased it linearly.
class Trainer:
    """Main class for model training over iterable dataset."""

    def __init__(
        self,
        model: nn.Module,
        batch_size: int,
        epochs: int,
        collate_fn: Callable,
        trn_dataset: datasets.Dataset,
        trn_sample: Optional[int],
        val_dataset: datasets.Dataset,
        val_sample: Optional[int],
        criterion: torch.nn.modules.loss._WeightedLoss,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
        device: torch.device,
        checkpoint_frequency: int,
        log_dir: Union[str, Path],
        model_dir: Union[str, Path],
        model_name: str,
    ):
        self.model = model
        self.batch_size = batch_size
        self.epochs = epochs
        self.collate_fn = collate_fn
        self.trn_dataset = trn_dataset
        self.trn_sample = trn_sample if trn_sample else None
        self.val_dataset = val_dataset
        self.val_sample = val_sample if val_sample else None

        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        self.device = device

        self.checkpoint_frequency = checkpoint_frequency
        self.model_dir = model_dir
        self.model_name = model_name

        self.loss = {"train": [], "val": []}
        self.writer = SummaryWriter(log_dir=log_dir)

        self.model.to(self.device)

    def train(self):
        """Train model."""
        for epoch in tqdm.trange(self.epochs):
            self._train_epoch(epoch)
            self._validate_epoch()
            print(
                f'Epoch: {epoch + 1}/{self.epochs}, Train Loss={self.loss["train"][-1]:.5f}, Val Loss={self.loss["val"][-1]:.5f}'
            )

            # self.writer.add_scalar(
            #     "Loss/train",
            #     self.loss["train"][-1],
            #     epoch + 1,
            # )
            # self.writer.add_scalar(
            #     "Loss/val",
            #     self.loss["val"][-1],
            #     epoch + 1,
            # )
            self.writer.add_scalars(
                "Loss",
                {"train": self.loss["train"][-1], "val": self.loss["val"][-1]},
                epoch,
            )

            self.lr_scheduler.step()

            if self.checkpoint_frequency:
                self._save_checkpoint(epoch)

    def _train_epoch(self, epoch: int):
        loader = DataLoader(
            self.trn_dataset.shuffle(),
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
        )

        self.model.train()
        running_loss = []

        # for i, batch_data in enumerate(loader, 1):
        for i, batch_data in enumerate(tqdm.tqdm(loader, total=self.trn_sample, desc="Training", leave=False)):
            inputs = batch_data[0].to(self.device)
            labels = batch_data[1].to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss.append(loss.item())

            if i % 100 == 0:
                # TODO:set global_step param in add_scalar
                self.writer.add_scalar(
                    f"tr_loss_{epoch}",
                    np.mean(running_loss[-100:]),
                )

        epoch_loss = np.mean(running_loss)
        self.loss["train"].append(epoch_loss)

    def _validate_epoch(self):
        loader = DataLoader(
            self.val_dataset.shuffle(),
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
        )

        self.model.eval()
        running_loss = []

        with torch.no_grad():
            # for i, batch_data in enumerate(loader, 1):
            for _, batch_data in enumerate(tqdm.tqdm(loader, total=self.val_sample, desc="Validation", leave=False)):
                inputs = batch_data[0].to(self.device)
                labels = batch_data[1].to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                running_loss.append(loss.item())

        epoch_loss = np.mean(running_loss)
        self.loss["val"].append(epoch_loss)

    def _save_checkpoint(self, epoch):
        """Save model checkpoint to `self.model_dir` directory."""
        epoch_num = epoch + 1
        if epoch_num % self.checkpoint_frequency == 0:
            model_path = f"checkpoint_{str(epoch_num).zfill(len(str(self.epochs)))}.pt"
            model_path = os.path.join(self.model_dir, model_path)
            torch.save(self.model, model_path)

    def save_model(self):
        """Save final model to `self.model_dir` directory."""
        model_path = os.path.join(self.model_dir, "model.pt")
        torch.save(self.model, model_path)

    def save_loss(self):
        """Save train/val loss as json file to `self.model_dir` directory."""
        loss_path = os.path.join(self.model_dir, "loss.json")
        with open(loss_path, "w") as fp:
            json.dump(self.loss, fp)
