import pandas as pd

import transformers
import torch
import pytorch_lightning as pl

from tqdm.auto import tqdm

from .dataset import Dataset

class Dataloader(pl.LightningDataModule):
    def __init__(self, model_name, batch_size, shuffle, train_path, dev_path, test_path, predict_path, num_workers):
        super().__init__()
        self.model_name = model_name
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.train_path = train_path
        self.dev_path = dev_path
        self.test_path = test_path
        self.predict_path = predict_path
        self.num_workers = num_workers

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.predict_dataset = None

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, max_length=160)
        self.target_columns = ['label']
        self.delete_columns = ['id']
        self.text_columns = ['sentence_1', 'sentence_2']

    def tokenizing(self, dataframe):
        data, attention_masks = [], []
        for idx, item in tqdm(dataframe.iterrows(), desc='tokenizing', total=len(dataframe)):
            text = '[SEP]'.join([item[text_column] for text_column in self.text_columns])
            outputs = self.tokenizer(text, add_special_tokens=True, padding='max_length', truncation=True)
            data.append(outputs['input_ids'])
            attention_masks.append(outputs['attention_mask'])

        return data, attention_masks

    def preprocessing(self, data):
        data = data.drop(columns=self.delete_columns)
        
        try:
            targets = data[self.target_columns].values.tolist()
        except:
            targets = []
        inputs, attention_masks = self.tokenizing(data)

        return inputs, attention_masks, targets


    def setup(self, stage='fit'):
        if stage == 'fit':
            train_data = pd.read_csv(self.train_path)
            val_data = pd.read_csv(self.dev_path)

            train_inputs, train_attention_masks, train_targets = self.preprocessing(train_data)

            val_inputs, val_attention_masks, val_targets = self.preprocessing(val_data)

            self.train_dataset = Dataset(train_inputs, train_attention_masks, train_targets)
            self.val_dataset = Dataset(val_inputs, val_attention_masks, val_targets)
        else:
            test_data = pd.read_csv(self.test_path)
            test_inputs, test_attention_masks, test_targets = self.preprocessing(test_data)
            self.test_dataset = Dataset(test_inputs, test_attention_masks, test_targets)

            predict_data = pd.read_csv(self.predict_path)
            predict_inputs, predict_attention_masks, predict_targets = self.preprocessing(predict_data)
            self.predict_dataset = Dataset(predict_inputs, predict_attention_masks, [])

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=self.shuffle)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def predict_dataloader(self):
        return torch.utils.data.DataLoader(self.predict_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
