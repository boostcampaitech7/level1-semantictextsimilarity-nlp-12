import torch

class Dataset(torch.utils.data.Dataset):
    def __init__(self, inputs, attention_masks, targets=[]):
        self.inputs = inputs
        self.attention_masks = attention_masks
        self.targets = targets

    def __getitem__(self, idx):
        if len(self.targets) == 0:
            return torch.tensor(self.inputs[idx]), torch.tensor(self.attention_masks[idx])
        else:
            return torch.tensor(self.inputs[idx]), torch.tensor(self.attention_masks[idx]), torch.tensor(self.targets[idx])

    def __len__(self):
        return len(self.inputs)