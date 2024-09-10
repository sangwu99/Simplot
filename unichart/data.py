import json, os
import random
from typing import Any, List, Tuple
from PIL import Image
import torch
from torch.utils.data import Dataset
from transformers import DonutProcessor
# from datasets import load_dataset, load_from_disk

added_tokens = []

class ChartQADataset(Dataset):
    """
    """

    def __init__(
        self,
        dataset: str,
        images_folder: str,
        max_length: int,
        processor : DonutProcessor = None,
        split: str = "train",
        ignore_id: int = -100,
        prompt_end_token: str = None,
        task_prefix: str = '<chartqa>',
        sort_json_key: bool = True,
    ):
        super().__init__()

        self.max_length = max_length
        self.split = split
        self.ignore_id = ignore_id

        self.prompt_end_token = prompt_end_token 
        self.sort_json_key = sort_json_key
        self.images_folder = images_folder

  
        self.dataset = dataset
        self.dataset_length = len(self.dataset)

        self.processor = processor
        self.prompt_end_token_id = self.processor.tokenizer.convert_tokens_to_ids(self.prompt_end_token)
        self.task_prefix = task_prefix


    def __len__(self) -> int:
        return self.dataset_length

    def __getitem__(self, idx: int):

        sample = self.dataset[idx]

        # input_tensor
        img_path = os.path.join(self.images_folder, sample['imgname'])
        img = Image.open(img_path)
        pixel_values = self.processor(img.convert("RGB"), random_padding=self.split == "train", return_tensors="pt").pixel_values
        input_tensor = pixel_values.squeeze()

        # input_ids
        processed_parse = self.task_prefix + " " + sample['query'] + " " + self.prompt_end_token + " " + sample['label'] + self.processor.tokenizer.eos_token 
        input_ids = self.processor.tokenizer(
            processed_parse,
            add_special_tokens=False,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )["input_ids"].squeeze(0)

        if self.split == "train":
            labels = input_ids.clone()
            labels[
                labels == self.processor.tokenizer.pad_token_id
            ] = self.ignore_id  # model doesn't need to predict pad token
            labels[
                : torch.nonzero(labels == self.prompt_end_token_id).sum() + 1
            ] = self.ignore_id  # model doesn't need to predict prompt 
            return input_tensor, input_ids, labels
        else:
            prompt_end_index = torch.nonzero(
                input_ids == self.prompt_end_token_id
            ).sum()  # return prompt end index instead of target output labels
            return input_tensor, input_ids, prompt_end_index, processed_parse

class Phase1Dataset(Dataset):
    """
    """

    def __init__(
        self,
        dataset: str,
        max_length: int,
        processor : DonutProcessor = None,
        ignore_id: int = -100,
        prompt_end_token: str = "<s_answer>",
        task_prefix: str = '<extract_data_table>',
        split: str = "train",
    ):
        super().__init__()

        self.max_length = max_length
        self.ignore_id = ignore_id

        self.prompt_end_token = prompt_end_token 
          
        self.dataset = dataset
        self.dataset_length = len(self.dataset)

        self.processor = processor
        self.prompt_end_token_id = self.processor.tokenizer.convert_tokens_to_ids(self.prompt_end_token)
        self.task_prefix = task_prefix
        self.split = split


    def __len__(self) -> int:
        return self.dataset_length

    def __getitem__(self, idx: int):

        sample = self.dataset[idx]

        # input_tensor
        img = Image.open(sample['image'])
        indexes = sample['row']
        columns = sample['col']
        label = sample['text']
        
        pixel_values = self.processor(img.convert("RGB"), random_padding=self.split == "train", return_tensors="pt").pixel_values
        input_tensor = pixel_values.squeeze()

        # input_ids
        processed_parse = self.task_prefix + " <columns> " + columns + " <rows> " + indexes + " " + self.prompt_end_token + " " + label + self.processor.tokenizer.eos_token 
        
        input_ids = self.processor.tokenizer(
            processed_parse,
            add_special_tokens=False,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )["input_ids"].squeeze(0)

        labels = input_ids.clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = self.ignore_id  
        labels[:torch.nonzero(labels == self.prompt_end_token_id).sum() + 1] = self.ignore_id  
        
        return input_tensor, input_ids, labels


class Phase2Dataset(Dataset):
    """
    """

    def __init__(
        self,
        dataset: str,
        max_length: int,
        processor : DonutProcessor = None,
        ignore_id: int = -100,
        prompt_end_token: str = "<s_answer>",
        task_prefix: str = '<extract_data_table>',
        split: str = "train",
    ):
        super().__init__()

        self.max_length = max_length
        self.ignore_id = ignore_id

        self.prompt_end_token = prompt_end_token 
          
        self.dataset = dataset
        self.dataset_length = len(self.dataset)

        self.processor = processor
        self.prompt_end_token_id = self.processor.tokenizer.convert_tokens_to_ids(self.prompt_end_token)
        self.task_prefix = task_prefix
        self.split = split


    def __len__(self) -> int:
        return self.dataset_length

    def __getitem__(self, idx: int):

        sample = self.dataset[idx]

        # input_tensor
        img = Image.open(sample['image'])
        pos_img = Image.open(sample['positive_image'])
        neg_img = Image.open(sample['negative_image'])
        indexes = sample['row']
        columns = sample['col']
        label = sample['text']
        
        anchor_pixel_values = self.processor(img.convert("RGB"), random_padding=self.split == "train", return_tensors="pt").pixel_values
        pos_pixel_values = self.processor(pos_img.convert("RGB"), random_padding=self.split == "train", return_tensors="pt").pixel_values
        neg_pixel_values = self.processor(neg_img.convert("RGB"), random_padding=self.split == "train", return_tensors="pt").pixel_values
        
        anchor_input_tensor = anchor_pixel_values.squeeze()
        pos_input_tensor = pos_pixel_values.squeeze()
        neg_input_tensor = neg_pixel_values.squeeze()

        # input_ids
        processed_parse = self.task_prefix + " <columns> " + columns + " <rows> " + indexes + " " + self.prompt_end_token + " " + label + self.processor.tokenizer.eos_token 
        
        input_ids = self.processor.tokenizer(
            processed_parse,
            add_special_tokens=False,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )["input_ids"].squeeze(0)

        labels = input_ids.clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = self.ignore_id  # model doesn't need to predict pad token
        labels[:torch.nonzero(labels == self.prompt_end_token_id).sum() + 1] = self.ignore_id  # model doesn't need to predict prompt 
        
        return anchor_input_tensor, pos_input_tensor, neg_input_tensor, input_ids, labels
    
class SimplotTestDataset(Dataset):
    """
    """

    def __init__(
        self,
        dataset: str,
        processor : DonutProcessor = None,
        prompt_end_token: str = "<s_answer>",
        task_prefix: str = '<extract_data_table>',
    ):
        super().__init__()

        self.prompt_end_token = prompt_end_token 
          
        self.dataset = dataset
        self.dataset_length = len(self.dataset)

        self.processor = processor
        self.prompt_end_token_id = self.processor.tokenizer.convert_tokens_to_ids(self.prompt_end_token)
        self.task_prefix = task_prefix


    def __len__(self) -> int:
        return self.dataset_length

    def __getitem__(self, idx: int):

        sample = self.dataset[idx]

        # input_tensor
        img = Image.open(sample['image'])
        indexes = sample['row']
        columns = sample['col']
        label = sample['text']
        
        pixel_values = self.processor(img.convert("RGB"), return_tensors="pt").pixel_values
        input_tensor = pixel_values.squeeze()

        # input_ids
        processed_parse = self.task_prefix + " <columns> " + columns + " <rows> " + indexes + " " + self.prompt_end_token 
        
        input_ids = self.processor.tokenizer(
            processed_parse,
            add_special_tokens=False,
            return_tensors="pt",
        )["input_ids"].squeeze(0)

        
        return input_tensor, input_ids, label
    
class BaselineDataset(Dataset):
    """
    """

    def __init__(
        self,
        dataset: str,
        max_length: int,
        processor : DonutProcessor = None,
        ignore_id: int = -100,
        prompt_end_token: str = "<s_answer>",
        task_prefix: str = '<extract_data_table>',
        split: str = "train",
    ):
        super().__init__()

        self.max_length = max_length
        self.ignore_id = ignore_id

        self.prompt_end_token = prompt_end_token 
          
        self.dataset = dataset
        self.dataset_length = len(self.dataset)

        self.processor = processor
        self.prompt_end_token_id = self.processor.tokenizer.convert_tokens_to_ids(self.prompt_end_token)
        self.task_prefix = task_prefix
        self.split = split


    def __len__(self) -> int:
        return self.dataset_length

    def __getitem__(self, idx: int):

        sample = self.dataset[idx]

        # input_tensor
        img = Image.open(sample['image'])
        label = sample['text']
        
        pixel_values = self.processor(img.convert("RGB"), random_padding=self.split == "train", return_tensors="pt").pixel_values
        input_tensor = pixel_values.squeeze()

        # input_ids
        processed_parse = self.task_prefix + " " +  self.prompt_end_token + " " + label + self.processor.tokenizer.eos_token 
        
        input_ids = self.processor.tokenizer(
            processed_parse,
            add_special_tokens=False,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )["input_ids"].squeeze(0)

        labels = input_ids.clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = self.ignore_id  
        labels[:torch.nonzero(labels == self.prompt_end_token_id).sum() + 1] = self.ignore_id  
        
        return input_tensor, input_ids, labels
    

class BaselineTestDataset(Dataset):
    """
    """

    def __init__(
        self,
        dataset: str,
        processor : DonutProcessor = None,
        ignore_id: int = -100,
        prompt_end_token: str = "<s_answer>",
        task_prefix: str = '<extract_data_table>',
    ):
        super().__init__()

        self.ignore_id = ignore_id
        self.prompt_end_token = prompt_end_token 
        self.dataset = dataset
        self.dataset_length = len(self.dataset)

        self.processor = processor
        self.prompt_end_token_id = self.processor.tokenizer.convert_tokens_to_ids(self.prompt_end_token)
        self.task_prefix = task_prefix


    def __len__(self) -> int:
        return self.dataset_length

    def __getitem__(self, idx: int):

        sample = self.dataset[idx]

        # input_tensor
        img = Image.open(sample['image'])
        label = sample['text']
        
        pixel_values = self.processor(img.convert("RGB"), return_tensors="pt").pixel_values
        input_tensor = pixel_values.squeeze()

        # input_ids
        processed_parse = self.task_prefix + " " +  self.prompt_end_token
        
        input_ids = self.processor.tokenizer(
            processed_parse,
            add_special_tokens=False,
            return_tensors="pt",
        )["input_ids"].squeeze(0)

        return input_tensor, input_ids, label

class Phase1NoRenderDataset(Dataset):
    """
    """

    def __init__(
        self,
        dataset: str,
        max_length: int,
        processor : DonutProcessor = None,
        ignore_id: int = -100,
        prompt_end_token: str = "<s_answer>",
        task_prefix: str = '<extract_data_table>',
        split: str = "train",
    ):
        super().__init__()

        self.max_length = max_length
        self.ignore_id = ignore_id

        self.prompt_end_token = prompt_end_token 
          
        self.dataset = dataset
        self.dataset_length = len(self.dataset)

        self.processor = processor
        self.prompt_end_token_id = self.processor.tokenizer.convert_tokens_to_ids(self.prompt_end_token)
        self.task_prefix = task_prefix
        self.split = split


    def __len__(self) -> int:
        return self.dataset_length

    def __getitem__(self, idx: int):

        sample = self.dataset[idx]

        # input_tensor
        img = Image.open(sample['image'])
        indexes = sample['row']
        columns = sample['col']
        label = sample['text']
        
        pixel_values = self.processor(img.convert("RGB"), random_padding=self.split == "train", return_tensors="pt").pixel_values
        input_tensor = pixel_values.squeeze()

        # input_ids
        processed_parse = self.task_prefix + " " +  self.prompt_end_token + " " + label + self.processor.tokenizer.eos_token 
        
        input_ids = self.processor.tokenizer(
            processed_parse,
            add_special_tokens=False,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )["input_ids"].squeeze(0)

        labels = input_ids.clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = self.ignore_id  
        labels[:torch.nonzero(labels == self.prompt_end_token_id).sum() + 1] = self.ignore_id  
        
        return input_tensor, input_ids, labels
    

class Phase2NoRenderDataset(Dataset):
    """
    """

    def __init__(
        self,
        dataset: str,
        max_length: int,
        processor : DonutProcessor = None,
        ignore_id: int = -100,
        prompt_end_token: str = "<s_answer>",
        task_prefix: str = '<extract_data_table>',
        split: str = "train",
    ):
        super().__init__()

        self.max_length = max_length
        self.ignore_id = ignore_id

        self.prompt_end_token = prompt_end_token 
          
        self.dataset = dataset
        self.dataset_length = len(self.dataset)

        self.processor = processor
        self.prompt_end_token_id = self.processor.tokenizer.convert_tokens_to_ids(self.prompt_end_token)
        self.task_prefix = task_prefix
        self.split = split


    def __len__(self) -> int:
        return self.dataset_length

    def __getitem__(self, idx: int):

        sample = self.dataset[idx]

        # input_tensor
        img = Image.open(sample['image'])
        pos_img = Image.open(sample['positive_image'])
        neg_img = Image.open(sample['negative_image'])
        indexes = sample['row']
        columns = sample['col']
        label = sample['text']
        
        anchor_pixel_values = self.processor(img.convert("RGB"), random_padding=self.split == "train", return_tensors="pt").pixel_values
        pos_pixel_values = self.processor(pos_img.convert("RGB"), random_padding=self.split == "train", return_tensors="pt").pixel_values
        neg_pixel_values = self.processor(neg_img.convert("RGB"), random_padding=self.split == "train", return_tensors="pt").pixel_values
        
        anchor_input_tensor = anchor_pixel_values.squeeze()
        pos_input_tensor = pos_pixel_values.squeeze()
        neg_input_tensor = neg_pixel_values.squeeze()

        # input_ids
        processed_parse = self.task_prefix + " " +  self.prompt_end_token + " " + label + self.processor.tokenizer.eos_token 
        
        input_ids = self.processor.tokenizer(
            processed_parse,
            add_special_tokens=False,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )["input_ids"].squeeze(0)

        labels = input_ids.clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = self.ignore_id  # model doesn't need to predict pad token
        labels[:torch.nonzero(labels == self.prompt_end_token_id).sum() + 1] = self.ignore_id  # model doesn't need to predict prompt 
        
        return anchor_input_tensor, pos_input_tensor, neg_input_tensor, input_ids, labels
    

class UniformatPhase1Dataset(Dataset):
    """
    """

    def __init__(
        self,
        dataset: str,
        max_length: int,
        processor : DonutProcessor = None,
        ignore_id: int = -100,
        prompt_end_token: str = "<s_answer>",
        task_prefix: str = '<extract_data_table>',
        split: str = "train",
    ):
        super().__init__()

        self.max_length = max_length
        self.ignore_id = ignore_id

        self.prompt_end_token = prompt_end_token 
          
        self.dataset = dataset
        self.dataset_length = len(self.dataset)

        self.processor = processor
        self.prompt_end_token_id = self.processor.tokenizer.convert_tokens_to_ids(self.prompt_end_token)
        self.task_prefix = task_prefix
        self.split = split


    def __len__(self) -> int:
        return self.dataset_length

    def __getitem__(self, idx: int):

        sample = self.dataset[idx]

        # input_tensor
        img = Image.open(sample['image'])
        indexes = sample['row']
        columns = sample['col']
        label = sample['text']
        
        pixel_values = self.processor(img.convert("RGB"), random_padding=self.split == "train", return_tensors="pt").pixel_values
        input_tensor = pixel_values.squeeze()

        # input_ids
        processed_parse = self.task_prefix + " columns are " + columns + " rows are " + indexes + " " + self.prompt_end_token + " " + label + self.processor.tokenizer.eos_token 
        
        input_ids = self.processor.tokenizer(
            processed_parse,
            add_special_tokens=False,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )["input_ids"].squeeze(0)

        labels = input_ids.clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = self.ignore_id  
        labels[:torch.nonzero(labels == self.prompt_end_token_id).sum() + 1] = self.ignore_id  
        
        return input_tensor, input_ids, labels
    

class UniformatTestDataset(Dataset):
    """
    """

    def __init__(
        self,
        dataset: str,
        processor : DonutProcessor = None,
        ignore_id: int = -100,
        prompt_end_token: str = "<s_answer>",
        task_prefix: str = '<extract_data_table>',
    ):
        super().__init__()

        self.ignore_id = ignore_id

        self.prompt_end_token = prompt_end_token 
          
        self.dataset = dataset
        self.dataset_length = len(self.dataset)

        self.processor = processor
        self.prompt_end_token_id = self.processor.tokenizer.convert_tokens_to_ids(self.prompt_end_token)
        self.task_prefix = task_prefix


    def __len__(self) -> int:
        return self.dataset_length

    def __getitem__(self, idx: int):

        sample = self.dataset[idx]

        # input_tensor
        img = Image.open(sample['image'])
        indexes = sample['row']
        columns = sample['col']
        label = sample['text']
        
        pixel_values = self.processor(img.convert("RGB"), return_tensors="pt").pixel_values
        input_tensor = pixel_values.squeeze()

        # input_ids
        processed_parse = self.task_prefix + " columns are " + columns + " rows are " + indexes + " " + self.prompt_end_token 
        
        input_ids = self.processor.tokenizer(
            processed_parse,
            add_special_tokens=False,
            return_tensors="pt",
        )["input_ids"].squeeze(0)
        
        return input_tensor, input_ids, label
    
