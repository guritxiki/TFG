import os
import pandas as pd
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from PIL import Image
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from collections import Counter
import random
import ast
random.seed(42)
class CustomDatasetFromCSV(Dataset):
    def __init__(self, data_dir, captions_csv, vocab_csv, transform=None, percentage=1):
        self.data_dir = data_dir
        self.transform = transform

        # Load captions from CSV file
        captions_df = pd.read_csv(captions_csv)
        
        
        # Remove brackets from captions
        captions_df['caption'] = captions_df['caption'].str.strip('[]').replace(' ', '').str.split(',')

        #captions_df['caption'] = captions_df['caption'].apply(lambda x: ast.literal_eval(x))
        self.data = captions_df.to_dict(orient='records')
        # Load vocabulary from CSV file
        vocab_df = pd.read_csv(vocab_csv)
        self.vocab = {row['token']: row['id'] for _, row in vocab_df.iterrows()}  # Use token as key and id as value
        
        # Reverse the vocabulary for mapping id to token
        self.id_to_token = {row['id']: row['token'] for _, row in vocab_df.iterrows()}
        
        # ENSURE ALL DATA IS OKAY
        self.maxCaptionLength= self.findLongestCaption()
        



        # Use the specified percentage of the data
        if percentage < 100:
            data_size = len(self.data)
            sample_size = int(data_size * (percentage / 100))
            self.data = random.sample(self.data, sample_size)
        
        self.split_data(percentage=100)
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        entry = self.data[idx]
        image_id = entry["id"]
        caption_ids = entry["caption"]
        caption_ids = [int(token_id) for token_id in caption_ids]  # Convert string tokens to integers
        image_path = os.path.join(self.data_dir, f"{image_id:012d}.jpg")

        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        # Now, you can safely create a tensor with the list of integers
        caption_tensor = torch.tensor(caption_ids, dtype=torch.long)

        return image, caption_tensor
        
    
    def split_data(self, percentage):
        data_size = len(self.data)
        train_size = int(data_size * 0.8)
        val_size = int(data_size * 0.1)
        test_size = data_size - train_size - val_size

        if percentage < 100:
            train_size = int(train_size * (percentage / 100))
            val_size = int(val_size * (percentage / 100))
            test_size = int(test_size * (percentage / 100))

        # Randomly shuffle the data before splitting
        random.shuffle(self.data)

        self.train_data = self.data[:train_size]
        self.val_data = self.data[train_size:(train_size + val_size)]
        self.test_data = self.data[(train_size + val_size):(train_size + val_size + test_size)]

    

    def print_image_with_caption(self, idx):
        entry = self.data[idx]
        image_id = entry["id"]
        image_path = os.path.join(self.data_dir, f"{image_id:012d}.jpg")
        image = Image.open(image_path).convert("RGB")
        caption = self.get_text_caption(idx)
        

        plt.figure(figsize=(8, 8))
        plt.imshow(image)
        plt.axis('off')
        plt.title(f"Caption: {caption}")
        plt.show()

    def get_image(self, idx):
        entry = self.data[idx]
        image_id = entry["id"]
        image_path = os.path.join(self.data_dir, f"{image_id:012d}.jpg")
        return Image.open(image_path).convert("RGB")

    def get_caption(self, idx):
        entry = self.data[idx]
        caption = entry["caption"]
        return caption

    def get_text_caption(self, idx):
        caption_ids = self.data[idx]['caption']
        
        # Assuming that self.id_to_token is a dictionary mapping token IDs to text tokens
        token_text_list = [self.id_to_token[int(token_id)] for token_id in caption_ids if self.id_to_token[int(token_id)] != '<PAD>']
        return ' '.join(token_text_list)

    

    ###  FUNCTIONS TO ENSURE THAT THE DATA IS GOOD
    def allCaptionsSameSize(self):
        max_caption_length = len(self.data[0]["caption"])
        min_caption_length = len(self.data[0]["caption"])

        for idx in range(1, len(self.data)):
            entry = self.data[idx]
            caption_length = len(entry["caption"])

            if caption_length > max_caption_length:
                max_caption_length = caption_length
            if caption_length < min_caption_length:
                min_caption_length = caption_length

        # Check if all captions are the same length
        is_same_length = max_caption_length == min_caption_length

        return is_same_length
    
    def findLongestCaption(self):
        max_caption_length = len(self.data[0]["caption"])
        longest_caption = self.data[0]["caption"]

        for idx in range(1, len(self.data)):
            entry = self.data[idx]
            caption_length = len(entry["caption"])

            if caption_length > max_caption_length:
                max_caption_length = caption_length
                longest_caption = entry["caption"]

        print("Longest Caption:", longest_caption)
        print("Length:", max_caption_length)

        return max_caption_length
    
    

        
    





