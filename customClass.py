import os
import numpy as np
import random
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from nltk.tokenize import word_tokenize
from PIL import Image
from tqdm import tqdm
from collections import Counter
import matplotlib.pyplot as plt
import pandas as pd


class CustomDataset2(Dataset):
    def __init__(self, data_dir, captions_file, transform=None, percentage=100):
        self.data_dir = data_dir
        self.transform = transform
        self.batch_size = 32
        self.image_size= 224
        self.min_word_frequency= 5
        self.percentage = percentage
        self.caption_padding= True
        
        # Load captions from the JSON file
        with open(captions_file, "r") as file:
            self.data = json.load(file)["annotations"]

        # Use the specified percentage of the data
        if percentage < 100:
            data_size = len(self.data)
            sample_size = int(data_size * (percentage / 100))
            self.data = random.sample(self.data, sample_size)

        
    
        #Now the train/val/test
        self.train_data, self.val_data, self.test_data = self.split_data()
        
        # Initialize vocabulary
        self.vocab = {
            "<PAD>": 0,
            "<START>": 1,
            "<END>": 2,
            "<UNK>": 3,
        }

        # Tokenize captions and build vocabulary
        self.build_vocab()
    

    
    def build_vocab(self):
        all_tokens = []

        for entry in self.data:
            caption = entry["caption"]
            tokens = word_tokenize(caption.lower())
            all_tokens.extend(tokens)

        token_counter = Counter(all_tokens)

        for token, freq in token_counter.items():
            if freq >= self.min_word_frequency:
                self.vocab[token] = len(self.vocab)
        # Compute the maximum caption length
        self.max_caption_length = max(len(word_tokenize(entry["caption"])) + 2 for entry in self.data)

    def save_captions_to_csv(self, filename):
        captions_data = []
        max_caption_length = 0

        for entry in self.data:
            caption = entry["caption"]
            tokens = word_tokenize(caption.lower())
            caption_encoded = [self.vocab.get(token, self.vocab["<UNK>"]) for token in tokens]
            caption_encoded = [self.vocab["<START>"]] + caption_encoded + [self.vocab["<END>"]]

            if len(caption_encoded) > max_caption_length:
                max_caption_length = len(caption_encoded)

            captions_data.append({"id": entry["image_id"], "caption": caption_encoded})

        # Pad captions to the maximum length
        for caption_data in captions_data:
            caption_data["caption"] += [self.vocab["<PAD>"]] * (max_caption_length - len(caption_data["caption"]))

        df = pd.DataFrame(captions_data)
        df.to_csv(filename, index=False)

    def save_vocab_to_csv(self, filename):
        vocab_data = [{"token": token, "id": token_id} for token, token_id in self.vocab.items()]
        df = pd.DataFrame(vocab_data)
        df.to_csv(filename, index=False)
    def split_data(self):
        num_samples = len(self.data)
        train_size = int(0.8 * num_samples)  # 80% for training
        val_size = int(0.1 * num_samples)  # 10% for validation
        test_size = num_samples - train_size - val_size  # Remaining 10% for testing

        train_data, val_data, test_data = torch.utils.data.random_split(
            self.data, [train_size, val_size, test_size]
        )

        return train_data, val_data, test_data

    def __len__(self):
        return len(self.data)
    



    #Getters

    def __getitem__(self, idx):
        entry = self.data[idx]
        image_id = entry["image_id"]
        caption = entry["caption"]
        image_path = os.path.join(self.data_dir, f"{image_id:012d}.jpg")

        image = Image.open(image_path).convert("RGB")

        # Apply image transformations if specified
        if self.transform is not None:
            image = self.transform(image)

        # Tokenize and encode caption with start and end tokens
        tokens = word_tokenize(caption.lower())
        caption_encoded = [self.vocab.get(token, self.vocab["<UNK>"]) for token in tokens]
        caption_encoded = [self.vocab["<START>"]] + caption_encoded + [self.vocab["<END>"]]

        # Pad caption sequence to max_caption_length
        if self.caption_padding:
            if len(caption_encoded) < self.max_caption_length:
                caption_encoded.extend([self.vocab["<PAD>"]] * (self.max_caption_length - len(caption_encoded)))
            else:
                caption_encoded = caption_encoded[:self.max_caption_length]

        # Convert caption to a tensor
        caption_tensor = torch.tensor(caption_encoded,dtype=torch.long)

        return image, caption_tensor
    

    def getImagePath(self,idx):
        entry = self.data[idx]
        image_id = entry["image_id"]
        return   os.path.join(self.data_dir, f"{image_id:012d}.jpg")
    def getImage(self,idx):
        entry = self.data[idx]
        image_id = entry["image_id"]
        image_path = os.path.join(self.data_dir, f"{image_id:012d}.jpg")
        return Image.open(image_path).convert("RGB")
    def getCaption(self, idx):
        entry = self.data[idx]
        caption = entry["caption"]

        # Tokenize and encode caption with start and end tokens
        tokens = word_tokenize(caption.lower())
        caption_encoded = [self.vocab.get(token, self.vocab["<UNK>"]) for token in tokens]
        caption_encoded = [self.vocab["<START>"]] + caption_encoded + [self.vocab["<END>"]]

        return caption, caption_encoded
    def print_image_with_caption(self, idx):
        entry = self.data[idx]
        image_id = entry["image_id"]
        caption = entry["caption"]
        image_path = os.path.join(self.data_dir, f"{image_id:012d}.jpg")
        image = Image.open(image_path).convert("RGB")
        # Display the image with its caption
        plt.figure(figsize=(8, 8))
        plt.imshow(image)
        plt.axis('off')
        plt.title(f"Caption: {caption}")
        plt.show()
    def print_image_with_caption_with_model(self,model,idx):
        entry = self.data[idx]
        image_id = entry["image_id"]
        caption = entry["caption"]
        
        image_path = os.path.join(self.data_dir, f"{image_id:012d}.jpg")
        image = Image.open(image_path).convert("RGB")

        generated_caption = model.generate_caption(image).to(self.device)
        generated_caption_words = [self.vocab[token_id] for token_id in generated_caption]
        # Display the image with its caption
        plt.figure(figsize=(8, 8))
        plt.imshow(image)
        plt.axis('off')
        plt.title(f"Caption: {caption}")
        plt.text("Generated caption:",generated_caption_words)
        plt.show()























































