import random
import os
import torch
from .file_dataset import FileDataset
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from io import BytesIO
import base64
import glob
import json

# turn off image boom detection
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
ImageFile.MAX_IMAGE_PIXELS = None
Image.MAX_IMAGE_PIXELS = None


def whole_word_mask(tokenizer, input_text, mask_probability=0.15, input_ids=None):
    mask_id = tokenizer.mask_token_id
    vocab_size = tokenizer.vocab_size

    # Tokenize the input text
    tokenized_input = tokenizer(input_text, return_tensors='pt', add_special_tokens=True)
    input_ids = tokenized_input['input_ids'][0]
    raw_input_ids = input_ids.clone()

    # Get special tokens mask
    special_tokens_mask = tokenizer.get_special_tokens_mask(input_ids.tolist(), already_has_special_tokens=True)
    special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)

    probability_matrix = torch.full(input_ids.shape, mask_probability)
    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()

    # Create mask for whole words
    words = input_text.split()
    word_start_indices = []
    current_index = 0

    for word in words:
        start_index = current_index
        current_index += len(tokenizer.encode(word, add_special_tokens=False))
        word_start_indices.append((start_index, current_index))

    for start_index, end_index in word_start_indices:
        if start_index < len(masked_indices) and masked_indices[start_index]:
            masked_indices[start_index:end_index] = True

    # Apply masking
    for i in range(len(input_ids)):
        if masked_indices[i]:
            rand = random.random()
            if rand < 0.8:  # 80% of the time, replace with mask id
                input_ids[i] = mask_id
            elif rand < 0.9:  # 10% of the time, leave it as is
                continue
            else:  # 10% of the time, replace with a random token
                input_ids[i] = random.randint(0, vocab_size - 1)

    return raw_input_ids, input_ids


class MMDataset(Dataset):
    def __init__(self, 
                 text_tokenizer,
                 image_dir,
                 text_dir,
                 transforms,
                 max_length=100,
                 image_index=None,
                 text_index=None
                 ):
        if image_index is None:
            self.image_index = [0, 1]
        else:
            self.image_index = image_index

        if text_index is None:
            self.text_index = [0, 1]
        else:
            self.text_index = text_index
        
        self.image_dataset = FileDataset(image_dir, self.image_index, cached_index=True) if image_dir is not None else None
        self.text_dataset = FileDataset(text_dir, text_index, cached_index=True) if text_dir is not None else None
        self.text_tokenizer = text_tokenizer
        self.trans = transforms
        self.max_length = max_length

    def __len__(self):
        return len(self.image_dataset)

    def process_image(self, index):

        index = index % len(self.image_dataset)

        while True:
            try:
                uniq_id, image = self.image_dataset[index]  # imagenet21k
                image = Image.open(BytesIO(base64.urlsafe_b64decode(image))).convert("RGB")
                break
            except:
                print(f"image {uniq_id} failed!")
                index = random.randint(0, len(self.image_dataset) - 1)

        image, second_image, img_mask = self.trans(image)

        sample = {
            "image": image,
            "second_image": second_image,
            "image_mask": img_mask
            }

        return sample

    def process_text(self, index):
        
        while True:
            index = index % len(self.text_dataset)
            uniq_id, text = self.text_dataset[index]
            text_input_ids, masked_ids = whole_word_mask(self.text_tokenizer, text, mask_probability=0.15)
            
            if len(text_input_ids) > 30:
                break
            else:
                index += 1

        # if the length exceed max_len
        if len(text_input_ids) > self.max_length:
            
            start_pos = random.randint(1, len(text_input_ids) - self.max_length)
            
            bos = text_input_ids[0].unsqueeze(0)  
            middle_part = text_input_ids[start_pos:start_pos + self.max_length - 2]  
            eos = text_input_ids[-1].unsqueeze(0)  
            text_input_ids = torch.cat((bos, middle_part, eos))

            bos = masked_ids[0].unsqueeze(0)  
            middle_part = masked_ids[start_pos:start_pos + self.max_length - 2]  
            eos = masked_ids[-1].unsqueeze(0)  
            masked_ids = torch.cat((bos, middle_part, eos))


        num_tokens = len(masked_ids)
        text_masked_ids = list(masked_ids.numpy()) + [self.text_tokenizer.pad_token_id] * (self.max_length - num_tokens)
        text_input_ids = list(text_input_ids.numpy()) + [self.text_tokenizer.pad_token_id] * (self.max_length - num_tokens)
        padding_mask = [0] * num_tokens + [1] * (self.max_length - num_tokens)

        sample = {"text_input_ids": torch.tensor(text_input_ids),
                  "text_masked_ids": torch.tensor(text_masked_ids),
                  "text_padding_mask": torch.tensor(padding_mask)
                  }

        return sample

    def __getitem__(self, index):
        ret = dict()

        if self.image_dataset is not None:
            image_sample = self.process_image(index)
            ret.update(image_sample)

        if self.text_dataset is not None:
            text_sample = self.process_text(index)
            ret.update(text_sample)

        return ret



class Quilt1mDataset(Dataset):
    def __init__(self, 
                 text_tokenizer,
                 image_dir,
                 text_dir,
                 transforms,
                 max_length=100,
                 image_index=None,
                 text_index=None
                 ):
        if image_index is None:
            self.image_index = [0, 1]
        else:
            self.image_index = image_index

        if text_index is None:
            self.text_index = [0, 1]
        else:
            self.text_index = text_index
        
        self.text_counter = 0  # Initialize a counter for text data

        if os.path.exists(image_dir):
            self.image_dataset = list(glob.glob(f"{image_dir}/*"))
        else:
            self.image_dataset = None

        if os.path.exists(text_dir):
            items = []
            with open(text_dir, mode="r", encoding="utf-8") as reader:
                for line in reader:
                    data = json.loads(line)
                    items.append(data)
                print("Load %d image-text pairs from %s. " % (len(items), text_dir))
            self.text_dataset = items
        else:
            self.text_dataset = None

        self.text_tokenizer = text_tokenizer
        self.trans = transforms
        self.max_length = max_length
        self.bos = text_tokenizer.convert_tokens_to_ids(text_tokenizer.bos_token)
        self.eos = text_tokenizer.convert_tokens_to_ids(text_tokenizer.eos_token)

    def __len__(self):
        return len(self.image_dataset)

    def process_image(self, index):
        index = index % len(self.image_dataset)
        image = Image.open(self.image_dataset[index]).convert("RGB")
        image, second_image, img_mask = self.trans(image)
        sample = {
            "image": image,
            "second_image": second_image,
            "image_mask": img_mask
            }
        return sample

    def process_text(self, index):
        self.text_counter = self.text_counter % len(self.text_dataset)
        text = self.text_dataset[self.text_counter]['text_segment']
        text = self.text_tokenizer.decode([self.bos] +  text + [self.eos])

        text_input_ids, masked_ids = whole_word_mask(self.text_tokenizer, text, mask_probability=0.15)

        # if the length exceed max_len
        if len(text_input_ids) > self.max_length:
            
            start_pos = random.randint(1, len(text_input_ids) - self.max_length)
            
            bos = text_input_ids[0].unsqueeze(0)  
            middle_part = text_input_ids[start_pos:start_pos + self.max_length - 2]  
            eos = text_input_ids[-1].unsqueeze(0)  
            text_input_ids = torch.cat((bos, middle_part, eos))

            bos = masked_ids[0].unsqueeze(0)  
            middle_part = masked_ids[start_pos:start_pos + self.max_length - 2]  
            eos = masked_ids[-1].unsqueeze(0)  
            masked_ids = torch.cat((bos, middle_part, eos))


        num_tokens = len(masked_ids)
        text_masked_ids = list(masked_ids.numpy()) + [self.text_tokenizer.pad_token_id] * (self.max_length - num_tokens)
        text_input_ids = list(text_input_ids.numpy()) + [self.text_tokenizer.pad_token_id] * (self.max_length - num_tokens)
        padding_mask = [0] * num_tokens + [1] * (self.max_length - num_tokens)

        sample = {"text_input_ids": torch.tensor(text_input_ids),
                  "text_masked_ids": torch.tensor(text_masked_ids),
                  "text_padding_mask": torch.tensor(padding_mask)
                  }
        
        self.text_counter += 1

        return sample

    def __getitem__(self, index):
        ret = dict()

        if self.image_dataset is not None:
            image_sample = self.process_image(index)
            ret.update(image_sample)

        if self.text_dataset is not None:
            text_sample = self.process_text(index)
            ret.update(text_sample)

        return ret

