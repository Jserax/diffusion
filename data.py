from typing import Optional, Tuple

import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
from transformers.models.bert.tokenization_bert_fast import BertTokenizerFast


class DiffDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        caption_path: str,
        img_dir: str,
        tokenizer: BertTokenizerFast,
        max_length: int = 64,
        transforms: Optional[transforms.Compose] = None,
    ) -> None:
        super().__init__()
        data = pd.read_csv(caption_path)
        self.images = (
            img_dir + "/" + data.image_id.astype("str").str.zfill(12) + ".jpg"
        ).tolist()
        data = tokenizer.batch_encode_plus(
            data.caption.tolist(),
            add_special_tokens=True,
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
            return_token_type_ids=False,
        )
        self.tokens = data["input_ids"]
        self.attn_mask = data["attention_mask"]

        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        image = Image.open(self.images[idx])
        if self.transforms:
            image = self.transforms(image)
        tokens = self.tokens[idx]
        attn_mask = self.attn_mask[idx]
        return image, tokens, attn_mask
