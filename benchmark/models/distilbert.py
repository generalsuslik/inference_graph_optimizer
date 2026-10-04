import torch
from transformers import AutoModel


SEQUENCE_LENGTH = 128
INPUT_NAMES = ["input_ids", "attention_mask"]


class LastHiddenState(torch.nn.Module):
    """DistilBERT returning a plain tensor rather than transformers' output object."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)[0]


def build():
    # Eager attention: the default SDPA path exports through scaled_dot_product_attention,
    # which needs opset 14. Below opset 17 each of its 13 LayerNorms is exported decomposed.
    return LastHiddenState(AutoModel.from_pretrained("distilbert-base-uncased", attn_implementation="eager"))


def example_inputs():
    # One unpadded sequence of token ids from the middle of the vocabulary.
    input_ids = torch.randint(1000, 2000, (1, SEQUENCE_LENGTH))
    return input_ids, torch.ones_like(input_ids)
