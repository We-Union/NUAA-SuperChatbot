from typing import Dict, Generator, List, Tuple
import torch
from random import shuffle
from ChatBot.constant import *
from itertools import zip_longest

# make a index matrix with PAD_TOKEN to compliment pad position
def align_batch_with_padded(batch : List[List[int]], return_mask : bool = False, pad_token : int = PAD_TOKEN) -> Tuple[torch.LongTensor, torch.BoolTensor]:
    zipped_list = list(zip_longest(*batch, fillvalue=pad_token))
    padded_tensor = torch.LongTensor(zipped_list).t()
    # mask for loss calculation
    if return_mask:
        mask_tensor = torch.BoolTensor(zipped_list).t()
    else:
        mask_tensor = None
    return padded_tensor, mask_tensor

def DataLoader(pairs : List[List[int]], batch_size : int = BATCH_SIZE, use_shuffle : bool = True) -> Generator:
    """
        batch_info field:
        - input_batch_tensor
        - input_length_tensor
        - output_batch_tensor
        - mask_tensor
        - max_output_length
    """
    if use_shuffle:
        shuffle(pairs)
    
    batch = []
    for _, pair in enumerate(pairs):
        batch.append(pair)
        if len(batch) == batch_size:
            # sort for pack
            batch.sort(key=lambda x : len(x[0]), reverse=True)

            input_batch  = [p[0] for p in batch]
            output_batch, max_output_length = [], 0
            for p in batch:
                output_batch.append(p[1])
                max_output_length = max(max_output_length, len(p[1]))

            input_length_tensor = torch.LongTensor([len(b) for b in input_batch])

            input_batch_tensor, _            = align_batch_with_padded(batch=input_batch, return_mask=False)
            output_batch_tensor, mask_tensor = align_batch_with_padded(batch=output_batch, return_mask=True)
            batch.clear()

            yield {
                "input_batch_tensor" : input_batch_tensor,
                "input_length_tensor" : input_length_tensor,
                "output_batch_tensor" : output_batch_tensor,
                "mask_tensor" : mask_tensor,
                "max_output_length" : max_output_length,
                "batch_size" : batch_size
            }

