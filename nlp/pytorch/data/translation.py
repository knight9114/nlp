# -------------------------------------------------------------------------
#   Translation Dataset
# -------------------------------------------------------------------------
# Imports
from typing import Tuple, Callable, Union, List, Optional
import torch
import torchtext as tt
from nlp.pytorch.utils import constants


# -------------------------------------------------------------------------
#   Globals
# -------------------------------------------------------------------------
OptInt = Optional[int]


# -------------------------------------------------------------------------
#   Create Translation Dataset
# -------------------------------------------------------------------------
class TranslationDataset(tt.data.Dataset):
    def __init__(
            self,
            src_path:str,
            tgt_path:str,
            src_field:tt.data.Field,
            tgt_field:tt.data.Field,
            src_encoding:str='utf-8',
            tgt_encoding:str='utf-8',
            **kwargs):
        """
        """
        # Prepare Fields
        fields = [('src', src_field), ('tgt', tgt_field)]

        # Read Files
        examples = []
        with open(src_path, encoding=src_encoding) as sfp, open(tgt_path, encoding=tgt_encoding) as tfp:
            for src_line, tgt_line in zip(sfp, tfp):
                src_line, tgt_line = src_line.strip(), tgt_line.strip()
                if src_line != '' and tgt_line != '':
                    examples.append(tt.data.Example.fromlist([src_line, tgt_line], fields))

        super().__init__(examples, fields, **kwargs)


# -------------------------------------------------------------------------
#   Data Processing Functions
# -------------------------------------------------------------------------
def create_dataset(
            src_path:str,
            tgt_path:str,
            src_encoding:str='utf-8',
            tgt_encoding:str='utf-8',
            src_tokenizer:Union[Callable[[str], List[str]], str]=str.split,
            tgt_tokenizer:Union[Callable[[str], List[str]], str]=str.split,
            max_src_len:OptInt=None,
            max_tgt_len:OptInt=None,
            **kwargs) -> Tuple[tt.data.Dataset, tt.data.Field, tt.data.Field]:
       """
       """
       # Create Fields
       src_field = tt.data.ReversibleField(
               sequential=True,
               use_vocab=True,
               init_token=constants.SOS_TKN,
               eos_token=constants.EOS_TKN,
               fix_length=max_src_len,
               dtype=torch.int64,
               tokenize=src_tokenizer,
               batch_first=True,
               pad_token=constants.PAD_TKN,
               unk_token=constants.UNK_TKN)

       tgt_field = tt.data.ReversibleField(
               sequential=True,
               use_vocab=True,
               init_token=constants.SOS_TKN,
               eos_token=constants.EOS_TKN,
               fix_length=max_tgt_len,
               dtype=torch.int64,
               tokenize=tgt_tokenizer,
               batch_first=True,
               pad_token=constants.PAD_TKN,
               unk_token=constants.UNK_TKN)

       # Create Dataset
       dataset = TranslationDataset(
               src_path,
               tgt_path,
               src_field,
               tgt_field,
               src_encoding,
               tgt_encoding,
               **kwargs)

       # Build Vocabularies
       src_field.build_vocab(dataset)
       tgt_field.build_vocab(dataset)

       return dataset, src_field, tgt_field

# -------------------------------------------------------------------------
#   Iterator Functions
# -------------------------------------------------------------------------
def create_train_and_test_iterator(
        dataset:tt.data.Dataset,
        batch_size:int,
        test_split:float=0.25,
        ctx:torch.device=constants.DEFAULT_DEVICE) -> Tuple[tt.data.Iterator, tt.data.Iterator]:
        """
        """
        # Split Dataset
        train_data, test_data = dataset.split(1. - test_split)

        # Create Iterator
        train = tt.data.BucketIterator( 
                dataset=train_data,
                batch_size=batch_size, 
                shuffle=True, 
                device=ctx,
                sort=False)

        test = tt.data.BucketIterator( 
                dataset=test_data,
                batch_size=batch_size, 
                shuffle=False, 
                repeat=True,
                device=ctx,
                sort=False)

        return train, test
