from sys import path
from typing import Dict, List
path.append(".")

from itertools import zip_longest
from os import listdir
import json
import jieba
import tqdm
from ChatBot.constant import *

class Vocab(object):
    def __init__(self, name : str, vocab : Dict=None, min_count : int = 0) -> None:
        super().__init__()
        self.__name = name 
        self.__build = False
        self.__prune = False
        self.__min_count = min_count

        self.__reserve_word_index = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN]
        self.__reserve_word_name = ["PAD", "SOS", "EOS", "UNK"]

        if vocab:
            self.__word2index = vocab["word2index"]
            self.__index2word = vocab["index2word"]
            self.__word_count = vocab["word_count"]
            self.__vocab_size = vocab["vocab_size"]
        else:
            self.__word2index = {n : i for i, n in zip(self.__reserve_word_index, self.__reserve_word_name)}
            self.__index2word = {i : n for i, n in zip(self.__reserve_word_index, self.__reserve_word_name)}
            self.__word_count = {i : 0 for i, n in zip(self.__reserve_word_index, self.__reserve_word_name)}
            self.__vocab_size = 4         
        # we have self._reserve_word_name as the origin four words, so the size of 
        # vocab is 4 in the beginning
    
    # You can rewrite your logic of read data there
    def build_vocab(self, path : str):
        for line in tqdm.tqdm(open(path, "r", encoding="utf-8"), ncols=90):
            line = line.strip()
            for word in jieba.lcut(line):
                self.__word_count[word] = self.__word_count.get(word, 0) + 1
        
        print("\033[32m", end="")
        for key in tqdm.tqdm(self.__word_count, ncols=90):
            if self.__word_count[key] >= self.__min_count:
                self.__word2index[key] = self.__vocab_size
                self.__index2word[self.__vocab_size] = key
                self.__vocab_size += 1
        print("\033[0m", end="")
        self.__build = True

    # dump word2index, index2word and wordcount to the path as json
    def dump_to_vocab(self, path : str, reminder : bool = True):
        vocab = {
            "vocab_size" : self.__vocab_size,
            "word2index" : self.__word2index,
            "index2word" : self.__index2word,
            "word_count" : self.__word_count
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj=vocab, fp=f, **JSON_IO_PARAMETER)
        if reminder:
            print(f"Sucessfully dump to\033[32m{path}\033[0m! vocab size:\033[32m{self.__vocab_size}\033[0m")
    
    # load pairs and transform to index sequence (unalign)
    def dump_to_index_seq(self, text_path : str, index_seq_path : str, use_local_vocab : bool = True, vocab : dict = None, reminder : bool = True):        
        if use_local_vocab:
            if self.__build == False:
                raise PermissionError("You must build vocab before load data, try to use `Vocab.build_vocab`!!!")
            word2index : Dict = self.__word2index
        else:
            word2index : Dict = vocab
        
        result_json = {
            "pairs_num" : 0,
            "max_input_dialog_length" : 0, 
            "max_output_dialog_length" : 0,
            "index_pairs" : []
        }

        for line in open(text_path, "r", encoding="utf-8"):
            s1, s2 = line.strip().split(SPLIT_DIVIDER)
            s1_index_seq = [word2index.get(word, UNK_TOKEN) for word in jieba.lcut(s1)]
            s2_index_seq = [word2index.get(word, UNK_TOKEN) for word in jieba.lcut(s2)]
            result_json["max_input_dialog_length"] = max(result_json["max_input_dialog_length"], len(s1_index_seq))
            result_json["max_output_dialog_length"] = max(result_json["max_output_dialog_length"], len(s2_index_seq))
            result_json["pairs_num"] += 1
            result_json["index_pairs"].append([s1_index_seq, s2_index_seq])
        
        with open(index_seq_path, "w", encoding="utf-8") as f:
            json.dump(obj=result_json, fp=f, **JSON_IO_PARAMETER)
        
        if reminder:
            print(f"Sucessfully dump to\033[32m{index_seq_path}\033[0m!")

        

def transform(data_dir : str):
    for file in tqdm.tqdm(listdir(f"./data/ChatBot/{data_dir}"), ncols=90):
        cur_pair = []
        total_pairs = []
        file_path = f"./data/ChatBot/{data_dir}/{file}"
        for line in open(file_path, "r", encoding="utf-8"):
            line = line.strip()
            if cur_pair:
                total_pairs.append(cur_pair[0] + SPLIT_DIVIDER + line + "\n")
                cur_pair.clear()
            else:
                cur_pair.append(line)
        with open(file_path, "w", encoding="utf-8") as f:
            f.writelines(total_pairs)

def merge_csv(new_file : str, *data_dir):
    total_pairs = []
    for _dir in data_dir:
        for file in listdir(f"./data/ChatBot/{_dir}"):
            file_path = f"./data/ChatBot/{_dir}/{file}"
            for line in open(file_path, "r", encoding="utf-8"):
                total_pairs.append(line)
    with open(new_file, "w", encoding="utf-8") as f:
        f.writelines(total_pairs)

# plain csv -> index sequence and json
def main(pairs_csv_path : str, target_index_path : str, target_vocab_path : str):
    vocab = Vocab(name="ChatBot", vocab=None, min_count=5)
    vocab.build_vocab(pairs_csv_path)
    vocab.dump_to_index_seq(pairs_csv_path, target_index_path)
    vocab.dump_to_vocab(target_vocab_path)

if __name__ == "__main__":
    name = "ensemble"
    main(
        pairs_csv_path=f"./data/ChatBot/ensemble/{name}.csv",
        target_index_path=f"./data/ChatBot/ensemble/{name}_pairs.json",
        target_vocab_path=f"./data/ChatBot/ensemble/{name}_vocab.json"    
    )