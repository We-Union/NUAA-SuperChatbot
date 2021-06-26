from pprint import pprint
from ChatBot.network import *
from ChatBot.dataloader import *
import json

def test_CBNet():
    network = CBNet(
        embedding_dim=10,
        encoder_hidden_size=64,
        decoder_hidden_size=64,
        attention_score_name="dot",
        vocab_size=15,
        n_layer=1,
        dropout=0
    )

    sentence = torch.tensor([[10, 9, 2, 1, 4, 6], [10, 9, 2, 1, 4, 0], [10, 9, 2, 1, 4, 0]], dtype=torch.int64)
    length = torch.tensor([6, 5, 5])
    result = network(sentence, length, length.max().item(), None, False)
    print(result)


def test_vocab():
    with open("./data/vocab.json", "r", encoding="utf-8") as f:
        a = json.load(fp=f)
    print(a["word2index"]["人工智能"])


def test_dataloader():
    ...