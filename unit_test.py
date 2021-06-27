from pprint import pprint
from torch import optim
from ChatBot.network import *
from ChatBot.dataloader import *
from ChatBot.train import *
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
    with open("./data/ChatBot/ensemble/small_samples_pairs.json", "r", encoding="utf-8") as f:
        data_dict = json.load(fp=f)
    pairs = data_dict["index_pairs"]
    loader = DataLoader(pairs, batch_size=3)
    for batch in loader:
        x = batch["input_batch_tensor"]
        y = batch["output_batch_tensor"]
        length = batch["input_length_tensor"]
        mask = batch["mask_tensor"]
        print(x)
        print(y)
        print(length)
        print(mask)
        return 

def test_get_datatime_info():
    info = get_datetime_info()
    print(info)

def test_train():
    with open("./data/ChatBot/ensemble/small_samples_pairs.json", "r", encoding="utf-8") as f:
        data_dict = json.load(fp=f)
    with open("./data/ChatBot/ensemble/small_samples_vocab.json", "r", encoding="utf-8") as f:
        vocab_dict = json.load(fp=f)

    model = CBNet(
        embedding_dim=128,
        encoder_hidden_size=256,
        decoder_hidden_size=256,
        attention_score_name="dot",
        vocab_size=vocab_dict["vocab_size"],
    )
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)
    train(
        version="0.0.0",
        pairs=data_dict["index_pairs"],
        Epoch=50,
        model=model,
        optimizer=optimizer,
        batch_size=BATCH_SIZE,
        save_dir="./dist/ChatBot",
        save_model=False,   
        save_optimizer=True,
        clip_threshold=50.0,
        TF_ratio=TEACHER_FORCING_RATE,
        save_interval=5,
        display_progress_bar=True
    )

test_train()