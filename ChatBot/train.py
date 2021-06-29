from numpy.lib.function_base import average
import tqdm
from random import random
import time
from typing import Dict, Generator, Tuple, Union
from torch import optim
import os
import json
from ChatBot.dataloader import DataLoader
from ChatBot.network import *

def ensure_create_folder(dir_name : str):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

def get_datetime_info(day_spliter : str = "-", time_spliter : str = "_") -> Dict:
    tlc = time.localtime()
    today = [str(tlc.tm_year), str(tlc.tm_mon), str(tlc.tm_mday)]
    today = day_spliter.join(today)
    cur_time = [str(tlc.tm_hour), str(tlc.tm_min), str(tlc.tm_sec)]
    cur_time = time_spliter.join(cur_time)
    return {
        "today" : today, "cur_time" : cur_time
    }

def color_wrapper(s : object, color : Union[int, str]) -> str:
    color_dict = {
        "RED"    : 31,
        "GREEN"  : 32,
        "ORANGE" : 33,
        "BLUE"   : 34,
        "PURPLE" : 35,
        "CYAN"   : 36,
        "WHITE"  : 37
    }
    if isinstance(color, str):
        color = color_dict[color]
    return "\033[{}m{}\033[0m".format(color, s)

def maskNLLLoss(output : torch.FloatTensor, target : torch.LongTensor, mask : torch.LongTensor) -> Tuple[torch.FloatTensor, int]:
    """
    output:     [batch_size, max_length, output_size]
    target:     [batch_size, max_length]
    mask: mask matrix  the same shape as target
    """
    target = target.type(torch.int64).to(DEVICE)
    mask = mask.type(torch.BoolTensor).to(DEVICE)

    total_word_num = mask.sum()  
    crossEntropy = -torch.log(torch.gather(output, dim=2, index=target.unsqueeze(2)))
    # crossEntropy : [batch_size, max_length, 1]
    loss = crossEntropy.squeeze(2).masked_select(mask).mean()
    loss = loss.to(DEVICE)
    return loss, total_word_num.item()

def train_batch(batch : Dict, model : CBNet, optimizer : optim.Optimizer, clip_threshold : float, TF_ratio : float) -> Dict:
    """
        train batch dict field:
        - loss : mask_loss
    """
    mask_loss : torch.FloatTensor = None

    X             = batch["input_batch_tensor"].to(DEVICE)
    length_tensor = batch["input_length_tensor"].to(DEVICE)
    Y             = batch["output_batch_tensor"].to(DEVICE)
    mask          = batch["mask_tensor"].to(DEVICE)

    predict_result = model.forward(
        index_seq=X,
        index_seq_length=length_tensor,
        max_label_sentence_length=batch["max_output_length"],
        use_teacher_forcing=bool(random() < TF_ratio),
        label_sentence=Y
    )
    
    mask_loss, word_num = maskNLLLoss(
        output=predict_result, 
        target=Y,
        mask=mask
    )

    nn.utils.clip_grad_norm_(model.parameters(), clip_threshold)

    mask_loss.backward()
    optimizer.step()

    return {
        "mask_loss" : mask_loss.item(),
        "label_word_num" : word_num
    }

def train_loader(loader : Generator, batch_num : int, model : CBNet, optimizer : optim.Optimizer, clip_threshold : float, TF_ratio : float = TEACHER_FORCING_RATE, display_progress_bar : bool = False, cur_epoch : int = None) -> Dict:
    if display_progress_bar and cur_epoch is None:
        raise ValueError("If you choose display_progress_bar mode, please make sure argument cur_epoch is not None!!!")
    word_wise_losses = []
    index_iter = range(batch_num)
    if display_progress_bar:
        index_iter = tqdm.tqdm(index_iter, **BATCH_LOOP_TQDM)
    
    for batch_index, batch in zip(index_iter, loader):
        train_batch_info = train_batch(
            batch=batch,
            model=model,
            optimizer=optimizer,
            clip_threshold=clip_threshold,
            TF_ratio=TF_ratio
        )
        word_wise_losses.append(train_batch_info["mask_loss"] / train_batch_info["label_word_num"])
        if display_progress_bar:
            if batch_index == batch_num - 1:
                index_iter.set_description_str("{} {}".format(
                    color_wrapper("Finish Epoch", GREEN),
                    color_wrapper(cur_epoch, GREEN)
                ))
            else:
                index_iter.set_description_str("{} {:<3}".format(
                    color_wrapper("Batch", BLUE), 
                    color_wrapper(batch_index, ORANGE)
                ))
            index_iter.set_postfix_str("{}:{}".format(
                color_wrapper("word_loss", BLUE), 
                color_wrapper(round(word_wise_losses[-1], ndigits=LOSS_DISPLAY_BIT), GREEN)
            ))

    return {
        "word_wise_losses" : word_wise_losses
    }

def train(version : str, pairs, Epoch : int, model : CBNet, optimizer : optim.Optimizer, batch_size : int = BATCH_SIZE, save_dir : str = "./dist/", 
            save_model : bool = True, save_optimizer : bool = False, clip_threshold : float = 50, TF_ratio : float = TEACHER_FORCING_RATE, save_interval : int = 10, 
            display_progress_bar : bool = False):
    """
        pairs : data
        Epoch : epoch of training
        model : CBNet
        optimizer : derived class of optim.Optimizer
        save_dir : dir to dump model's parameter
        save_optimizer : whether save the argument of optimizer
        clip_threshold : threshold to clip the gradient
        TF_ratio : possibility to adopt strategy of teacher forcing
        save_interval : interval of epoch to save the model
        display_progress_bar : whether to show
    """
    today = get_datetime_info()["today"]
    save_dir = os.path.join(save_dir, today)
    ensure_create_folder(save_dir)
    start_time = time.time()
    word_wise_losses = []
    all_cur_time = []
    iter_body = tqdm.tqdm(range(Epoch), **EPOCH_LOOP_TQDM) if display_progress_bar else range(Epoch)
    parameters = {
        "embedding_dim" : EMBEDDING_DIM,
        "encoder_hidden_size" : ENCODER_HIDDEN_SIZE,
        "decoder_hidden_size" : DECODER_HIDDEN_SIZE,
        "attention_score_name" : ATTENTION_SCORE_NAME
    }
    with open(os.path.join(save_dir, "parameter.json"), "w", encoding="utf-8") as f:
        json.dump(obj=parameters, fp=f, **JSON_IO_PARAMETER)
    for epoch in iter_body:
        loader = DataLoader(pairs=pairs, batch_size=batch_size)
        train_info = train_loader(
            loader=loader,
            batch_num=len(pairs) // batch_size,
            model=model,
            optimizer=optimizer,
            clip_threshold=clip_threshold,
            TF_ratio=TF_ratio,
            display_progress_bar=True,
            cur_epoch=epoch
        )
        word_wise_losses.append(train_info["word_wise_losses"])
        if display_progress_bar:
            average_loss = sum(train_info["word_wise_losses"]) / len(train_info["word_wise_losses"])
            iter_body.set_description_str("{} {:<3}".format(
                color_wrapper("Epoch", BLUE),
                color_wrapper(epoch, CYAN))
            )
            iter_body.set_postfix_str("{}:{}".format(
                color_wrapper("Average loss", BLUE),
                color_wrapper(round(average_loss, ndigits=LOSS_DISPLAY_BIT), GREEN)
            ))
    
        if ((epoch + 1) % save_interval == 0 or epoch + 1 == Epoch) and save_model:
            cost_time = time.time() - start_time
            time_info = get_datetime_info()
            today = time_info["today"]
            cur_time = time_info["cur_time"]
            all_cur_time.append(cur_time)
            save_dict = {
                "save_date" : today,
                "save_time" : cur_time,
                "cost_time" : cost_time,
                "version" : version,
                "Epoch" : epoch,
                "model_state_dict" : model.state_dict(),
                "optimizer_state_dict" : optimizer.state_dict() if save_optimizer else None,
                "loss" : train_info["word_wise_losses"]
            }
            save_folder = os.path.join(save_dir, "[{}]loss={:.5f}".format(cur_time, average_loss))
            check_point_save_path = os.path.join(save_folder, "model.tar")
            ensure_create_folder(save_folder)
            torch.save(obj=save_dict, f=check_point_save_path)
    
    # TODO : finish here
    loss_dict = {
        "word_wise_losses" : word_wise_losses
    }
    
    loss_file_name = "{}to{}.json".format(all_cur_time[0], all_cur_time[-1])
    with open(os.path.join(save_dir, loss_file_name), "w", encoding="utf-8") as f:
        json.dump(loss_dict, f)
    
    return word_wise_losses