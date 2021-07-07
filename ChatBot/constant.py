import torch 

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PAD_TOKEN             = 0         # pad for absent position
SOS_TOKEN             = 1         # token for the beginning of one sentence
EOS_TOKEN             = 2         # token for the end of one sentence
UNK_TOKEN             = 3         # work that doesn't exist in the vocab

SPLIT_DIVIDER         = " | "     # parameter of split when divide the pair string
JSON_IO_PARAMETER     = {"ensure_ascii" : False, "indent" : 4}     # position arguments when call json IO API

"""
    parameter for training
"""
EPOCH_NUM             = 30
MAX_RESPONSE_LENGTH   = 20        # max length of the response sentence
BATCH_SIZE            = 128
MAX_LENGTH            = 20
MIN_COUNT             = 3        #TODO: wait for a math model of N(\mu,\Sigma)
TEACHER_FORCING_RATE  = 0.9      # possibility to adopt teacher forcing strategy
CLIP_THRESHOLD        = 40.0     # clip of gradient
LEARNING_RATE         = 1e-4

PRINT_INTERVAL        = 50
SAVE_INTERVAL         = 5

"""
    parameter for network
"""
EMBEDDING_DIM         = 512
ENCODER_HIDDEN_SIZE   = 512
DECODER_HIDDEN_SIZE   = 512
ENCODER_N_LAYER       = 10
DECODER_N_LAYER       = 10
DROPOUT               = 0.1
ATTENTION_SCORE_NAME  = "dot"     # field: dot, general, concat


"""
    parameter with respect for inference
"""
QUIT_WORDS = ["q", "quit", "exit", "退出"]



"""
    unimportant argument
"""
EPOCH_LOOP_TQDM = {
    "ncols" : 90,        # max length of progressbar
    "desc" : "Start Epoch Loop...",
}

BATCH_LOOP_TQDM = {
    "ncols" : 90,        # max length of progressbar
    "desc" : "Start New Batch Loop...",
}

LOSS_DISPLAY_BIT = 5

# ASCII color index
RED    = 31
GREEN  = 32
ORANGE = 33
BLUE   = 34
PURPLE = 35
CYAN   = 36 
WHITE  = 37