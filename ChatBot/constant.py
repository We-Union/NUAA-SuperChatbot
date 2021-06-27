import torch 

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PAD_TOKEN             = 0         # pad for absent position
SOS_TOKEN             = 1         # token for the beginning of one sentence
EOS_TOKEN             = 2         # token for the end of one sentence
UNK_TOKEN             = 3         # work that doesn't exist in the vocab

SPLIT_DIVIDER         = " | "     # parameter of split when divide the pair string
JSON_IO_PARAMETER     = {"ensure_ascii" : False, "indent" : 4}     # position arguments when call json IO API

MAX_RESPONSE_LENGTH   = 20        # max length of the response sentence

BATCH_SIZE            = 64
MAX_LENGTH            = 20
MIN_COUNT             = 3

TEACHER_FORCING_RATE  = 0.9      # possibility to adopt teacher forcing strategy
CLIP_THRESHOLD        = 50.0     # clip of gradient
LEARNING_RATE         = 1e-4

EMBEDDING_DIM         = 512
HIDDEN_SIZE           = 512
ENCODER_N_LAYER       = 2
DECODER_N_LAYER       = 2
DROPOUT               = 0.1

EPOCH_NUM             = 15
PRINT_INTERVAL        = 50
SAVE_INTERVAL         = 900