from numpy.core.defchararray import encode
from torch import nn, Tensor
import torch
import numpy as np
from torch.nn import Embedding
from attention import Attention
from constant import *

class Encoder(nn.Module):
    def __init__(self, embedding_dim : int, hidden_size : int, n_layer : int = 1, dropout : float = 0.):
        """
            embedding_dim : length of the word vector
            hidden_size : feature dim conveyed in RNN
            n_layer : layer of RNN
            dropout : possibility of that each RNN cell is dropped
        """
        self._embedding_dim = embedding_dim
        self._hidden_size = hidden_size
        self._n_layer = n_layer
        self._dropout = dropout
        super().__init__()

        self.__rnn = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=n_layer,
            dropout=(0 if n_layer == 1 else dropout),
            bidirectional=True,
            batch_first=True                        # this is necessary
        )

    def forward(self, embedded_index_seq : Tensor, index_seq_length : Tensor, h0 : Tensor = None):
        """
            embedded_index_seq : a batch of embedded word index sequence(zero padded for align), 
                                 shape is (B, max_length, embedding_dim)
            index_seq_length : represent valid length of each line in index_sequence, shape is (B, )
            h0 : first hidden tensor in RNN
        """

        # in order to avoid the same zero padded mapped into different value instead of making a sparse matrix
        # we should compress the sequence of word vectors, using pack_padded_sequence api in nn.utils.rnn
        packed = nn.utils.rnn.pack_padded_sequence(
            input=embedded_index_seq,
            lengths=index_seq_length,
            batch_first=True
        )
        # through RNN
        outputs, hidden = self.__rnn(packed, h0)
        # decompress the outputs to get origin uncompressed outputs
        outputs, _ = nn.utils.rnn.pad_packed_sequence(
            sequence=outputs,
            batch_first=True,
            padding_value=0
        )
        # outputs shape : (B, max_length, 2 * hidden_size)
        # transform bidirectional rnn into one channel
        outputs = outputs[..., : self._hidden_size] + outputs[..., self._hidden_size : ]
        # outputs shape : (B, max_length, hidden_size)
        # hidden shape : (B, 2, hidden_size)
        return outputs, hidden

class Decoder(nn.Module):
    def __init__(self, score_name : str, hidden_size : int, vocab_size : int, n_layer : int = 1, dropout : float = 0.):
        """
            score_name : name of score function used in attention module
            embedding : embedding matrix
            hidden_size : hidden size of RNN
            vocab_size : size of vocab
            n_layer : layer of RNN
            dropout : possibility of that each cell is dropped
        """
        
        self._score_name = score_name
        self._hidden_size = hidden_size
        self._vocab_size = vocab_size
        self._n_layer = n_layer
        self._dropout = dropout

        # 定义或获取前向传播需要的算子
        # self.__embedding_dropout = nn.Dropout(dropout)
        self.__rnn = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=n_layer,
            dropout=(0 if n_layer == 1 else dropout),
            batch_first=True
        )
        self.__attention = Attention(score_name, hidden_size)
        self.__hidden2output = nn.Linear(hidden_size, vocab_size)


    def forward(self, pre_embedded_predict_index : Tensor, pre_hidden : Tensor, encoder_output : Tensor):
        """
            pre_embedded_predict_index:  pre index sequence                   (B, 1)
            pre_hidden: hidden layer of pre decoder             (B, 2, hidden_size)
            encoder_output:  total output of encoder            (B, max_length, hidden_size)
        """
        current_output, current_hidden = self.__rnn(pre_embedded_predict_index, pre_hidden)
        # current_output : [B, 1, hidden_size]
        # current_hidden : [1, B, hidden_size]

        # Here, current_output is current_hidden to do attention calculate
        # of course, you can directly use current_hidden
        attention_vector = self.__attention(current_output, encoder_output)
        # attention_vector : [B, 1, hidden_size]

        # map the dim of attention_vector to vocab_size
        output = self.__hidden2output(attention_vector)
        # output : [B, 1, vocab_size]
        output = nn.functional.softmax(output.squeeze(1), dim=1)
        # output : [B, vocab_size]
        return output, current_hidden

class CBNet(nn.Module):
    def __init__(self, embedding_dim : int, encoder_hidden_size : int, decoder_hidden_size : int,
        attention_score_name : str, vocab_size : int, n_layer : int = 1, dropout : float = 0.):
        """
            embedding_dim : length of word vector
            encoder_hidden_size : size of hidden layer in encoder's RNN
            decoder_hidden_size : size of hidden layer in decoder's RNN
            attention_score_name : score function adopted in Luong attention, value scope is ["dot", "general", "concat"]
            vocab_size : size of vocab
            n_layer : layer of RNN in both encoder and decoder
            dropout : possibility of drop of each cell
        """
        if encoder_hidden_size < decoder_hidden_size:
            raise ValueError("encoder's hidden size must be greater than that of decoder!!!")
        self._embedding_dim = embedding_dim
        self._encoder_hidden_size = encoder_hidden_size
        self._decoder_hidden_size = decoder_hidden_size
        self._attention_score_name = attention_score_name
        self._vocab_size = vocab_size
        self._n_layer = n_layer
        self._dropout = dropout
        super().__init__()
        self.__encoder = Encoder(
            embedding_dim=embedding_dim,
            hidden_size=encoder_hidden_size,
            n_layer=n_layer,
            dropout=dropout
        )
        self.__decoder = Decoder(
            score_name=attention_score_name,
            hidden_size=decoder_hidden_size,
            vocab_size=vocab_size,
            n_layer=n_layer,
            dropout=dropout
        )
        self.__embedding = Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=0
        )
    
    def refill_embedding(self, embedding : Embedding):
        self.__embedding = embedding

    """
        The difference between forward and predict is that former need to consider the pad and thus need 
        to pack the sentence. The latter don't need to consider the pad and the batch of the predict is 
        usually one, which input of forward is always shaped as (1, sentence_length, embedding_dim)
    """
    def forward(self, index_seq : Tensor, index_seq_length : Tensor, max_label_sentence_length : int, h0 : Tensor = None, use_teacher_forcing : bool = False, label_sentence : Tensor = None):
        if use_teacher_forcing and label_sentence is None:
            raise ValueError("You adopt teacher forcing strategy but not provide label sentence!!!")
        # index_seq shape : (B, max_length)
        embedded_seq = self.__embedding(index_seq)
        # embedded_seq shape : (B, max_length, embedding_dim)
        encoder_output, encoder_hidden = self.__encoder(embedded_seq, index_seq_length, h0)

        # initial the first input of decoder
        decoder_input = SOS_TOKEN * torch.ones(size=[BATCH_SIZE], dtype=torch.int64)
        decoder_input = decoder_input.to(DEVICE)

        # clip the pre decoder_hidden_size of the hidden size of encoder
        decoder_hidden = encoder_hidden[:self._decoder_hidden_size]

        predict_result = torch.FloatTensor()
        predict_result.to(DEVICE)

        for i in range(max_label_sentence_length):
            output, hidden = self.__decoder(
                pre_embedded_predict_index=self.__embedding(decoder_input),     # transform index to embedded  
                pre_hidden=decoder_hidden, 
                encoder_output=encoder_output
            )
            # output shape : (B, vocab_size)
            # predict_result : (B, i, vocab_size)
            predict_result = torch.cat([predict_result, output.unsqueeze(1)], dim=1)
            decoder_input = label_sentence[:, i] if use_teacher_forcing else torch.argmax(output, dim=1)
            decoder_input = decoder_input.reshape([-1, 1])
            decoder_hidden = hidden

        return predict_result

    # for inference
    def predict(self, index_seq : Tensor, index_seq_length : Tensor, h0 : Tensor = None):
        # for the forward logic of encoder, we must use index_seq_length
        encoder_output, encoder_hidden = self.__encoder(self.__embedding(index_seq), index_seq_length, h0)
        # initial the input of decoder
        decoder_hidden = encoder_hidden[:self._decoder_hidden_size]
        decoder_input = torch.tensor([[SOS_TOKEN]], dtype=torch.int64)

        predict_index = []
        predict_index_confidence = []

        for i in range(MAX_RESPONSE_LENGTH):
            output, hidden = self.__decoder(
                pre_embedded_predict_index=self.__embedding(decoder_input),     # transform index to embedded  
                pre_hidden=decoder_hidden, 
                encoder_output=encoder_output
            )
            max_p, index = torch.max(output, dim=1)
            predict_index.append(index.item())
            predict_index_confidence.append(max_p.item())
            decoder_input = index.unsqueeze(0)
            decoder_hidden = hidden
        return predict_index, predict_index_confidence


    def encoder(self, *args, **kwargs):
        return self.__encoder(*args, **kwargs)

    def decoder(self, *args, **kwargs):
        return self.__decoder(*args, **kwargs)

    @property
    def encoder(self):
        return self.__encoder
    
    @property
    def decoder(self):
        return self.__decoder

if __name__ == "__main__":
    
    m = Embedding(
        num_embeddings=5,
        embedding_dim=10
    )

    network = CBNet(
        embedding_dim=10,
        encoder_hidden_size=64,
        decoder_hidden_size=64,
        attention_score_name="dot",
        vocab_size=3000,
        n_layer=1,
        dropout=0
    )

    