from torch import nn, Tensor
import torch
from attention import Attention

class Encoder(nn.Module):
    def __init__(self, hidden_size : int, embedding : nn.Embedding, n_layer : int = 1, dropout : float = 0.):
        """
            hidden_size : feature dim conveyed in RNN
            embedding : embedding matrix, which is collection of word vector
            n_layer : layer of RNN
            dropout : possibility of that each RNN cell is dropped
        """
        self._hidden_size = hidden_size
        self._embedding = embedding
        self._n_layer = n_layer
        self._dropout = dropout
        super().__init__()

        self._rnn = nn.GRU(
            input_size=embedding.weight.data.shape[1],
            hidden_size=hidden_size,
            num_layers=n_layer,
            dropout=(0 if n_layer == 1 else dropout),
            bidirectional=True,
            batch_first=True                        # this is necessary
        )

    def forward(self, input_seq : Tensor, input_len : Tensor, h0 : Tensor = None):
        """
            input_seq : a batch of word index sequence(zero padded for align), shape is (B, embedding_dim)
            input_len : represent valid length of each line in input_seq, shape is (B, )
            h0 : first hidden tensor in RNN
        """
        # embedding to word vector first
        embedding = self._embedding(input_seq)
        # embedding shape : (B, max_length, embedding_dim)

        # in order to avoid the same zero padded mapped into different value instead of making a sparse matrix
        # we should compress the sequence of word vectors, using pack_padded_sequence api in nn.utils.rnn
        packed = nn.utils.rnn.pack_padded_sequence(
            input=embedding,
            lengths=input_len,
            batch_first=True
        )
        # through RNN
        outputs, hidden = self._rnn(packed, h0)
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
    def __init__(self, score_name : str, embedding : Tensor, hidden_size : int, output_size : int, n_layer : int = 1, dropout : float = 0.):
        """
            score_name : name of score function used in attention module
            embedding : embedding matrix
            hidden_size : hidden size of RNN
            output_size : size of the result of attention
            n_layer : layer of RNN
            dropout : possibility of that each cell is dropped
        """
        
        self.score_name = score_name
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layer = n_layer
        self.dropout = dropout

        # 定义或获取前向传播需要的算子
        self.embedding = embedding
        self.embedding_dropout = nn.Dropout(dropout)
        self._gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=n_layer,
            dropout=(0 if n_layer == 1 else dropout),
            batch_first=True
        )
        self.attention = Attention(score_name, hidden_size)
        self.hidden2output = nn.Linear(hidden_size, output_size)


    def forward(self, input_single_seq, last_hidden, encoder_output):
        """
            input_single_seq:  pre index sequence               (B, 1)
            last_hidden: hidden layer of pre decoder            (B, 2, hidden_size)
            encoder_output:  total output of encoder            (B, max_length, hidden_size)
        """

        embedded = self.embedding(input_single_seq)   # [B, max_length, hidden_size]
        embedded = self.embedding_dropout(embedded)     # [B, max_length, hidden_size]

        current_output, current_hidden = self._gru(embedded, last_hidden)
        # current_output : [B, 1, hidden_size]
        # current_hidden : [1, B, hidden_size]

        # Here, current_output is current_hidden to do attention calculate
        # of course, you can directly use current_hidden
        attention_vector = self.attention(current_output, encoder_output)
        # attention_vector : [B, 1, hidden_size]

        # map the dim of attention_vector to output_size
        output = self.hidden2output(attention_vector)
        # output : [B, 1, output_size]
        output = nn.functional.softmax(output.squeeze(1), dim=1)
        # output : [B, output_size]
        return output, current_hidden

if __name__ == "__main__":
    ...