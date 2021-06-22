from torch import nn, Tensor
import torch

class Attention(nn.Module):
    def __init__(self, score_name : str, hidden_size : int):
        """
            score_name : name of score function, Union["dot", "general", "concat"]
            hidden_size : hidden size of RNN
        """
        self._score_name = score_name
        self._hidden_size = hidden_size
        super().__init__()
        score_fields = ["dot", "general", "concat"]
        if score_name not in score_fields:
            raise ValueError(f'score function must be within "dot", "general", "concat"! Receive {score_name} instead!')

        self.Wa = None
        self.Wc = None

        if score_name == "general":
            self.Wa = nn.Linear(hidden_size, hidden_size)
        elif score_name == "concat":
            self.Wa = nn.Linear(hidden_size * 2, hidden_size)
            self.v = nn.Paramter(torch.FloatTensor(hidden_size))
        
        self.Wc = nn.Linear(hidden_size * 2, hidden_size)

        # score function registered
        self.score_dict = {
            "dot" : self.__dot_score,
            "general" : self.__general_score,
            "concat" : self.__concat_score
        }

    def forward(self, current_hidden : Tensor, encoder_output : Tensor):
        # calculate the score
        score = self.score_dict[self._score_name](current_hidden, encoder_output)
        # use softmax to map score into weight alpha
        alpha = nn.functional.softmax(score, dim=1)
        # accomplish this through broadcast calculation
        context = torch.sum(encoder_output * alpha.unsqueeze(2), dim=1)
        attention = self.Wc(torch.cat([context.unsqueeze(1), current_hidden], dim=2)).tanh()
        return attention


    def __dot_score(self, current_hidden : Tensor, encoder_output : Tensor):
        return torch.sum(encoder_output * current_hidden, dim=2)
    
    def __general_score(self, current_hidden : Tensor, encoder_output : Tensor):
        energy = self.Wa(encoder_output)
        return torch.sum(energy * current_hidden, dim=2)
    
    def __concat_score(self, current_hidden : Tensor, encoder_output : Tensor):
        concat = torch.cat([current_hidden.expand(encoder_output.shape), encoder_output], dim=2)
        energy = self.Wa(concat).tanh()
        return torch.sum(self.v * energy, dim=2)