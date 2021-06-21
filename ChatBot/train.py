from network import *

def maskNLLLoss(self, output, target, mask):
    """
    output:     [batch_size, max_length, output_size]
    target:     [batch_size, max_length]
    mask: mask matrix  the same shape as target
    """
    target = target.type(torch.int64).to(DEVICE)
    mask = mask.type(torch.BoolTensor).to(DEVICE)

    total_word = mask.sum()  
    crossEntropy = -torch.log(torch.gather(output, dim=2, index=target.unsqueeze(2)))
    # crossEntropy : [batch_size, max_length, 1]
    loss = crossEntropy.squeeze(2).masked_select(mask).mean()
    loss = loss.to(DEVICE)
    return loss, total_word.item()

