import torch

#https://blog.csdn.net/tsq292978891/article/details/79364140

class SegmentConsensus(torch.autograd.Function):

    def __init__(self, consensus_type, dim=1):
        self.consensus_type = consensus_type
        self.dim = dim
        self.shape = None

    def forward(self, input_tensor):
        self.shape = input_tensor.size()

        output = input_tensor.mean(dim=self.dim, keepdim=True)

        return output

    def backward(self, grad_output):
        grad_in = grad_output.expand(self.shape) / float(self.shape[self.dim])
        return grad_in


class ConsensusModule(torch.nn.Module):

    def __init__(self, consensus_type, dim=1):
        super(ConsensusModule, self).__init__()
        self.consensus_type = consensus_type
        self.dim = dim

    def forward(self, input):
        return SegmentConsensus(self.consensus_type, self.dim)(input)
