# Reference: https://github.com/guacomolia/ptr_net
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def to_var(x):
    return Variable(x)

class PointerNetwork(nn.Module):
    def __init__(self, input_size, answer_seq_len, weight_size=128, hidden_size=128):
        super(PointerNetwork, self).__init__()

        self.hidden_size = hidden_size
        self.answer_seq_len = answer_seq_len
        self.weight_size = weight_size
        self.input_size = input_size

        self.enc = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dec = nn.LSTMCell(input_size, hidden_size) # LSTMCell's input is always batch first

        ## Reference: https://zhuanlan.zhihu.com/p/30860157
        self.W1 = nn.Linear(hidden_size, weight_size, bias=False) # blending encoder
        self.W2 = nn.Linear(hidden_size, weight_size, bias=False) # blending decoder
        self.vt = nn.Linear(weight_size, 1, bias=False)

    def forward(self, input):
        batch_size = input.size(0)
        # import pdb;pdb.set_trace()
        # input = self.emb(input) # (batch_size, L, embd_size)
        new_input = input.view(batch_size, self.answer_seq_len, 1)


        # Encoding
        encoder_states, hc = self.enc(new_input.float()) # encoder_state: (batch_size, L, H)
        encoder_states = encoder_states.transpose(1, 0) # (L, batch_size, H)


        # Decoding states initialization
        decoder_input = to_var(torch.zeros(batch_size, self.input_size)) # (batch_size, embd_size)
        hidden = to_var(torch.zeros([batch_size, self.hidden_size]))   # (batch_size, h)
        cell_state = encoder_states[-1]                                # (batch_size, h)

        probs = []
        # Decoding
        for i in range(self.answer_seq_len): # range(M)
            hidden, cell_state = self.dec(decoder_input, (hidden, cell_state)) # (batch_size, h), (batch_size, h)

            ## Reference: https://zhuanlan.zhihu.com/p/30860157
            blend1 = self.W1(encoder_states)          # (L, bs, W)
            blend2 = self.W2(hidden)                  # (bs, W)
            blend_sum = F.tanh(blend1 + blend2)    # (L, bs, W)
            out = self.vt(blend_sum).squeeze()        # (L, bs)
            out = F.log_softmax(out.transpose(0, 1).contiguous(), -1) # (bs, L)
            
            decoder_input = torch.max(out, 1)[1].view(batch_size, -1).float() # (bs, input_size)
            probs.append(out)

        probs = torch.stack(probs, dim=1)           # (bs, M, L)

        return probs
