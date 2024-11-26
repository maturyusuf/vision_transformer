import torch
import torch.nn as nn

torch.manual_seed(0)

class MSA(nn.Module):
  def __init__(self, hidden_d,n_heads = 2, ):
    super().__init__()
    self.n_heads = n_heads
    self.hidden_d = hidden_d

    assert self.hidden_d % self.n_heads == 0, "Sequence must be divisible by number of heads"

    self.d_heads = self.hidden_d // self.n_heads

    self.query_mapper = nn.ModuleList([nn.Linear(self.d_heads,self.d_heads) for _ in range(self.n_heads)])
    self.key_mapper = nn.ModuleList([nn.Linear(self.d_heads,self.d_heads) for _ in range(self.n_heads)])
    self.value_mapper = nn.ModuleList([nn.Linear(self.d_heads,self.d_heads) for _ in range(self.n_heads)])

    self.softmax = nn.Softmax(dim=-1)
  def forward(self, sequences):

    # SEQUENCE : (N, SEQ_LEN, token_dim) ---> (N, SEQ_LEN, HEADS, token_dim / HEADS)
    results = []
    for sequence in (sequences):
      seq_results = []
      for head in range(self.n_heads):
        query_mapping = self.query_mapper[head]
        key_mapping = self.key_mapper[head]
        value_mapping = self.value_mapper[head]

        query = query_mapping(sequence[:, head * self.d_heads:(head + 1) * self.d_heads]) # query = seq_len x d_heads
        key = key_mapping(sequence[:, head * self.d_heads:(head + 1) * self.d_heads]) # key = seq_len x d_heads
        value = value_mapping(sequence[:, head * self.d_heads:(head + 1) * self.d_heads]) # value = seq_len x d_heads

        attention = (self.softmax((query @ key.T) / (self.d_heads**0.5))) @ value
        seq_results.append(attention)
        # seq_results:
      # results: seq_len
      results.append(torch.hstack(seq_results))

    return torch.cat([torch.unsqueeze(r, dim=0) for r in results])

# msa = MSA(8,2)

# msa(torch.randn(1, 50, 8)).shape