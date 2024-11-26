import math
import torch
from torch import nn
import encoder_block
from encoder_block import EncoderBlock

# Image size: (N, C, H, W)
# For MNIST: (N, 1, 28, 28)

# We'll divide each image to 7x7 patches, 28/7 = 4, so each patch is 4x4
# 7*7 = 49 patch in total

# (N, C, H, W)  --->  (N, P², HWC/P²)
# (N, P², HWC/P²) = (N, 7x7, 4x4) = (N, 49, 16)

# We flattened each image to 1*4*4 = 16, vectors with 16 dimension

def patchify(images, n_patches=7):
      n, c, h, w = images.shape
      assert h == w, "Patchify method only works for square images"

      patches = torch.zeros(n, n_patches**2, (h*w*c)//(n_patches**2))
      patch_size = h // n_patches # 4 in our case

      for idx, image in enumerate(images):
        for i in range(n_patches):
          for j in range(n_patches):
            patch = image[:, i*patch_size : (i + 1) * patch_size, j * patch_size : (j + 1) * patch_size]
            patches[idx, i*n_patches + j] = patch.flatten()
      return patches



# images = torch.randint(0, 255, (1, 1, 28, 28))
# patches = patchify(images)
# print(patches.shape)

def positional_embedding(sequence_length, d):
  pos_emb_matrix = torch.ones(sequence_length, d)
  for i in range(len(pos_emb_matrix)):
    for j in range(d):
      if j % 2 == 0:
        pos_emb_matrix[i][j] = math.sin(i/(10000**(j/d)))
      else:
        pos_emb_matrix[i][j] = math.cos(i/(10000**((j-1)/d)))
  return pos_emb_matrix


class MyViT(nn.Module):
  def __init__(self, chw, n_patches=7, n_blocks=2, hidden_d=8, n_heads=2,out_d=10  ):
    super().__init__()
    self.chw = chw
    self.n_patches = n_patches
    self.hidden_d = hidden_d
    assert chw[1] % n_patches == 0, "Input shape must be square"
    assert chw[2] % n_patches == 0, "Input shape must be square"
    self.n_blocks = n_blocks
    self.n_heads = n_heads
    self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Input Shape = (N x H x W)

    self.patch_size = (chw[1] // n_patches,chw[2] // n_patches) #
    # 1- Linear Mapping
    self.input_d = int(self.patch_size[0]*self.patch_size[1]* self.chw[0]) # p x p x n
    self.linear_mapper = nn.Linear(self.input_d, self.hidden_d)
    # 2- Adding Class Token (Learnable Parameter)
    self.class_token = nn.Parameter(torch.rand(1, self.hidden_d))
    # 3- Positional Encoding (Non-Learnable Parameter)
    self.register_buffer('positional_embeddings', positional_embedding(n_patches ** 2 + 1, hidden_d), persistent=False)
    self.encoder_blocks = nn.ModuleList([EncoderBlock(self.hidden_d, self.n_heads) for _ in range(n_blocks)])

    # 5) Classification MLPk
    self.mlp = nn.Sequential(
      nn.Linear(self.hidden_d, out_d),
      nn.Softmax(dim=-1)
    )
  def forward(self, images):
    patches = patchify(images, self.n_patches)
    tokens = self.linear_mapper(patches.to(self.DEVICE))
    # len(tokens) -> Number of images
    # tokens[i] -> each patch of current image
    # Using torch.vstack, we concatenate one image's patch matrix with class token
    # Then wrapping it with torch.stack , we concatenate each images' patch matrix (n = 10 in this case)
    # CLASS TOKEN HAVE TO BE THE FIRST TOKEN OF EACH PATCH MATRIX
    tokens = torch.stack([torch.vstack((self.class_token, tokens[i])) for i in range(len(tokens))])
    # We have to add Positional Embedding to each of the images' patch matrix
    # So we use torch.repeat to create n amount of positional embedding instances to add each of the image patch matrix
    pos_embs = self.positional_embeddings.repeat(len(images), 1, 1)
    out = tokens + pos_embs
    for block in self.encoder_blocks:
      out = block(out)

    # Get only classification token
    out = out[:, 0]

    return self.mlp(out)

# model = MyViT((1, 28, 28))
# model(torch.randn(1, 1, 28, 28)).shape


