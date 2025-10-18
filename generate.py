import torch
from v2 import BigramLanguageModel, Block, FeedForward, MultiHeadAttention, Head

max_new_tokens = 30000
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = torch.load('model-5000.pt', weights_only=False)
model.to(device)
model.eval()

context = torch.zeros((1, 1), dtype=torch.long, device=device)
with torch.no_grad():
    output = model.generate(context, max_new_tokens=max_new_tokens)[0].tolist()
generated_text = model.decode(output)

print(generated_text)

with open(f'generated_text-5000-{max_new_tokens}.txt', 'w', encoding='utf-8') as f:
    f.write(generated_text)
