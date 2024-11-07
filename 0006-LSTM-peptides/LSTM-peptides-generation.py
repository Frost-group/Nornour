import torch
import torch.nn.functional as F

import random


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

model = ''

def temperature_sampling(logits, temperature):

    padding_index = 21
    logits[:, padding_index] = float('-inf')  # Mask padding index

    scaled_logits = logits / temperature
    probabilities = F.softmax(scaled_logits, dim=-1)

    next_index = torch.multinomial(probabilities, num_samples=1).item()
    return next_index


def gen_peptides(model, seed, number_aa, vocab, device, temperature=1.0):
    model.eval()

    to_index = {a: i for i, a in enumerate(vocab)}
    index_to_amino = {i: a for i, a in enumerate(vocab)}

    seed_indices = [to_index[aa] for aa in seed if aa in to_index and to_index[aa] != 21]  # Avoid padding in seed
    input_tensor = torch.LongTensor(seed_indices).unsqueeze(0).to(device)

    state_h, state_c = model.init_state(1)
    state_h, state_c = state_h.to(device), state_c.to(device)

    gen_seq = seed

    for _ in range(number_aa):
        with torch.no_grad():
            y_pred, (state_h, state_c) = model(input_tensor, (state_h, state_c))

        next_index = temperature_sampling(y_pred[:, -1, :], temperature)

        if next_index == 21:
            break

        next_amino = index_to_amino[next_index]
        gen_seq += next_amino

        input_tensor = torch.cat((input_tensor, torch.tensor([[next_index]]).to(device)), dim=1)

    return gen_seq


vocab = ['R', 'H', 'K', 'D', 'E', 'S', 'T', 'N', 'Q', 'C', 'U',
         'G', 'P', 'A', 'I', 'L', 'M', 'F', 'W', 'Y', 'V', '_']
len_vocab = len(vocab)

gen_sequences = set()
temperature = 1
while len(gen_sequences) < 100:
    number_aa = random.randint(2, 15)
    seed = ['R', 'W', 'W']

    gen_pep = gen_peptides(model, seed, number_aa, vocab, device, temperature)

    gen_pep_str = ''.join(gen_pep)

    print(gen_pep_str)

    # Add to set to ensure uniqueness
    gen_sequences.add(gen_pep_str)

    print('Len sequences:', len(gen_sequences))


gen_sequences = list(gen_sequences)
for pep in gen_sequences:
    print(pep)