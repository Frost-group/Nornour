import torch
import torch.nn.functional as F
import argparse
import random


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


def load_model(model_path):
    model = torch.load(model_path)
    model.to(device)
    model.eval()
    return model


def open_file(filepath):
    try:
        with open(filepath, 'r') as doc:
            peptides = doc.readlines()

        peptides = [pep.strip() for pep in peptides if pep.strip()]

        long_pep = max(len(pep) for pep in peptides)

        return peptides, long_pep

    except FileNotFoundError:
        print(f"File not found: {filepath}")
        return None, None



def temperature_sampling(logits, temperature):
    padding_index = 21
    logits[:, padding_index] = float('-inf')
    scaled_logits = logits / temperature
    probabilities = F.softmax(scaled_logits, dim=-1)
    next_index = torch.multinomial(probabilities, num_samples=1).item()
    return next_index


def gen_peptides(model, seed, max_length, vocab, device, temperature=1.0):
    model.eval()

    to_index = {a: i for i, a in enumerate(vocab)}
    index_to_amino = {i: a for i, a in enumerate(vocab)}

    seed_indices = [to_index[aa] for aa in seed if aa in to_index and to_index[aa] != 21]
    input_tensor = torch.LongTensor(seed_indices).unsqueeze(0).to(device)

    state_h, state_c = model.init_state(1)
    state_h, state_c = state_h.to(device), state_c.to(device)

    gen_seq = seed

    for _ in range(max_length):
        with torch.no_grad():
            y_pred, (state_h, state_c) = model(input_tensor, (state_h, state_c))

        next_index = temperature_sampling(y_pred[:, -1, :], temperature)

        if next_index == 21:
            break

        next_amino = index_to_amino[next_index]
        gen_seq += next_amino

        input_tensor = torch.cat((input_tensor, torch.tensor([[next_index]]).to(device)), dim=1)

    return gen_seq


def main():

    parser = argparse.ArgumentParser(description="Generate peptide sequences using a trained LSTM model")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the saved model file")
    parser.add_argument('--dataset_path', type=str, required=True, help="Path to dataset used for training the model")
    parser.add_argument('--temperature', type=float, default=1.0, help="Sampling temperature (default: 1.0)")
    parser.add_argument('--num_sequences', type=int, default=100, help="Number of unique sequences to generate")
    parser.add_argument('--max_length', type=int, default=15, help="Maximum peptide length")
    parser.add_argument('--seed', type=list, default=['R', 'W', 'W'],help="Seed sequence to start generation as a list (default: ['R', 'W', 'W'])")
    args = parser.parse_args()

    model = load_model(args.model_path)

    peptides, _ = open_file(args.dataset_path)

    vocab = ['R', 'H', 'K', 'D', 'E', 'S', 'T', 'N', 'Q', 'C', 'U',
             'G', 'P', 'A', 'I', 'L', 'M', 'F', 'W', 'Y', 'V', '_']

    gen_sequences = set()
    iteration = 0
    while len(gen_sequences) < args.num_sequences:
        gen_pep = gen_peptides(model, args.seed, args.max_length, vocab, device, args.temperature)
        gen_sequences.add(gen_pep)
        iteration += 1
        if iteration > (2 * args.num_sequences):
            break

    peptides_to_remove = set()

    print(f"Generated {len(gen_sequences)} unique peptide sequences:")
    for pep in gen_sequences:
        print(pep)
        if pep in peptides:
            print('Peptide already in dataset \n Peptide marked for removal from generated sequences')
            peptides_to_remove.add(pep)

    gen_sequences -= peptides_to_remove

    print(f'Generated {len(gen_sequences)} new unique peptide sequences')


if __name__ == "__main__":
    main()
