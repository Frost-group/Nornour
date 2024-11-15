from typing import Dict, Any
from huggingface_hub import login

from transformers import AutoTokenizer
from datasets import Dataset
from transformers import AutoModelForCausalLM

hf_token = 
login(token=hf_token, add_to_git_credential=True)

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

def padding(peptide, long_pep):
    pad_length = long_pep - len(peptide)

    if pad_length > 0:
        peptide = peptide.strip() + '_' * pad_length

    return peptide

class PeptideTokenizer:
    def __init__(self, vocab: list, max_seq_len: int = 30):
        self.start_id = 's'
        self.pad_id = '_'
        self.end_id = 'e'
        vocab.append(self.start_id)
        vocab.append(self.pad_id)
        vocab.append(self.end_id)
        self.vocab = vocab
        self.vocab_ids = {a: i for i, a in enumerate(vocab)}
        self.id_to_vocab = {i: a for a, i in self.vocab_ids.items()}
        self.max_seq_len = max_seq_len

    def encode(self, peptide: str, **kwargs: Dict[str, Any]) -> Dict[str, list]:
        """Convert text (peptide sequence) into token IDs."""
        pad_pep = padding(peptide, self.max_seq_len)
        peptide_form = "s" + pad_pep + "e"
        input_ids = [self.vocab_ids[amino] for amino in peptide_form if amino in self.vocab_ids]
        encoded_inputs = {
            'input_ids': input_ids,
            'token_type_ids': [0 for i in input_ids],
            'attention_mask': [1 for i in input_ids]
        }
        return encoded_inputs

    def decode(self, input_ids: list[int]) -> str:
        """Convert token IDs back into the original peptide sequence."""
        tokens = [self.id_to_vocab[token_id] for token_id in input_ids]

        tokens = [token for token in tokens if token not in {"<|start|>", "<|end|>", "<|pad|>"}]

        peptide = ''.join(tokens)
        return peptide

    def to_sequence(self, peptide_str: str) -> str:
        """Remove start, end, and pad tokens from a given peptide string."""
        # Replace special tokens with an empty string
        cleaned_sequence = peptide_str.replace(self.start_id, '').replace(self.pad_id, '').replace(self.end_id, '')
        return cleaned_sequence


model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-3.2-1B')

model_checkpoint = 
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, token=hf_token)

dataset_path = '/Users/igorgonteri/Desktop/LLM_fine_tuning/re_datasets/RW_lexicon.txt'
peptides, long_pep = open_file(dataset_path)
peptides_token = tokenizer(peptides, padding='longest')

dataset = Dataset.from_dict(peptides_token)
split_dataset = dataset.train_test_split(test_size=0.2)

print(dataset)




