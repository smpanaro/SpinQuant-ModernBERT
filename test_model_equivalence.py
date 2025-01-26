import torch
from torch import nn
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoConfig
from train_utils.modeling_modernbert_quant import ModernBertForMaskedLM as ModernBertForMaskedLMQuant

from train_utils.main import prepare_model
from utils.hadamard_utils import random_hadamard_matrix
from utils.process_args import parser_gen

"""
Simple test that the modified modernbert matches the original model
before any training.
"""

torch.set_grad_enabled(False)

class RotateModule(nn.Module):
    # Copied from optimize_rotation.py
    def __init__(self, R_init):
        super(RotateModule, self).__init__()
        self.weight = nn.Parameter(R_init.to(torch.float32).to(torch.device("cuda" if torch.cuda.is_available() else "cpu")))

    def forward(self, x, transpose=False):
        if transpose:
            return x @ self.weight
        else:
            return self.weight @ x

model_name = "answerdotai/ModernBERT-base"
ptq_args, _ = parser_gen()

tok = AutoTokenizer.from_pretrained(model_name)
ref = AutoModelForMaskedLM.from_pretrained(model_name)

qc = AutoConfig.from_pretrained(model_name)
qc.tie_word_embeddings = False
quant = ModernBertForMaskedLMQuant.from_pretrained(model_name, config=qc)
quant.decoder.weight.data = quant.model.embeddings.tok_embeddings.weight.data.clone() # Untie embeddings!
quant = prepare_model(ptq_args, quant)
R1 = random_hadamard_matrix(quant.config.hidden_size, "cuda" if torch.cuda.is_available() else "cpu")
quant.R1 = RotateModule(R1)

inputs = tok("The capital of France is [MASK].", return_tensors="pt")

ref_outputs = ref(**inputs)
quant_outputs = quant(**inputs)

print(ref_outputs)
print(quant_outputs)
