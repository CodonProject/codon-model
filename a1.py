from codon.motif import MotifA1, MotifA1Tokenizer

model = MotifA1().from_remote()
tokenizer = MotifA1Tokenizer().from_remote()

from codon.utils.service import ModelCard, Service

Service([
    ModelCard(model, tokenizer, 'motif-a1', 'CodonProject')
]).run()