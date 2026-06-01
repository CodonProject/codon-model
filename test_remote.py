from codon.motif import MotifA1, MotifA1Tokenizer
from codon.utils.service import Service, ModelCard


model = MotifA1().from_remote().to('cuda')
tokenizer = MotifA1Tokenizer().from_remote()


Service([
    ModelCard(
        model=model,
        tokenizer=tokenizer,
        model_id='Motif-A1',
        owned='CodonProject'
    )
]).run(port=11305)
