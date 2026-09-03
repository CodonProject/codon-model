from codon.motif import MotifA1, MotifA1Tokenizer

tokenizer = MotifA1Tokenizer().from_remote()
model = MotifA1(num_layers=1)

from codon.pipeline.pretrain import PretrainConfig, PretrainPipeline

pipeline = PretrainPipeline(
    model=model,
    tokenizer=tokenizer,
    config=PretrainConfig(
        compiled=False,
        target_context=128,
        step_mode='min',
        global_batch_tokens=512,
        ckpt_dir='./dev/ckpt/'
    )
)

from codon.utils.data.text import TextFileDataset

dataset = TextFileDataset('D:/Datasets/NSFW_Pre', recursive=True, tokenizer=tokenizer, seq_len=1024, drop_last=False, shuffle=True)

pipeline.train(dataset)