from codon.motif import MotifA1
from codon.motif.onnx import CausalLanguageModelONNXWrapper

model = MotifA1().from_remote().cpu()

wrapper = CausalLanguageModelONNXWrapper(model).eval()

wrapper.export('motifa1.onnx')