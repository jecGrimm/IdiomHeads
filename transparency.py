import sys
sys.path.append("llm-transparency-tool")

from llm_transparency_tool.models.tlens_model import TransformerLensTransparentLlm
import torch as t
from llm_transparency_tool.routes.contributions import get_attention_contributions
import argparse
from transformers import HfArgumentParser
from llm_tranparency_tool.server.app import LlmViewerConfig, App
from llm_tranparency_tool.server.monitor import SystemMonitor

model = TransformerLensTransparentLlm("meta-llama/Llama-3.2-1B-Instruct", dtype = t.bfloat16)
run_model = model.run(["Test this"]).copy()
print(run_model._last_run)
# # VON LEA
# layer, B0, source_token = 0
# resid_pre = model.residual_in(layer)[B0].unsqueeze(0)
# print("resid_pre:", resid_pre.size())
# resid_mid = model.residual_after_attn(layer)[B0].unsqueeze(0)
# print("resid_mid:", resid_mid.size())
# decomposed_attn = model.decomposed_attn(B0, layer).unsqueeze(0)
# print("decomposed_attn:", decomposed_attn.size())

# head_contrib, _ = get_attention_contributions(resid_pre, resid_mid, decomposed_attn)
# print("head_contrib:", head_contrib.size())

# # [batch pos key_pos head] -> [head]
# flat_contrib = head_contrib[0, -1, source_token, :] # #TODO token 2 is the noun, do this dynamically
# print("flat_contrib:", flat_contrib.size())


# if total_attn_contributions is None:
#     total_attn_contributions = t.zeros_like(flat_contrib)
# total_attn_contributions += flat_contrib

# if __name__ == "__main__":
#     top_parser = argparse.ArgumentParser()
#     top_parser.add_argument("config_file")
#     args = top_parser.parse_args()

#     parser = HfArgumentParser([LlmViewerConfig])
#     config = parser.parse_json_file(args.config_file)[0]

#     with SystemMonitor(config.debug) as prof:
#         app = App(config)
#         app.run()