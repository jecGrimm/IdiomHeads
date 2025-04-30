from llm_transparency_tool.models.tlens_model import TransformerLensTransparentLlm
import torch as t

model = TransformerLensTransparentLlm("EleutherAI/pythia-14m", dtype = t.bfloat16)


# VON LEA
# resid_pre = model.residual_in(layer)[B0].unsqueeze(0)
# resid_mid = model.residual_after_attn(layer)[B0].unsqueeze(0)
# decomposed_attn = model.decomposed_attn(B0, layer).unsqueeze(0)

# head_contrib, _ = get_attention_contributions(resid_pre, resid_mid, decomposed_attn)
# # [batch pos key_pos head] -> [head]
# flat_contrib = head_contrib[0, -1, source_token, :] # #TODO token 2 is the noun, do this dynamically


# if total_attn_contributions is None:
#     total_attn_contributions = torch.zeros_like(flat_contrib)
# total_attn_contributions += flat_contrib