from typing import List
import re
from dataclasses import dataclass

def parse_trace_file(filepath: str):
    js_weights = []
    safetensors_weights = []
    with open(filepath, "r") as f:
        for line in f:
            if not line.strip():
                continue
                
            parts = line.strip().split()
            # print(f"Processing line: {line.strip()}")
            # print(f"Parts: {parts}")
            
            framework = parts[2]  # jetstream_pt
            weight_name = parts[4]  # the actual weight name
            # print(f"Framework: {framework}, Weight: {weight_name}")
            
            if framework == "jetstream_pt":
                js_weights.append(weight_name)
            else:
                assert framework == "safetensors"
                safetensors_weights.append(weight_name)

    print(f"\nFound {len(js_weights)} jetstream weights")
    print(f"Found {len(safetensors_weights)} safetensors weights")
    # print("\nSample jetstream weights:", js_weights[:3])
    # print("Sample safetensors weights:", safetensors_weights[:3])
    
    return js_weights, safetensors_weights

def _hf_mapping(layer_idx: int = -1, expert_idx: int = -1) -> dict:
  # pylint: disable=line-too-long
  return {
      "tok_embeddings.weight": "model.embed_tokens.weight",
      "norm.weight": "model.norm.weight",
      "output.weight": "lm_head.weight",
      # MOE model
      f"layers.{layer_idx}.attention_norm.weight": f"model.layers.{layer_idx}.input_layernorm.weight",
      f"layers.{layer_idx}.ffn_norm.weight": f"model.layers.{layer_idx}.post_attention_layernorm.weight",
      f"layers.{layer_idx}.attention.wq.weight": f"model.layers.{layer_idx}.self_attn.q_proj.weight",
      f"layers.{layer_idx}.attention.wk.weight": f"model.layers.{layer_idx}.self_attn.k_proj.weight",
      f"layers.{layer_idx}.attention.wv.weight": f"model.layers.{layer_idx}.self_attn.v_proj.weight",
      f"layers.{layer_idx}.attention.wo.weight": f"model.layers.{layer_idx}.self_attn.o_proj.weight",
      f"layers.{layer_idx}.feed_forward.gate.weight": f"model.layers.{layer_idx}.block_sparse_moe.gate.weight",
      f"layers.{layer_idx}.feed_forward.experts.{expert_idx}.w1.weight": f"model.layers.{layer_idx}.block_sparse_moe.experts.{expert_idx}.w1.weight",
      f"layers.{layer_idx}.feed_forward.experts.{expert_idx}.w2.weight": f"model.layers.{layer_idx}.block_sparse_moe.experts.{expert_idx}.w2.weight",
      f"layers.{layer_idx}.feed_forward.experts.{expert_idx}.w3.weight": f"model.layers.{layer_idx}.block_sparse_moe.experts.{expert_idx}.w3.weight",
      # dense model
      f"layers.{layer_idx}.feed_forward.w1.weight": f"model.layers.{layer_idx}.mlp.gate_proj.weight",
      f"layers.{layer_idx}.feed_forward.w2.weight": f"model.layers.{layer_idx}.mlp.down_proj.weight",
      f"layers.{layer_idx}.feed_forward.w3.weight": f"model.layers.{layer_idx}.mlp.up_proj.weight",
  }

def get_js_mapping(key):
    fields = key.split(".")
    num_fields = [int(field) for field in fields if re.match(r"[0-9]+", field) is not None]
    mapping = _hf_mapping(*num_fields)
    return mapping

def map_js_to_safetensors_weight_name(key: str) -> str:
    mapping = get_js_mapping(key)
    if key not in mapping:
      raise ValueError(f"Key `{key}` is missing from the mapping.")
    return mapping[key]

def map_safetensors_to_js_weight_name(key: str) -> str:
    mapping = get_js_mapping(key)
    inv_mapping = {v: k for k, v in mapping.items()}
    if key not in inv_mapping:
      raise ValueError(f"Key `{key}` is missing from the original collection and from the mapping.")
    return inv_mapping[key]





def convert_safetensors_to_js(safetensors_weights: List[str]) -> List[str]:
    """Convert safetensors weight names to jetstream format."""
    # mapper = _HFNamespaceMapper({})
    output_weights = []
    
    for weight in safetensors_weights:
        try:
            
            js_weight = map_safetensors_to_js_weight_name(weight)
            # print(f"Mapped to: {js_weight}")
            output_weights.append(js_weight)
        except ValueError as e:
            print(f"Warning: Could not map weight {weight}: {str(e)}")
            continue
            
    return output_weights

def convert_js_to_safetensors(js_weights: List[str]) -> List[str]:
    """Convert jetstream weight names to safetensors format."""
    output_weights = []
    for weight in js_weights:
        try:
            safetensors_weight = map_js_to_safetensors_weight_name(weight)
            output_weights.append(safetensors_weight)
        except ValueError as e:
            print(f"Warning: Could not map weight {weight}: {str(e)}")
            continue
            
    return output_weights

def compare_weights(js_weights: List[str], converted_weights: List[str]) -> bool:
    """Compare two lists of weight names, ignoring order."""
    return set(js_weights) == set(converted_weights)

# Usage
if __name__ == "__main__":
    js_weights, safetensors_weights = parse_trace_file("trace.txt")
    # output_weights = convert_safetensors_to_js(safetensors_weights)
    output_weights = convert_js_to_safetensors(js_weights)
    if not compare_weights(safetensors_weights, output_weights):
    # if not compare_weights(js_weights, output_weights):
        print("Weight conversion mismatch!")
        print("Missing in converted:", set(safetensors_weights) - set(output_weights))
        print("Extra in converted:", set(output_weights) - set(safetensors_weights))
    else:
        print("Weight conversion successful!")

