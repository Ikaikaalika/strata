# gemma3-12B Image+Text example

import torch
from lokahi import Inference, file_get_contents, TextStreamer
o = Inference("gemma3-12B", device="cuda:0", logging=True, multimodality=True)
o.ini_model(models_dir="/media/mega4alik/ssd/models/", force_download=False)
o.offload_layers_to_cpu(layers_num=12) #offload some layers to CPU for speed boost
past_key_values = None #o.DiskCache(cache_dir="/media/mega4alik/ssd/kv_cache/") #uncomment for large context
text_streamer = TextStreamer(o.tokenizer, skip_prompt=True, skip_special_tokens=False)	

messages = [
    {"role": "system","content": [{"type": "text", "text": "You are a helpful assistant."}]},
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "https://raw.githubusercontent.com/OpenBMB/MiniCPM-V/main/assets/minicpm-v-4dot5-framework.png"},
            {"type": "text", "text": "Describe this image in detail."}
        ]
    }
]
inputs = o.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to(o.device, dtype=torch.bfloat16)
outputs = o.model.generate(**inputs, past_key_values=past_key_values, max_new_tokens=500, streamer=text_streamer).cpu()
answer = o.tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=False)
print(answer)