from lokahi import Inference, file_get_contents, TextStreamer

o = Inference("llama3-1B-chat", device="cuda:0", logging=True) #llama3-1B-chat(3B, 8B) | gpt-oss-20B
o.ini_model(models_dir="/media/mega4alik/ssd/models/", force_download=False)
o.offload_layers_to_cpu(layers_num=2) #offload some layers to CPU for speed increase
past_key_values = o.DiskCache(cache_dir="/media/mega4alik/ssd/kv_cache/")
text_streamer = TextStreamer(o.tokenizer, skip_prompt=True, skip_special_tokens=False)

sm, um = "You are helpful AI assistant", "List planets starting from Mercury"
#sm, um = file_get_contents("./samples/85k_sample.txt"), "What's common between these articles?"
messages = [{"role":"system", "content":sm}, {"role":"user", "content":um}]
input_ids = o.tokenizer.apply_chat_template(messages, reasoning_effort="minimal", tokenize=True, add_generation_prompt=True, return_tensors="pt").to(o.device)
outputs = o.model.generate(input_ids=input_ids,  past_key_values=past_key_values, max_new_tokens=100, streamer=text_streamer).cpu()
answer = o.tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=False)
print(answer)