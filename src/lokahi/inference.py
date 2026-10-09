import os, requests, zipfile
from transformers import AutoTokenizer, AutoProcessor
from .utils import Stats, file_get_contents
from .backends import select_backend

class Inference:
	def __init__(self, model_id, device=None, logging=True, multimodality=False, tracer=None, memory_budget_bytes=None):
		self.model_id = model_id
		self.backend_selection = select_backend(device)
		self.backend = self.backend_selection.backend
		self.device = self.backend_selection.resolved_device
		self.device_request = self.backend_selection.device_request
		self.multimodality = multimodality
		self.stats = Stats() if logging else None
		self.tracer = tracer
		self.memory_budget_bytes = memory_budget_bytes

	def download_and_unpack(self, models_dir: str):
		os.makedirs(models_dir, exist_ok=True)
		urls = {
			"llama3-1B-chat": "https://lokahi.s3.us-east-1.amazonaws.com/models/llama3-1B-chat.zip",
			"llama3-3B-chat": "https://lokahi.s3.us-east-1.amazonaws.com/models/llama3-3B-chat.zip",
			"llama3-8B-chat": "https://lokahi.s3.us-east-1.amazonaws.com/models/llama3-8B-chat.zip",
			"gpt-oss-20B":    "https://lokahi.s3.us-east-1.amazonaws.com/models/gpt-oss-20B.zip"
		}
		url = urls[self.model_id]
		
		# Extract filename from URL
		filename = url.split("/")[-1]
		zip_path = os.path.join(models_dir, filename)

		# Download the file
		print(f"Downloading {url} ...")
		response = requests.get(url, stream=True)
		response.raise_for_status()
		with open(zip_path, "wb") as f:
			for chunk in response.iter_content(chunk_size=8192):
				f.write(chunk)
		print(f"Downloaded to {zip_path}")

		# Unzip
		print(f"Unpacking {zip_path} ...")
		with zipfile.ZipFile(zip_path, 'r') as zip_ref:
			zip_ref.extractall(models_dir)
		print(f"Unpacked to {models_dir}")

		os.remove(zip_path) # Optional: remove the zip file after extraction

	
	def hf_download(self, model_dir):
		from huggingface_hub import snapshot_download
		urls = {
			"llama3-1B-chat": "meta-llama/Llama-3.2-1B-Instruct",
			"llama3-3B-chat": "meta-llama/Llama-3.2-3B-Instruct",
			"llama3-8B-chat": "meta-llama/Llama-3.1-8B-Instruct",
			"gpt-oss-20B": "openai/gpt-oss-20b",
			"qwen3-next-80B": "Qwen/Qwen3-Next-80B-A3B-Instruct",
			"gemma3-12B": "google/gemma-3-12b-it",
			"deepseek-coder-1.3b": "deepseek-ai/deepseek-coder-1.3b-instruct",
			"deepseek-coder-6.7b": "deepseek-ai/deepseek-coder-6.7b-instruct",
			"deepseek-llm-7b": "deepseek-ai/deepseek-llm-7b-chat"
		}
		url = urls[self.model_id]
		print(f"Downloading {url} ...")
		snapshot_download(
		    repo_id=url,
		    local_dir=model_dir,
		    local_dir_use_symlinks=False
		)

	
	def ini_model(self, models_dir="./models/", force_download=False):
		models_list = ["llama3-1B-chat", "llama3-3B-chat", "llama3-8B-chat", "gpt-oss-20B", "qwen3-next-80B", "gemma3-12B", "deepseek-coder-1.3b", "deepseek-coder-6.7b", "deepseek-llm-7b"]
		if self.model_id not in models_list:
			raise ValueError("Incorrect model id. It must be one of", models_list)

		model_dir = os.path.join(models_dir, self.model_id)
		if os.path.exists(model_dir)==False or force_download==True:
			# All models now download from HuggingFace
			self.hf_download(model_dir)

		print("loading model from", model_dir)

		# Load model using MLX backend
		self._load_mlx_model(model_dir)

		# Load tokenizer if not already loaded by mlx_lm
		if not hasattr(self, '_using_mlx_lm'):
			self.tokenizer = AutoTokenizer.from_pretrained(model_dir)

	def _load_mlx_model(self, model_dir: str):
		"""Load model using MLX backend."""
		from transformers import AutoConfig

		if self.model_id.startswith("llama"):
			# Use mlx_lm's standard loading for llama models
			try:
				if self.memory_budget_bytes is not None:
					raise ImportError("Lokahi memory governor requested")
				from mlx_lm import load
				print(f"Loading {self.model_id} using mlx_lm...")

				# Map our model IDs to HuggingFace model IDs
				hf_model_ids = {
					"llama3-1B-chat": "meta-llama/Llama-3.2-1B-Instruct",
					"llama3-3B-chat": "meta-llama/Llama-3.2-3B-Instruct",
					"llama3-8B-chat": "meta-llama/Llama-3.1-8B-Instruct",
				}

				hf_model_id = hf_model_ids.get(self.model_id)
				if not hf_model_id:
					raise ValueError(f"Unknown model ID: {self.model_id}")

				# Load model using mlx_lm
				self.model, self.tokenizer = load(hf_model_id)
				self._using_mlx_lm = True
				print(f"✅ MLX Llama model loaded: {self.model_id}")
				return

			except ImportError:
				if self.memory_budget_bytes is not None:
					print("Lokahi memory governor requested; using the custom layer loader")
				else:
					print("mlx_lm not available, falling back to custom loader")
			except Exception as e:
				print(f"Failed to load with mlx_lm: {e}")
				print("Falling back to custom loader")

			# Fallback to custom loader (for models with gds_export)
			from . import llama_mlx
			from .mlx_loader import MLXWeights

			# Load config
			config = AutoConfig.from_pretrained(model_dir)

			# Initialize MLX loader
			gds_export_path = os.path.join(model_dir, "gds_export")
			if not os.path.exists(gds_export_path):
				detail = (
					"The memory governor currently requires a Lokahi gds_export manifest."
					if self.memory_budget_bytes is not None
					else "Please install mlx_lm: pip install mlx-lm"
				)
				raise FileNotFoundError(
					f"Custom loader requires gds_export directory at {gds_export_path}. "
					f"{detail}"
				)

			llama_mlx.loader = MLXWeights(gds_export_path, device=self.device)
			llama_mlx.stats = self.stats

			# Create MLX model
			self.model = llama_mlx.MLXLlamaForCausalLM(
				config,
				tracer=self.tracer,
				weight_loader=llama_mlx.loader,
				memory_budget_bytes=self.memory_budget_bytes,
			)

			# Load embeddings and LM head from safetensors using MLX utils
			import mlx.core as mx
			import numpy as np

			# Use MLX's load function which handles all dtypes properly
			try:
				from mlx.utils import tree_map
				# Try loading from safetensors
				st_file = os.path.join(model_dir, "model.safetensors")
				if os.path.exists(st_file):
					weights_dict = mx.load(st_file)

					# Extract specific weights we need
					if "model.embed_tokens.weight" in weights_dict:
						self.model.model.embed_tokens_weight = weights_dict["model.embed_tokens.weight"]
					if "lm_head.weight" in weights_dict:
						self.model.model.lm_head_weight = weights_dict["lm_head.weight"]
					if "model.norm.weight" in weights_dict:
						self.model.model.norm.weight = weights_dict["model.norm.weight"]
				else:
					# Try sharded format
					safetensors_files = [f for f in os.listdir(model_dir) if f.endswith('.safetensors')]
					if safetensors_files:
						weights_dict = {}
						for st_file in safetensors_files:
							file_weights = mx.load(os.path.join(model_dir, st_file))
							weights_dict.update(file_weights)

						if "model.embed_tokens.weight" in weights_dict:
							self.model.model.embed_tokens_weight = weights_dict["model.embed_tokens.weight"]
						if "lm_head.weight" in weights_dict:
							self.model.model.lm_head_weight = weights_dict["lm_head.weight"]
						if "model.norm.weight" in weights_dict:
							self.model.model.norm.weight = weights_dict["model.norm.weight"]
					else:
						raise FileNotFoundError(f"No safetensors files found in {model_dir}")
			except Exception as e:
				print(f"Warning: Could not load embeddings from safetensors: {e}")
				print("Using default initialization")

			print(f"MLX Llama model loaded: {self.model_id}")

		elif self.model_id.startswith("deepseek"):
			from . import deepseek_mlx
			from .mlx_loader import MLXMoEWeightsLoader

			# Load config
			config = AutoConfig.from_pretrained(model_dir)

			# DeepSeek models use safetensors
			deepseek_mlx.loader = MLXMoEWeightsLoader(model_dir, device=self.device)
			deepseek_mlx.stats = self.stats

			# Create MLX model
			self.model = deepseek_mlx.MLXDeepSeekForCausalLM(
				config,
				tracer=self.tracer,
				weight_loader=deepseek_mlx.loader,
				memory_budget_bytes=self.memory_budget_bytes,
			)

			# Load embeddings and LM head from safetensors using MLX utils
			import mlx.core as mx
			import numpy as np

			# Use MLX's load function which handles all dtypes properly
			try:
				from mlx.utils import tree_map
				# Load all weights from safetensors
				st_file = os.path.join(model_dir, "model.safetensors")
				if os.path.exists(st_file):
					weights_dict = mx.load(st_file)

					# Extract specific weights we need
					if "model.embed_tokens.weight" in weights_dict:
						self.model.model.embed_tokens_weight = weights_dict["model.embed_tokens.weight"]
					if "lm_head.weight" in weights_dict:
						self.model.model.lm_head_weight = weights_dict["lm_head.weight"]
					if "model.norm.weight" in weights_dict:
						self.model.model.norm.weight = weights_dict["model.norm.weight"]
				else:
					print(f"Warning: {st_file} not found, using default initialization")
			except Exception as e:
				print(f"Warning: Could not load embeddings from safetensors: {e}")
				print("Using default initialization")

			print(f"MLX DeepSeek model loaded: {self.model_id}")

		elif self.model_id == "qwen3-next-80B":
			raise NotImplementedError(
				"Qwen3-Next MLX support not yet implemented."
			)
		elif self.model_id == "gemma3-12B":
			raise NotImplementedError(
				"Gemma3 MLX support not yet implemented."
			)
		elif self.model_id == "gpt-oss-20B":
			raise NotImplementedError(
				"GPT-OSS MLX support not yet implemented (requires mxfp4 unpacking)."
			)

	
	def DiskCache(self, cache_dir="./kvcache"):
		if self.model_id in ["gpt-oss-20B"]:
			print(f"{self.model_id} DiskCache is not supported at the moment. Using default DynamicCache instead")
			return None
		elif self.model_id=="qwen3-next-80B":
			from .mlx_kvcache import MLXQwen3NextDiskCache
			return MLXQwen3NextDiskCache(self.model.config, cache_dir=cache_dir, stats=self.stats)
		else:
			return self.backend.create_kv_cache(
				cache_dir=cache_dir,
				stats=self.stats,
				config=self.model.config,
				model_id=self.model_id
			)
