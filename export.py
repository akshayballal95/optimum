from transformers import AutoModel, AutoConfig
from colpali_engine.models.late_interaction.colpali_architecture import ColPali
from transformers import GemmaForCausalLM,PaliGemmaForConditionalGeneration
import torch
model = "vidore/colpali-v1.2-merged"
config = AutoConfig.from_pretrained(model)
model = ColPali.from_pretrained(model, config=config, device_map="cuda").eval()

from optimum.exporters import TasksManager
from pathlib import Path
from optimum.exporters.onnx.convert import export

onnx_path = Path("colpali/model.onnx")

onnx_config_constructor = TasksManager.get_exporter_config_constructor("onnx", model, task="feature-extraction")
onnx_config = onnx_config_constructor(model.config)
onnx_inputs, onnx_outputs = export(model, onnx_config, onnx_path, 16, device="cuda")