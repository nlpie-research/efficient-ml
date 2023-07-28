# taken from https://github.com/huggingface/peft/issues/183
import os
import warnings
import json

import torch
from accelerate import dispatch_model, infer_auto_device_map
from accelerate.hooks import AlignDevicesHook, add_hook_to_module, remove_hook_from_submodules
from accelerate.utils import get_balanced_memory
from peft import PrefixTuningConfig, LoraConfig, PrefixEncoder, LoraModel
from peft import TRANSFORMERS_MODELS_TO_PREFIX_TUNING_POSTPROCESS_MAPPING
from peft.mapping import _prepare_lora_config, _prepare_prompt_learning_config
from transformers import PreTrainedModel


WEIGHTS_NAME = "adapter_model.bin"
CONFIG_NAME2CONFIG_CLS = {
    "pv2_config": PrefixTuningConfig,
    "lora_config": LoraConfig
}



def get_peft_model_state_dict(model, state_dict=None):
    if state_dict is None:
        state_dict = model.state_dict()

    bias = model.lora_config.bias
    if bias == "none":
        to_return = {k: state_dict[k] for k in state_dict if "lora_" in k}
    elif bias == "all":
        to_return = {k: state_dict[k] for k in state_dict if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        for k in state_dict:
            if "lora_" in k:
                to_return[k] = state_dict[k]
                bias_name = k.split("lora_")[0] + "bias"
                if bias_name in state_dict:
                    to_return[bias_name] = state_dict[bias_name]
    else:
        raise NotImplementedError

    if model.pv2_config.inference_mode:
        prompt_embeddings = model.prompt_encoder.embedding.weight
    else:
        prompt_embeddings = model.get_prompt_embedding_to_save()
    to_return["prompt_embeddings"] = prompt_embeddings

    return to_return


def set_peft_model_state_dict(model, peft_model_state_dict):
    model.load_state_dict(peft_model_state_dict, strict=False)
    model.prompt_encoder.embedding.load_state_dict(
        {"weight": peft_model_state_dict["prompt_embeddings"]}, strict=True
    )
    return model


def config_save_pretrained(config, save_directory, **kwargs):
    if os.path.isfile(save_directory):
        raise AssertionError(f"Provided path ({save_directory}) should be a directory, not a file")

    os.makedirs(save_directory, exist_ok=True)

    output_dict = config.__dict__
    output_path = os.path.join(save_directory, "pv2_config" if isinstance(config, PrefixTuningConfig) else "lora_config")

    # save it
    with open(output_path, "w") as writer:
        writer.write(json.dumps(output_dict, indent=2, sort_keys=True))


def config_from_pretrained(config_name, pretrained_model_name_or_path, **kwargs):
    config_file = os.path.join(pretrained_model_name_or_path, config_name)

    loaded_attributes = CONFIG_NAME2CONFIG_CLS[config_name].from_json_file(config_file)

    config = CONFIG_NAME2CONFIG_CLS[config_name](**kwargs)

    for key, value in loaded_attributes.items():
        if hasattr(config, key):
            setattr(config, key, value)

    return config


class Pv2LoraCausalModel(torch.nn.Module):
    def __init__(self, model, pv2_config: PrefixTuningConfig, lora_config: LoraConfig):
        super().__init__()

        self.config = model.config
        pv2_config.base_model_name_or_path = model.__dict__.get("name_or_path", None)
        lora_config.base_model_name_or_path = model.__dict__.get("name_or_path", None)
        self.pv2_config = _prepare_prompt_learning_config(pv2_config, model.config.to_dict())
        self.lora_config = _prepare_lora_config(lora_config, model.config.to_dict())

        self.base_model = LoraModel(lora_config, model)
        self._setup_prompt_encoder()
        self.base_model_prepare_inputs_for_generation = self.base_model.model.prepare_inputs_for_generation

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def _freeze_base_model(self):
        for name, module in self.base_model.named_children():
            for param in module.parameters():
                param.requires_grad = False

    def _setup_prompt_encoder(self):
        transformer_backbone = None
        for name, module in self.base_model.model.named_children():
            if isinstance(module, PreTrainedModel):
                if transformer_backbone is None:
                    transformer_backbone = module
                    self.transformer_backbone_name = name

        if self.pv2_config.num_transformer_submodules is None:
            self.pv2_config.num_transformer_submodules = 1

        for named_param, value in list(transformer_backbone.named_parameters()):
            if value.shape[0] == self.base_model.config.vocab_size:
                self.word_embeddings = transformer_backbone.get_submodule(named_param.replace(".weight", ""))
                break

        prompt_encoder = PrefixEncoder(self.pv2_config)
        self.prompt_encoder = prompt_encoder
        self.prompt_tokens = torch.arange(
            self.pv2_config.num_virtual_tokens * self.pv2_config.num_transformer_submodules
        ).long()

    def get_prompt(self, batch_size):
        prompt_tokens = self.prompt_tokens.unsqueeze(0).expand(batch_size, -1).to(self.device)

        prompt_tokens = prompt_tokens[:, : self.pv2_config.num_virtual_tokens]
        if self.pv2_config.inference_mode:
            past_key_values = self.prompt_encoder.embedding.weight.repeat(batch_size, 1, 1)
        else:
            past_key_values = self.prompt_encoder(prompt_tokens)
        past_key_values = past_key_values.view(
            batch_size,
            self.pv2_config.num_virtual_tokens,
            self.pv2_config.num_layers * 2,
            self.pv2_config.num_attention_heads,
            self.pv2_config.token_dim // self.pv2_config.num_attention_heads,
        )
        if self.pv2_config.num_transformer_submodules == 2:
            past_key_values = torch.cat([past_key_values, past_key_values], dim=2)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(
            self.pv2_config.num_transformer_submodules * 2
        )
        if TRANSFORMERS_MODELS_TO_PREFIX_TUNING_POSTPROCESS_MAPPING.get(self.config.model_type, None) is not None:
            post_process_fn = TRANSFORMERS_MODELS_TO_PREFIX_TUNING_POSTPROCESS_MAPPING[self.config.model_type]
            past_key_values = post_process_fn(past_key_values)
        return past_key_values

    def get_prompt_embedding_to_save(self):
        prompt_tokens = self.prompt_tokens.unsqueeze(0).expand(1, -1).to(self.device)
        prompt_tokens = prompt_tokens[:, : self.pv2_config.num_virtual_tokens]
        prompt_embeddings = self.prompt_encoder(prompt_tokens)
        return prompt_embeddings[0].detach().cpu()

    def print_trainable_parameters(self):
        trainable_params = 0
        all_param = 0
        for _, param in self.named_parameters():
            num_params = param.numel()
            # if using DS Zero 3 and the weights are initialized empty
            if num_params == 0 and hasattr(param, "ds_numel"):
                num_params = param.ds_numel

            all_param += num_params
            if param.requires_grad:
                trainable_params += num_params
        print(
            f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
        )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        batch_size = input_ids.shape[0]
        if attention_mask is not None:
            # concat prompt attention mask
            prefix_attention_mask = torch.ones(batch_size, self.pv2_config.num_virtual_tokens).to(self.device)
            attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)

        if kwargs.get("position_ids", None) is not None:
            warnings.warn("Position ids are not supported for parameter efficient tuning. Ignoring position ids.")
            kwargs["position_ids"] = None
        if kwargs.get("token_type_ids", None) is not None:
            warnings.warn("Token type ids are not supported for parameter efficient tuning. Ignoring token type ids")
            kwargs["token_type_ids"] = None
        kwargs.update(
            {
                "attention_mask": attention_mask,
                "labels": labels,
                "output_attentions": output_attentions,
                "output_hidden_states": output_hidden_states,
                "return_dict": return_dict,
            }
        )

        past_key_values = self.get_prompt(batch_size)
        return self.base_model(input_ids=input_ids, past_key_values=past_key_values, **kwargs)

    def generate(self, **kwargs):
        self.base_model.model.prepare_inputs_for_generation = self.prepare_inputs_for_generation
        try:
            if "input_ids" not in kwargs:
                raise ValueError("input_ids must be provided for Peft model generation")
            if kwargs.get("attention_mask", None) is not None:
                # concat prompt attention mask
                prefix_attention_mask = torch.ones(
                    kwargs["input_ids"].shape[0], self.pv2_config.num_virtual_tokens
                ).to(kwargs["input_ids"].device)
                kwargs["attention_mask"] = torch.cat((prefix_attention_mask, kwargs["attention_mask"]), dim=1)

            if kwargs.get("position_ids", None) is not None:
                warnings.warn(
                    "Position ids are not supported for parameter efficient tuning. Ignoring position ids."
                )
                kwargs["position_ids"] = None
            if kwargs.get("token_type_ids", None) is not None:
                warnings.warn(
                    "Token type ids are not supported for parameter efficient tuning. Ignoring token type ids"
                )
                kwargs["token_type_ids"] = None

            outputs = self.base_model.generate(**kwargs)
        except:
            self.base_model.model.prepare_inputs_for_generation = self.base_model_prepare_inputs_for_generation
            raise
        else:
            self.base_model.model.prepare_inputs_for_generation = self.base_model_prepare_inputs_for_generation
            return outputs

    def prepare_inputs_for_generation(self, *args, **kwargs):
        model_kwargs = self.base_model_prepare_inputs_for_generation(*args, **kwargs)
        past_key_values = self.get_prompt(batch_size=model_kwargs["input_ids"].shape[0])
        model_kwargs["past_key_values"] = past_key_values

        return model_kwargs

    def save_pretrained(self, save_directory, **kwargs):
        if os.path.isfile(save_directory):
            raise ValueError(f"Provided path ({save_directory}) should be a directory, not a file")
        os.makedirs(save_directory, exist_ok=True)

        # save only the trainable weights
        output_state_dict = get_peft_model_state_dict(self, kwargs.get("state_dict", None))
        torch.save(output_state_dict, os.path.join(save_directory, WEIGHTS_NAME))

        # save the config and change the inference mode to `True`
        for config in [self.lora_config, self.pv2_config]:
            if config.base_model_name_or_path is None:
                config.base_model_name_or_path = self.base_model.model.__dict__.get("name_or_path", None)
            inference_mode = config.inference_mode
            config.inference_mode = True
            config_save_pretrained(config, save_directory)
            config.inference_mode = inference_mode

    @classmethod
    def from_pretrained(cls, model, model_id, **kwargs):
        # load the config
        pv2_config = config_from_pretrained("pv2_config", model_id)
        lora_config = config_from_pretrained("lora_config", model_id)

        if getattr(model, "hf_device_map", None) is not None:
            remove_hook_from_submodules(model)

        model = Pv2LoraCausalModel(model, pv2_config, lora_config)

        # load weights if any
        filename = os.path.join(model_id, WEIGHTS_NAME)

        adapters_weights = torch.load(
            filename, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        # load the weights into the model
        model = set_peft_model_state_dict(model, adapters_weights)
        if getattr(model, "hf_device_map", None) is not None:
            device_map = kwargs.get("device_map", "auto")
            max_memory = kwargs.get("max_memory", None)
            no_split_module_classes = model._no_split_modules
            if device_map != "sequential":
                max_memory = get_balanced_memory(
                    model,
                    max_memory=max_memory,
                    no_split_module_classes=no_split_module_classes,
                    low_zero=(device_map == "balanced_low_0"),
                )
            if isinstance(device_map, str):
                device_map = infer_auto_device_map(
                    model, max_memory=max_memory, no_split_module_classes=no_split_module_classes
                )
            model = dispatch_model(model, device_map=device_map)
            hook = AlignDevicesHook(io_same_device=True)

            remove_hook_from_submodules(model.prompt_encoder)
            add_hook_to_module(model.base_model.model, hook)
        return model