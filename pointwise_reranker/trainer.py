# trainer.py
import os
import logging
from typing import Optional

import torch
from transformers.trainer import Trainer
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from peft import get_peft_model_state_dict

logger = logging.getLogger(__name__)

class RerankerTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_amp = self.args.bf16 or self.args.fp16

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        output_dir = output_dir or self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info("Saving model checkpoint to %s", output_dir)

        # Save only student PEFT adapter in standard runs
        self.model.save(output_dir)

        # For ZeRO-3 we need to gather LoRA state dict
        if is_deepspeed_zero3_enabled():
            if state_dict is None:
                state_dict = self.model.state_dict()
            # drop teacher.*
            state_dict = {k: v for k, v in state_dict.items() if not k.startswith("teacher.")}
            # remove 'student.' prefix so PEFT can find correct keys
            prefix = "student."
            assert all(k.startswith(prefix) for k in state_dict.keys()), "Unexpected keys in state_dict."
            state_dict = {k[len(prefix):]: v for k, v in state_dict.items()}
            lora_state = get_peft_model_state_dict(self.model.student, state_dict)
            if self.args.process_index <= 0:
                torch.save(lora_state, os.path.join(output_dir, "adapter_model.bin"))
                print(f"Saved PEFT adapter to {output_dir}/adapter_model.bin")

    def compute_loss(self, model, inputs, return_outputs=False):
        pair_student = inputs["student"]
        pair_teacher = inputs["teacher"]
        outputs = model(pair_student, pair_teacher)
        return outputs.loss
