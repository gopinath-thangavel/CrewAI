from crewai.tools import BaseTool
from typing import Dict, Any
import os
import re

class FineTuningTool(BaseTool):
    name: str = "FineTuningTool"
    description: str = ("""Fine-tunes any Hugging Face-compatible Large Language Model (LLM) using the Unsloth library for 
        fast and memory-efficient training. This tool allows full customization of the fine-tuning process, 
        enabling use-case specific adaptation of foundation models through parameter-efficient techniques 
        like LoRA (Low-Rank Adaptation).
        
        The agent must provide all required training and LoRA configuration parameters, including:
        - base_model_name_or_path: The Hugging Face model to fine-tune.
        - load_in_4bit: Boolean to enable 4-bit quantized loading (reduces memory usage).
        - max_seq_length: Maximum token length for input sequences.
        - batch_size and gradient_accumulation_steps: To control memory usage and effective batch size.
        - num_epochs: Total number of training epochs.
        - learning_rate: Learning rate for the optimizer.
        - r, lora_alpha, target_modules: LoRA-specific hyperparameters.
        - finetune_vision_layers and finetune_language_layers: Booleans to specify which layers to fine-tune.
        - training_dataset: Dataset to be used for model fine-tuning (can be Hugging Face name or local path).
        
        This tool includes built-in support for dynamic batch size adjustment to avoid out-of-memory (OOM)
        errors. It uses Unsloth’s PEFT-enabled architecture to provide a scalable and efficient fine-tuning 
        workflow across a wide range of hardware configurations."""
    )

    def _run(
        self,
        base_model_name: str,
        dataset_id: str,
        finetune_settings: Dict[str, Any]
    ) -> str:
        import torch
        from unsloth import FastLanguageModel
        from datasets import load_dataset
        from transformers import TrainingArguments
        from trl import SFTTrainer

        def slugify(text):
            return re.sub(r'\W+', '-', text.lower())

        try:
            dataset = load_dataset(dataset_id, split="train")

            def formatting(example):
                return [
                    f"<|system|>\nYou are a helpful assistant</s>\n"
                    f"<|user|>\n{example.get('instruction', '')}</s>\n"
                    f"<|assistant|>\n{example.get('output', '')}</s>"
                ]

            dataset = dataset.map(lambda x: {"text": formatting(x)})
        except Exception as e:
            return f"Failed to load or process dataset '{dataset_id}': {e}"

        output_dir = f"./output/finetuned-models/{slugify(base_model_name)}-{slugify(dataset_id)}"
        os.makedirs(output_dir, exist_ok=True)

        try:
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=base_model_name,
                max_seq_length=finetune_settings.get("max_seq_length", 2048),
                dtype=None,
                load_in_4bit=True,
            )

            model = FastLanguageModel.get_peft_model(
                model,
                r=finetune_settings["lora_r"],
                target_modules=finetune_settings["target_modules"],
                lora_alpha=finetune_settings["lora_alpha"],
                lora_dropout=finetune_settings["lora_dropout"],
                bias="none",
                use_gradient_checkpointing=True,
                random_state=42,
                use_rslora=False,
                loftq_config=None,
            )

            args = TrainingArguments(
                per_device_train_batch_size=finetune_settings["batch_size"],
                gradient_accumulation_steps=finetune_settings["gradient_accumulation_steps"],
                warmup_steps=5,
                max_steps=finetune_settings["max_steps"],
                learning_rate=finetune_settings["learning_rate"],
                fp16=finetune_settings["fp16"],
                logging_steps=1,
                output_dir=output_dir,
                optim=finetune_settings.get("optim", "paged_adamw_8bit"),
                lr_scheduler_type=finetune_settings.get("lr_scheduler_type", "constant"),
                save_total_limit=1,
            )

            trainer = SFTTrainer(
                model=model,
                tokenizer=tokenizer,
                train_dataset=dataset,
                dataset_text_field="text",
                max_seq_length=finetune_settings["max_seq_length"],
                args=args,
                packing=False,
                formatting_func=formatting,
            )

            trainer.train()

            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)

            return (
                f"Fine-tuning complete!\n"
                f"Model: {base_model_name}\n"
                f"Dataset: {dataset_id}\n"
                f"Saved to: {output_dir}"
            )

        except Exception as e:
            return f"Fine-tuning failed: {str(e)}"
