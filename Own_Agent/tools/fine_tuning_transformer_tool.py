from crewai.tools import BaseTool
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    Trainer,
    TrainingArguments,
)
from datasets import load_dataset
import os
import json


class FTTool(BaseTool):
    name: str = "FTTool"
    description: str = (
        "Fine-tunes a HuggingFace model for a given task type. "
        "Requires: model_name, dataset_name_or_path, task_type (causal_lm, sequence_classification, token_classification, etc.), "
        "and optional training arguments."
    )

    def _run(
        self, model_name: str, dataset_name_or_path: str, task_type: str, **kwargs
    ) -> str:
        try:
            model_loader_map = {
                "causal_lm": AutoModelForCausalLM,
                "sequence_classification": AutoModelForSequenceClassification,
                "token_classification": AutoModelForTokenClassification,
            }
            if task_type not in model_loader_map:
                return json.dumps(
                    {
                        "status": "error",
                        "message": f"Unsupported task_type '{task_type}'. Supported: {list(model_loader_map.keys())}",
                    },
                    indent=2,
                )

            tokenizer = AutoTokenizer.from_pretrained(model_name)
            if task_type == "causal_lm" and tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            model = model_loader_map[task_type].from_pretrained(model_name)

            # Load dataset
            if os.path.exists(dataset_name_or_path):
                dataset = load_dataset("json", data_files=dataset_name_or_path)
            else:
                dataset = load_dataset(dataset_name_or_path)

            # Ensure validation split exists
            if "validation" not in dataset:
                if "test" in dataset:
                    dataset["validation"] = dataset["test"]
                elif "train" in dataset:
                    split_dataset = dataset["train"].train_test_split(test_size=0.1)
                    dataset["train"] = split_dataset["train"]
                    dataset["validation"] = split_dataset["test"]

            def tokenize_function(examples):
                max_len = kwargs.get(
                    "max_length", 512 if task_type == "causal_lm" else 256
                )
                return tokenizer(
                    examples["text"],
                    truncation=True,
                    padding="max_length",
                    max_length=max_len,
                )

            tokenized_datasets = dataset.map(tokenize_function, batched=True)

            # Adjust evaluation strategy
            eval_strategy = kwargs.get(
                "evaluation_strategy",
                "epoch" if "validation" in tokenized_datasets else "no",
            )

            output_dir = "./output/fine_tuning_results"
            train_args = TrainingArguments(
                output_dir=output_dir,
                eval_strategy=eval_strategy,
                learning_rate=kwargs.get("learning_rate", 5e-5),
                per_device_train_batch_size=kwargs.get("train_batch_size", 4),
                per_device_eval_batch_size=kwargs.get("eval_batch_size", 4),
                num_train_epochs=kwargs.get("num_epochs", 3),
                weight_decay=kwargs.get("weight_decay", 0.01),
                logging_dir=os.path.join(output_dir, "logs"),
                logging_steps=kwargs.get("logging_steps", 50),
                save_strategy="epoch",
                push_to_hub=False,
            )

            trainer = Trainer(
                model=model,
                args=train_args,
                train_dataset=tokenized_datasets["train"],
                eval_dataset=tokenized_datasets.get("validation"),
                tokenizer=tokenizer,
            )

            trainer.train()

            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)

            return json.dumps(
                {
                    "status": "success",
                    "output_dir": output_dir,
                    "message": f"Fine-tuned {model_name} for {task_type} saved at {output_dir}",
                },
                indent=2,
            )

        except Exception as e:
            return json.dumps({"status": "error", "message": str(e)}, indent=2)
