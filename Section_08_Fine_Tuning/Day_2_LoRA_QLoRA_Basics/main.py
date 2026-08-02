"""
Day 2 - LoRA/QLoRA first fine-tune  (Colab / GPU environment)
--------------------------------------------------------------
NOTE: this script assumes a CUDA GPU + unsloth. It will NOT run on CPU.
Best run in Google Colab (T4 GPU) as a notebook, but the shape is here.
"""

from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel


def main() -> None:
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Llama-3.2-3B-Instruct",
        max_seq_length=2048,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=16, lora_alpha=16, lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    ds = load_dataset("json", data_files="triage_train.jsonl", split="train")
    ds = ds.map(lambda r: {"text": tokenizer.apply_chat_template(
        r["messages"], tokenize=False, add_generation_prompt=False)})

    trainer = SFTTrainer(
        model=model, tokenizer=tokenizer, train_dataset=ds,
        args=SFTConfig(
            output_dir="outputs",
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            num_train_epochs=3,
            learning_rate=2e-4,
            warmup_steps=5,
            logging_steps=5,
            save_strategy="no",
            report_to="none",
            fp16=True,
        ),
    )
    trainer.train()
    model.save_pretrained("triage-lora")
    tokenizer.save_pretrained("triage-lora")
    print("Saved LoRA adapter to ./triage-lora")


if __name__ == "__main__":
    main()
