"""
Day 3 - Fine-tune Llama 3.2 3B on Dolly-15k (Colab / GPU)
----------------------------------------------------------
"""

from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel


def dolly_to_chat(row: dict) -> dict:
    user = row["instruction"] + (f"\n\n{row['context']}" if row["context"] else "")
    return {"messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": user},
        {"role": "assistant", "content": row["response"]},
    ]}


def main() -> None:
    ds = load_dataset("databricks/databricks-dolly-15k", split="train[:1000]")
    ds = ds.map(dolly_to_chat, remove_columns=ds.column_names)
    ds = ds.train_test_split(test_size=0.1, seed=42)

    model, tokenizer = FastLanguageModel.from_pretrained(
        "unsloth/Llama-3.2-3B-Instruct", max_seq_length=2048, load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model, r=16, lora_alpha=16,
        target_modules=["q_proj","k_proj","v_proj","o_proj",
                        "gate_proj","up_proj","down_proj"],
        use_gradient_checkpointing="unsloth", random_state=42,
    )

    fmt = lambda r: {"text": tokenizer.apply_chat_template(r["messages"], tokenize=False)}
    train_ds = ds["train"].map(fmt)
    eval_ds  = ds["test"].map(fmt)

    trainer = SFTTrainer(
        model=model, tokenizer=tokenizer,
        train_dataset=train_ds, eval_dataset=eval_ds,
        args=SFTConfig(
            output_dir="dolly-out",
            per_device_train_batch_size=2, gradient_accumulation_steps=4,
            num_train_epochs=1, learning_rate=2e-4,
            eval_strategy="steps", eval_steps=50,
            logging_steps=10, save_strategy="no",
            report_to="none", fp16=True,
        ),
    )
    trainer.train()
    model.save_pretrained("dolly-lora")
    tokenizer.save_pretrained("dolly-lora")


if __name__ == "__main__":
    main()
