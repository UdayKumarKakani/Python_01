"""
Day 6 - Capstone pipeline scaffold (Colab / GPU)
-------------------------------------------------
Fill in the TODOs for your chosen track. This file is a shape, not a runnable
end-to-end (each track needs its own to_chat and eval logic).
"""

from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel


DATASET_ID = "TODO: pick from the class list"
RUN_NAME   = "capstone-v1"


def to_chat(row: dict) -> dict:
    # TODO: adapt to your dataset's columns
    return {"messages": [
        {"role": "system",    "content": "You are a helpful assistant."},
        {"role": "user",      "content": row["input"]},
        {"role": "assistant", "content": row["output"]},
    ]}


def main() -> None:
    ds = load_dataset(DATASET_ID, split="train[:1000]").map(to_chat)
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
    trainer = SFTTrainer(
        model=model, tokenizer=tokenizer,
        train_dataset=ds["train"].map(fmt),
        eval_dataset=ds["test"].map(fmt),
        args=SFTConfig(
            output_dir="capstone-out", run_name=RUN_NAME,
            per_device_train_batch_size=2, gradient_accumulation_steps=4,
            num_train_epochs=2, learning_rate=2e-4,
            eval_strategy="steps", eval_steps=100,
            logging_steps=20, save_strategy="no",
            report_to="wandb", fp16=True,
        ),
    )
    trainer.train()

    merged = model.merge_and_unload()
    merged.save_pretrained_gguf(
        "capstone-gguf", tokenizer, quantization_method="q4_k_m",
    )
    print("Done. Push to HF Hub next.")


if __name__ == "__main__":
    main()
