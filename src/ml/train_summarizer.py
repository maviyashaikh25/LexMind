from datasets import load_dataset
from transformers import BartForConditionalGeneration, BartTokenizer
import evaluate, numpy as np
import os
import torch
from torch.utils.data import DataLoader
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

MODEL_NAME = "facebook/bart-large-cnn"

# Lazy-loaded / fallback tokenizer and model
_tokenizer = None
_model = None

def get_default_model_and_tokenizer():
    global _tokenizer, _model
    if _tokenizer is None or _model is None:
        _tokenizer = BartTokenizer.from_pretrained("models/legal_summarizer")
        _model = BartForConditionalGeneration.from_pretrained("models/legal_summarizer")
    return _model, _tokenizer

def summarize(text: str, model=None, tokenizer=None, max_len: int = 200) -> str:
    if model is None or tokenizer is None:
        model, tokenizer = get_default_model_and_tokenizer()
    inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
    device = model.device
    ids    = model.generate(inputs["input_ids"].to(device), max_length=max_len, num_beams=4, early_stopping=True)
    return tokenizer.decode(ids[0], skip_special_tokens=True)

def get_mock_summarizer_data():
    from datasets import Dataset, DatasetDict
    judgments = [
        "The appellant filed a petition challenging the order of the High Court. The High Court had dismissed the bail application of the appellant. After hearing the arguments from both parties, the Supreme Court allowed the appeal and granted bail to the appellant.",
        "This is a civil breach of contract dispute where the plaintiff alleges that the defendant failed to deliver the goods on time. The defendant claims force majeure. The court ruled that the defendant breached the contract and ordered payment of damages.",
        "The prosecution alleged that the accused committed cheating under Section 420 IPC by inducing the victim to invest money in a fake business. The defense argued that there was no criminal intent. The court found the accused guilty."
    ] * 20
    summaries = [
        "Supreme Court allows appeal and grants bail to the appellant, reversing the High Court's dismissal.",
        "Court finds defendant in breach of contract and orders payment of damages to the plaintiff.",
        "Accused found guilty under Section 420 IPC for cheating by inducing victim to invest in a fake business."
    ] * 20
    train_ds = Dataset.from_dict({"judgment": judgments[:50], "summary": summaries[:50]})
    test_ds = Dataset.from_dict({"judgment": judgments[50:60], "summary": summaries[50:60]})
    return DatasetDict({"train": train_ds, "test": test_ds})

# Custom Collate function for PyTorch DataLoader
class CollateFn:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        input_ids = [item["input_ids"] for item in batch]
        attention_mask = [item["attention_mask"] for item in batch]
        labels = [item["labels"] for item in batch]

        # Pad sequences
        input_ids = self.tokenizer.pad({"input_ids": input_ids}, return_tensors="pt")["input_ids"]
        attention_mask = self.tokenizer.pad({"input_ids": attention_mask}, return_tensors="pt")["input_ids"]
        labels = self.tokenizer.pad({"input_ids": labels}, return_tensors="pt")["input_ids"]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

if __name__ == "__main__":
    print("CUDA available:", torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    if torch.cuda.is_available():
        print("GPU name:", torch.cuda.get_device_name(0))

    tokenizer  = BartTokenizer.from_pretrained(MODEL_NAME)
    model      = BartForConditionalGeneration.from_pretrained(MODEL_NAME).to(device)

    print("Loading LegalSum dataset...")
    try:
        dataset = load_dataset("d0r1h/LegalSum", token=os.getenv("HF_TOKEN"))
        if "test" not in dataset:
            dataset = dataset["train"].train_test_split(test_size=0.1)
        train_dataset = dataset["train"].select(range(min(150, len(dataset["train"]))))
        eval_dataset = dataset["test"].select(range(min(30, len(dataset["test"]))))
    except Exception as e:
        print(f"Failed to load LegalSum from Hugging Face: {e}")
        print("Falling back to generating local mock dataset for training...")
        dataset = get_mock_summarizer_data()
        train_dataset = dataset["train"]
        eval_dataset = dataset["test"]

    def preprocess(batch):
        inputs = tokenizer(
            batch["judgment"], max_length=1024,
            truncation=True, padding="max_length"
        )
        labels = tokenizer(
            batch["summary"], max_length=256,
            truncation=True, padding="max_length"
        )
        # Convert pad token ids to -100 to ignore loss
        inputs["labels"] = [
            [(-100 if t == tokenizer.pad_token_id else t) for t in ids]
            for ids in labels["input_ids"]
        ]
        return inputs

    print("Preprocessing datasets...")
    tokenized_train = train_dataset.map(preprocess, batched=True, remove_columns=train_dataset.column_names)
    tokenized_eval = eval_dataset.map(preprocess, batched=True, remove_columns=eval_dataset.column_names)

    # Use PyTorch DataLoader
    # We set format to torch
    tokenized_train.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    tokenized_eval.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    collate_fn = CollateFn(tokenizer)
    train_dataloader = DataLoader(tokenized_train, batch_size=1, shuffle=True, collate_fn=collate_fn)
    eval_dataloader = DataLoader(tokenized_eval, batch_size=1, collate_fn=collate_fn)

    # Setup Optimizer & Scheduler
    from torch.optim import AdamW
    optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    
    epochs = 2
    gradient_accumulation_steps = 4
    
    print("Starting custom PyTorch BART fine-tuning on GPU...")
    model.train()
    for epoch in range(epochs):
        print(f"\n--- Epoch {epoch+1}/{epochs} ---")
        epoch_loss = 0.0
        optimizer.zero_grad()
        
        for step, batch in enumerate(train_dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss / gradient_accumulation_steps
            loss.backward()
            
            epoch_loss += loss.item() * gradient_accumulation_steps
            
            if (step + 1) % gradient_accumulation_steps == 0 or (step + 1) == len(train_dataloader):
                optimizer.step()
                optimizer.zero_grad()
                
            if (step + 1) % 10 == 0:
                print(f"  Step {step+1}/{len(train_dataloader)} | Avg Loss: {epoch_loss / (step+1):.4f}")
                
        print(f"Epoch {epoch+1} Complete | Average Loss: {epoch_loss / len(train_dataloader):.4f}")

    # Evaluate ROUGE
    print("\nEvaluating model on validation subset...")
    model.eval()
    rouge = evaluate.load("rouge")
    predictions, references = [], []
    
    with torch.no_grad():
        for batch in eval_dataloader:
            input_ids = batch["input_ids"].to(device)
            ids = model.generate(input_ids, max_length=256, num_beams=4, early_stopping=True)
            
            decoded_preds = tokenizer.batch_decode(ids, skip_special_tokens=True)
            # Replace -100 in labels
            labels = batch["labels"].clone()
            labels[labels == -100] = tokenizer.pad_token_id
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
            
            predictions.extend(decoded_preds)
            references.extend(decoded_labels)

    eval_results = rouge.compute(predictions=predictions, references=references, use_stemmer=True)
    print("Evaluation Results:")
    metrics = {}
    for k, v in eval_results.items():
        val = float(v)
        metrics[k] = val
        print(f"  {k}: {val*100:.4f}")

    # Save fine-tuned model
    os.makedirs("models/legal_summarizer", exist_ok=True)
    model.save_pretrained("models/legal_summarizer")
    tokenizer.save_pretrained("models/legal_summarizer")
    print("Summarizer trained and saved successfully.")

    # Save metrics
    metrics_path = "models/legal_summarizer/metrics.json"
    import json
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved evaluation metrics to {metrics_path}")