from transformers import MarianMTModel, MarianTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
from transformers import DataCollatorForSeq2Seq


tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-es")

model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-es")

dataset = load_dataset('json', data_files={'train': 'data/tzotzil_spanish_dataset.json'})

print(dataset['train'][0])

def tokenize_and_format(examples):
    
    inputs = examples['input'] if isinstance(examples['input'], list) else [examples['input']]
    targets = examples['target'] if isinstance(examples['target'], list) else [examples['target']]
    
    tokenized_inputs = tokenizer(inputs, truncation=True, padding='max_length', max_length=128, return_tensors=None)
    tokenized_labels = tokenizer(targets, truncation=True, padding='max_length', max_length=128, return_tensors=None)
    
    return {
        'input_ids': tokenized_inputs['input_ids'],
        'labels': tokenized_labels['input_ids']
    }

dataset = dataset.map(tokenize_and_format, batched=True)

training_args = TrainingArguments(
    output_dir="./results",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=10_000,
    save_total_limit=2,
    prediction_loss_only=True,
)

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset['train']
)

trainer.train()

trainer.save_model("model/")
tokenizer.save_pretrained("model/")
