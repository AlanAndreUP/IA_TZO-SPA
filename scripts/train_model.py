from transformers import MarianMTModel, MarianTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
from transformers import DataCollatorForSeq2Seq

# Cargar el tokenizador para MarianMT
tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-es")

# Cargar el modelo preentrenado
model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-es")

# Cargar el dataset tokenizado en formato JSON
dataset = load_dataset('json', data_files={'train': 'data/tzotzil_spanish_dataset.json'})

# Verificar una muestra del dataset para entender su estructura
print(dataset['train'][0])

# Ajustar las columnas del dataset y tokenizar
def tokenize_and_format(examples):
    # Asegúrate de que 'input' y 'target' sean listas
    inputs = examples['input'] if isinstance(examples['input'], list) else [examples['input']]
    targets = examples['target'] if isinstance(examples['target'], list) else [examples['target']]
    
    # Tokenizar inputs y labels
    tokenized_inputs = tokenizer(inputs, truncation=True, padding='max_length', max_length=128, return_tensors=None)
    tokenized_labels = tokenizer(targets, truncation=True, padding='max_length', max_length=128, return_tensors=None)
    
    # Asegurarse de que 'input_ids' y 'labels' estén en formato de lista
    return {
        'input_ids': tokenized_inputs['input_ids'],
        'labels': tokenized_labels['input_ids']
    }

dataset = dataset.map(tokenize_and_format, batched=True)

# Configuración del entrenamiento
training_args = TrainingArguments(
    output_dir="./results",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=10_000,
    save_total_limit=2,
    prediction_loss_only=True,
)

# Data collator para seq2seq
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model
)

# Crear el Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset['train']
)

# Entrenar el modelo
trainer.train()

# Guardar el modelo y el tokenizador
trainer.save_model("model/")
tokenizer.save_pretrained("model/")
