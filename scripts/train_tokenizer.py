import pandas as pd
from transformers import MarianTokenizer

# Cargar el tokenizador
tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-es")

def tokenize_data(input_file, output_file):
    # Cargar el dataset
    df = pd.read_csv(input_file)

    # Asegúrate de que las columnas 'input' y 'target' sean de tipo cadena
    df['input'] = df['input'].astype(str)
    df['target'] = df['target'].astype(str)

    # Tokenizar los datos
    inputs = tokenizer(list(df['input']), padding='max_length', truncation=True, max_length=128, return_tensors='pt')
    targets = tokenizer(list(df['target']), padding='max_length', truncation=True, max_length=128, return_tensors='pt')

    # Convertir a DataFrame
    tokenized_df = pd.DataFrame({
        'input_ids': [ids.tolist() for ids in inputs['input_ids']],
        'attention_mask': [mask.tolist() for mask in inputs['attention_mask']],
        'labels': [ids.tolist() for ids in targets['input_ids']]
    })

    # Guardar el DataFrame tokenizado
    tokenized_df.to_csv(output_file, index=False)

if __name__ == "__main__":
    input_file = 'data/tzotzil_spanish.csv'  # Ajusta el nombre del archivo según sea necesario
    output_file = 'data/tzotzil_spanish_tokenized.csv'
    tokenize_data(input_file, output_file)
