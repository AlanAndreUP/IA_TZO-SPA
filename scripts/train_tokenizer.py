import pandas as pd
from transformers import MarianTokenizer


tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-es")

def tokenize_data(input_file, output_file):
  
    df = pd.read_csv(input_file)

   
    df['input'] = df['input'].astype(str)
    df['target'] = df['target'].astype(str)

 
    inputs = tokenizer(list(df['input']), padding='max_length', truncation=True, max_length=128, return_tensors='pt')
    targets = tokenizer(list(df['target']), padding='max_length', truncation=True, max_length=128, return_tensors='pt')

    tokenized_df = pd.DataFrame({
        'input_ids': [ids.tolist() for ids in inputs['input_ids']],
        'attention_mask': [mask.tolist() for mask in inputs['attention_mask']],
        'labels': [ids.tolist() for ids in targets['input_ids']]
    })

  
    tokenized_df.to_csv(output_file, index=False)

if __name__ == "__main__":
    input_file = 'data/tzotzil_spanish.csv'  
    output_file = 'data/tzotzil_spanish_tokenized.csv'
    tokenize_data(input_file, output_file)
