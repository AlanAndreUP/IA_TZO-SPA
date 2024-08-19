from transformers import MarianMTModel, MarianTokenizer

# Cargar el modelo y el tokenizador guardados
model = MarianMTModel.from_pretrained("model/")
tokenizer = MarianTokenizer.from_pretrained("model/")

# Ejemplo de entrada
input_text = "Quiero volar como un pájaro."

# Tokenizar entrada
inputs = tokenizer(input_text, return_tensors="pt")

# Generar salida
outputs = model.generate(**inputs)

# Decodificar salida
translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Texto traducido: {translated_text}")
