from transformers import MarianMTModel, MarianTokenizer

model = MarianMTModel.from_pretrained("model/")
tokenizer = MarianTokenizer.from_pretrained("model/")


input_text = "Quiero volar como un pájaro."


inputs = tokenizer(input_text, return_tensors="pt")


outputs = model.generate(**inputs)


translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Texto traducido: {translated_text}")
