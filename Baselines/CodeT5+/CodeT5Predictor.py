from transformers import T5ForConditionalGeneration, T5Tokenizer

class CodeT5Predictor:
    def __init__(self, model_path="./codet5p-finetuned"):
        self.tokenizer = T5Tokenizer.from_pretrained(model_path)
        self.model = T5ForConditionalGeneration.from_pretrained(model_path)

    def predict_arguments(self, preceding_code):
        """Predict arguments for a given preceding code snippet."""
        input_text = f"Predict arguments: {preceding_code}"
        inputs = self.tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
        outputs = self.model.generate(inputs["input_ids"], max_length=128)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
