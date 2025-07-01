from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

class UniXcoderPredictor:
    def __init__(self, model_path="./unixcoder-finetuned"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

    def predict_arguments(self, preceding_code):
        """Predict arguments for a given preceding code snippet."""
        input_text = f"Predict arguments: {preceding_code}"
        inputs = self.tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
        outputs = self.model.generate(inputs["input_ids"], max_length=128)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
