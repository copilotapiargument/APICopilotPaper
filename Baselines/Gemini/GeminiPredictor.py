import google.generativeai as genai

class GeminiPredictor:
    def __init__(self, api_key):
        self.api_key = api_key
        genai.configure(api_key=self.api_key)

    def predict_arguments(self, preceding_code, few_shot_examples):
        """Predict arguments using Gemini Flash 2.0."""
        prompt = self._build_prompt(preceding_code, few_shot_examples)
        model = genai.GenerativeModel("gemini-flash-2.0")
        response = model.generate_content(prompt)
        return response.text

    def _build_prompt(self, preceding_code, few_shot_examples):
        """Build a prompt with few-shot examples."""
        prompt = "Predict the missing arguments for the following API method call:\n\n"
        for example in few_shot_examples:
            prompt += f"Input: {example['input']}\nOutput: {example['output']}\n\n"
        prompt += f"Input: {preceding_code}\nOutput:"
        return prompt
