import requests

class LlamaPredictor:
    def __init__(self, api_key, api_url="https://api.llama.ai/v1/chat"):
        self.api_key = api_key
        self.api_url = api_url

    def predict_arguments(self, preceding_code, few_shot_examples):
        """Predict arguments using Llama 3 70B."""
        prompt = self._build_prompt(preceding_code, few_shot_examples)
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        data = {
            "model": "llama-3-70b",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
            "max_tokens": 128,
        }
        response = requests.post(self.api_url, headers=headers, json=data)
        return response.json()["choices"][0]["message"]["content"]

    def _build_prompt(self, preceding_code, few_shot_examples):
        """Build a prompt with few-shot examples."""
        prompt = "Predict the missing arguments for the following API method call:\n\n"
        for example in few_shot_examples:
            prompt += f"Input: {example['input']}\nOutput: {example['output']}\n\n"
        prompt += f"Input: {preceding_code}\nOutput:"
        return prompt
