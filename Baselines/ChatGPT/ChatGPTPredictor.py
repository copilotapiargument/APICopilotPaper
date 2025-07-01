import openai
class ChatGPTPredictor:
    def __init__(self, api_key):
        self.api_key = api_key
        openai.api_key = self.api_key

    def predict_arguments(self, preceding_code, few_shot_examples):
        """Predict arguments using ChatGPT-4o."""
        prompt = self._build_prompt(preceding_code, few_shot_examples)
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=128,
        )
        return response.choices[0].message['content']

    def _build_prompt(self, preceding_code, few_shot_examples):
        """Build a prompt with few-shot examples."""
        prompt = "Predict the missing arguments for the following API method call:\n\n"
        for example in few_shot_examples:
            prompt += f"Input: {example['input']}\nOutput: {example['output']}\n\n"
        prompt += f"Input: {preceding_code}\nOutput:"
        return prompt
