import re
import openai
from openai import OpenAI

class ArgumentRecommender:
    def __init__(self, api_key: str, expected_types: list):
        """
        Initialize the recommender with OpenAI API key and expected argument types
        
        Args:
            api_key: OpenAI API key
            expected_types: List of expected types for each argument position
                             e.g. [str, int] for (String, int) parameters
        """
        self.client = OpenAI(api_key=api_key)
        self.expected_types = expected_types
        self.type_checks = {
            str: self._is_string,
            int: self._is_integer,
            float: self._is_float,
            bool: self._is_boolean
        }
        self.defaults = {
            str: '""',
            int: '0',
            float: '0.0',
            bool: 'false'
        }

    def _is_string(self, arg: str) -> bool:
        """Check if argument is a valid string literal"""
        return re.match(r'^["\'].*["\']$', arg) is not None

    def _is_integer(self, arg: str) -> bool:
        """Check if argument is a valid integer literal"""
        return re.match(r'^[+-]?\d+$', arg) is not None

    def _is_float(self, arg: str) -> bool:
        """Check if argument is a valid float literal"""
        return re.match(r'^[+-]?\d+\.?\d*([eE][+-]?\d+)?$', arg) is not None

    def _is_boolean(self, arg: str) -> bool:
        """Check if argument is a valid boolean literal"""
        return arg.lower() in ['true', 'false']

    def _validate_argument(self, arg: str, expected_type: type) -> bool:
        """Validate if argument matches expected type"""
        checker = self.type_checks.get(expected_type)
        return checker(arg) if checker else True

    def _post_process(self, generated_args: list) -> list:
        """Validate and fix generated arguments based on expected types"""
        processed_args = []
        
        for expected_type, arg in zip(self.expected_types, generated_args):
            # Check if argument matches expected type
            if self._validate_argument(arg, expected_type):
                processed_args.append(arg)
            else:
                # Replace with type-appropriate default
                default = self.defaults.get(expected_type, 'null')
                processed_args.append(default)
        
        # Handle missing/extra arguments
        while len(processed_args) < len(self.expected_types):
            default = self.defaults.get(self.expected_types[len(processed_args)], 'null')
            processed_args.append(default)
            
        return processed_args[:len(self.expected_types)]

    def _parse_arguments(self, llm_output: str) -> list:
        """Extract arguments from LLM output"""
        # Look for method call pattern
        match = re.search(r'\b\w+\((.*?)\);?$', llm_output)
        if not match:
            return []
            
        args_str = match.group(1)
        # Split arguments while ignoring commas inside parentheses
        args = []
        current = []
        paren_level = 0
        for char in args_str:
            if char == '(':
                paren_level += 1
            elif char == ')':
                paren_level -= 1
            if char == ',' and paren_level == 0:
                args.append(''.join(current).strip())
                current = []
            else:
                current.append(char)
        if current:
            args.append(''.join(current).strip())
            
        return args

    def recommend_arguments(self, prompt: str) -> list:
        """
        Generate and validate arguments using LLM
        
        Returns:
            List of processed arguments with type validation
        """
        # Get LLM completion
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "user",
                "content": prompt
            }],
            temperature=0.2,
            max_tokens=256
        )
        
        # Parse and validate arguments
        llm_output = response.choices[0].message.content
        raw_args = self._parse_arguments(llm_output)
        processed_args = self._post_process(raw_args)
        
        return processed_args

# Example usage
if __name__ == "__main__":
    # Example configuration
    API_KEY = "your-openai-api-key"
    METHOD_SIGNATURE = [str, int]  # updateSettings(String, int)
    
    # Example prompt from previous phase
    EXAMPLE_PROMPT = """
// ========== Contextual Knowledge ==========
// Input Graph Triples:
(originalImage, typeOf, Image)
(originalImage, hasValue, "path/to/image.jpg")
(transformer, typeOf, ImageTransformer)
(transformer.resize, takesArgument, originalImage)

// ========== Best Examples ==========
// Example 1
(t.resize, takesArgument, img)
Method call: t.resize(img, 300, 200)
Arguments: [('img', 0), ('300', 1), ('200', 2)]

// ========== Code Context ==========
Image originalImage = new Image("path/to/image.jpg");
ImageTransformer transformer = new ImageTransformer();

// ========== Completion Query ==========
Complete the following method call by filling missing arguments.
Only output the completed method call with arguments.

transformer.resize(originalImage, /* Missing Arguments */

// ========== Current Arguments ==========
Existing arguments: [(None, 1), (None, 2)]
"""

    # Initialize recommender
    recommender = ArgumentRecommender(API_KEY, METHOD_SIGNATURE)
    
    # Get recommendations
    recommended_args = recommender.recommend_arguments(EXAMPLE_PROMPT)
    
    print("Recommended arguments:", recommended_args)
    # Example output for invalid LLM suggestion ["admin", "high"]:
    # ["admin", "0"]
