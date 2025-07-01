import openai
import re
from typing import List, Tuple

class KnowledgeTripleExtractor:
    def __init__(self, api_key: str, model: str = "gpt-4o"):
        """
        Initialize the extractor with OpenAI API credentials
        """
        openai.api_key = api_key
        self.model = model
        self.triple_pattern = re.compile(r'\(([^,]+),\s*([^,]+),\s*([^)]+)\)')

    def _format_prompt(self, ar: dict) -> str:
        """
        Create the prompt for GPT-4o using the AR context
        """
        code_snippet = f"{ar['P']}\n{ar['mcall']}"
        return f"""As an expert in code understanding, analyze the following code snippet:
{code_snippet}

Extract knowledge triples (subject, predicate, object) that describe the API call relationships within this code. Focus on capturing:
- API calls and their arguments
- Variable types and declarations
- Method invocations and relationships
- Argument positions and data flow

Output each extracted triple on a new line, formatted as: (Subject, Predicate, Object)."""

    def _parse_response(self, response: str) -> List[Tuple[str, str, str]]:
        """
        Parse the model response into structured triples
        """
        triples = []
        for line in response.split('\n'):
            line = line.strip()
            match = self.triple_pattern.match(line)
            if match:
                triples.append(tuple(match.groups()))
        return triples

    def extract_triples(self, ar: dict) -> List[Tuple[str, str, str]]:
        """
        Extract knowledge triples from an Argument Request (AR)
        Returns list of (subject, predicate, object) tuples
        """
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[{
                    "role": "user",
                    "content": self._format_prompt(ar)
                }],
                temperature=0.1,
                max_tokens=1000
            )
            return self._parse_response(response.choices[0].message['content'])
        except Exception as e:
            print(f"Error extracting triples: {e}")
            return []

    @staticmethod
    def format_triples(triples: List[Tuple[str, str, str]]) -> str:
        """
        Format triples for display/output
        """
        return '\n'.join([f"({s}, {p}, {t})" for s, p, t in triples])

# Example usage
if __name__ == "__main__":
    # Initialize with your OpenAI API key
    extractor = KnowledgeTripleExtractor(api_key="your-api-key-here")

    # Example input AR from Listing 3
    input_ar = {
        'P': """Image originalImage = new Image("path/to/image.jpg");
ImageTransformer transformer = new ImageTransformer();""",
        'mcall': "transformer.resize(originalImage, /* Missing Arguments */",
        'Args': [(None, 1), (None, 2)]
    }

    # Extract knowledge triples
    triples = extractor.extract_triples(input_ar)
    
    print("Extracted Knowledge Triples:")
    print(KnowledgeTripleExtractor.format_triples(triples))

    # Example output would look like:
    # (originalImage, typeOf, Image)
    # (originalImage, hasValue, "path/to/image.jpg")
    # (transformer, typeOf, ImageTransformer)
    # (transformer.resize, takesArgument, originalImage)
    # (transformer.resize, takesArgument, null)
    # (transformer.resize, takesArgument, null)
