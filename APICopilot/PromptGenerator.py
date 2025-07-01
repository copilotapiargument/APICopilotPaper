class PromptGenerator:
    def __init__(self, input_ar, top_graphs, example_ars):
        self.input_ar = input_ar
        self.top_graphs = top_graphs  # From graph matching phase
        self.example_ars = example_ars  # Best examples from retrieval
        
    def _format_triples(self, triples):
        return "\n".join([f"({s}, {p}, {t})" for s, p, t in triples])
    
    def _format_examples(self):
        examples_str = ""
        for i, example in enumerate(self.example_ars[:3]):  # Top 3 examples
            examples_str += f"// Example {i+1}\n"
            examples_str += self._format_triples(example['knowledge_triples']) + "\n"
            examples_str += f"Method call: {example['ar']['mcall']}\n"
            examples_str += f"Arguments: {example['ar']['Args']}\n\n"
        return examples_str
    
    def generate_prompt(self):
        # 1. Knowledge Triples
        input_triples = self._format_triples(self.input_ar['knowledge_triples'])
        graph_triples = "\n".join([
            self._format_triples(graph['knowledge_triples']) 
            for graph in self.top_graphs
        ])
        
        # 2. Best Examples
        examples_section = self._format_examples()
        
        # 3. Query and Input Code
        query = self.input_ar['mcall']
        preceding_code = self.input_ar['P']
        
        prompt_template = f"""
// ========== Contextual Knowledge ==========
// Input Graph Triples:
{input_triples}

// Top Matching Graph Triples:
{graph_triples}

// ========== Best Examples ==========
{examples_section}
// ========== Code Context ==========
{preceding_code}

// ========== Completion Query ==========
Complete the following method call by filling missing arguments.
Only output the completed method call with arguments.

{query}

// ========== Current Arguments ==========
Existing arguments: {self.input_ar['Args']}
"""

        return prompt_template.strip()

# Example usage
if __name__ == "__main__":
    # Mock input data from previous phases
    input_ar = {
        'P': 'Image originalImage = new Image("path/to/image.jpg");\nImageTransformer transformer = new ImageTransformer();',
        'mcall': 'transformer.resize(originalImage, /* Missing Arguments */',
        'Args': [(None, 1), (None, 2)],
        'knowledge_triples': [
            ('originalImage', 'typeOf', 'Image'),
            ('originalImage', 'hasValue', '"path/to/image.jpg"'),
            ('transformer', 'typeOf', 'ImageTransformer'),
            ('transformer.resize', 'takesArgument', 'originalImage')
        ]
    }
    
    top_graphs = [
        {'knowledge_triples': [
            ('anotherImage', 'typeOf', 'Image'),
            ('anotherTransformer', 'typeOf', 'ImageTransformer'),
            ('anotherTransformer.resize', 'takesArgument', 'anotherImage'),
            ('anotherTransformer.resize', 'takesArgument', '200'),
            ('anotherTransformer.resize', 'takesArgument', '150')
        ]}
    ]
    
    example_ars = [
        {'ar': {
            'P': 'Image img = new Image("test.png");\nImageTransformer t = new ImageTransformer();',
            'mcall': 't.resize(img, 300, 200)',
            'Args': [('img', 0), ('300', 1), ('200', 2)]
        },
        'knowledge_triples': [
            ('img', 'typeOf', 'Image'),
            ('t', 'typeOf', 'ImageTransformer'),
            ('t.resize', 'takesArgument', 'img'),
            ('t.resize', 'takesArgument', '300'),
            ('t.resize', 'takesArgument', '200')
        ]
    ]
    
    generator = PromptGenerator(input_ar, top_graphs, example_ars)
    prompt = generator.generate_prompt()
    
    print("Generated Prompt:\n")
    print(prompt)
