import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
from scipy.spatial.distance import cosine

class ExampleRetriever:
    def __init__(self, training_ars, model_name="codellama/CodeLlama-7b-hf"):
        """
        Initialize with training ARs and load CodeLlama model
        """
        self.training_ars = training_ars
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Precompute training embeddings
        self.training_embeddings = self._precompute_embeddings()

    def _get_code_context(self, ar):
        """Combine preceding code and method call for embedding"""
        return f"{ar['P']}\n{ar['mcall']}"

    def _embed_text(self, text):
        """Generate embedding for text using CodeLlama"""
        inputs = self.tokenizer(text, return_tensors="pt", 
                              truncation=True, max_length=2048).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).cpu().numpy()[0]

    def _precompute_embeddings(self):
        """Precompute embeddings for all training ARs"""
        return [self._embed_text(self._get_code_context(ar)) for ar in self.training_ars]

    def calculate_similarity(self, input_ar, top_k=3):
        """
        Calculate semantic similarity between input AR and training ARs
        Returns top-k similar ARs with scores
        """
        input_embedding = self._embed_text(self._get_code_context(input_ar))
        
        # Calculate cosine similarities
        similarities = []
        for train_emb in self.training_embeddings:
            sim = 1 - cosine(input_embedding, train_emb)
            similarities.append(sim)
            
        # Get indices of top-k similarities
        sorted_indices = np.argsort(similarities)[::-1][:top_k]
        
        return [(self.training_ars[i], similarities[i]) for i in sorted_indices]

    def retrieve_examples(self, input_ar, top_k=3):
        """
        Retrieve top-k similar AR examples with their knowledge triples
        """
        similar_ars = self.calculate_similarity(input_ar, top_k)
        
        results = []
        for ar, score in similar_ars:
            # Extract knowledge triples from AR (implementation depends on KG construction)
            triples = self._extract_knowledge_triples(ar)
            results.append({
                'ar': ar,
                'similarity_score': score,
                'knowledge_triples': triples
            })
            
        return results

    def _extract_knowledge_triples(self, ar):
        """
        Extract knowledge triples from AR (simplified example implementation)
        This would be replaced with actual KG extraction logic
        """
        # Placeholder implementation showing triple extraction pattern
        triples = []
        
        # Extract variable-type relationships
        for line in ar['P'].split('\n'):
            if '=' in line and 'new' in line:
                var = line.split('=')[0].strip()
                cls = line.split('new ')[1].split('(')[0].strip()
                triples.append(f"({var}, typeOf, {cls})")
                
        # Extract method call relationships
        if 'mcall' in ar:
            method_parts = ar['mcall'].split('(')[0].split('.')
            if len(method_parts) > 1:
                obj = method_parts[0]
                method = method_parts[1]
                triples.append(f"({obj}, hasMethod, {method})")
                
        return triples

# Example usage
if __name__ == "__main__":
    # Example training ARs (would come from ARExtractor)
    training_ars = [
        {'P': 'Image a = new Image("test.jpg");\nImageTransformer t = new ImageTransformer();', 
         'mcall': 't.resize(a, 100, 200)',
         'Args': [...]},
        # ... more training examples
    ]
    
    # Initialize retriever with training data
    retriever = ExampleRetriever(training_ars)
    
    # Example input AR (from Listing 3)
    input_ar = {
        'P': 'Image originalImage = new Image("path/to/image.jpg");\nImageTransformer transformer = new ImageTransformer();',
        'mcall': 'transformer.resize(originalImage, /* Missing Arguments */',
        'Args': [...]
    }
    
    # Retrieve top-2 examples
    examples = retriever.retrieve_examples(input_ar, top_k=2)
    
    # Display results
    for example in examples:
        print(f"Similarity Score: {example['similarity_score']:.4f}")
        print("Knowledge Triples:")
        for triple in example['knowledge_triples']:
            print(f"  {triple}")
        print("\nMethod Call:", example['ar']['mcall'])
        print("="*50 + "\n")
