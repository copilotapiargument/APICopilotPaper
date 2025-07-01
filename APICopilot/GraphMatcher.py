import networkx as nx
from networkx.algorithms import isomorphism
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch

class GraphMatcher:
    def __init__(self, kg_examples, g_input):
        self.kg_examples = kg_examples
        self.g_input = g_input
        self.tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-7b-hf")
        self.model = AutoModel.from_pretrained("codellama/CodeLlama-7b-hf")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
    def _get_embedding(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).cpu().numpy()[0]

    def _node_matcher(self, node1, node2):
        emb1 = self._get_embedding(node1)
        emb2 = self._get_embedding(node2)
        return 1 - cosine(emb1, emb2) > 0.8  # Semantic similarity threshold

    def _edge_matcher(self, edge1, edge2):
        e1_data = self.kg_examples.get_edge_data(*edge1)
        e2_data = self.g_input.get_edge_data(*edge2)
        return e1_data.get('label', '') == e2_data.get('label', '')

    def find_isomorphic_subgraphs(self):
        matcher = isomorphism.GraphMatcher(
            self.kg_examples,
            self.g_input,
            node_match=self._node_matcher,
            edge_match=self._edge_matcher
        )
        return list(matcher.subgraph_isomorphisms_iter())

    def calculate_nerp(self, subgraph_mapping):
        node_similarities = []
        edge_similarities = []
        
        # Calculate node similarities
        for input_node, example_node in subgraph_mapping.items():
            node_emb_input = self._get_embedding(input_node)
            node_emb_example = self._get_embedding(example_node)
            node_similarities.append(1 - cosine(node_emb_input, node_emb_example))
        
        # Calculate edge similarities
        for input_edge in self.g_input.edges(data=True):
            example_edge = (subgraph_mapping[input_edge[0]], 
                          subgraph_mapping[input_edge[1]], 
                          input_edge[2])
            
            # Create edge representation strings
            input_edge_str = f"{input_edge[0]}-{input_edge[2]['label']}-{input_edge[1]}"
            example_edge_str = f"{example_edge[0]}-{example_edge[2]['label']}-{example_edge[1]}"
            
            edge_emb_input = self._get_embedding(input_edge_str)
            edge_emb_example = self._get_embedding(example_edge_str)
            edge_similarities.append(1 - cosine(edge_emb_input, edge_emb_example))
        
        return sum(node_similarities) + sum(edge_similarities)

    def get_top_k_subgraphs(self, top_k=3):
        isomorphic_mappings = self.find_isomorphic_subgraphs()
        scored_subgraphs = []
        
        for mapping in isomorphic_mappings:
            nerp_score = self.calculate_nerp(mapping)
            scored_subgraphs.append((mapping, nerp_score))
            
        # Sort by NERP score and return top-k
        scored_subgraphs.sort(key=lambda x: x[1], reverse=True)
        return scored_subgraphs[:top_k]

# Example usage
if __name__ == "__main__":
    # Example graphs from previous construction
    kg_examples = nx.MultiDiGraph()
    kg_examples.add_edges_from([
        ("anotherImage", "Image", {"label": "typeOf"}),
        ("anotherImage", '"path/to/another.png"', {"label": "hasValue"}),
        ("anotherTransformer", "ImageTransformer", {"label": "typeOf"}),
        ("resizedAnotherImage", "anotherTransformer.resize", {"label": "assignedFrom"}),
        ("anotherTransformer.resize", "anotherImage", {"label": "takesArgument"})
    ])

    g_input = nx.MultiDiGraph()
    g_input.add_edges_from([
        ("originalImage", "Image", {"label": "typeOf"}),
        ("originalImage", '"path/to/image.jpg"', {"label": "hasValue"}),
        ("transformer", "ImageTransformer", {"label": "typeOf"}),
        ("transformer.resize", "originalImage", {"label": "takesArgument"})
    ])

    # Initialize matcher
    matcher = GraphMatcher(kg_examples, g_input)
    
    # Get top 2 subgraphs by NERP
    top_subgraphs = matcher.get_top_k_subgraphs(top_k=2)
    
    print("Top matching subgraphs:")
    for idx, (mapping, score) in enumerate(top_subgraphs):
        print(f"\nSubgraph {idx+1} (NERP: {score:.2f}):")
        for input_node, example_node in mapping.items():
            print(f"  {input_node} -> {example_node}")
