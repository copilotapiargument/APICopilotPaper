import networkx as nx

class KnowledgeGraphBuilder:
    def __init__(self):
        """
        Initialize empty knowledge graphs
        """
        self.kg_examples = nx.MultiDiGraph()
        self.g_input = nx.MultiDiGraph()

    def _add_triples_to_graph(self, graph: nx.MultiDiGraph, triples: list) -> None:
        """
        Add triples to a graph with validation
        """
        for triple in triples:
            if len(triple) != 3:
                continue  # Skip invalid triples
            h, r, t = triple
            graph.add_edge(h, t, label=r)
            
            # Add nodes explicitly in case they don't have edges
            graph.add_node(h)
            graph.add_node(t)

    def build_kg_examples(self, example_ars: list) -> None:
        """
        Construct KG_examples from multiple example ARs
        """
        for ar in example_ars:
            if 'knowledge_triples' in ar:
                self._add_triples_to_graph(self.kg_examples, ar['knowledge_triples'])

    def build_g_input(self, input_ar: dict) -> None:
        """
        Construct G_input from input AR
        """
        if 'knowledge_triples' in input_ar:
            self._add_triples_to_graph(self.g_input, input_ar['knowledge_triples'])

    def get_kg_examples(self) -> nx.MultiDiGraph:
        """
        Return the constructed KG_examples
        """
        return self.kg_examples

    def get_g_input(self) -> nx.MultiDiGraph:
        """
        Return the constructed G_input
        """
        return self.g_input

    @staticmethod
    def visualize_graph(graph: nx.MultiDiGraph) -> None:
        """
        Visualize the knowledge graph (requires matplotlib)
        """
        import matplotlib.pyplot as plt
        
        pos = nx.spring_layout(graph)
        edge_labels = {(u, v): d['label'] for u, v, d in graph.edges(data=True)}
        
        plt.figure(figsize=(12, 8))
        nx.draw_networkx_nodes(graph, pos, node_size=2000, node_color='skyblue')
        nx.draw_networkx_edges(graph, pos, arrowstyle='-|>', arrowsize=20)
        nx.draw_networkx_labels(graph, pos)
        nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels)
        plt.axis('off')
        plt.show()

# Example usage
if __name__ == "__main__":
    # Example knowledge triples from previous phases
    example_ars = [
        {'knowledge_triples': [
            ('anotherImage', 'typeOf', 'Image'),
            ('anotherImage', 'hasValue', '"path/to/another.png"'),
            ('anotherTransformer', 'typeOf', 'ImageTransformer'),
            ('resizedAnotherImage', 'assignedFrom', 'anotherTransformer.resize'),
            ('anotherTransformer.resize', 'takesArgument', 'anotherImage')
        ]},
        {'knowledge_triples': [
            ('myImage', 'typeOf', 'Image'),
            ('myImage', 'hasValue', '"path/to/my.jpeg"'),
            ('myTransformer', 'typeOf', 'ImageTransformer'),
            ('myTransformer.resize', 'takesArgument', '300'),
            ('myTransformer.resize', 'takesArgument', '200')
        ]}
    ]

    input_ar = {
        'knowledge_triples': [
            ('originalImage', 'typeOf', 'Image'),
            ('originalImage', 'hasValue', '"path/to/image.jpg"'),
            ('transformer', 'typeOf', 'ImageTransformer'),
            ('transformer.resize', 'takesArgument', 'originalImage'),
            ('transformer.resize', 'takesArgument', 'null')
        ]
    }

    # Build knowledge graphs
    kgb = KnowledgeGraphBuilder()
    kgb.build_kg_examples(example_ars)
    kgb.build_g_input(input_ar)

    # Access the constructed graphs
    kg_examples = kgb.get_kg_examples()
    g_input = kgb.get_g_input()

    print("KG_examples nodes:", kg_examples.nodes())
    print("KG_examples edges:", list(kg_examples.edges(data=True)))
    print("\nG_input nodes:", g_input.nodes())
    print("G_input edges:", list(g_input.edges(data=True)))

    # Visualize the graphs (optional)
    # kgb.visualize_graph(kg_examples)
    # kgb.visualize_graph(g_input)
