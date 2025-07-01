class APICopilot:
    def __init__(self, dataset_type, dataset_path, openai_api_key):
        """
        Initialize APICopilot with dataset and OpenAI API key.

        Args:
            dataset_type (str): Type of dataset ('eclipse', 'netbeans', 'py150').
            dataset_path (str): Path to the dataset.
            openai_api_key (str): OpenAI API key for LLM-based predictions.
        """
        self.dataset_type = dataset_type
        self.dataset_path = dataset_path
        self.openai_api_key = openai_api_key

        # Initialize preprocessing module based on dataset type
        if self.dataset_type == "eclipse":
            self.preprocessor = EclipsePreprocessing(dataset_path)
        elif self.dataset_type == "netbeans":
            self.preprocessor = NetBeansPreprocessing(dataset_path)
        elif self.dataset_type == "py150":
            self.preprocessor = PY150Preprocessing(dataset_path)
        else:
            raise ValueError("Unsupported dataset type. Use 'eclipse', 'netbeans', or 'py150'.")

        # Initialize other components
        self.ar_extractor = ARExtractor()
        self.example_retriever = ExampleRetriever()
        self.knowledge_triple_extractor = KnowledgeTripleExtractor()
        self.knowledge_graph_builder = KnowledgeGraphBuilder()
        self.graph_matcher = GraphMatcher()
        self.prompt_generator = PromptGenerator()
        self.argument_recommender = ArgumentRecommender(openai_api_key)

    def preprocess_dataset(self):
        """Preprocess the dataset using the appropriate preprocessor."""
        print("Preprocessing dataset...")
        self.preprocessed_data = self.preprocessor.preprocess()
        print(f"Preprocessing complete. Found {len(self.preprocessed_data)} files.")

    def extract_argument_requests(self):
        """Extract Argument Requests (ARs) from preprocessed data."""
        print("Extracting Argument Requests...")
        self.ar_tuples = self.ar_extractor.extract_ar(self.preprocessed_data)
        print(f"Extracted {len(self.ar_tuples)} AR tuples.")

    def retrieve_examples(self):
        """Retrieve similar examples for each AR."""
        print("Retrieving similar examples...")
        self.example_ars = []
        for ar in self.ar_tuples:
            examples = self.example_retriever.retrieve_examples(ar)
            self.example_ars.append(examples)
        print(f"Retrieved examples for {len(self.example_ars)} ARs.")

    def extract_knowledge_triples(self):
        """Extract knowledge triples from ARs and examples."""
        print("Extracting knowledge triples...")
        self.knowledge_triples = []
        for ar, examples in zip(self.ar_tuples, self.example_ars):
            ar_triples = self.knowledge_triple_extractor.extract_triples(ar)
            example_triples = [self.knowledge_triple_extractor.extract_triples(ex) for ex in examples]
            self.knowledge_triples.append((ar_triples, example_triples))
        print("Knowledge triples extracted.")

    def build_knowledge_graphs(self):
        """Build knowledge graphs from knowledge triples."""
        print("Building knowledge graphs...")
        self.knowledge_graphs = []
        for ar_triples, example_triples in self.knowledge_triples:
            kg_input = self.knowledge_graph_builder.build_g_input(ar_triples)
            kg_examples = self.knowledge_graph_builder.build_kg_examples(example_triples)
            self.knowledge_graphs.append((kg_input, kg_examples))
        print("Knowledge graphs constructed.")

    def perform_graph_matching(self):
        """Perform graph matching to find similar subgraphs."""
        print("Performing graph matching...")
        self.matched_subgraphs = []
        for kg_input, kg_examples in self.knowledge_graphs:
            matched = self.graph_matcher.find_isomorphic_subgraphs(kg_input, kg_examples)
            self.matched_subgraphs.append(matched)
        print(f"Found {sum(len(m) for m in self.matched_subgraphs)} matched subgraphs.")

    def generate_prompts(self):
        """Generate prompts for LLM-based argument completion."""
        print("Generating prompts...")
        self.prompts = []
        for ar, matched_subgraphs in zip(self.ar_tuples, self.matched_subgraphs):
            prompt = self.prompt_generator.generate_prompt(ar, matched_subgraphs)
            self.prompts.append(prompt)
        print(f"Generated {len(self.prompts)} prompts.")

    def recommend_arguments(self):
        """Recommend arguments using LLM-based prediction."""
        print("Recommending arguments...")
        self.recommended_arguments = []
        for prompt in self.prompts:
            args = self.argument_recommender.recommend_arguments(prompt)
            self.recommended_arguments.append(args)
        print(f"Recommended arguments for {len(self.recommended_arguments)} ARs.")

    def run_pipeline(self):
        """Run the full APICopilot pipeline."""
        print("Starting APICopilot pipeline...")
        self.preprocess_dataset()
        self.extract_argument_requests()
        self.retrieve_examples()
        self.extract_knowledge_triples()
        self.build_knowledge_graphs()
        self.perform_graph_matching()
        self.generate_prompts()
        self.recommend_arguments()
        print("APICopilot pipeline completed.")

        # Display results
        for ar, args in zip(self.ar_tuples, self.recommended_arguments):
            print(f"\nMethod Call: {ar['mcall']}")
            print(f"Recommended Arguments: {args}")

# Example usage
if __name__ == "__main__":
    # Initialize APICopilot with Eclipse dataset
    api_copilot = APICopilot(
        dataset_type="eclipse",
        dataset_path="/path/to/eclipse/project",
        openai_api_key="your-openai-api-key"
    )
    
    # Run the full pipeline
    api_copilot.run_pipeline()
