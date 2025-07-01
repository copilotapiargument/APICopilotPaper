from preprocessing.EclipsePreprocessing import EclipsePreprocessing
from extraction.ARExtractor import ARExtractor
from codet5.CodeT5FineTuner import CodeT5FineTuner
from codet5.CodeT5Predictor import CodeT5Predictor

class APICopilot:
    def __init__(self, dataset_type, dataset_path):
        self.dataset_type = dataset_type
        self.dataset_path = dataset_path
        self.preprocessor = self._initialize_preprocessor()
        self.ar_extractor = ARExtractor()
        self.codet5_finetuner = CodeT5FineTuner()
        self.codet5_predictor = None

    def _initialize_preprocessor(self):
        """Initialize the appropriate preprocessor based on dataset type."""
        if self.dataset_type == "eclipse":
            return EclipsePreprocessing(self.dataset_path)
        elif self.dataset_type == "netbeans":
            return NetBeansPreprocessing(self.dataset_path)
        elif self.dataset_type == "py150":
            return PY150Preprocessing(self.dataset_path)
        else:
            raise ValueError("Unsupported dataset type.")

    def run_pipeline(self):
        """Run the full APICopilot pipeline."""
        print("Preprocessing dataset...")
        preprocessed_data = self.preprocessor.preprocess()

        print("Extracting Argument Requests...")
        ar_tuples = self.ar_extractor.extract_ar(preprocessed_data)

        print("Preparing data for CodeT5+ fine-tuning...")
        preceding_code_list = [ar["P"] for ar in ar_tuples]
        arguments_list = [", ".join([arg[0] for arg in ar["Args"] if arg[0]]) for ar in ar_tuples]
        dataset = self.codet5_finetuner.preprocess_data(preceding_code_list, arguments_list)

        print("Fine-tuning CodeT5+...")
        self.codet5_finetuner.fine_tune(dataset["train"], dataset["test"])

        print("Testing the fine-tuned model...")
        self.codet5_predictor = CodeT5Predictor()
        for ar in ar_tuples[:5]:  # Test on the first 5 ARs
            predicted_args = self.codet5_predictor.predict_arguments(ar["P"])
            print(f"Input: {ar['P']}")
            print(f"Predicted Arguments: {predicted_args}\n")

if __name__ == "__main__":
    api_copilot = APICopilot(dataset_type="eclipse", dataset_path="/path/to/eclipse/project")
    api_copilot.run_pipeline()
