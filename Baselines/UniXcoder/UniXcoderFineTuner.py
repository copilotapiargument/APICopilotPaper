from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import Dataset

class UniXcoderFineTuner:
    def __init__(self, model_name="microsoft/unixcoder-base"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def preprocess_data(self, preceding_code_list, arguments_list):
        """Prepare and tokenize the dataset."""
        data = {"input": preceding_code_list, "output": arguments_list}
        dataset = Dataset.from_dict(data)
        return dataset.train_test_split(test_size=0.2)

    def fine_tune(self, train_dataset, eval_dataset, output_dir="./unixcoder-finetuned"):
        """Fine-tune the UniXcoder model."""
        training_args = Seq2SeqTrainingArguments(
            output_dir=output_dir,
            evaluation_strategy="epoch",
            learning_rate=5e-5,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=3,
            weight_decay=0.01,
            save_total_limit=2,
            predict_with_generate=True,
            fp16=True,
        )

        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
        )

        trainer.train()
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
