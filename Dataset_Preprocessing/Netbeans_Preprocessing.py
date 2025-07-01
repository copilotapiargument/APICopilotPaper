import os
import re

def preprocess_netbeans_code(root_dir, output_file, remove_comments=True, sequence_length=128):
    """
    Preprocesses Java code files from a Netbeans project for code completion.

    Args:
        root_dir (str): The root directory of the Netbeans project.
        output_file (str): The path to the output file where preprocessed code will be written.
        remove_comments (bool): Whether to remove comments from the code.
        sequence_length (int): The desired length of code sequences for training.
    """
    all_code_sequences = []

    def remove_java_comments(code):
        # Remove single-line comments
        code = re.sub(r'//.*', '', code)
        # Remove multi-line comments
        code = re.sub(r'/\*[\s\S]*?\*/', '', code)
        return code

    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(".java"):
                filepath = os.path.join(dirpath, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        code = f.read()

                    if remove_comments:
                        code = remove_java_comments(code)

                    # Basic tokenization (you might need a more sophisticated tokenizer)
                    tokens = code.split()
                    if not tokens:
                        continue

                    # Create sequences for code completion (next token prediction)
                    for i in range(0, len(tokens) - sequence_length):
                        sequence = " ".join(tokens[i:i + sequence_length])
                        next_token = tokens[i + sequence_length]
                        all_code_sequences.append((sequence, next_token))

                except Exception as e:
                    print(f"Error reading or processing file: {filepath} - {e}")

    # Write the preprocessed sequences to the output file
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for sequence, next_token in all_code_sequences:
            outfile.write(f"{sequence}\t{next_token}\n") # Tab-separated for easy parsing

    print(f"Preprocessing complete. {len(all_code_sequences)} sequences written to {output_file}")

if __name__ == "__main__":
    netbeans_project_root = "/path/to/your/NetbeansProject"  # Replace with the actual path
    output_file_path = "preprocessed_netbeans_code.txt"
    preprocess_netbeans_code(netbeans_project_root, output_file_path, remove_comments=True, sequence_length=128)
    print(f"Preprocessed data saved to: {output_file_path}")
