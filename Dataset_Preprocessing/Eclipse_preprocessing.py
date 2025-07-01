import os
import re

def find_java_files(root_dir):
    """Recursively finds all .java files within the given root directory."""
    java_files = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".java"):
                java_files.append(os.path.join(root, file))
    return java_files

def preprocess_java_code(file_path):
    """Reads a Java file and performs basic preprocessing.

    Args:
        file_path (str): The path to the Java file.

    Returns:
        str: The preprocessed Java code, or None if an error occurred.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()

        # Remove single-line comments
        code = re.sub(r'//.*', '', code)
        # Remove multi-line comments
        code = re.sub(r'/\*[\s\S]*?\*/', '', code)
        # Remove import statements (optional, depending on your task)
        # code = re.sub(r'^import\s+.*?;', '', code, flags=re.MULTILINE)
        # Remove package declaration (optional)
        # code = re.sub(r'^package\s+.*?;', '', code, flags=re.MULTILINE)
        # Normalize whitespace (remove leading/trailing and reduce multiple spaces to single)
        code = ' '.join(code.split())
        # Add a newline at the end (for consistency)
        code += '\n'

        return code
    except Exception as e:
        print(f"Error processing file: {file_path} - {e}")
        return None

def extract_code_completion_pairs(preprocessed_code):
    """(Placeholder) Extracts context-target pairs for code completion.

    This is a simplified example. The actual logic will depend heavily on
    your specific code completion task and how you define context and target.

    Args:
        preprocessed_code (str): The preprocessed Java code.

    Returns:
        list: A list of tuples, where each tuple is (context, target).
    """
    completion_pairs = []
    # Example: Split the code into lines and consider the last token of a line
    # as context and the next token (if any) as the target.
    tokens = preprocessed_code.split()
    for i in range(len(tokens) - 1):
        context = " ".join(tokens[:i+1])
        target = tokens[i+1]
        completion_pairs.append((context, target))
    return completion_pairs

if __name__ == "__main__":
    eclipse_project_path = "/path/to/your/eclipse/project"  # Replace with the actual path
    output_file = "eclipse_preprocessed_data.txt"

    java_files = find_java_files(eclipse_project_path)
    all_completion_pairs = []

    for java_file in java_files:
        print(f"Processing: {java_file}")
        preprocessed_code = preprocess_java_code(java_file)
        if preprocessed_code:
            # For a basic task, you might just want to save the cleaned code
            # with open(output_file, 'a', encoding='utf-8') as outfile:
            #     outfile.write(preprocessed_code)

            # For a code completion task, you'll likely need to extract context-target pairs
            completion_pairs = extract_code_completion_pairs(preprocessed_code)
            all_completion_pairs.extend(completion_pairs)

    # Save the extracted completion pairs
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for context, target in all_completion_pairs:
            outfile.write(f"context: {context}\ttarget: {target}\n")

    print(f"\nPreprocessing complete. Results saved to {output_file}")
    print(f"Found {len(java_files)} Java files.")
    print(f"Extracted {len(all_completion_pairs)} potential completion pairs (example-based).")

    print("\n--- Important Considerations for a Real Code Completion Task ---")
    print("1. **More Sophisticated Parsing:** For accurate context and target extraction, especially for code completion, consider using a dedicated Java parsing library like JavaParser (as mentioned in the README recommendations). Libraries like this can provide you with the Abstract Syntax Tree (AST) of the code, which allows for much more precise identification of code elements and completion points.")
    print("2. **Defining Completion Points:** You need a clear definition of where you want to predict the next code element. This could be after a '.', after a type declaration, within a method body, etc.")
    print("3. **Tokenization:** Instead of simple whitespace splitting, you might want to use a proper tokenizer that understands the syntax of Java (e.g., separating operators, identifiers, keywords).")
    print("4. **Handling Context Length:** For training models, you'll need to consider the maximum context length your model can handle. You might need to truncate or window the code context.")
    print("5. **Filtering and Cleaning:** Further filtering of the extracted pairs might be necessary based on the specific requirements of your code completion task (e.g., removing very short contexts, specific types of completions).")
