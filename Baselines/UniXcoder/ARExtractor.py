import ast
import re

class ARExtractor:
    @staticmethod
    def split_arguments(s):
        args = []
        current = []
        level = 0
        for c in s.strip():
            if c == '(':
                level += 1
            elif c == ')':
                level -= 1
            if c == ',' and level == 0:
                args.append(''.join(current).strip())
                current = []
            else:
                current.append(c)
        if current:
            args.append(''.join(current).strip())
        return args

    @staticmethod
    def is_placeholder(arg):
        arg = arg.strip()
        return (arg.startswith('/*') and arg.endswith('*/')) or arg.startswith('//') or arg == ''

    @classmethod
    def extract_java_ar(cls, java_code):
        ar_list = []
        pattern = re.compile(r'(\w+\.\w+)\s*\(')
        pos = 0
        while pos < len(java_code):
            match = pattern.search(java_code, pos)
            if not match:
                break
            method_part = match.group(0)
            start_method = match.start()
            pos_call = start_method + len(method_part) - 1
            paren_count = 1
            end_pos = pos_call
            while end_pos < len(java_code) and paren_count > 0:
                end_pos += 1
                char = java_code[end_pos] if end_pos < len(java_code) else ''
                if char == '(':
                    paren_count += 1
                elif char == ')':
                    paren_count -= 1
            if paren_count != 0:
                pos = end_pos + 1
                continue
            arg_list_str = java_code[pos_call + 1:end_pos]
            args = cls.split_arguments(arg_list_str)
            args_with_pos = []
            for arg_pos, arg in enumerate(args):
                if cls.is_placeholder(arg):
                    args_with_pos.append((None, arg_pos))
                else:
                    args_with_pos.append((arg.strip(), arg_pos))
            P = java_code[:start_method]
            mcall = java_code[start_method:end_pos + 1]
            ar_list.append({'P': P, 'mcall': mcall, 'Args': args_with_pos})
            pos = end_pos + 1
        return ar_list

    @classmethod
    def extract_python_ar(cls, python_code):
        try:
            tree = ast.parse(python_code)
        except SyntaxError:
            return []
        ar_list = []
        lines = python_code.split('\n')
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                lineno = getattr(node, 'lineno', 1) - 1
                col_offset = getattr(node, 'col_offset', 0)
                line_start = sum(len(line) + 1 for line in lines[:lineno]) if lineno > 0 else 0
                start_index = line_start + col_offset
                P = python_code[:start_index]
                try:
                    mcall = ast.unparse(node)
                except AttributeError:
                    continue
                args = []
                for arg_pos, arg_node in enumerate(node.args):
                    try:
                        arg_code = ast.unparse(arg_node)
                    except AttributeError:
                        arg_code = ""
                    args.append((arg_code, arg_pos))
                ar_list.append({'P': P, 'mcall': mcall, 'Args': args})
        return ar_list

    @classmethod
    def extract_ar(cls, code, language):
        if language == 'java':
            return cls.extract_java_ar(code)
        elif language == 'python':
            return cls.extract_python_ar(code)
        else:
            raise ValueError(f"Unsupported language: {language}")

def main():
    java_example = """
    public class ImageEditor {
        public static void main(String[] args) {
            BufferedImage specialImage = ImageIO.read(new File("path/to/special.gif"));
            ImageEditor editor = new ImageEditor();
            BufferedImage croppedImage = editor.crop(specialImage, 100, 100, 200, 200);
        }
    }
    """
    ar_java = ARExtractor.extract_ar(java_example, 'java')
    print("Java AR Example:")
    for ar in ar_java:
        print(f"P: {ar['P'][-50:]}...")  # Show last 50 chars of P for brevity
        print(f"mcall: {ar['mcall']}")
        print(f"Args: {ar['Args']}\n")

    python_example = """
    def example():
        a = some_function(arg1, arg2)
        b = another_function()
    """
    ar_python = ARExtractor.extract_ar(python_example, 'python')
    print("\nPython AR Example:")
    for ar in ar_python:
        print(f"P: {ar['P'][-50:]}...")
        print(f"mcall: {ar['mcall']}")
        print(f"Args: {ar['Args']}\n")

if __name__ == "__main__":
    main()
