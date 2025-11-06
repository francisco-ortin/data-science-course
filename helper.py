import os
from ipykernel.zmqshell import ZMQInteractiveShell
import json


def run_notebook_subdir(subdir: str, notebook_file_name: str, full_notebook_file_path: str,
                        ipython_shell: ZMQInteractiveShell) -> None:
    """
    Run a Jupyter notebook in a subdirectory.
    :param subdir: The subdirectory, relative to the current directory, where the notebook is located.
    :param notebook_file_name: The name of the notebook file.
    :param full_notebook_file_path: The full path to the source notebook file.
    :param ipython_shell: The IPython shell object used by the notebook.
    """
    # get the directory of the source notebook
    source_path = os.path.dirname(full_notebook_file_path)
    # join the current path with the subdirectory
    destination_path = os.path.join(source_path, subdir)
    # change directory to the destination path
    os.chdir(destination_path)
    # programmatically run the notebook with %run notebook_file_name
    ipython_shell.run_line_magic('run', notebook_file_name)
    os.chdir(source_path)  # go back to the original directory


def fix_notebook_metadata(source_notebook_path: str, destination_notebook_path: str) -> None:
    """
    Fix the notebook metadata to ensure it contains the necessary 'widgets' state.
    This function reads a Jupyter notebook file, checks its metadata for the 'widgets' key,
    and adds an empty 'state' dictionary if it is missing.
    """
    # Load the notebook
    with open(source_notebook_path, 'r', encoding='utf-8') as f_input:
        notebook = json.load(f_input)
    # Fix the metadata
    if 'widgets' in notebook.get('metadata', {}):
        if 'state' not in notebook['metadata']['widgets']:
            notebook['metadata']['widgets']['state'] = {}
    # Save the fixed notebook
    with open(destination_notebook_path, 'w', encoding='utf-8') as f_output:
        json.dump(notebook, f_output, indent=2)

if __name__ == "__main__":
    fix_notebook_metadata("deep-learning/cnn/hugging_face.ipynb", "deep-learning/cnn/hugging_face_fixed.ipynb")
