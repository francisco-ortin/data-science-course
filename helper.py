import os
from ipykernel.zmqshell import ZMQInteractiveShell


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
