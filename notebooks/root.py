import os

def set_project_rood_dir():
    notebook_dir = os.path.dirname(os.path.abspath("__file__"))
    paths = notebook_dir.split("/")

    # remove directory unless notebook/s directory is found
    while len(paths) > 0:
        if paths[-1] == 'notebook' or paths[-1] == 'notebooks':
            paths.pop()
            break
        paths.pop()

    # show error if paths is empty
    if len(paths) == 0:
        print("Current directory: ", notebook_dir)
        raise ValueError("Unable to find notebook/s directory in path")

    root = "/".join(paths)
    os.chdir(root)
    print("Successfully changed working directory: ", root)
    print("Current working directory: ", os.getcwd())


set_project_rood_dir()
