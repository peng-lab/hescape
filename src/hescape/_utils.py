import os


def find_root():
    dir_path = os.path.abspath(os.curdir)
    while True:
        if os.path.exists(os.path.join(dir_path, ".git")):
            return dir_path
        else:
            dir_path = os.path.dirname(dir_path)
            if dir_path == os.path.abspath(os.path.dirname(os.curdir)):
                raise Exception("Not a git repository")
