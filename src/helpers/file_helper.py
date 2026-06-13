import os


def create_directory(path):

    result = False
    if os.path.exists(path):
        return True

    try:
        # create file directory
        os.mkdir(path)
        result = True
    except OSError as error:
        print(error)

    return result

def move_file(source_path, target_path):
    result = False

    try:
        # move file from source_path to target_path
        os.replace(source_path, target_path)
        result = True
    except OSError as error:
        print(error)

    return result

def get_files(path):

    files = None
    try:
        files = os.listdir(path)
        files.sort()
    except OSError as error:
        print(error)

    return files

def exists(path):
    result = False
    if os.path.exists(path):
        result = True

    return result

def delete(path):
    if exists(path):
        os.remove(path)
