from pathlib import Path


def increment_filename(path):
    p_path = Path(path)
    suffix = p_path.suffix
    parent = p_path.parent
    stem = p_path.stem
    i = 1
    while p_path.is_file():
        p_path = Path(parent / f'{stem}_{i}{suffix}')
        i += 1
    return p_path
