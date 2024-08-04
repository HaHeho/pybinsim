from pathlib import Path


def parse_soundfile_list(soundfile_list: str) -> list[Path]:
    """Parse a soundfile list separated by '#' to a list of paths"""
    if soundfile_list:
        return list(map(Path, soundfile_list.split("#")))
    else:
        return list()


def parse_boolean(value):
    if isinstance(value, str) and value.lower() == "true":
        return True
    elif isinstance(value, str) and value.lower() == "false":
        return False
    elif isinstance(value, bool):
        return value
    elif isinstance(value, int):
        return value != 0  # same as bool(input)
    else:
        return None
