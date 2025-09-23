import re
from typing import Any, Dict, List, Optional, Union


def divider(title: str = "", char: str = "-", width: int = 79, show: bool = True) -> str:
    """
    Creates a horizontal divider line with an optional centered title.

    Args:
        title (str): Optional title to center in the divider.
        char (str): Character to use for the divider line.
        width (int): Total width of the divider.
        show (bool): If True, prints the divider directly. If False, returns the string.

    Returns:
        str: The formatted divider line (if show=False).
    """
    clean_title = title.strip()
    line = f" {clean_title} " if clean_title else ""
    side_len = (width - len(line)) // 2
    full_line = f"{char * side_len}{line}{char * (width - side_len - len(line))}"

    if show:
        print(full_line)
    else:
        return full_line


def plain_divider(char: str = "-", width: int = 79, show: bool = True) -> str:
    """
    Prints a plain horizontal divider.

    Args:
        char (str): Character to repeat.
        width (int): Width of the divider.
        show (bool): Whether to print or return the string.

    Returns:
        str: The divider string (if show=False).
    """
    line = char * width
    if show:
        print(line)
    else:
        return line


def print_dict(d: dict, title: str = "", indent: int = 0) -> None:
    """
    Prints a dictionary in a formatted way.

    Args:
        d (dict): The dictionary to print.
        title (str): Optional title for the dictionary.
        indent (int): Number of spaces to indent the output.
    """
    if title:
        print(f"{' ' * indent}{title}:")
    for key, value in d.items():
        if isinstance(value, dict):
            print(f"{' ' * (indent + 2)}{key}:")
            print_dict(value, indent=indent + 4)
        else:
            print(f"{' ' * (indent + 2)}{key}: {value}")


def is_matched_key(pattern, key, reject_pattern=None):
    is_matched = re.search(rf"(^|_)({re.escape(pattern)})(_|$)", key)
    is_rejected = False
    if reject_pattern:
        is_rejected = re.search(rf"(^|_)({re.escape(reject_pattern)})(_|$)", key)
    return is_matched and not is_rejected


def find_all_keys(
    patterns: Union[str, List[str]],
    keys: Union[List[str], Dict[str, Any]],
    reject: Optional[List[str]] = None,
    strict: bool = True,
    only_one_match: bool = False,
) -> List[str]:
    if isinstance(patterns, str):
        patterns = [patterns]
    if reject is None:
        reject = []

    matched = []
    for key in keys:
        for pat in patterns:
            if strict:
                # match with underscore boundaries or start/end of string
                if re.search(rf"(^|_)({re.escape(pat)})(_|$)", key):
                    matched.append(key)
                    break
            else:
                if pat in key:
                    matched.append(key)
                    break

    for k in matched:
        for r in reject:
            if re.search(rf"(^|_)({re.escape(r)})(_|$)", k):
                matched.remove(k)
                break
            
    if only_one_match:
        if len(matched) == 0:
            return None
        if len(matched) > 1:
            raise ValueError(f"Multiple matches found: {matched}. Expected only one match.")
        return matched[0]
    return matched


if __name__ == "__main__":
    # Example usage
    key = "eef_pos"
    pattern = "pos"
    reject_pattern = "delta"
    print(is_matched_key(pattern, key, reject_pattern))  # True