# https://en.wikipedia.org/wiki/Burrows%E2%80%93Wheeler_transform

def bwt(s: str) -> str:
    """Apply Burrows–Wheeler transform to input string."""
    assert "\002" not in s and "\003" not in s, "Input string cannot contain STX and ETX characters"
    s = "\002" + s + "\003"  # Add start and end of text marker
    table = sorted(s[i:] + s[:i] for i in range(len(s)))  # Table of rotations of string
    last_column = [row[-1:] for row in table]  # Last characters of each row
    return "".join(last_column)  # Convert list of characters into string

def ibwt(r: str) -> str:
    """Apply inverse Burrows–Wheeler transform."""
    table = [""] * len(r)  # Make empty table
    for _ in range(len(r)):
        table = sorted(r[i] + table[i] for i in range(len(r)))  # Add a column of r

    s = next((row for row in table if row.endswith("\003")), "") # Iterate over and check whether last character ends with ETX or not
    return s.rstrip("\003").strip("\002")  # Retrive data from array and get rid of start and end markers
