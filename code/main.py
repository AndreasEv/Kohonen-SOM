"""
EPL442 Assignment 3: Kohonen Self-Organizing Map (SOM)
Author: <Name Surname>, Student ID: <AM>

Entry point. Looks for parameters.txt in the current working folder and
runs one SOM training with the settings found there.
"""

import os
import sys

PARAMETERS_FILE = "parameters.txt"


def find_parameters_file(folder="."):
    """Return the path of parameters.txt in `folder`, or None if it is missing."""
    path = os.path.join(folder, PARAMETERS_FILE)
    return path if os.path.isfile(path) else None


def main():
    """Run the program: locate parameters.txt, then train (later tasks)."""
    path = find_parameters_file()
    if path is None:
        print(f"Error: {PARAMETERS_FILE} not found in {os.getcwd()}", file=sys.stderr)
        return 1
    print(f"Using {path}. Training is not implemented yet.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
