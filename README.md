# Kohonen SOM (EPL442 Assignment 3)

Self-Organizing Map written from scratch in Python + NumPy (matplotlib only for plots).
It clusters the letter data from Assignment 2.

## Layout

```
code/
  parameters.txt   all run settings (read from the current folder)
  main.py          entry point
  tests/           unit tests (standard-library unittest)
```

## Run

```
cd code
python3 main.py
```

## Tests

```
cd code
python3 -m unittest discover -s tests -v
```
