# Test Helpers

This directory holds scripts that (re)generate the
reference test data in `../data`.

**Important:** 
- The files in `tests/data` are checked-in fixtures used
  by the test suite.
- You **must not** run these helpers during normal test runs.

---

## Directory Layout

- `tests/data/`: Static npz files loaded as reference in the test suite.
- `tests/helpers/`: Python scripts that create or refresh the files
  in `tests/data/`.

--

## Running the Test Suite

Simply run:
```bash
pytest
```

## Updating Fixtures
When you have made a deliberate change to the core algorithm
and need new reference outputs:
1. (Optionally) Update or tweak the helper scripts in `tests/helpers/`.
2. Run the helper scripts manually to regenerate the data files.
   ```bash
   python tests/helpers/<script_name>.py
   ```
   This will overwrite files in `tests/data/`.
3. Verify the new data under `tests/data/`.
4. Commit both the updated helper script(s) and the new data files
   with a clear message, e.g.

