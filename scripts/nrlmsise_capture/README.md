# NRLMSISE-00 oracle harness

Regenerates the fixtures in `tests/fixtures/nrlmsise00/` and the
coefficient module `pytcl/atmosphere/_nrlmsise00_data.py` from the
vendored reference C implementation in `csrc/nrlmsise00/` (which is
also what `setup.py` compiles into the runtime extension).

```sh
# Build the oracle driver against the vendored sources
cc -O2 -DINLINE -Icsrc/nrlmsise00 -o /tmp/nrlmsise_oracle \
   scripts/nrlmsise_capture/oracle_driver.c \
   csrc/nrlmsise00/nrlmsise-00.c csrc/nrlmsise00/nrlmsise-00_data.c -lm

# Regenerate the input grid and oracle outputs
uv run python scripts/nrlmsise_capture/generate_grid.py /tmp/nrlmsise_oracle

# Regenerate the coefficient data module
uv run python scripts/nrlmsise_capture/convert_data_tables.py
```

The grid covers every internal model boundary (72.5/85/110/120 km),
five geographic sites, three solar regimes, storm-mode (7-element Ap)
records, `gtd7d` effective-density records, and 300 seeded-random
fills; outputs are written at 17 significant digits.
