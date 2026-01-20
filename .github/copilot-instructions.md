# Copilot Instructions for GIST Intern PSL

## Project Overview
This is a Power Systems Lab (PSL) project for analyzing electrical power grids using matrix-based calculations. The codebase implements power flow analysis by reading bus and transmission line data, then performing matrix multiplication operations to compute network behavior.

## Architecture

### Core Components
- **main.py**: Entry point that orchestrates file reading and data processing
- **function.py**: Utility functions for:
  - `read_bus_file()` / `read_line_file()`: Parse CSV data with header validation
  - `input_matrix()`: User input with NumPy array reshaping and error handling
  - `matrix_judgment()`: Validates matrix dimensions for multiplication (A_cols == B_rows)
  - `matrix_multipulication()`: Performs NumPy matrix operations

### Data Flow (see Diagram/2_file_reading.puml)
1. `main.py` defines configuration dict with file path + headers
2. Functions read whitespace-separated CSV files using `pd.read_csv(sep='\s+')`
3. Header validation: raise error if column count ≠ header count
4. Returns pandas DataFrames for bus/line data

### Data Format (Power Systems Standard)
- **Bus Data** (Bus.txt): 8 columns with power generation/load in per-unit
  - Col 1: Bus ID | Col 2: Bus type (1=Slack, 2=PV, 3=PQ)
  - Cols 3-4: Active/Reactive power generation | Cols 5-6: Load
  - Cols 7-8: Voltage magnitude and angle (degrees)
  
- **Line Data** (Line.txt): 5 columns for transmission lines
  - Cols 1-2: Connected bus IDs | Cols 3-5: Resistance, Reactance, Susceptance (per-unit)

## Developer Workflows

### Running the Project
```bash
python main.py
```
Currently reads and prints bus/line data. Matrix multiplication code is commented out (legacy matrix calculator code remains for reference).

### Testing Data Processing
- Bus/Line CSV files use **whitespace delimiters** (not comma)
- Headers are injected programmatically, not from file
- Row/column mismatch triggers ValueError → caught and printed, returns None/empty

### Key Functions and Error Handling
- All file I/O wrapped in `try/except` blocks
- `input_matrix()` loops until valid float list + row count provided
- Division check: `if mat_a.shape[1] == mat_b.shape[0]` required before multiplication

## Project Conventions

### Code Style
- CamelCase for DataFrames: `BUS_VAL`, `LINE_MATRIX`
- Docstring comments in Korean and English (mixed)
- Unused matrix multiplication code commented but preserved for reference

### Data Handling
- All CSV imports use `sep='\s+'` for flexible whitespace parsing
- Headers defined in `main.py` config dicts (not file-based)
- Pandas DataFrames are primary working structure

### Naming Patterns
- `INF_DICT_*`: Configuration dictionaries (path + headers metadata)
- `*_VAL` or `*_MATRIX`: Data containers after reading
- `input_matrix()`: Loops and validates user input
- `*_judgment()`: Boolean validators for operation feasibility

## Integration Points

### External Dependencies
- `numpy`: Matrix operations and array reshaping
- `pandas`: CSV reading and DataFrame manipulation
- Windows-style path separators in file configs (e.g., `"DATA\Bus.txt"`)

### Cross-file Communication
- `main.py` imports all from `function.py` via `from function import *`
- Functions are stateless utilities (no global state)
- Data flows through function returns only

## Common Tasks

### Adding New Data Sources
1. Create `INF_DICT_*` in `main.py` with path + headers
2. Call `read_*_file(path, headers)` with proper header count
3. Validate DataFrame shape matches header length

### Debugging Data Mismatches
- Check CSV delimiter (should be whitespace, not comma)
- Verify header list length equals actual CSV columns
- Print DataFrame shape: `df.shape` returns (rows, cols)

### Extending Matrix Operations
- Matrix validation already exists in `matrix_judgment()` - reuse it
- New operations should follow same pattern: validate → calculate → return
- Use `np.matmul()` for multiplication (not `*` operator on arrays)
