# CLAUDE.md - Coding Guidelines

## Commands
- Run tests: `pytest test_file/test_*.py`
- Run single test: `pytest test_file/test_main.py::test_function_name -v`
- Profile code: `python -m cProfile -o output.prof script.py`
- View profile: `python -m pstats output.prof`

## Code Style
- Indentation: 4 spaces, no tabs
- Classes: CamelCase (WaveManager)
- Functions/variables: snake_case (process_data, pivot_arr)
- Constants: UPPERCASE_WITH_UNDERSCORES
- Line length: ≤ 120 characters
- Imports order: stdlib, third-party, local (separate groups with newline)

## Patterns
- Data handling: pandas for DataFrame operations, numpy for numerical computations
- State management: transitions library for state machine implementation
- Documentation: Docstrings for public functions and classes
- Error handling: Use try/except for external operations, conditional checks for control flow
- Type hints: Add return type annotations and parameter types for key functions
- Testing: pytest with assertions, use CSV fixtures for data-driven tests