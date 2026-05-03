# PyPI Preparation Plan for zigzag-dt

## Objective
Publish the zigzag peak/valley detection package to PyPI under the name `zigzag-dt` without conflicting with existing packages (`zigzag`, `npzigzag`, etc.).

## Planned Changes

### 1. Package Directory Rename
- Rename `zigzag/` → `zigzag_dt/`
- This ensures `import zigzag_dt` is unambiguous when installed alongside other zigzag packages.

### 2. PyPI Name
- Update `pyproject.toml`: `name = "zigzag-dt"`
- Keep version at `0.4.0` for continuity (bump later if desired).

### 3. Function Aliases (zz_ prefix)
To avoid name collisions when users import from multiple zigzag-like packages:

| Original | Alias |
|----------|-------|
| `peak_valley_pivots` | `zz_pivots` |
| `pivots_to_modes` | `zz_modes` |
| `compute_segment_returns` | `zz_segment_returns` |
| `compute_performance` | `zz_performance` |
| `compute_performance_nd` | `zz_performance_nd` |
| `max_drawdown` | `zz_max_drawdown` |
| `zigzag` (line builder) | `zz_line` |

Constants (`PEAK`, `VALLEY`, `SIDEMOVE`, `EPS`) keep their original names — they are unlikely to clash.

### 4. Files to Update

| File | Changes |
|------|---------|
| `pyproject.toml` | `name`, `authors`, `urls` |
| `zigzag_dt/__init__.py` | Add `zz_*` aliases; update `__all__` |
| `zigzag_dt/core.py` | Fix internal import: `from zigzag_dt.__init__ import ...` |
| `zigzag_dt/plotting.py` | No changes needed (no package imports) |
| `test_core.py` | Update imports to `zigzag_dt` |
| `test_plain_python.py` | Update imports to `zigzag_dt` |
| `README.md` | Update install instructions and package references |
| `CHANGES.txt` | Add v0.4.0 / v0.5.0 entry for rename |

### 5. Backwards Compatibility
- Original function names remain available inside `zigzag_dt`.
- Users can `from zigzag_dt import zz_pivots` or continue using `peak_valley_pivots`.
- No breaking change for internal code if imports are updated.

### 6. Post-Publish
- Update `data-prep` function_mapping to use `zigzag_dt` imports if needed.
- Tag release on GitHub.
- Update any notebooks that `import zigzag`.

## Verification Steps
1. `python -m pytest test_core.py` passes.
2. `python -m pytest test_plain_python.py` passes.
3. `python -c "from zigzag_dt import zz_pivots, zz_line; print('OK')"` works.
4. `pip install -e .` succeeds with new package name.
