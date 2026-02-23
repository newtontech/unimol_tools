# [BUG] concurrency/race condition in unimol_tools/data/conformer.py causes prediction to hang

## Describe the bug
When `multi_process=True`, the `predict` method in `unimol_tools/data/conformer.py` can hang intermittently. The hang occurs when calling `MolPredict.predict()`; the process stops at `pool.imap()` with no error output. Memory usage looks normal while the CPU is idle, and the process must be terminated manually.

Root cause analysis:
1. **Missing pool.join() call**: In the two `transform()` methods in `conformer.py` (around lines 191 and 466), the code calls `pool.close()` but never calls `pool.join()`. This can allow the main process to continue before worker processes have finished, causing a race condition.
2. **Insufficient exception handling**: An `except:` at line 279 catches all exceptions (including `KeyboardInterrupt`) and uses `print` instead of a logger to record the error.
3. **No context manager usage**: The process pool should be used with a context manager (`with Pool() as pool:`) to ensure proper cleanup.
4. **No timeout mechanism**: `pool.imap()` is used without any timeout handling; if a worker blocks, the main process cannot recover.

## unimol_tools Version

0.1.5

## Expected behavior

When `multi_process=True`, multiprocessing should operate reliably and not hang. Specifically:
1. All worker processes should finish before the main process continues.
2. Exceptions should be logged and handled appropriately.
3. Processes should be terminable by signals such as `KeyboardInterrupt`.
4. Long-running or blocking tasks should be handled with timeouts.

## To Reproduce

Steps to reproduce:
1. Set `multi_process=True`.
2. Call `MolPredict.predict()` again and again.
3. Observe the process state.

Minimal reproduction snippet:

```python
from unimol_tools import MolPredict
from pathlib import Path
from typing import List

def UniMolPredict(model_dir: Path, csv_path: Path) -> List[float]:
    logger.info(f"   Start predicting: {csv_path}")
    clf = MolPredict(load_model=model_dir)
    logger.info("   Prediction model: clf is a MolPredict object")
    y_pred = clf.predict(str(csv_path))  # Hangs here
    logger.info(f"   Prediction result: {y_pred}")
    return y_pred
```

Observed behavior:
1. The process hangs at `pool.imap()`.
2. No error messages are printed.
3. Memory usage remains normal while CPU is idle.
4. The process must be terminated manually.
5. The issue occurs randomly and is not tied to a specific molecule or dataset.

## Environment

- OS: macOS Darwin 25.1.0
- Python: 3.13.2
- Dependencies:
  - PyTorch >= 2.4.0
  - RDKit >= 2024.3.4
  - NumPy < 2.3.0, >= 2.0.0
  - Pandas >= 2.2.2
  - scikit-learn >= 1.5.0
- Hardware:
  - CPU: available
  - Memory: ~1.5 GB free at the time of the issue

## Additional context

Characteristics of the issue:
1. Not related to MCP (Model Context Protocol).
2. Not caused by out-of-memory (there was ~1.5 GB free).
3. Prediction runs on CPU and CPU usage is low when the hang occurs.
4. The behavior is intermittent; sometimes the prediction finishes successfully.

Successful log sample:

```
2025-12-02 18:25:13 | unimol_tools/data/datahub.py | 187 | INFO | Uni-Mol Tools | conf_cache_level is 1, saving conformers to /unimol_weight_perovskite/fixed_hyper_param/20251202_182513_smiles.sdf
```

Log when the process hangs:

```
0it [00:00, ?it/s]
2it [00:00, 35.32it/s]
2025-12-02 18:36:00 | unimol_tools/data/conformer.py | 197 | INFO | Uni-Mol Tools | Succeeded in generating conformers for 100.00% of molecules.
2025-12-02 18:36:00 | unimol_tools/data/conformer.py | 214 | INFO | Uni-Mol Tools | Succeeded in generating 3d conformers for 100.00% of molecules.
# The process hangs here with no further output
```

Relevant code locations:

- `/Users/yhm/Desktop/code/unimol_tools/unimol_tools/data/conformer.py`
  - Line ~191: `pool.close()` (missing `pool.join()`)
  - Line ~279: `except:` (exception handling issue)
  - Line ~466: same issue in another `transform()` method

Temporary workaround:

Setting `multi_process=False` avoids the hang, but sacrifices the performance benefits of multiprocessing.

Recommended fixes:
1. Add `pool.join()` after `pool.close()` to ensure worker processes finish before the main process continues.
2. Improve exception handling to `except Exception as e:` and log errors instead of using `print`, so `KeyboardInterrupt` is not swallowed.
3. Use a context manager: `with Pool() as pool:` to ensure the pool is cleaned up properly.
4. Add a timeout mechanism or other protections around `pool.imap()` so the main process can recover if a worker blocks.