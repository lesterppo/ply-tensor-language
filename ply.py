#!/usr/bin/env python3
"""
Ply — Tensor-Native Language v2.0
Usage: ply [--ir] [file.ply]   (or pipe source via stdin)
  --ir    Use IR compiler with kernel fusion
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from runtime import run as run_eager


def main():
    use_ir = False
    args = sys.argv[1:]

    if '--ir' in args:
        use_ir = True
        args.remove('--ir')

    if args:
        source = Path(args[0]).read_text()
    elif not sys.stdin.isatty():
        source = sys.stdin.read()
    else:
        print("Ply v2.0 — Tensor-Native Language")
        print("Usage: ply [--ir] <file.ply>")
        print("   or: echo 'program' | ply")
        print()
        print("No control flow. No variables. Pure tensor geometry.")
        print("─────────────────────────────────────────────────")
        print("REPL mode (type 'exit' to quit)")
        lines = []
        while True:
            try:
                line = input("ply> " if not lines else ".... ")
                if line.strip() == 'exit':
                    break
                if line.strip() == '':
                    if lines:
                        source = '\n'.join(lines)
                        result = run_eager(source, use_ir=use_ir)
                        if result is not None:
                            _show_result(result)
                        lines = []
                    continue
                lines.append(line)
            except (EOFError, KeyboardInterrupt):
                print()
                break
        return 0

    if use_ir:
        from ir import compile_and_run
        from parser import parse
        from runtime import Runtime
        import time
        program = parse(source)
        rt = Runtime()
        result = rt.eval(program)

        if result is not None and hasattr(result, 'data'):
            # Try IR compilation for comparison
            try:
                t0 = time.perf_counter()
                ir_result = compile_and_run(result, apply_fusion=True)
                t1 = time.perf_counter()
                print(f"[IR compiled with fusion in {(t1-t0)*1000:.1f}ms]")
                _show_result_raw(ir_result)
            except Exception as e:
                print(f"[IR compilation skipped: {e}]")
                _show_result(result)
        elif result is not None:
            _show_result(result)
    else:
        result = run_eager(source)
        if result is not None:
            _show_result(result)

    return 0


def _show_result(result):
    import numpy as np
    from tensor import Tensor
    if isinstance(result, Tensor):
        print(f"shape={result.shape} dtype={result.dtype}")
        data = result.data
        if data.size <= 20:
            np.set_printoptions(precision=4, suppress=True)
            print(data)
        else:
            flat = data.flat
            print(f"[{flat[0]:.4f} {flat[1]:.4f} ... {flat[-2]:.4f} {flat[-1]:.4f}]")
    elif isinstance(result, np.ndarray):
        _show_result_raw(result)


def _show_result_raw(arr):
    import numpy as np
    print(f"shape={arr.shape} dtype={arr.dtype}")
    if arr.size <= 20:
        np.set_printoptions(precision=4, suppress=True)
        print(arr)
    else:
        flat = arr.flat
        print(f"[{flat[0]:.4f} {flat[1]:.4f} ... {flat[-2]:.4f} {flat[-1]:.4f}]")


if __name__ == '__main__':
    sys.exit(main())
