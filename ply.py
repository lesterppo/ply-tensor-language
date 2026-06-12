#!/usr/bin/env python3
"""
Ply — Tensor-Native Language
Usage: ply [file.ply]   (or pipe source via stdin)
"""
import sys
from pathlib import Path

# Add ply dir to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from runtime import run


def main():
    if len(sys.argv) > 1:
        source = Path(sys.argv[1]).read_text()
    elif not sys.stdin.isatty():
        source = sys.stdin.read()
    else:
        print("Ply — Tensor-Native Language")
        print("Usage: ply <file.ply>")
        print("   or: echo 'program' | ply")
        print()
        print("No control flow. No variables. Pure tensor geometry.")
        print("─────────────────────────────────────────────────")
        # Simple REPL
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
                        result = run(source)
                        if result is not None:
                            _show_result(result)
                        lines = []
                    continue
                lines.append(line)
            except (EOFError, KeyboardInterrupt):
                print()
                break
        return 0

    result = run(source)
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
        print(f"shape={result.shape} dtype={result.dtype}")
        if result.size <= 20:
            np.set_printoptions(precision=4, suppress=True)
            print(result)
        else:
            flat = result.flat
            print(f"[{flat[0]:.4f} {flat[1]:.4f} ... {flat[-2]:.4f} {flat[-1]:.4f}]")


if __name__ == '__main__':
    sys.exit(main())
