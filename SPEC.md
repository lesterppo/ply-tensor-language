# Ply Language Specification v1.0

## A Tensor-Native Programming Language

Ply is a tensor-native language where **everything is a tensor**. There are no
variables, no control flow keywords (if/else/for/while), and no sequential
execution. Programs are static computation graphs that describe relationships
between tensors through shape-algebraic operations.

### Design Philosophy

| Principle | Implementation |
|-----------|---------------|
| No control flow | Branching via tensor masking (`? :`), iteration via broadcasting |
| No variables | Bindings (`:=`) are immutable — assigned once, never mutated |
| No side effects | Every expression is pure — same input always yields same output |
| Shape-aware | Shapes are first-class citizens, inferred at parse time |
| Hardware-mappable | The computation graph maps directly to GPU tensor cores |

## 1. Lexical Structure

### Comments
```
-- Single-line comments start with double dash
```

### Tokens
| Token | Pattern | Example |
|-------|---------|---------|
| ID | `[a-zA-Z_][a-zA-Z0-9_]*` | `A`, `randn`, `softmax` |
| INT | `[0-9]+` | `42`, `128` |
| FLOAT | `[0-9]+\.[0-9]*([eE][+-]?[0-9]+)?` | `3.14`, `1e-5` |
| STRING | `'...'` or `"..."` | `'ij,jk->ik'` |

### Operators
| Symbol | Meaning | Precedence |
|--------|---------|------------|
| `? :` | Ternary (mask-based select) | 1 (lowest) |
| `\|` | Element-wise logical OR | 2 |
| `&` | Element-wise logical AND | 3 |
| `== != < > <= >=` | Comparison | 4 |
| `+ -` | Element-wise add/subtract | 5 |
| `* / @` | Mul, div, matrix multiply | 6 |
| `- !` | Unary negate, logical not | 7 |
| `[]` | Slicing | 8 |
| `.` | Attribute/method | 8 |

## 2. Syntax

### Program Structure
```
program    := (binding | blank_line)* [bare_expression]

binding    := ID ':=' expression
bare_expr  := expression           -- allowed only as the final line
```

### Expressions
```
expression := ternary
ternary    := logical_or ('?' expression ':' expression)?
logical_or := logical_and ('|' logical_and)*
logical_and:= comparison ('&' comparison)*
comparison := addsub (('>' | '<' | '>=' | '<=' | '==' | '!=') addsub)?
addsub     := muldiv (('+' | '-') muldiv)*
muldiv     := unary (('*' | '/' | '@') unary)*
unary      := ('-' | '!') unary | postfix
postfix    := primary ('[' slice_spec ']' | '.' ID)*
primary    := ID | INT | FLOAT | STRING | funcall | list_lit | '(' expression ')'
funcall    := ID '(' arglist ')'
list_lit   := '[' arglist ']'
slice_spec := [start] ':' [end] [':' step]  |  index
arglist    := expression (',' expression)*
```

## 3. Semantics

### Binding (`:=`)
A binding associates a name with a tensor expression. Names are immutable —
once bound, they cannot be reassigned. The expression is evaluated immediately
in program order and the result is stored.

```
A := randn(3, 4)        -- binds tensor to A
B := A @ A.T            -- uses A, binds result to B
```

### Bare Expression
The final line of a program may be a bare expression (no `:=`). It is evaluated
and returned as the program's result.

```
A := randn(3, 4)
sum(A)                  -- returns the sum of all elements
```

### Operators (element-wise by default)
All binary operators (`+`, `-`, `*`, `/`, comparisons) are **element-wise**
with NumPy-style broadcasting rules. The `@` operator performs matrix
multiplication (generalized to batched matmul for 3D+ tensors).

```
A + B     -- element-wise addition (broadcasting)
A @ B     -- matrix multiply (2D) or batched matmul (3D+)
A > 0     -- element-wise comparison, produces boolean tensor
```

### Ternary (`? :`)
The only conditional in Ply. Equivalent to `where(cond, true_val, false_val)`.
All three sub-expressions are always evaluated — there is no short-circuit.

```
mask := A > 0
B := mask ? A : -1.0    -- element-wise select
```

### Logical Operators (`&`, `|`, `!`)
Element-wise logical operations on boolean tensors. No short-circuit.

```
valid := (A > 0) & (A < 100)
```

### Slicing (`[]`)
NumPy-style slicing with start:end:step.

```
A[0:2, 0:4]       -- slice first 2 rows, first 4 columns
A[:, 0]            -- first column (all rows)
A[0:10:2]          -- every other element in first 10
A[5]               -- single index (row 5)
```

### Attributes (`.`)
Method-style access for common operations.

```
A.T                -- transpose
```

## 4. Built-in Functions

### Tensor Creation
| Function | Description |
|----------|-------------|
| `randn(d0, d1, ...)` | Random normal (μ=0, σ=1) |
| `zeros(d0, d1, ...)` | All zeros |
| `ones(d0, d1, ...)` | All ones |
| `full(value, d0, d1, ...)` | Constant value |
| `eye(n)` | Identity matrix |
| `arange(start, [stop], [step])` | Linear range |

### Shape Manipulation
| Function | Description |
|----------|-------------|
| `reshape(tensor, d0, d1, ...)` | Reshape tensor |
| `permute(tensor, axis0, axis1, ...)` | Generalized transpose |
| `transpose(tensor)` | 2D transpose (alias: `.T`) |
| `concat(t1, t2, ..., dim)` | Concatenate along axis |
| `stack(t1, t2, ..., dim)` | Stack along new axis |
| `broadcast(tensor, d0, d1, ...)` | Broadcast to shape |

### Reductions
All reductions take an optional `dim` argument.

| Function | Description |
|----------|-------------|
| `sum(tensor, [dim])` | Sum of elements |
| `mean(tensor, [dim])` | Mean |
| `max(tensor, [dim])` | Maximum |
| `min(tensor, [dim])` | Minimum |
| `std(tensor, [dim])` | Standard deviation |

### Activation Functions
| Function | Description |
|----------|-------------|
| `relu(tensor)` | ReLU: max(0, x) |
| `gelu(tensor)` | GELU: Gaussian Error Linear Unit |
| `sigmoid(tensor)` | Sigmoid: 1/(1+e⁻ˣ) |
| `tanh(tensor)` | Hyperbolic tangent |
| `softmax(tensor, dim=-1)` | Softmax along dimension |

### Element-wise Math
| Function | Description |
|----------|-------------|
| `exp(tensor)` | eˣ |
| `log(tensor)` | Natural log (clamped > 0) |
| `sqrt(tensor)` | Square root (clamped ≥ 0) |
| `abs(tensor)` | Absolute value |
| `sin(tensor)` | Sine |
| `cos(tensor)` | Cosine |
| `clip(tensor, lo, hi)` | Clip to range |

### Tensor Operations
| Function | Description |
|----------|-------------|
| `where(cond, a, b)` | Element-wise select |
| `matmul(a, b)` | Matrix multiply (alias: `@`) |
| `einsum(spec, t1, t2, ...)` | Einstein summation |
| `layernorm(tensor)` | Layer normalization (last axis) |

### Utilities
| Function | Description |
|----------|-------------|
| `shape(tensor)` | Returns shape as 1D float tensor |
| `print(tensor, ...)` | Print tensor info to stdout |

## 5. Examples

### Scaled Dot-Product Attention
The core of every Transformer model — expressed in 22 lines:
```
Q := X @ W_Q
K := X @ W_K
V := X @ W_V
scores := Q @ K.T / sqrt(d_k)
attn := softmax(scores, -1)
output := attn @ V
```

### 2-Layer MLP with GELU
The feedforward block from GPT architectures:
```
h := X @ W1 + b1
a := gelu(h)
out := a @ W2 + b2
final := layernorm(out) + X
```

### Generating a 128×128 Tensor Core Dispatch Matrix
See `examples/hmtl.ply` — 128×128 FP8 weight matrix synthesized from four
orthogonal manifold axes (topology, task flow, resource saturation, dynamic
swapping) with Hadamard filter masking.

## 6. Runtime Characteristics

| Property | Value |
|----------|-------|
| Backend | NumPy |
| Precision | float32 (default) |
| Execution | Immediate (not lazy) |
| Memory | Everything in RAM |
| Max tensor size | Limited by system memory |

## 7. What Ply DOES NOT Have

- ❌ Variables (only immutable bindings)
- ❌ if/else statements (use ternary `? :`)
- ❌ for/while loops (use broadcasting + reduction)
- ❌ Functions or procedures (use bindings for sub-expressions)
- ❌ Classes or objects
- ❌ Mutation or side effects
- ❌ I/O (except `print` for debugging)
- ❌ Exceptions or error handling

## 8. Grammar (Formal)

```
program      → (binding NEWLINE)* [expression NEWLINE*] EOF
binding      → ID ':=' expression
expression   → ternary
ternary      → logical_or ('?' expression ':' expression)?
logical_or   → logical_and ('|' logical_and)*
logical_and  → comparison ('&' comparison)*
comparison   → addsub (('>'|'<'|'>='|'<='|'=='|'!=') addsub)?
addsub       → muldiv (('+'|'-') muldiv)*
muldiv       → unary (('*'|'/'|'@') unary)*
unary        → ('-'|'!') unary | postfix
postfix      → primary (('[' slices ']') | ('.' ID))*
primary      → ID | NUMBER | STRING | '(' expression ')' | funcall | list_lit
funcall      → ID '(' [expression (',' expression)*] ')'
list_lit     → '[' [expression (',' expression)*] ']'
slices       → slice_spec (',' slice_spec)*
slice_spec   → [expression] ':' [expression] [':' expression] | expression
```
