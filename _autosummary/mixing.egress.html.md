# mixing.egress

Canonical *egress*: route a produced result to its destination.

Across `mixing`, every result-producing function takes a single `output`
parameter whose **role** is constant — “what to do with the result” — while its
**type** is open:

| `output` value   | behavior                                                                                                 |
|------------------|----------------------------------------------------------------------------------------------------------|
| `None`           | return the in-memory result (object producers) or<br/>save next to the input (file producers); see below |
| a **file path**  | write the result there; return the `Path`                                                                |
| a **directory**  | write with an auto-derived name; return the `Path`                                                       |
| a **callable**   | return `output(result)` — the general escape hatch                                                       |

There are two flavors of producer, modeled by two helpers:

- [`deliver()`](#mixing.egress.deliver) — *object-first*: `output=None` returns the in-memory
  object (e.g. an `Audio`). Used by editing functions that
  build a value you may want to keep chaining.
- [`write_egress()`](#mixing.egress.write_egress) — *file-first*: `output=None` saves to a default path
  (typically beside the input) and returns it. Used by file→file operations
  whose whole point is to produce a file.

Both accept the same `output` forms, so callers learn one mental model.
`Output` is the shared type alias.

### Module Attributes

| [`Output`](#mixing.egress.Output)   | nothing, a filesystem path/dir, or a result-consuming sink.   |
|-----------------------------------------------------------|---------------------------------------------------------------|

### Functions

| [`is_path_output`](#mixing.egress.is_path_output)(output)                           | True if `output` denotes a filesystem path (`str` / `os.PathLike`).       |
|---------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------|
| [`is_sink`](#mixing.egress.is_sink)(output)                                  | True if `output` is a result-consuming callable (not a path).             |
| [`resolve_output_path`](#mixing.egress.resolve_output_path)(output, \*, default_name)    | Resolve a path/dir `output` to a concrete file `Path` (parents created).  |
| [`deliver`](#mixing.egress.deliver)(result, output, \*, write, default_name) | Object-first egress: route an in-memory `result` to `output`.             |
| [`write_egress`](#mixing.egress.write_egress)(output, \*, default_path, write)    | File-first egress: for operations whose primary effect is writing a file. |

### mixing.egress.Output

nothing, a filesystem path/dir, or a result-consuming sink.

* **Type:**
  An egress target

alias of [`None`](https://docs.python.org/3/builtins/constants.html#None) | [`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | `os.PathLike[str]` | `Callable`[[[`Any`](https://docs.python.org/3/library/typing.html#typing.Any)], [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]

### mixing.egress.deliver(result, output, , write, default_name)

Object-first egress: route an in-memory `result` to `output`.

- `None` → return `result` unchanged (no I/O).
- sink callable → return `output(result)`.
- path/dir → `write(result, path)` then return the `Path`.

* **Parameters:**
  * **result** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – the produced in-memory value (e.g. an `Audio`).
  * **output** ([`Union`](https://docs.python.org/3/library/typing.html#typing.Union)[[`None`](https://docs.python.org/3/builtins/constants.html#None), [`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`PathLike`](https://docs.python.org/3/library/os.html#os.PathLike)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)], [`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[[`Any`](https://docs.python.org/3/library/typing.html#typing.Any)], [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]]) – the egress target (see module docstring).
  * **write** ([`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[[`Any`](https://docs.python.org/3/library/typing.html#typing.Any), [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)], [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]) – `write(result, path)` persists `result` to `path`.
  * **default_name** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – filename to use when `output` is a directory.
* **Return type:**
  [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)

### mixing.egress.is_path_output(output)

True if `output` denotes a filesystem path (`str` / `os.PathLike`).

* **Return type:**
  [`bool`](https://docs.python.org/3/builtins/functions.html#bool)

### mixing.egress.is_sink(output)

True if `output` is a result-consuming callable (not a path).

* **Return type:**
  [`bool`](https://docs.python.org/3/builtins/functions.html#bool)

### mixing.egress.resolve_output_path(output, , default_name)

Resolve a path/dir `output` to a concrete file `Path` (parents created).

A directory `output` (existing, or with a trailing separator) is joined
with `default_name`. Only call this when `output` is a path — not
`None` and not a sink.

* **Return type:**
  [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)

### mixing.egress.write_egress(output, , default_path, write)

File-first egress: for operations whose primary effect is writing a file.

- `None` → write to `default_path` (e.g. beside the input) and return it.
- sink callable → write to `default_path`, then return `output(path)`.
- path/dir → write to that file (dir → `default_path`’s name inside it).

* **Parameters:**
  * **output** ([`Union`](https://docs.python.org/3/library/typing.html#typing.Union)[[`None`](https://docs.python.org/3/builtins/constants.html#None), [`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`PathLike`](https://docs.python.org/3/library/os.html#os.PathLike)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)], [`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[[`Any`](https://docs.python.org/3/library/typing.html#typing.Any)], [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]]) – the egress target (see module docstring).
  * **default_path** ([`Union`](https://docs.python.org/3/library/typing.html#typing.Union)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`PathLike`](https://docs.python.org/3/library/os.html#os.PathLike)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]]) – where to write when `output` is `None` / a sink, and
    the source of the auto filename when `output` is a directory.
  * **write** ([`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[[`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)], [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]) – `write(path)` performs the actual encode to `path`.
* **Return type:**
  [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)
