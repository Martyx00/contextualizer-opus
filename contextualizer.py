"""
C Code Contextualizer for LLM Prompts.

Builds structured context from a C codebase for LLM-based security analysis.
Given a target file and line number, identifies the enclosing function and
produces two context sections:
  1. Dependencies: functions, macros, and global variables used by the target.
  2. Code paths: call chains that can lead to the target function's execution.
"""

import os
import re
from dataclasses import dataclass, field

import tree_sitter_c as tsc
import tree_sitter as ts

# ---------------------------------------------------------------------------
# Standard C library identifiers to exclude from dependency resolution
# ---------------------------------------------------------------------------
_LIBC_FUNCTIONS = frozenset([
    # <stdio.h>
    "printf", "fprintf", "sprintf", "snprintf", "vprintf", "vfprintf",
    "vsprintf", "vsnprintf", "scanf", "fscanf", "sscanf", "fopen", "fclose",
    "fread", "fwrite", "fgets", "fputs", "fgetc", "fputc", "getc", "putc",
    "getchar", "putchar", "puts", "gets", "ungetc", "fseek", "ftell",
    "rewind", "feof", "ferror", "clearerr", "perror", "remove", "rename",
    "tmpfile", "tmpnam", "fflush", "freopen", "setbuf", "setvbuf",
    "fsetpos", "fgetpos",
    # <stdlib.h>
    "malloc", "calloc", "realloc", "free", "abort", "exit", "_Exit",
    "atexit", "at_quick_exit", "quick_exit", "system", "getenv",
    "atoi", "atol", "atoll", "atof", "strtol", "strtoll", "strtoul",
    "strtoull", "strtof", "strtod", "strtold", "abs", "labs", "llabs",
    "div", "ldiv", "lldiv", "rand", "srand", "qsort", "bsearch",
    "mblen", "mbtowc", "wctomb", "mbstowcs", "wcstombs",
    # <string.h>
    "memcpy", "memmove", "memset", "memcmp", "memchr",
    "strcpy", "strncpy", "strcat", "strncat", "strcmp", "strncmp",
    "strchr", "strrchr", "strstr", "strtok", "strlen", "strerror",
    "strspn", "strcspn", "strpbrk", "strcoll", "strxfrm", "strnlen",
    "strdup", "strndup",
    # <math.h>
    "sin", "cos", "tan", "asin", "acos", "atan", "atan2",
    "sinh", "cosh", "tanh", "exp", "log", "log10", "log2",
    "pow", "sqrt", "ceil", "floor", "fabs", "fmod", "round",
    "trunc", "remainder", "copysign", "nan", "isnan", "isinf",
    "isfinite", "fpclassify",
    # <ctype.h>
    "isalpha", "isdigit", "isalnum", "isspace", "isupper", "islower",
    "ispunct", "isprint", "iscntrl", "isxdigit", "isgraph",
    "toupper", "tolower",
    # <assert.h>
    "assert",
    # <errno.h> / <signal.h> / <setjmp.h>
    "signal", "raise", "setjmp", "longjmp",
    # <stdarg.h>
    "va_start", "va_end", "va_arg", "va_copy",
    # <unistd.h> (POSIX, common in C projects)
    "read", "write", "open", "close", "lseek", "pipe", "dup", "dup2",
    "fork", "execv", "execve", "execvp", "wait", "waitpid",
    "sleep", "usleep", "nanosleep", "access", "unlink", "rmdir",
    "getcwd", "chdir", "getpid", "getppid", "isatty",
    # <pthread.h>
    "pthread_create", "pthread_join", "pthread_exit", "pthread_detach",
    "pthread_mutex_init", "pthread_mutex_lock", "pthread_mutex_unlock",
    "pthread_mutex_destroy", "pthread_cond_init", "pthread_cond_wait",
    "pthread_cond_signal", "pthread_cond_broadcast", "pthread_cond_destroy",
    # Common macros
    "NULL", "EOF", "stdin", "stdout", "stderr", "errno",
    "EXIT_SUCCESS", "EXIT_FAILURE", "BUFSIZ",
    "SEEK_SET", "SEEK_CUR", "SEEK_END",
    "INT_MAX", "INT_MIN", "UINT_MAX", "LONG_MAX", "LONG_MIN",
    "ULONG_MAX", "LLONG_MAX", "LLONG_MIN", "ULLONG_MAX",
    "SIZE_MAX", "CHAR_BIT", "CHAR_MAX", "CHAR_MIN",
    "SCHAR_MAX", "SCHAR_MIN", "UCHAR_MAX",
    "SHRT_MAX", "SHRT_MIN", "USHRT_MAX",
    "FLT_MAX", "FLT_MIN", "FLT_EPSILON",
    "DBL_MAX", "DBL_MIN", "DBL_EPSILON",
    "LDBL_MAX", "LDBL_MIN", "LDBL_EPSILON",
    "true", "false", "bool",
    "SIGINT", "SIGTERM", "SIGKILL", "SIGSEGV", "SIGABRT",
    # Common type-related keywords treated as known
    "size_t", "ssize_t", "ptrdiff_t", "intptr_t", "uintptr_t",
    "int8_t", "int16_t", "int32_t", "int64_t",
    "uint8_t", "uint16_t", "uint32_t", "uint64_t",
    "FILE", "va_list", "pid_t", "off_t", "time_t",
])

# C keywords and type specifiers that should never be resolved
_C_KEYWORDS = frozenset([
    "auto", "break", "case", "char", "const", "continue", "default", "do",
    "double", "else", "enum", "extern", "float", "for", "goto", "if",
    "inline", "int", "long", "register", "restrict", "return", "short",
    "signed", "sizeof", "static", "struct", "switch", "typedef", "union",
    "unsigned", "void", "volatile", "while", "_Alignas", "_Alignof",
    "_Atomic", "_Bool", "_Complex", "_Generic", "_Imaginary",
    "_Noreturn", "_Static_assert", "_Thread_local",
    "typeof", "__attribute__", "__asm__", "__volatile__",
    "__extension__", "__inline__", "__restrict__",
])

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Snippet:
    """A source code snippet with its location metadata."""
    name: str
    kind: str          # "function", "macro", "global", "type", "enum"
    file_path: str     # relative to root
    start_line: int
    end_line: int
    text: str

    @property
    def location(self) -> str:
        return f"{self.file_path}:{self.start_line}"


@dataclass
class CallerInfo:
    """A call-site that invokes the target (directly or via pointer)."""
    caller_snippet: Snippet          # the enclosing function
    call_line: int                   # line of the call expression
    via_pointer: str | None = None   # if through a pointer, its name


@dataclass
class CodePath:
    """A chain of callers leading to the target function."""
    chain: list[CallerInfo] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Tree-sitter helpers
# ---------------------------------------------------------------------------

def _make_parser() -> ts.Parser:
    return ts.Parser(ts.Language(tsc.language()))


def _parse_file(parser: ts.Parser, source: bytes) -> ts.Tree:
    return parser.parse(source)


def _node_text(node: ts.Node) -> str:
    return node.text.decode("utf-8", errors="replace")


def _source_lines(source: bytes) -> list[str]:
    return source.decode("utf-8", errors="replace").splitlines(keepends=True)


# ---------------------------------------------------------------------------
# Indexing: extract top-level definitions from a single file
# ---------------------------------------------------------------------------

def _index_file(parser: ts.Parser, rel_path: str, source: bytes) -> list[Snippet]:
    """Extract all top-level definitions from a C source file."""
    tree = _parse_file(parser, source)
    root = tree.root_node
    snippets: list[Snippet] = []

    for node in root.children:
        snippets.extend(_node_to_snippets(node, rel_path))
    return snippets


def _node_to_snippets(node: ts.Node, rel_path: str) -> list[Snippet]:
    """Convert a top-level tree-sitter node to Snippet(s) if applicable.

    Returns multiple snippets when a typedef contains enum enumerators or
    struct/union members that should be independently searchable — each
    enumerator maps back to the full typedef block.
    """
    ntype = node.type
    start = node.start_point[0] + 1  # 1-based
    end = node.end_point[0] + 1
    text = _node_text(node)
    results: list[Snippet] = []

    if ntype == "function_definition":
        name = _extract_function_name(node)
        if name:
            results.append(Snippet(name, "function", rel_path, start, end, text))

    elif ntype == "declaration":
        name = _extract_declaration_name(node)
        if name:
            results.append(Snippet(name, "global", rel_path, start, end, text))

    elif ntype == "preproc_def":
        name = _extract_preproc_name(node)
        if name:
            results.append(Snippet(name, "macro", rel_path, start, end, text))

    elif ntype == "preproc_function_def":
        name = _extract_preproc_name(node)
        if name:
            results.append(Snippet(name, "macro", rel_path, start, end, text))

    elif ntype == "type_definition":
        name = _extract_typedef_name(node)
        if name:
            results.append(Snippet(name, "type", rel_path, start, end, text))
        # Also index enum enumerator names so STATUS_OK -> full typedef block
        enumerators = _extract_enumerator_names(node)
        for ename in enumerators:
            results.append(Snippet(ename, "enum", rel_path, start, end, text))
        # Also index inner struct/union field names for pointer resolution
        inner_struct = _find_child_of_type(node, "struct_specifier")
        if inner_struct is None:
            inner_struct = _find_child_of_type(node, "union_specifier")
        if inner_struct:
            sname = _extract_struct_union_name(inner_struct)
            if sname and sname != name:
                results.append(Snippet(sname, "type", rel_path, start, end, text))

    elif ntype == "enum_specifier":
        name = _extract_enum_name(node)
        if name:
            results.append(Snippet(name, "enum", rel_path, start, end, text))
        enumerators = _extract_enumerator_names(node)
        for ename in enumerators:
            results.append(Snippet(ename, "enum", rel_path, start, end, text))

    elif ntype in ("struct_specifier", "union_specifier"):
        name = _extract_struct_union_name(node)
        if name:
            results.append(Snippet(name, "type", rel_path, start, end, text))

    elif ntype in ("preproc_ifdef", "preproc_if"):
        pass  # handled by walking children in deep index

    return results


def _extract_function_name(node: ts.Node) -> str | None:
    """Get function name from a function_definition node."""
    declarator = node.child_by_field_name("declarator")
    if declarator is None:
        return None
    return _find_identifier_in_declarator(declarator)


def _find_identifier_in_declarator(node: ts.Node) -> str | None:
    """Recursively find the identifier in a (possibly nested) declarator."""
    if node.type == "identifier":
        return _node_text(node)
    if node.type == "type_identifier":
        return _node_text(node)
    if node.type == "field_identifier":
        return _node_text(node)
    # function_declarator -> declarator field contains the name
    decl = node.child_by_field_name("declarator")
    if decl:
        return _find_identifier_in_declarator(decl)
    # Fallback: look for first identifier child
    for child in node.children:
        if child.type == "identifier":
            return _node_text(child)
        if child.type in ("function_declarator", "pointer_declarator",
                          "array_declarator", "parenthesized_declarator"):
            result = _find_identifier_in_declarator(child)
            if result:
                return result
    return None


def _extract_declaration_name(node: ts.Node) -> str | None:
    """Get the primary name from a top-level declaration."""
    declarator = node.child_by_field_name("declarator")
    if declarator:
        return _find_identifier_in_declarator(declarator)
    # Multiple declarators
    for child in node.children:
        if child.type in ("init_declarator",):
            decl = child.child_by_field_name("declarator")
            if decl:
                return _find_identifier_in_declarator(decl)
    return None


def _extract_preproc_name(node: ts.Node) -> str | None:
    """Get the macro name from a preproc_def or preproc_function_def."""
    name_node = node.child_by_field_name("name")
    if name_node:
        return _node_text(name_node)
    return None


def _extract_typedef_name(node: ts.Node) -> str | None:
    """Get the typedef'd name from a type_definition node."""
    declarator = node.child_by_field_name("declarator")
    if declarator:
        return _find_identifier_in_declarator(declarator)
    # Fallback: last identifier or type_identifier child
    for child in reversed(node.children):
        if child.type == "type_identifier":
            return _node_text(child)
        if child.type == "identifier":
            return _node_text(child)
    return None


def _extract_enum_name(node: ts.Node) -> str | None:
    name_node = node.child_by_field_name("name")
    if name_node:
        return _node_text(name_node)
    return None


def _extract_enumerator_names(node: ts.Node) -> list[str]:
    """Extract all enumerator constant names from an enum or typedef-enum."""
    names: list[str] = []
    _walk_enumerators(node, names)
    return names


def _walk_enumerators(node: ts.Node, names: list[str]) -> None:
    stack = [node]
    while stack:
        current = stack.pop()
        if current.type == "enumerator":
            name_node = current.child_by_field_name("name")
            if name_node:
                names.append(_node_text(name_node))
            continue
        for child in reversed(current.children):
            stack.append(child)


def _find_child_of_type(node: ts.Node, type_name: str) -> ts.Node | None:
    """Find the first direct or nested child with the given node type."""
    stack = list(node.children)
    while stack:
        current = stack.pop(0)
        if current.type == type_name:
            return current
        stack.extend(current.children)
    return None


def _extract_struct_union_name(node: ts.Node) -> str | None:
    name_node = node.child_by_field_name("name")
    if name_node:
        return _node_text(name_node)
    return None


# ---------------------------------------------------------------------------
# Identifier extraction: collect all identifiers used inside a function body
# ---------------------------------------------------------------------------

def _collect_identifiers_in_node(node: ts.Node) -> set[str]:
    """Collect all identifier tokens within a tree-sitter node (recursive)."""
    ids: set[str] = set()
    _walk_identifiers(node, ids)
    return ids


def _walk_identifiers(node: ts.Node, ids: set[str]) -> None:
    """Iteratively walk the AST collecting identifiers."""
    stack = [node]
    while stack:
        current = stack.pop()
        if current.type == "identifier":
            ids.add(_node_text(current))
        elif current.type == "type_identifier":
            ids.add(_node_text(current))
        elif current.type == "field_identifier":
            # field access like s->field — we track the field name too
            ids.add(_node_text(current))
        stack.extend(current.children)


def _collect_call_identifiers(node: ts.Node) -> set[str]:
    """Collect identifiers that appear as direct function calls."""
    calls: set[str] = set()
    _walk_calls(node, calls)
    return calls


def _walk_calls(node: ts.Node, calls: set[str]) -> None:
    stack = [node]
    while stack:
        current = stack.pop()
        if current.type == "call_expression":
            func = current.child_by_field_name("function")
            if func and func.type == "identifier":
                calls.add(_node_text(func))
        stack.extend(current.children)


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

_C_EXTENSIONS = {".c", ".h", ".cc", ".cpp", ".cxx", ".hpp", ".hxx", ".inl"}


def _find_c_files(root_path: str) -> list[str]:
    """Find all C/C++ source files under root_path. Returns relative paths."""
    result = []
    for dirpath, dirnames, filenames in os.walk(root_path):
        # Skip hidden and build directories
        dirnames[:] = [
            d for d in dirnames
            if not d.startswith(".") and d not in ("build", "node_modules", "__pycache__")
        ]
        for fname in filenames:
            ext = os.path.splitext(fname)[1].lower()
            if ext in _C_EXTENSIONS:
                full = os.path.join(dirpath, fname)
                result.append(os.path.relpath(full, root_path))
    return sorted(result)


# ---------------------------------------------------------------------------
# Codebase index
# ---------------------------------------------------------------------------

@dataclass
class CodebaseIndex:
    """Index of all top-level definitions across the codebase."""
    snippets_by_name: dict[str, list[Snippet]] = field(default_factory=dict)
    all_snippets: list[Snippet] = field(default_factory=list)
    # file path -> list of function snippets (for caller search)
    functions_by_file: dict[str, list[Snippet]] = field(default_factory=dict)
    # file path -> parsed source bytes
    sources: dict[str, bytes] = field(default_factory=dict)

    def add(self, snippet: Snippet) -> None:
        self.all_snippets.append(snippet)
        self.snippets_by_name.setdefault(snippet.name, []).append(snippet)
        if snippet.kind == "function":
            self.functions_by_file.setdefault(snippet.file_path, []).append(snippet)

    def lookup(self, name: str) -> list[Snippet]:
        return self.snippets_by_name.get(name, [])


def _build_index(root_path: str, parser: ts.Parser) -> CodebaseIndex:
    """Build a full codebase index."""
    index = CodebaseIndex()
    c_files = _find_c_files(root_path)
    for rel_path in c_files:
        full_path = os.path.join(root_path, rel_path)
        try:
            with open(full_path, "rb") as f:
                source = f.read()
        except (OSError, IOError):
            continue
        index.sources[rel_path] = source
        snippets = _index_file(parser, rel_path, source)
        for s in snippets:
            index.add(s)
    return index


# ---------------------------------------------------------------------------
# Preprocessor conditional blocks: extract definitions inside #ifdef etc.
# ---------------------------------------------------------------------------

def _index_file_deep(parser: ts.Parser, rel_path: str, source: bytes) -> list[Snippet]:
    """Like _index_file but also looks inside preprocessor conditional blocks."""
    tree = _parse_file(parser, source)
    snippets: list[Snippet] = []
    _walk_for_definitions(tree.root_node, rel_path, snippets)
    return snippets


def _walk_for_definitions(node: ts.Node, rel_path: str, snippets: list[Snippet]) -> None:
    """Walk the entire AST looking for definitions at any depth (iterative)."""
    stack = [node]
    while stack:
        current = stack.pop()
        found = _node_to_snippets(current, rel_path)
        if found:
            snippets.extend(found)
            # don't recurse into this node's children for more defs
            continue
        # Push children in reverse so we process them in order
        for child in reversed(current.children):
            stack.append(child)


def _build_index_deep(root_path: str, parser: ts.Parser) -> CodebaseIndex:
    """Build index including definitions inside preprocessor conditionals."""
    index = CodebaseIndex()
    c_files = _find_c_files(root_path)
    for rel_path in c_files:
        full_path = os.path.join(root_path, rel_path)
        try:
            with open(full_path, "rb") as f:
                source = f.read()
        except (OSError, IOError):
            continue
        index.sources[rel_path] = source
        snippets = _index_file_deep(parser, rel_path, source)
        for s in snippets:
            index.add(s)
    return index


# ---------------------------------------------------------------------------
# Target function identification
# ---------------------------------------------------------------------------

def _find_target_function(
    parser: ts.Parser, rel_path: str, source: bytes, line: int
) -> Snippet | None:
    """Find the function definition that encloses the given line number."""
    tree = _parse_file(parser, source)
    # line is 1-based, tree-sitter uses 0-based rows
    target_row = line - 1
    candidates: list[Snippet] = []
    _find_functions_containing_line(tree.root_node, rel_path, target_row, candidates)
    if not candidates:
        return None
    # Pick the innermost (smallest range) function
    candidates.sort(key=lambda s: s.end_line - s.start_line)
    return candidates[0]


def _find_functions_containing_line(
    node: ts.Node, rel_path: str, target_row: int, results: list[Snippet]
) -> None:
    """Iteratively find function_definition nodes containing the target row."""
    stack = [node]
    while stack:
        current = stack.pop()
        if current.type == "function_definition":
            if current.start_point[0] <= target_row <= current.end_point[0]:
                name = _extract_function_name(current)
                if name:
                    results.append(Snippet(
                        name=name,
                        kind="function",
                        file_path=rel_path,
                        start_line=current.start_point[0] + 1,
                        end_line=current.end_point[0] + 1,
                        text=_node_text(current),
                    ))
        stack.extend(current.children)


# ---------------------------------------------------------------------------
# Section 1: Dependency resolution
# ---------------------------------------------------------------------------

def _resolve_dependencies(
    target: Snippet, index: CodebaseIndex, parser: ts.Parser
) -> list[Snippet]:
    """Find all project-defined functions, macros, globals used by the target."""
    source = index.sources.get(target.file_path, b"")
    tree = _parse_file(parser, source)

    # Find the function node again to extract identifiers from its body
    target_row = target.start_line - 1
    func_node = _find_func_node_at(tree.root_node, target_row)
    if func_node is None:
        return []

    # Get the function body
    body = func_node.child_by_field_name("body")
    if body is None:
        body = func_node

    used_ids = _collect_identifiers_in_node(body)

    # Collect identifiers from the full function signature (return type, params)
    # by scanning the entire function node but excluding the body
    for child in func_node.children:
        if child != body:
            used_ids |= _collect_identifiers_in_node(child)

    # Remove the function's own name, C keywords, and libc identifiers
    used_ids.discard(target.name)
    used_ids -= _C_KEYWORDS
    used_ids -= _LIBC_FUNCTIONS

    # Also remove local variable names declared inside the function body
    local_vars = _collect_local_declarations(body)
    used_ids -= local_vars

    # Also remove parameter names
    param_names = _collect_param_names(func_node)
    used_ids -= param_names

    # Resolve each identifier against the index
    # Deduplicate by file+line so the same block (e.g. a typedef enum with
    # multiple enumerators) is only included once.
    resolved: dict[str, Snippet] = {}
    for ident in sorted(used_ids):
        matches = index.lookup(ident)
        for m in matches:
            key = f"{m.file_path}:{m.start_line}"
            if key not in resolved:
                resolved[key] = m

    return sorted(resolved.values(), key=lambda s: (s.kind, s.file_path, s.start_line))


def _find_func_node_at(node: ts.Node, target_row: int) -> ts.Node | None:
    """Find the function_definition node at the given row (iterative)."""
    stack = [node]
    while stack:
        current = stack.pop()
        if current.type == "function_definition":
            if current.start_point[0] <= target_row <= current.end_point[0]:
                return current
        stack.extend(current.children)
    return None


def _collect_local_declarations(body_node: ts.Node) -> set[str]:
    """Collect locally declared variable names inside a compound_statement."""
    locals_: set[str] = set()
    _walk_local_decls(body_node, locals_)
    return locals_


def _walk_local_decls(node: ts.Node, locals_: set[str]) -> None:
    stack = [node]
    while stack:
        current = stack.pop()
        if current.type == "declaration":
            declarator = current.child_by_field_name("declarator")
            if declarator:
                name = _find_identifier_in_declarator(declarator)
                if name:
                    locals_.add(name)
            # Handle multiple declarators (e.g., int a, b;)
            for child in current.children:
                if child.type == "init_declarator":
                    decl = child.child_by_field_name("declarator")
                    if decl:
                        name = _find_identifier_in_declarator(decl)
                        if name:
                            locals_.add(name)
        # Don't recurse into nested function definitions
        if current.type == "function_definition":
            continue
        stack.extend(current.children)


def _collect_param_names(func_node: ts.Node) -> set[str]:
    """Collect parameter names from a function definition."""
    params: set[str] = set()
    declarator = func_node.child_by_field_name("declarator")
    if declarator is None:
        return params
    # Find parameter_list
    param_list = declarator.child_by_field_name("parameters")
    if param_list is None:
        return params
    for param in param_list.children:
        if param.type == "parameter_declaration":
            decl = param.child_by_field_name("declarator")
            if decl:
                name = _find_identifier_in_declarator(decl)
                if name:
                    params.add(name)
    return params


# ---------------------------------------------------------------------------
# Section 2: Caller / code-path tracing
# ---------------------------------------------------------------------------

def _find_callers(
    target_name: str,
    index: CodebaseIndex,
    parser: ts.Parser,
) -> list[CallerInfo]:
    """Find all functions that call target_name (directly or via pointer)."""
    callers: list[CallerInfo] = []
    seen: set[str] = set()

    # 1. Find direct callers by searching for call expressions
    for rel_path, source in index.sources.items():
        tree = _parse_file(parser, source)
        funcs_in_file = index.functions_by_file.get(rel_path, [])
        for func_snippet in funcs_in_file:
            if func_snippet.name == target_name:
                continue
            func_node = _find_func_node_at(tree.root_node, func_snippet.start_line - 1)
            if func_node is None:
                continue
            body = func_node.child_by_field_name("body")
            if body is None:
                continue
            calls = _collect_call_identifiers(body)
            if target_name in calls:
                key = f"{func_snippet.name}:{func_snippet.file_path}:{func_snippet.start_line}"
                if key not in seen:
                    seen.add(key)
                    call_line = _find_call_line(body, target_name)
                    callers.append(CallerInfo(
                        caller_snippet=func_snippet,
                        call_line=call_line or func_snippet.start_line,
                    ))

    # 2. Find function pointer assignments/storage and trace their usage
    pointer_callers = _find_pointer_callers(target_name, index, parser, seen)
    callers.extend(pointer_callers)

    return callers


def _find_call_line(node: ts.Node, func_name: str) -> int | None:
    """Find the line number where func_name is called within node (iterative)."""
    stack = [node]
    while stack:
        current = stack.pop()
        if current.type == "call_expression":
            func = current.child_by_field_name("function")
            if func and func.type == "identifier" and _node_text(func) == func_name:
                return current.start_point[0] + 1
        stack.extend(current.children)
    return None


def _find_pointer_callers(
    target_name: str,
    index: CodebaseIndex,
    parser: ts.Parser,
    seen: set[str],
) -> list[CallerInfo]:
    """
    Find callers that invoke target_name through function pointers.

    Strategy:
    1. Search the codebase for assignments like `ptr = target_name` or
       `.field = target_name` or `array[i] = target_name`.
    2. Identify the pointer/field/array name.
    3. Find functions that call through that pointer.
    """
    callers: list[CallerInfo] = []

    # Collect the names of pointers/fields/arrays that store target_name
    pointer_names: set[str] = set()
    struct_field_assignments: list[tuple[str, str]] = []  # (struct_type_or_var, field)

    for rel_path, source in index.sources.items():
        tree = _parse_file(parser, source)
        _find_pointer_assignments(
            tree.root_node, target_name, pointer_names, struct_field_assignments
        )

    if not pointer_names and not struct_field_assignments:
        return callers

    # Now find functions that call through these pointers
    for rel_path, source in index.sources.items():
        tree = _parse_file(parser, source)
        funcs_in_file = index.functions_by_file.get(rel_path, [])
        for func_snippet in funcs_in_file:
            func_node = _find_func_node_at(tree.root_node, func_snippet.start_line - 1)
            if func_node is None:
                continue
            body = func_node.child_by_field_name("body")
            if body is None:
                continue

            # Check for calls through pointer names
            pointer_call = _find_pointer_call(body, pointer_names, struct_field_assignments)
            if pointer_call:
                ptr_name, call_line = pointer_call
                key = f"{func_snippet.name}:{func_snippet.file_path}:{func_snippet.start_line}"
                if key not in seen:
                    seen.add(key)
                    callers.append(CallerInfo(
                        caller_snippet=func_snippet,
                        call_line=call_line,
                        via_pointer=ptr_name,
                    ))

    return callers


def _find_pointer_assignments(
    node: ts.Node,
    target_name: str,
    pointer_names: set[str],
    struct_field_assignments: list[tuple[str, str]],
) -> None:
    """Find where target_name is assigned to a pointer, struct field, or array."""
    stack = [node]
    while stack:
        current = stack.pop()

        if current.type == "assignment_expression":
            right = current.child_by_field_name("right")
            if right and right.type == "identifier" and _node_text(right) == target_name:
                left = current.child_by_field_name("left")
                if left:
                    if left.type == "identifier":
                        pointer_names.add(_node_text(left))
                    elif left.type == "field_expression":
                        field = left.child_by_field_name("field")
                        if field:
                            pointer_names.add(_node_text(field))
                    elif left.type == "subscript_expression":
                        arr = left.child_by_field_name("argument")
                        if arr and arr.type == "identifier":
                            pointer_names.add(_node_text(arr))

        elif current.type == "init_declarator":
            value = current.child_by_field_name("value")
            if value and value.type == "identifier" and _node_text(value) == target_name:
                decl = current.child_by_field_name("declarator")
                if decl:
                    name = _find_identifier_in_declarator(decl)
                    if name:
                        pointer_names.add(name)

        elif current.type == "initializer_list":
            # Check each element for target_name references
            for child in current.children:
                if child.type == "initializer_pair":
                    value_nodes = [c for c in child.children if c.type == "identifier"]
                    designators = [c for c in child.children if c.type == "field_designator"]
                    for v in value_nodes:
                        if _node_text(v) == target_name:
                            for d in designators:
                                fname = _node_text(d).lstrip(".")
                                pointer_names.add(fname)

        stack.extend(current.children)


def _find_pointer_call(
    body: ts.Node,
    pointer_names: set[str],
    struct_field_assignments: list[tuple[str, str]],
) -> tuple[str, int] | None:
    """
    Find a call expression within body that invokes through one of the
    pointer_names (e.g., ptr(...), s->field(...), arr[i](...)).
    """
    result = _walk_pointer_calls(body, pointer_names)
    return result


def _walk_pointer_calls(
    node: ts.Node, pointer_names: set[str]
) -> tuple[str, int] | None:
    stack = [node]
    while stack:
        current = stack.pop()
        if current.type == "call_expression":
            func = current.child_by_field_name("function")
            if func:
                # Direct pointer call: ptr(...)
                if func.type == "identifier" and _node_text(func) in pointer_names:
                    return (_node_text(func), current.start_point[0] + 1)
                # Field access call: s->field(...) or s.field(...)
                if func.type == "field_expression":
                    field = func.child_by_field_name("field")
                    if field and _node_text(field) in pointer_names:
                        return (_node_text(field), current.start_point[0] + 1)
                # Array subscript call: arr[i](...)
                if func.type == "subscript_expression":
                    arr = func.child_by_field_name("argument")
                    if arr and arr.type == "identifier" and _node_text(arr) in pointer_names:
                        return (_node_text(arr), current.start_point[0] + 1)
        stack.extend(current.children)
    return None


def _trace_code_paths(
    target_name: str,
    index: CodebaseIndex,
    parser: ts.Parser,
    depth: int,
) -> list[CodePath]:
    """
    Trace code paths leading to target_name up to `depth` steps.

    Function pointers do not count as a step — the actual call site through
    the pointer counts as the step.
    """
    paths: list[CodePath] = []
    _trace_recursive(target_name, index, parser, depth, [], set(), paths)
    return paths


def _trace_recursive(
    func_name: str,
    index: CodebaseIndex,
    parser: ts.Parser,
    remaining_depth: int,
    current_chain: list[CallerInfo],
    visited: set[str],
    paths: list[CodePath],
) -> None:
    if remaining_depth <= 0:
        if current_chain:
            paths.append(CodePath(chain=list(current_chain)))
        return

    callers = _find_callers(func_name, index, parser)

    if not callers:
        # End of the road — this is an entry point or has no callers
        if current_chain:
            paths.append(CodePath(chain=list(current_chain)))
        return

    for caller in callers:
        caller_key = f"{caller.caller_snippet.name}:{caller.caller_snippet.file_path}"
        if caller_key in visited:
            # Avoid cycles
            if current_chain:
                paths.append(CodePath(chain=list(current_chain)))
            continue

        visited.add(caller_key)
        current_chain.append(caller)

        # Function pointer indirection does not consume a depth step.
        # Only direct calls consume depth.
        if caller.via_pointer is not None:
            # The pointer itself is not a step; trace who calls through it
            # without decrementing depth
            _trace_recursive(
                caller.caller_snippet.name, index, parser,
                remaining_depth, current_chain, visited, paths,
            )
        else:
            _trace_recursive(
                caller.caller_snippet.name, index, parser,
                remaining_depth - 1, current_chain, visited, paths,
            )

        current_chain.pop()
        visited.discard(caller_key)


# ---------------------------------------------------------------------------
# Context formatting
# ---------------------------------------------------------------------------

def _format_context(
    target: Snippet,
    dependencies: list[Snippet],
    code_paths: list[CodePath],
) -> str:
    """Format the collected context into a structured string for LLM input."""
    sections: list[str] = []

    # Header
    sections.append("=" * 72)
    sections.append("TARGET FUNCTION")
    sections.append("=" * 72)
    sections.append(f"Function: {target.name}")
    sections.append(f"Location: {target.location}")
    sections.append("")
    sections.append(target.text)
    sections.append("")

    # Section 1: Dependencies
    sections.append("=" * 72)
    sections.append("SECTION 1: DEPENDENCIES")
    sections.append("Functions, macros, types, and global variables used by the target")
    sections.append("=" * 72)

    if not dependencies:
        sections.append("(No project-specific dependencies found)")
    else:
        grouped: dict[str, list[Snippet]] = {}
        for dep in dependencies:
            grouped.setdefault(dep.kind, []).append(dep)

        kind_order = ["macro", "type", "enum", "global", "function"]
        kind_labels = {
            "macro": "Macros",
            "type": "Types (structs, unions, typedefs)",
            "enum": "Enumerations",
            "global": "Global Variables / Declarations",
            "function": "Functions",
        }

        for kind in kind_order:
            items = grouped.get(kind, [])
            if not items:
                continue
            sections.append("")
            sections.append(f"--- {kind_labels.get(kind, kind)} ---")
            sections.append("")
            for item in items:
                sections.append(f"// {item.location}")
                sections.append(item.text)
                sections.append("")

    # Section 2: Code paths
    sections.append("=" * 72)
    sections.append("SECTION 2: CODE PATHS LEADING TO TARGET")
    sections.append(f"Traced callers up to the specified depth")
    sections.append("=" * 72)

    if not code_paths:
        sections.append("(No callers found — the target may be an entry point)")
    else:
        # Collect unique caller snippets to avoid repeating the same function
        emitted_snippets: set[str] = set()
        path_descriptions: list[str] = []

        for i, path in enumerate(code_paths, 1):
            desc_parts = []
            for step in reversed(path.chain):
                ci = step
                if ci.via_pointer:
                    desc_parts.append(
                        f"{ci.caller_snippet.name} (via pointer '{ci.via_pointer}', "
                        f"line {ci.call_line})"
                    )
                else:
                    desc_parts.append(
                        f"{ci.caller_snippet.name} "
                        f"(line {ci.call_line})"
                    )
            desc_parts.append(target.name)
            path_descriptions.append(f"Path {i}: " + " -> ".join(desc_parts))

        sections.append("")
        for desc in path_descriptions:
            sections.append(desc)

        sections.append("")
        sections.append("--- Caller function bodies ---")
        sections.append("")

        for path in code_paths:
            for step in path.chain:
                s = step.caller_snippet
                key = f"{s.name}:{s.file_path}:{s.start_line}"
                if key not in emitted_snippets:
                    emitted_snippets.add(key)
                    sections.append(f"// {s.location}")
                    sections.append(s.text)
                    sections.append("")

    return "\n".join(sections)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def create_context(
    root_path: str,
    file_path: str,
    line: int,
    depth: int = 1,
) -> str:
    """
    Build LLM prompt context for a C function at the given location.

    Parameters
    ----------
    root_path : str
        Absolute path to the root of the C code repository.
    file_path : str
        Relative path from root_path to the target C source file.
    line : int
        Line number (1-based) within the target file.
    depth : int
        Maximum number of caller-chain steps to trace (default 1).
        Function pointer indirections do not count as a step.

    Returns
    -------
    str
        Formatted context string containing the target function, its
        dependencies, and code paths leading to it.
    """
    root_path = os.path.abspath(root_path)
    full_target = os.path.join(root_path, file_path)

    if not os.path.isfile(full_target):
        raise FileNotFoundError(f"Target file not found: {full_target}")

    parser = _make_parser()

    # Read target file
    with open(full_target, "rb") as f:
        target_source = f.read()

    # 1. Identify the target function
    target = _find_target_function(parser, file_path, target_source, line)
    if target is None:
        raise ValueError(
            f"No function found at line {line} in {file_path}. "
            f"Ensure the line number points to a line within a function body."
        )

    # 2. Build codebase index (includes definitions inside #ifdef blocks)
    index = _build_index_deep(root_path, parser)

    # 3. Resolve dependencies (Section 1)
    dependencies = _resolve_dependencies(target, index, parser)

    # 4. Trace code paths (Section 2)
    code_paths = _trace_code_paths(target.name, index, parser, depth)

    # 5. Format and return
    return _format_context(target, dependencies, code_paths)

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 4:
        print("Usage: python c_context_builder.py <repo_root> <file_path> <line> [depth]")
        print()
        print("  repo_root  – path to the C repository root")
        print("  file_path  – path to the target file, relative to repo_root")
        print("  line       – 1-based line number")
        print("  depth      – call-stack depth (default: 1)")
        sys.exit(1)

    _root = sys.argv[1]
    _file = sys.argv[2]
    _line = int(sys.argv[3])
    _depth = int(sys.argv[4]) if len(sys.argv) > 4 else 1

    print(create_context(_root, _file, _line, _depth))