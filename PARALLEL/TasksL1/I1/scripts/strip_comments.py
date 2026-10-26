#!/usr/bin/env python3
import argparse
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

C_LIKE_EXTS = {".c", ".h", ".cpp", ".hpp", ".cc", ".cxx", ".cu", ".cuh"}
HASH_EXTS    = {".py", ".pyw", ".cmake", ".mk", ".sh", ".yml", ".yaml", ".toml"}
PS1_EXTS     = {".ps1", ".psm1"}

HASH_KEEP_PREFIXES = (
    "#include", "#define", "#ifdef", "#ifndef", "#if",
    "#elif", "#else", "#endif", "#pragma",
    "#error", "#warning", "#line", "#undef",
)

def strip_c_like(src: str) -> str:
    out = []
    i, n = 0, len(src)
    in_line  = False
    in_block = False
    in_str   = False
    in_ch    = False
    quote    = ""
    while i < n:
        c = src[i]
        nxt = src[i + 1] if i + 1 < n else ""

        if in_line:
            if c == "\n":
                in_line = False
                out.append(c)
            i += 1
            continue

        if in_block:
            if c == "*" and nxt == "/":
                in_block = False
                i += 2
            else:
                i += 1
            continue

        if in_str:
            out.append(c)
            if c == "\\" and i + 1 < n:
                out.append(src[i + 1]); i += 2; continue
            if c == quote:
                in_str = False
            i += 1
            continue

        if in_ch:
            out.append(c)
            if c == "\\" and i + 1 < n:
                out.append(src[i + 1]); i += 2; continue
            if c == "'":
                in_ch = False
            i += 1
            continue

        if c == "/" and nxt == "/":
            in_line = True
            i += 2
            continue
        if c == "/" and nxt == "*":
            in_block = True
            i += 2
            continue
        if c in ('"', "'"):
            quote = c
            if c == '"': in_str = True
            else:        in_ch  = True
            out.append(c); i += 1
            continue
        out.append(c); i += 1

    text = "".join(out)

    text = re.sub(r"\n{3,}", "\n\n", text)
    return text

def strip_hash_like(src: str, is_python: bool = True) -> str:
    lines = src.splitlines(keepends=True)
    out = []
    for idx, line in enumerate(lines):

        if idx == 0 and line.startswith("#!"):
            out.append(line); continue
        stripped = line.lstrip()

        if any(stripped.startswith(p) for p in HASH_KEEP_PREFIXES):
            out.append(line); continue

        in_str = False; quote = ""; res = []
        i = 0
        while i < len(line):
            c = line[i]
            if in_str:
                res.append(c)
                if c == "\\" and i + 1 < len(line):
                    res.append(line[i + 1]); i += 2; continue
                if c == quote:
                    in_str = False
                i += 1
                continue
            if c in ('"', "'"):
                in_str = True; quote = c
                res.append(c); i += 1; continue
            if c == "#":

                break
            res.append(c); i += 1

        cleaned = "".join(res).rstrip()

        if line.endswith("\n"):
            cleaned += "\n"
        out.append(cleaned)

    text = "".join(out)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text

def strip_ps1(src: str) -> str:

    out = []
    i, n = 0, len(src)
    in_block = False
    in_str = False; quote = ""
    while i < n:
        c = src[i]; nxt = src[i + 1] if i + 1 < n else ""
        if in_block:
            if c == "#" and nxt == ">":
                in_block = False; i += 2
            else:
                i += 1
            continue
        if in_str:
            out.append(c)
            if c == "\\" and i + 1 < n:
                out.append(src[i + 1]); i += 2; continue
            if c == quote:
                in_str = False
            i += 1
            continue
        if c == "<" and nxt == "#":
            in_block = True; i += 2; continue
        if c in ('"', "'"):
            in_str = True; quote = c; out.append(c); i += 1; continue
        if c == "#":
            while i < n and src[i] != "\n":
                i += 1
            continue
        out.append(c); i += 1
    text = "".join(out)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text

def dispatch(path: Path, src: str) -> str:
    ext = path.suffix.lower()
    if ext in C_LIKE_EXTS:
        return strip_c_like(src)
    if ext in PS1_EXTS:
        return strip_ps1(src)
    if ext in HASH_EXTS or path.name.lower() in ("cmakelists.txt", "makefile", "dockerfile"):
        return strip_hash_like(src)
    return src

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="перезаписать файлы")
    ap.add_argument("--backup", action="store_true", help="создать .bak")
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--skip-dirs", nargs="*",
                    default=[".git", "build", "__pycache__", ".venv", "venv",
                             "data", "node_modules", "cmake-build-debug"],
                    help="папки, которые не трогаем")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    skip = set(args.skip_dirs)

    changed = 0
    saved_bytes = 0
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if any(part in skip for part in p.parts):
            continue
        ext = p.suffix.lower()
        if ext not in C_LIKE_EXTS | HASH_EXTS | PS1_EXTS and \
           p.name.lower() not in ("cmakelists.txt", "makefile", "dockerfile"):
            continue

        try:
            src = p.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue

        new = dispatch(p, src)
        if new == src:
            continue

        removed = len(src) - len(new)
        changed += 1
        saved_bytes += removed
        print(f"{p.relative_to(root)}  -{removed} байт")

        if args.apply:
            if args.backup:
                shutil.copy2(p, p.with_suffix(p.suffix + ".bak"))
            p.write_text(new, encoding="utf-8")

    mode = "ПЕРЕЗАПИСАНО" if args.apply else "DRY-RUN (файлы не тронуты)"
    print(f"\n{mode}: файлов с изменениями: {changed}, "
          f"удалено байт всего: {saved_bytes}")
    if not args.apply and changed:
        print("Чтобы применить: python scripts/strip_comments.py --apply [--backup]")

if __name__ == "__main__":
    main()