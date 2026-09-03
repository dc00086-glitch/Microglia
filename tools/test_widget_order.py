#!/usr/bin/env python3
"""Fail if a widget is used in the same function before it is created.

MMPS builds its interface in long setup functions, and enabling a button is
often mirrored to a second button somewhere else in the same function. Put the
mirrored line above the line that creates the widget and the app raises
AttributeError while its window is being built -- so it does not start at all,
and the traceback names a line that looks perfectly correct.

Nothing else catches this: the file parses, imports, and passes every other
test, because the failure only happens once Qt is actually constructing.

    python3 tools/test_widget_order.py
"""
import ast
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TARGET = os.path.join(ROOT, 'MMPSv2.12.py')

# Widget attributes only. Ordinary state is legitimately read before being
# written in a function -- counters, caches, flags set up elsewhere.
SUFFIXES = ('_btn', '_spin', '_check', '_combo', '_slider', '_label',
            '_bar', '_scroll', '_edit', '_group')


class Scan(ast.NodeVisitor):
    def __init__(self):
        self.problems = []

    def visit_FunctionDef(self, node):
        assigned, used = {}, {}

        def walk(n, nested):
            # A lambda or inner def runs later, when everything exists, so a
            # reference inside one says nothing about construction order.
            if isinstance(n, (ast.Lambda, ast.FunctionDef, ast.AsyncFunctionDef)) \
                    and n is not node:
                nested = True
            if isinstance(n, ast.Attribute) and \
                    isinstance(n.value, ast.Name) and n.value.id == 'self' and \
                    n.attr.endswith(SUFFIXES):
                if isinstance(n.ctx, ast.Store):
                    assigned.setdefault(n.attr, n.lineno)
                elif not nested:
                    used.setdefault(n.attr, n.lineno)
            for c in ast.iter_child_nodes(n):
                walk(c, nested)

        for c in ast.iter_child_nodes(node):
            walk(c, False)

        for attr, first_use in used.items():
            born = assigned.get(attr)
            if born is not None and first_use < born:
                self.problems.append((node.name, attr, first_use, born))
        self.generic_visit(node)


def main():
    tree = ast.parse(open(TARGET).read(), TARGET)
    s = Scan()
    s.visit(tree)
    if s.problems:
        for fn, attr, use, born in s.problems:
            print(f"  {os.path.basename(TARGET)}:{use}  self.{attr} is used "
                  f"here but not created until line {born}  (in {fn})")
        sys.exit(f"\nFAIL: {len(s.problems)} widget(s) used before they exist. "
                 f"MMPS will raise AttributeError while building its window "
                 f"and will not start.")
    print("PASS: no widget is used before it is created.")


if __name__ == '__main__':
    main()
