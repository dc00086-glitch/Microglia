#!/usr/bin/env python3
"""Fail if a button is wired to a method that will accept Qt's checked flag.

QPushButton.clicked emits clicked(bool checked=False). PyQt inspects the slot
and passes that flag to anything whose signature will take it. So adding an
innocuous keyword argument to a method that happens to be a button slot
silently changes what every click does:

    def batch_generate_masks(self):                 # click -> ask for settings
    def batch_generate_masks(self, ask=True, ...):  # click -> ask=False

Nothing raises. The button keeps working. It just quietly does the other
thing -- in that real case, skipping the settings dialog and generating masks
with whatever was last stored. The fix is to keep the slot argument-free and
put the options on a separate method the code calls directly.

    python3 tools/test_signal_slots.py
"""
import ast
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TARGET = os.path.join(ROOT, 'MMPSv2.12.py')

# Signals that carry a value PyQt will hand to a willing slot.
VALUE_SIGNALS = {'clicked', 'toggled', 'triggered', 'stateChanged',
                 'valueChanged', 'currentIndexChanged', 'textChanged',
                 'currentTextChanged', 'activated'}


def slot_targets(tree):
    """{method name: (signal, line)} for every `<sig>.connect(self.method)`."""
    found = {}
    for n in ast.walk(tree):
        if not (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and n.func.attr == 'connect' and n.args):
            continue
        sig = n.func.value
        if not (isinstance(sig, ast.Attribute) and sig.attr in VALUE_SIGNALS):
            continue
        arg = n.args[0]
        # Only a bare `self.method` reference. A lambda already shields the
        # slot from the signal's argument, which is the other valid fix.
        if isinstance(arg, ast.Attribute) and isinstance(arg.value, ast.Name) \
                and arg.value.id == 'self':
            found.setdefault(arg.attr, (sig.attr, n.lineno))
    return found


def main():
    tree = ast.parse(open(TARGET).read(), TARGET)
    wired = slot_targets(tree)

    problems = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name not in wired:
            continue
        a = node.args
        # A slot written to RECEIVE the signal's value declares the parameter
        # without a default -- `def _on_opacity_changed(self, value)` is
        # correct and intended. The dangerous shape is a parameter WITH a
        # default: the default says the author expects some calls to omit it,
        # and every emission silently overrides it. `*args` is a deliberate
        # catch-all and ignores whatever arrives.
        params = (a.posonlyargs + a.args)[1:]
        defaulted = params[len(params) - len(a.defaults):] if a.defaults else []
        if defaulted:
            sig, line = wired[node.name]
            problems.append((node.name, sig, line, node.lineno,
                             [p.arg for p in defaulted]))

    if problems:
        for name, sig, cline, dline, extra in problems:
            print(f"  {os.path.basename(TARGET)}:{cline}  {sig} -> "
                  f"self.{name}, but it takes {', '.join(extra)} "
                  f"(defined line {dline})")
            print(f"      every emission passes the signal's value into "
                  f"'{extra[0]}'")
        sys.exit(
            f"\nFAIL: {len(problems)} slot(s) will receive the signal's "
            f"argument instead of their default. Keep the slot argument-free "
            f"and move the options to a separate method, or connect through "
            f"a lambda.")
    print(f"PASS: all {len(wired)} connected slots take no argument Qt "
          f"could fill.")


if __name__ == '__main__':
    main()
