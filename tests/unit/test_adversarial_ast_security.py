from __future__ import annotations

from atlas_memory.l3_procedural.skill_compiler import ASTSafetyScanner


def test_ast_scanner_catches_eval_and_exec():
    scanner = ASTSafetyScanner()
    code_eval = "res = eval('2 + 2')"
    code_exec = "exec('import os; os.system(\"rm -rf /\")')"

    v_eval = scanner.scan_code(code_eval)
    v_exec = scanner.scan_code(code_exec)

    assert len(v_eval) > 0
    assert "eval" in v_eval[0]
    assert len(v_exec) > 0
    assert "exec" in v_exec[0]


def test_ast_scanner_catches_subprocess_and_os_system():
    scanner = ASTSafetyScanner()
    code_sub = "import subprocess\nsubprocess.run(['ls', '-la'])"
    code_os = "import os\nos.system('echo pwned')"

    v_sub = scanner.scan_code(code_sub)
    v_os = scanner.scan_code(code_os)

    assert len(v_sub) > 0
    assert "subprocess" in v_sub[0]
    assert len(v_os) > 0
    assert "os.system" in v_os[0]


def test_ast_scanner_catches_open_write_modes():
    scanner = ASTSafetyScanner()
    code_w = "with open('secret.txt', 'w') as f: f.write('data')"
    code_a = "f = open('secret.txt', mode='a+')"
    code_r = "with open('readme.txt', 'r') as f: data = f.read()"

    assert len(scanner.scan_code(code_w)) > 0
    assert len(scanner.scan_code(code_a)) > 0
    # Read mode should be permitted
    assert len(scanner.scan_code(code_r)) == 0


def test_ast_scanner_catches_shutil_rmtree():
    scanner = ASTSafetyScanner()
    code = "import shutil\nshutil.rmtree('/important/dir')"
    violations = scanner.scan_code(code)
    assert len(violations) > 0
    assert "rmtree" in violations[0]


def test_ast_scanner_allows_pure_algorithmic_code():
    scanner = ASTSafetyScanner()
    safe_code = """
def compute_fib(n: int) -> int:
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(n - 1):
        a, b = b, a + b
    return b
"""
    violations = scanner.scan_code(safe_code)
    assert len(violations) == 0


def test_ast_scanner_syntax_error_graceful_handling():
    scanner = ASTSafetyScanner()
    invalid_code = "def broken_func( ::::"
    violations = scanner.scan_code(invalid_code)
    assert len(violations) > 0
    assert "SyntaxError" in violations[0]
