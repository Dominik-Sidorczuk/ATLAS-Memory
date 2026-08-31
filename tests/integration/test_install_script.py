"""Regression tests for scripts/install-hermes-plugin.sh (ARCH_AUDIT H1-H4)."""

import re
import subprocess
from pathlib import Path

INSTALL_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "install-hermes-plugin.sh"
PLACEHOLDERS = ["__HERMES_PYTHON__", "__HERMES_SITE__", "__ATLAS_SRC__"]


def _get_heredoc_content(script_text: str) -> str:
    match = re.search(r"<<'PYTHON'(.*?)^PYTHON$", script_text, re.DOTALL | re.MULTILINE)
    assert match is not None, "Nie znaleziono heredocu <<'PYTHON' ... PYTHON"
    return match.group(1)


def test_bash_syntax():
    """Assert 1: bash -n scripts/install-hermes-plugin.sh -> returncode 0."""
    assert INSTALL_SCRIPT.exists(), f"Skrypt {INSTALL_SCRIPT} nie istnieje"
    result = subprocess.run(
        ["bash", "-n", str(INSTALL_SCRIPT)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"bash -n failed with code {result.returncode}: {result.stderr}"


def test_no_hardcoded_user_home():
    """Assert 2: Plik NIE zawiera '/home/dominik' (H1 hardcoded path)."""
    content = INSTALL_SCRIPT.read_text(encoding="utf-8")
    assert "/home/dominik" not in content, "Hardcoded '/home/dominik' znaleziony w skrypcie instalacyjnym (H1)"


def test_placeholders_and_sed_substitutions():
    """Assert 4 & 5: Plik zawiera placeholdery i odpowiadające im sed -i podstawienia."""
    content = INSTALL_SCRIPT.read_text(encoding="utf-8")
    for placeholder in PLACEHOLDERS:
        assert placeholder in content, f"Brak wymaganego placeholderu {placeholder} w skrypcie"

    # Weryfikacja sed-ów po heredocu
    match = re.search(r"<<'PYTHON'.*?^PYTHON\n(.*)", content, re.DOTALL | re.MULTILINE)
    assert match is not None, "Nie znaleziono sekcji po heredocu"
    post_heredoc = match.group(1)

    for placeholder in PLACEHOLDERS:
        pattern = rf"sed\s+-i\s+.*{re.escape(placeholder)}"
        assert re.search(pattern, post_heredoc), (
            f"Brak podstawienia 'sed -i' dla placeholderu {placeholder} po heredocu"
        )


def test_wrapper_heredoc_no_insert0():
    """Assert 3: Heredoc wrappera NIE zawiera 'insert(0' i używa 'append' (H2+H3, Hard Rule 5)."""
    content = INSTALL_SCRIPT.read_text(encoding="utf-8")
    heredoc = _get_heredoc_content(content)

    assert "insert(0" not in heredoc, "Heredoc wrappera zawiera zakazane 'insert(0' (H2+H3)"
    assert "append" in heredoc, "Heredoc wrappera powinien używać 'append' dla sys.path"


def test_rendered_wrapper_template():
    """Assert 6: Render template wrappera NIE zawiera '/home/', '/dominik', 'insert(0' i ZAWIERA 'append'."""
    content = INSTALL_SCRIPT.read_text(encoding="utf-8")
    heredoc = _get_heredoc_content(content)

    rendered = heredoc
    test_python = "/custom/runtime/venv/bin/python3"
    test_site = "/custom/runtime/venv/lib/python3.11/site-packages"
    test_src = "/custom/workspace/src"

    rendered = rendered.replace("__HERMES_PYTHON__", test_python)
    rendered = rendered.replace("__HERMES_SITE__", test_site)
    rendered = rendered.replace("__ATLAS_SRC__", test_src)

    assert "/home/" not in rendered, "Renderowany wrapper zawiera '/home/'"
    assert "/dominik" not in rendered, "Renderowany wrapper zawiera '/dominik'"
    assert "insert(0" not in rendered, "Renderowany wrapper zawiera 'insert(0'"
    assert "append" in rendered, "Renderowany wrapper musi zawierać 'append'"
    assert test_python in rendered, "Renderowany wrapper nie zawiera podstawionej ścieżki Pythona"
    assert test_site in rendered, "Renderowany wrapper nie zawiera podstawionego site-packages"
    assert test_src in rendered, "Renderowany wrapper nie zawiera podstawionej ścieżki src"
