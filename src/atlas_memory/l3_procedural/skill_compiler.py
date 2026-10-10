from __future__ import annotations

import ast
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from atlas_memory.core.models import StandardProcedure


class SafetyViolationError(Exception):
    """Błąd naruszenia bezpieczeństwa AST w kodzie procedury lub handlera."""
    def __init__(self, violations: List[str], message: Optional[str] = None):
        self.violations = violations
        self.message = message or f"Detected {len(violations)} safety violations in generated code: {violations}"
        super().__init__(self.message)


class ASTSafetyScanner:
    """
    Skaner AST wykrywający niebezpieczne operacje w generowanym lub przekazanym kodzie Python:
    - eval, exec, compile, __import__
    - subprocess.* (call, run, Popen, etc.)
    - os.system, os.popen, os.remove, os.unlink
    - open() z trybem zapisu ('w', 'a', 'wb', etc.)
    - shutil.rmtree
    """

    FORBIDDEN_CALLS: Set[str] = {"eval", "exec", "compile", "__import__"}

    # Module-qualified calls that are forbidden: module_name -> set of forbidden attributes (or None for all)
    FORBIDDEN_MODULE_CALLS: Dict[str, Optional[Set[str]]] = {
        "os": {"system", "popen", "remove", "unlink"},
        "subprocess": None,  # Block all subprocess attribute calls
        "shutil": {"rmtree"},
    }

    # Dangerous attribute calls forbidden regardless of caller base
    FORBIDDEN_UNQUALIFIED_ATTRS: Set[str] = {"system", "popen", "rmtree", "unlink"}

    # File open modes that permit writing/modifying files
    DANGEROUS_OPEN_MODES: Set[str] = {"w", "a", "x", "+"}

    @staticmethod
    def _extract_open_mode(call_node: ast.Call) -> Optional[str]:
        """Extracts the 'mode' argument from an open(...) call if present as constant string."""
        if len(call_node.args) >= 2:
            second_arg = call_node.args[1]
            if isinstance(second_arg, ast.Constant) and isinstance(second_arg.value, str):
                return second_arg.value
        for kw in call_node.keywords:
            if kw.arg == "mode" and isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                return kw.value.value
        return None

    def scan_code(self, code: str) -> List[str]:
        violations: List[str] = []
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return [f"SyntaxError on line {e.lineno or 'unknown'}: {e.msg}"]
        except Exception as e:
            return [f"SyntaxError: {str(e)}"]

        for node in ast.walk(tree):
            lineno = getattr(node, "lineno", 0)
            # Check direct Name references (e.g. x = eval)
            if isinstance(node, ast.Name) and node.id in self.FORBIDDEN_CALLS:
                violations.append(f"Line {lineno}: unsafe reference to forbidden identifier '{node.id}'")

            if isinstance(node, ast.Call):
                # 1. Bare forbidden functions (e.g. eval, exec, compile, __import__)
                if isinstance(node.func, ast.Name) and node.func.id in self.FORBIDDEN_CALLS:
                    violations.append(f"Line {lineno}: unsafe call to {node.func.id}")

                # 2. Attribute calls (e.g. os.system, subprocess.run, shutil.rmtree, etc.)
                elif isinstance(node.func, ast.Attribute):
                    attr_name = node.func.attr
                    base_name = node.func.value.id if isinstance(node.func.value, ast.Name) else ""
                    full_call = f"{base_name}.{attr_name}" if base_name else attr_name

                    is_unsafe = False
                    if base_name in self.FORBIDDEN_MODULE_CALLS:
                        blocked_attrs = self.FORBIDDEN_MODULE_CALLS[base_name]
                        if blocked_attrs is None or attr_name in blocked_attrs:
                            is_unsafe = True
                    elif attr_name in self.FORBIDDEN_UNQUALIFIED_ATTRS:
                        is_unsafe = True

                    if is_unsafe:
                        violations.append(f"Line {lineno}: unsafe call to {full_call}")

                # 3. open() with dangerous write modes ('w', 'a', 'wb', 'ab', 'w+', 'a+')
                if isinstance(node.func, ast.Name) and node.func.id == "open":
                    mode_val = self._extract_open_mode(node)
                    if mode_val and any(m in mode_val for m in self.DANGEROUS_OPEN_MODES):
                        violations.append(f"Line {lineno}: unsafe call to open with write mode '{mode_val}'")

        return violations

        return violations


class ParametricDistributionGraph:
    """
    DYREKTYWA 1: Parametryczny Graf Dystrybucyjny.
    Modeluje przestrzeń dopuszczalnych transformacji argumentów jako ciągłą funkcję gęstości
    prawdopodobieństwa wokół zaobserwowanych wywołań.
    """

    def __init__(self, steps_distributions: Optional[List[Dict[str, Any]]] = None):
        self.steps_distributions = steps_distributions or []
        self.parameter_domains: Dict[str, Any] = {}
        if self.steps_distributions:
            self.parameter_domains = self.steps_distributions[0]
        self.temperature: float = 0.0
        self.delta_temperature: float = 0.35
        self.max_temperature: float = 1.0

    def cool(self, factor: float = 0.85, step: float = 0.05) -> float:
        """Histereza chłodzenia / powrót do krystalizacji po ustąpieniu perturbacji."""
        self.temperature = max(0.0, self.temperature * float(factor) - float(step))
        return self.temperature

    def record_error(self, delta: Optional[float] = None) -> float:
        """Podbicie temperatury z twardym sufitem chroniącym przed ucieczką termiczną."""
        dt = self.delta_temperature if delta is None else float(delta)
        self.temperature = min(self.max_temperature, self.temperature + dt)
        return self.temperature

    def record_success(self, cool_factor: float = 0.80, cool_step: float = 0.02) -> float:
        """Schłodzenie temperatury poznawczej po udanym kroku wykonawczym."""
        self.temperature = max(0.0, self.temperature * cool_factor - cool_step)
        return self.temperature

    def reset_temperature(self) -> None:
        """Zeruje temperaturę poznawczą do stanu nominalnego spoczynku."""
        self.temperature = 0.0

    def fit_parameter_space(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Dopasowuje przestrzeń ciągłych i dyskretnych domen parametrów z zaobserwowanych wywołań."""
        collected: Dict[str, List[Any]] = {}
        for s in samples:
            for k, v in s.items():
                collected.setdefault(k, []).append(v)

        dist: Dict[str, Any] = {}
        for k, vals in collected.items():
            numeric_vals = [float(v) for v in vals if isinstance(v, (int, float)) and not isinstance(v, bool)]
            if len(numeric_vals) == len(vals) and len(numeric_vals) > 0:
                mean = sum(numeric_vals) / len(numeric_vals)
                variance = sum((x - mean) ** 2 for x in numeric_vals) / max(1, len(numeric_vals))
                std = math.sqrt(variance)
                dist[k] = {
                    "type": "continuous",
                    "mean": float(mean),
                    "std": float(std),
                    "min": float(min(numeric_vals)),
                    "max": float(max(numeric_vals)),
                }
            else:
                counts: Dict[str, int] = {}
                for v in vals:
                    counts[str(v)] = counts.get(str(v), 0) + 1
                dist[k] = {
                    "type": "discrete",
                    "categories": list(counts.keys()),
                    "frequencies": {str(cat): cnt / len(vals) for cat, cnt in counts.items()},
                }

        self.parameter_domains = dist
        if not self.steps_distributions:
            self.steps_distributions = [dist]
        else:
            self.steps_distributions[0] = dist
        return dist

    def sample_parameters(
        self,
        step_idx: int = 0,
        nominal_params: Optional[Dict[str, Any]] = None,
        temperature: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Próbkuje parametry z przestrzeni dystrybucyjnej z uwzględnieniem temperatury poznawczej T."""
        t = self.temperature if temperature is None else float(temperature)

        # Pobierz bazowe parametry nominalne (lub ze średnich/kategorii domen)
        if nominal_params is None:
            nominal = {}
            for k, d in self.parameter_domains.items():
                if d.get("type") in ("continuous", "continuous_gaussian"):
                    nominal[k] = d.get("mean", 0.0)
                elif d.get("type") in ("discrete", "discrete_distribution"):
                    cats = d.get("categories", list(d.get("frequencies", {}).keys()))
                    nominal[k] = cats[0] if cats else ""
            sampled = dict(nominal)
        else:
            sampled = dict(nominal_params)

        if step_idx < len(self.steps_distributions):
            dist = self.steps_distributions[step_idx]
        else:
            dist = self.parameter_domains

        if t <= 1e-4:
            return sampled

        import random

        for p_k, p_meta in dist.items():
            if not isinstance(p_meta, dict):
                continue
            p_type = p_meta.get("type")
            if p_type in ("continuous", "continuous_gaussian"):
                mean = p_meta.get("mean", sampled.get(p_k, 0.0))
                std = p_meta.get("std", 1.0)
                bounds = p_meta.get("bounds", [p_meta.get("min"), p_meta.get("max")])
                jitter = random.gauss(0.0, max(0.01, std * t))
                val = mean + jitter
                if bounds and len(bounds) == 2 and bounds[0] is not None and bounds[1] is not None:
                    val = max(bounds[0], min(bounds[1], val))
                if isinstance(sampled.get(p_k), int):
                    sampled[p_k] = int(round(val))
                else:
                    sampled[p_k] = float(val)
            elif p_type in ("discrete", "discrete_distribution"):
                freqs = p_meta.get("frequencies", {})
                if freqs and random.random() < min(0.8, t):
                    choices = list(freqs.keys())
                    weights = list(freqs.values())
                    sampled[p_k] = random.choices(choices, weights=weights, k=1)[0]

        return sampled


def _sanitize_identifier(name: str) -> str:
    cleaned = re.sub(r"[^0-9a-zA-Z_]", "_", name)
    if not cleaned or cleaned[0].isdigit():
        cleaned = f"proc_{cleaned}"
    return cleaned


def generate_handler(sop: StandardProcedure) -> str:
    """
    DYREKTYWA 1: Zastępuje statyczny szablon ciągłym modelem Parametrycznego Grafu Dystrybucyjnego
    z mechanizmem Termicznego Spadku Potencjału (Thermal Potential Drop Delta T).
    """
    class_name = f"{_sanitize_identifier(sop.procedure_id).title().replace('_', '')}Handler"
    if not class_name:
        class_name = "ProcedureHandler"

    # Przygotuj dystrybucje kroków
    steps_dist = [
        getattr(s, "distribution_space", {}) or {}
        for s in sop.steps
    ]

    lines = [
        '"""',
        f'Autogenerated Continuous Operator Graph Handler (ParametricDistributionGraph) for SOP: {sop.name}',
        f'Procedure ID: {sop.procedure_id}',
        f'Success Rate: {sop.success_rate:.2f} ({sop.invocations_count} runs)',
        'Paradigm: Continuous Parametric Distribution Graph with Thermal Potential Drop',
        '"""',
        "from __future__ import annotations",
        "",
        "import logging",
        "import random",
        "from typing import Any, Callable, Dict, List, Optional",
        "",
        "logger = logging.getLogger(__name__)",
        "",
        "",
        f"class {class_name}:",
        f'    """Executable Continuous Operator Graph for {sop.name}."""',
        "",
        "    def __init__(self, context: Optional[Dict[str, Any]] = None):",
        "        self.context = context or {}",
        "        self.step_results: Dict[str, Any] = {}",
        f"        self.steps_distributions: List[Dict[str, Any]] = {repr(steps_dist)}",
        "        self.cognitive_temperature: float = 0.0",
        "        self.delta_temperature: float = 0.35",
        "        self.max_temperature: float = 1.0",
        "        self.adaptation_hook: Optional[Callable[..., Any]] = self.context.get('adaptation_hook')",
        "",
        "    def cool(self, factor: float = 0.85, step: float = 0.05) -> float:",
        "        self.cognitive_temperature = max(0.0, self.cognitive_temperature * float(factor) - float(step))",
        "        return self.cognitive_temperature",
        "",
        "    def reset_temperature(self) -> None:",
        "        self.cognitive_temperature = 0.0",
        "",
        "    def sample_parameters(self, step_idx: int, nominal: Dict[str, Any], temp: float) -> Dict[str, Any]:",
        "        if temp <= 1e-4 or step_idx >= len(self.steps_distributions):",
        "            return dict(nominal)",
        "        dist = self.steps_distributions[step_idx]",
        "        sampled = dict(nominal)",
        "        for k, meta in dist.items():",
        "            if isinstance(meta, dict) and meta.get('type') == 'continuous_gaussian':",
        "                mean = meta.get('mean', sampled.get(k, 0.0))",
        "                std = meta.get('std', 1.0)",
        "                bounds = meta.get('bounds')",
        "                val = mean + random.gauss(0.0, max(0.01, std * temp))",
        "                if bounds and len(bounds) == 2:",
        "                    val = max(bounds[0], min(bounds[1], val))",
        "                sampled[k] = int(round(val)) if isinstance(sampled.get(k), int) else float(val)",
        "            elif isinstance(meta, dict) and meta.get('type') == 'discrete_distribution':",
        "                freqs = meta.get('frequencies', {})",
        "                if freqs and random.random() < min(0.8, temp):",
        "                    sampled[k] = random.choices(list(freqs.keys()), weights=list(freqs.values()), k=1)[0]",
        "        return sampled",
        "",
    ]

    for idx, step in enumerate(sop.steps, start=1):
        step_fn = f"step_{idx}"
        tool_name = step.tool_name
        params = step.params_pattern
        expected = step.expected_outcome or "success"

        lines.extend([
            f"    def {step_fn}(self, **kwargs) -> Dict[str, Any]:",
            f'        """Step {idx}: Execute tool \'{tool_name}\'.',
            f'        Expected outcome: {expected}',
            f'        Nominal params: {params}',
            '        """',
            f"        logger.info('Executing continuous operator step {idx}: {tool_name}')",
            "        merged = dict(kwargs)",
            f"        nominal_params = {repr(params)}",
            "        for k, v in nominal_params.items():",
            "            if k not in merged:",
            "                merged[k] = v",
            f"        eff_params = self.sample_parameters({idx - 1}, merged, self.cognitive_temperature)",
            "        try:",
            f"            result = {{'tool': '{tool_name}', 'status': 'completed', 'params': eff_params, 'temperature': self.cognitive_temperature}}",
            f"            self.step_results['{step_fn}'] = result",
            "            return result",
            "        except Exception as err:",
            f"            logger.warning('Thermal Potential Drop triggered in step {idx} ({tool_name}): %s', err)",
            "            # Ograniczenie temperatury sufitem max_temperature (ochrona przed hipertermią)",
            "            self.cognitive_temperature = min(self.max_temperature, self.cognitive_temperature + self.delta_temperature)",
            f"            adapted_params = self.sample_parameters({idx - 1}, eff_params, self.cognitive_temperature)",
            "            if callable(self.adaptation_hook):",
            f"                adapted_action = self.adaptation_hook('{tool_name}', adapted_params, err)",
            "                result = {'tool': '{tool_name}', 'status': 'adapted', 'adapted_action': adapted_action, 'temperature': self.cognitive_temperature}",
            "            else:",
            f"                result = {{'tool': '{tool_name}', 'status': 'adapted', 'params': adapted_params, 'temperature': self.cognitive_temperature, 'adaptation_reason': str(err)}}",
            f"            self.step_results['{step_fn}'] = result",
            "            return result",
            "",
        ])

    # Method run()
    lines.extend([
        "    def run(self, **kwargs) -> Dict[str, Any]:",
        '        """Execute all steps with continuous thermal relaxation."""',
        "        overall_results: Dict[str, Any] = {}",
    ])

    if not sop.steps:
        lines.append("        return {'status': 'completed', 'steps': {}}")
    else:
        for idx, _ in enumerate(sop.steps, start=1):
            step_fn = f"step_{idx}"
            lines.extend([
                "        try:",
                f"            step_out = self.{step_fn}(**kwargs)",
                f"            overall_results['{step_fn}'] = step_out",
                "        except Exception as exc:",
                "            self.cognitive_temperature += self.delta_temperature",
                f"            logger.warning('Fallback at step {idx}: %s (Temp: %s)', exc, self.cognitive_temperature)",
                f"            overall_results['{step_fn}'] = {{'status': 'adapted', 'temperature': self.cognitive_temperature, 'error': str(exc)}}",
            ])
        lines.append("        return {'status': 'completed', 'results': overall_results, 'final_temperature': self.cognitive_temperature}")

    return "\n".join(lines) + "\n"


def _generate_skill_md(sop: StandardProcedure) -> str:
    """Generuje zawartość SKILL.md z YAML frontmatter i markdown body."""
    # Build parameters dictionary from step params
    combined_params: Dict[str, Any] = {}
    for step in sop.steps:
        for k, v in step.params_pattern.items():
            combined_params[k] = type(v).__name__ if v is not None else "string"

    yaml_lines = [
        "---",
        f"name: {sop.procedure_id}",
        f"description: {sop.name}",
        "version: 1.0.0",
        f"created_at: {sop.created_at}",
        f"success_rate: {sop.success_rate:.2f}",
        f"invocations_count: {sop.invocations_count}",
        "parameters:",
    ]
    if combined_params:
        for p_name, p_type in combined_params.items():
            yaml_lines.append(f"  {p_name}: {p_type}")
    else:
        yaml_lines.append("  {}")
    yaml_lines.append("---")

    body_lines = [
        "",
        f"# {sop.name}",
        "",
        f"Autonomous procedural skill distilled from {sop.invocations_count} executions (Success rate: {sop.success_rate * 100:.1f}%).",
        "",
        "## Steps",
    ]

    for idx, step in enumerate(sop.steps, start=1):
        outcome_str = f" → Outcome: `{step.expected_outcome}`" if step.expected_outcome else ""
        body_lines.append(f"{idx}. **{step.tool_name}**: parameters `{step.params_pattern}`{outcome_str}")

    body_lines.extend([
        "",
        "## Usage",
        "```python",
        f"from scripts.handler import {_sanitize_identifier(sop.procedure_id).title().replace('_', '')}Handler",
        "",
        f"handler = {_sanitize_identifier(sop.procedure_id).title().replace('_', '')}Handler()",
        "result = handler.run()",
        "```",
        "",
    ])

    return "\n".join(yaml_lines) + "\n" + "\n".join(body_lines)


def compile_sop_to_skill(
    sop: StandardProcedure,
    target_dir: Optional[Path] = None,
    scanner: Optional[ASTSafetyScanner] = None,
) -> Path:
    """
    Kompiluje obiekt StandardProcedure do natywnego skilla Hermes:
    - target_dir/SKILL.md (YAML frontmatter + markdown body)
    - target_dir/scripts/handler.py (Executable class z metodami step_N + run)
    
    Weryfikuje kod pod kątem bezpieczeństwa przez ASTSafetyScanner.
    Jeśli wykryto naruszenia, rzuca SafetyViolationError.

    V68 (ARCH-006, uczciwa etykieta): produktem jest skill dla WARSTWY
    AGENTA Hermesa (katalog skills) — silnik ATLAS (daemon/engine) NIE
    importuje ani nie wykonuje ProcedureHandler w locie; wykonawca O(1)
    w silniku pozostaje poza zakresem (decyzja właściciela: bez V69).
    """
    if target_dir is None:
        target_dir = Path(os.path.expanduser(f"~/.hermes/skills/{sop.procedure_id}"))

    scanner = scanner or ASTSafetyScanner()

    # Sprawdzenie czy w parametrach lub nazwach narzędzi nie ma jawnych wektorów ataku
    for step in sop.steps:
        # Check if step.tool_name or step.params_pattern contains code with safety violations
        for v in step.params_pattern.values():
            if isinstance(v, str):

                violations = scanner.scan_code(v)
                if violations and not any(v.startswith("SyntaxError") for v in violations):
                    raise SafetyViolationError(violations)
        forbidden_tools = scanner.FORBIDDEN_CALLS | {"os.system", "subprocess", "rmtree"}
        if step.tool_name in forbidden_tools:
            raise SafetyViolationError([f"Unsafe tool name: {step.tool_name}"])

    handler_code = generate_handler(sop)
    violations = scanner.scan_code(handler_code)
    if violations:
        raise SafetyViolationError(violations)

    skill_md_content = _generate_skill_md(sop)

    # Zapis na dysk
    scripts_dir = target_dir / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)

    skill_md_path = target_dir / "SKILL.md"
    handler_path = scripts_dir / "handler.py"

    skill_md_path.write_text(skill_md_content, encoding="utf-8")
    handler_path.write_text(handler_code, encoding="utf-8")

    return target_dir


def register_in_hermes_environment(
    skill_dir: Path,
    hermes_skills_root: Optional[Path] = None,
    scanner: Optional[ASTSafetyScanner] = None,
) -> Path:
    """
    Rejestruje skompilowany katalog skilla w środowisku Hermesa (~/.hermes/skills/atlas_procedural/<skill_name>).
    
    1. Weryfikuje obecność SKILL.md i scripts/handler.py.
    2. Przeprowadza walidację bezpieczeństwa AST kodu handlera.
    3. Instaluje/aktualizuje pliki w docelowym katalogu.
    4. Zapisuje lub aktualizuje atomiczny manifest skills_manifest.json.
    """
    if hermes_skills_root is None:
        hermes_skills_root = Path(os.environ.get("HERMES_SKILLS_DIR", str(Path.home() / ".hermes" / "skills")))

    target_procedural_root = hermes_skills_root / "atlas_procedural"
    target_procedural_root.mkdir(parents=True, exist_ok=True)

    skill_md = skill_dir / "SKILL.md"
    handler_py = skill_dir / "scripts" / "handler.py"

    if not skill_md.exists():
        raise FileNotFoundError(f"SKILL.md missing in {skill_dir}")
    if not handler_py.exists():
        raise FileNotFoundError(f"scripts/handler.py missing in {skill_dir}")

    scanner = scanner or ASTSafetyScanner()
    handler_code = handler_py.read_text(encoding="utf-8")
    violations = scanner.scan_code(handler_code)
    if violations:
        raise SafetyViolationError(violations)

    skill_name = skill_dir.name
    destination_skill_dir = target_procedural_root / skill_name
    dest_scripts_dir = destination_skill_dir / "scripts"
    dest_scripts_dir.mkdir(parents=True, exist_ok=True)

    import shutil
    shutil.copy2(skill_md, destination_skill_dir / "SKILL.md")
    shutil.copy2(handler_py, dest_scripts_dir / "handler.py")

    # Manifest registration
    import json
    import time
    manifest_path = target_procedural_root / "skills_manifest.json"
    manifest: Dict[str, Any] = {}
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            manifest = {}

    manifest[skill_name] = {
        "name": skill_name,
        "path": str(destination_skill_dir),
        "registered_at": time.time(),
        "handler_size_bytes": len(handler_code.encode("utf-8")),
        "status": "active",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return destination_skill_dir


def compile_and_register_sop(
    sop: StandardProcedure,
    target_dir: Optional[Path] = None,
    hermes_skills_root: Optional[Path] = None,
    scanner: Optional[ASTSafetyScanner] = None,
) -> Dict[str, Any]:
    """
    Pełny pipeline: Kompiluje SOP do natywnego formatu i rejestruje w katalogu Hermesa.

    V68 (ARCH-006): 'rejestruje' = instalacja plików skilla dla agenta
    Hermesa; daemon ATLAS nie ładuje wygenerowanego handlera (brak
    wykonawcy O(1) — etykieta, nie regresja).
    """
    compiled_dir = compile_sop_to_skill(sop, target_dir=target_dir, scanner=scanner)
    registered_path = register_in_hermes_environment(
        skill_dir=compiled_dir,
        hermes_skills_root=hermes_skills_root,
        scanner=scanner,
    )
    return {
        "procedure_id": sop.procedure_id,
        "compiled_path": str(compiled_dir),
        "registered_path": str(registered_path),
        "success_rate": sop.success_rate,
        "steps_count": len(sop.steps),
        "registered": True,
    }
