"""
Contract tests for Cut #2: Single canonical context renderer.

Validates:
  1. SUPERSEDED_BY predicate records are filtered out (marker, not fact)
  2. subject == object tautologies are deduplicated
  3. max_object_chars truncation works correctly
  4. max_total_chars budget is respected
  5. Both dict and model inputs produce correct output
  6. Empty records return empty string
"""

from atlas_memory.cognitive.retrieval.pipeline import RetrievalPipeline


class TestGenerateContextBlockCanonical:
    """Contract: generate_context_block is the single canonical renderer."""

    def test_superseded_by_not_rendered_as_fact(self):
        """SUPERSEDED_BY graph edges must NOT appear as facts in context block."""
        records = [
            {"subject": "rule:timeout:default", "predicate": "stated_memory", "object": "30 seconds"},
            {"subject": "rule:timeout:default", "predicate": "SUPERSEDED_BY", "object": "rule:timeout:updated"},
            {"subject": "rule:timeout:updated", "predicate": "stated_memory", "object": "60 seconds"},
        ]
        block = RetrievalPipeline.generate_context_block(records)
        assert "## ATLAS Memory Context" in block
        assert "rule:timeout:default" in block
        assert "rule:timeout:updated" in block
        assert "30 seconds" in block
        assert "60 seconds" in block
        # The SUPERSEDED_BY relation itself must NOT appear as a rendered fact line
        lines = block.strip().split("\n")
        for line in lines:
            if line.startswith("•"):
                assert "SUPERSEDED_BY" not in line, (
                    f"SUPERSEDED_BY appeared as a fact line: {line}"
                )

    def test_superseded_by_case_insensitive(self):
        """SUPERSEDED_BY filtering must be case-insensitive."""
        records = [
            {"subject": "a", "predicate": "superseded_by", "object": "b"},
            {"subject": "c", "predicate": "Superseded_By", "object": "d"},
            {"subject": "e", "predicate": "stated_memory", "object": "real fact"},
        ]
        block = RetrievalPipeline.generate_context_block(records)
        lines = [line for line in block.strip().split("\n") if line.startswith("•")]
        assert len(lines) == 1
        assert "real fact" in lines[0]

    def test_subject_equals_object_tautology_filtered(self):
        """Records where subject == object (self-referencing noise) must be removed."""
        records = [
            {"subject": "fact:A", "predicate": "stated_memory", "object": "fact:A"},
            {"subject": "fact:B", "predicate": "stated_memory", "object": "value B"},
        ]
        block = RetrievalPipeline.generate_context_block(records)
        lines = [line for line in block.strip().split("\n") if line.startswith("•")]
        assert len(lines) == 1
        assert "fact:B" in lines[0]
        assert "value B" in lines[0]

    def test_max_object_chars_truncation(self):
        """Long object values are truncated at max_object_chars with ...[truncated] marker."""
        long_value = "x" * 2000
        records = [{"subject": "key", "predicate": "stated_memory", "object": long_value}]
        block = RetrievalPipeline.generate_context_block(records, max_object_chars=100)
        assert "...[truncated]" in block
        # The truncated content should be ~100 chars + marker
        fact_line = [line for line in block.split("\n") if line.startswith("•")][0]
        # Remove prefix and check truncation
        content = fact_line[len("• [key] "):]
        assert content.startswith("x" * 100)
        assert content.endswith("...[truncated]")

    def test_no_truncation_when_zero(self):
        """max_object_chars=0 means no truncation."""
        long_value = "x" * 2000
        records = [{"subject": "key", "predicate": "stated_memory", "object": long_value}]
        block = RetrievalPipeline.generate_context_block(records, max_object_chars=0)
        assert "...[truncated]" not in block
        assert long_value in block

    def test_max_total_chars_budget(self):
        """Output is capped at max_total_chars."""
        records = [
            {"subject": f"key_{i}", "predicate": "stated_memory", "object": f"value {'y' * 200}"}
            for i in range(50)
        ]
        block = RetrievalPipeline.generate_context_block(records, max_total_chars=500)
        assert len(block) <= 500

    def test_empty_records_return_empty_string(self):
        """Empty record list returns empty string, not header-only block."""
        assert RetrievalPipeline.generate_context_block([]) == ""

    def test_dict_and_model_dual_path(self):
        """Both dict records and objects with attributes produce identical output."""
        from types import SimpleNamespace
        dict_rec = {"subject": "k1", "predicate": "stated_memory", "object": "v1"}
        model_rec = SimpleNamespace(subject="k1", predicate="stated_memory", object="v1")

        block_dict = RetrievalPipeline.generate_context_block([dict_rec])
        block_model = RetrievalPipeline.generate_context_block([model_rec])
        assert block_dict == block_model
        assert "• [k1] v1" in block_dict

    def test_header_only_when_all_filtered(self):
        """When all records are SUPERSEDED_BY or tautologies, only header remains."""
        records = [
            {"subject": "a", "predicate": "SUPERSEDED_BY", "object": "b"},
            {"subject": "c", "predicate": "stated_memory", "object": "c"},  # tautology
        ]
        block = RetrievalPipeline.generate_context_block(records)
        assert block == "## ATLAS Memory Context"

    def test_mixed_valid_and_filtered_records(self):
        """Mix of valid, SUPERSEDED_BY, and tautology records renders only valid ones."""
        records = [
            {"subject": "rule:git:workflow", "predicate": "stated_memory", "object": "use rebase"},
            {"subject": "rule:git:workflow", "predicate": "SUPERSEDED_BY", "object": "rule:git:workflow_v2"},
            {"subject": "rule:git:workflow_v2", "predicate": "stated_memory", "object": "use merge"},
            {"subject": "self_ref", "predicate": "stated_memory", "object": "self_ref"},
        ]
        block = RetrievalPipeline.generate_context_block(records)
        lines = [line for line in block.strip().split("\n") if line.startswith("•")]
        assert len(lines) == 2
        subjects = [line.split("]")[0].split("[")[1] for line in lines]
        assert "rule:git:workflow" in subjects
        assert "rule:git:workflow_v2" in subjects

    def test_superseded_flag_records_strictly_excluded(self):
        """CUT-08: Records with is_superseded=True (root or metadata) must be strictly excluded."""
        records = [
            {"subject": "old_key", "predicate": "stated_memory", "object": "old value", "is_superseded": True},
            {"subject": "meta_old_key", "predicate": "stated_memory", "object": "meta old", "metadata": {"is_superseded": True}},
            {"subject": "active_key", "predicate": "stated_memory", "object": "active value", "is_superseded": False},
        ]
        block = RetrievalPipeline.generate_context_block(records)
        lines = [line for line in block.strip().split("\n") if line.startswith("•")]
        assert len(lines) == 1
        assert "active_key" in lines[0]
        assert "active value" in lines[0]
        assert "old_key" not in block
        assert "meta_old_key" not in block

