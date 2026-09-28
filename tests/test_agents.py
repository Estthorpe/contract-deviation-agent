"""
Tests for the pipeline agents.

Model calls are replaced by fakes so the tests are free, fast and repeatable.
The one live test is skipped unless RUN_LIVE_TESTS=1 is set.
"""

import json
import os
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from agents.clause_extractor import (
    ClauseExtractorAgent,
    ClauseObject,
    ContractParseError,
    _locate,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_CONTRACT = (PROJECT_ROOT / "data" / "sample_contract.txt").read_text(encoding="utf-8")
SUB_CLAUSE = re.compile(r"^(\d+\.\d+)\s+(.+)$", re.MULTILINE)
SMALL_CONTRACT = (
    "1. SERVICES\n"
    "1.1 The Supplier shall provide the Services with reasonable skill and care.\n"
    "2. FEES\n"
    "2.1 The Customer shall pay the Charges within thirty days of receipt of invoice.\n"
)


class FakeMessages:
    """Stands in for client.messages: returns each numbered sub-clause as JSON."""

    def __init__(self) -> None:
        """Start the call counter at zero."""
        self.calls = 0

    def create(self, **kwargs: Any) -> SimpleNamespace:
        """
        Imitate a successful model response.

        Args:
            **kwargs: The arguments the agent passes to messages.create().

        Returns:
            SimpleNamespace: An object shaped like an Anthropic message response.
        """
        self.calls += 1
        section = kwargs["messages"][0]["content"]
        clauses = [
            {"section_number": number, "heading": None, "text": text.strip()}
            for number, text in SUB_CLAUSE.findall(section)
        ]
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=json.dumps(clauses))],
            usage=SimpleNamespace(input_tokens=100, output_tokens=50),
            stop_reason="end_turn",
        )


class BrokenMessages(FakeMessages):
    """Stands in for a model that answers with text that is not JSON."""

    def create(self, **kwargs: Any) -> SimpleNamespace:
        """
        Imitate an unparseable model response.

        Args:
            **kwargs: The arguments the agent passes to messages.create().

        Returns:
            SimpleNamespace: A response whose text contains no JSON array.
        """
        self.calls += 1
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text="Sorry, I could not do that.")],
            usage=SimpleNamespace(input_tokens=100, output_tokens=10),
            stop_reason="end_turn",
        )


def make_agent(messages: FakeMessages) -> ClauseExtractorAgent:
    """
    Build an agent whose model client is replaced by a fake.

    Args:
        messages (FakeMessages): The fake to answer model calls.

    Returns:
        ClauseExtractorAgent: An agent that makes no network calls.
    """
    agent = ClauseExtractorAgent(proxy_url="https://gateway.test", api_key="sk-test")
    agent._client = SimpleNamespace(messages=messages)
    return agent


def test_clause_extractor() -> None:
    """The sample contract yields at least 15 complete, validated clauses."""
    agent = make_agent(FakeMessages())
    clauses = agent.extract(SAMPLE_CONTRACT)

    assert len(clauses) >= 15
    for clause in clauses:
        assert isinstance(clause, ClauseObject)
        assert clause.section_number
        assert clause.heading
        assert len(clause.text) >= 10
        assert clause.character_start >= 0
        assert clause.word_count >= 1
    liability = [c for c in clauses if c.section_number == "9.1"]
    assert liability and "10,000" in liability[0].text


@pytest.mark.parametrize(
    "contract, expected_numbers",
    [
        ("I. DEFINITIONS\nWords have their meanings.\n"
         "II. LIMITATION OF LIABILITY\nLiability is capped.\n", ["I", "II"]),
        ("1 DEFINITIONS\nWords have their meanings.\n"
         "2 LIMITATION OF LIABILITY\nLiability is capped.\n", ["1", "2"]),
        ("ARTICLE 1 SERVICES\nThe Supplier provides services.\n"
         "ARTICLE 2 FEES\nThe Customer pays fees.\n", ["1", "2"]),
    ],
    ids=["roman-numerals", "numbers-only", "article-prefix"],
)
def test_split_sections_non_standard_headings(contract: str, expected_numbers: list[str]) -> None:
    """AGENT DECISION 1 recognises heading styles other than '9. TITLE'."""
    agent = make_agent(FakeMessages())
    sections = agent._split_sections(contract)
    assert [section["number"] for section in sections] == expected_numbers


def test_unstructured_contract_uses_fallback() -> None:
    """AGENT DECISION 1 falls back to paragraph chunks when there are no headings."""
    agent = make_agent(FakeMessages())
    text = "The supplier will do the work.\n\nThe customer will pay on time.\n"
    sections = agent._split_sections(text)
    assert len(sections) >= 1
    assert all(section["number"] is None for section in sections)


def test_hallucination_guard_rejects_invented_clause() -> None:
    """AGENT DECISION 2b: invented clauses are rejected and invented headings replaced."""
    agent = make_agent(FakeMessages())
    contract = "9. LIMITATION OF LIABILITY\n9.1 The Supplier's liability shall not exceed the Charges paid.\n"
    section = {"number": "9", "title": "LIMITATION OF LIABILITY", "text": contract}
    raw_clauses = [
        {"section_number": "9.1", "heading": "A HEADING NOT IN THE CONTRACT",
         "text": "The Supplier's liability shall not exceed the Charges paid."},
        {"section_number": "9.2", "heading": None,
         "text": "The Supplier accepts unlimited liability for everything."},
    ]
    result = agent._validate_clauses(raw_clauses, section, contract)

    assert [clause.section_number for clause in result] == ["9.1"]
    assert result[0].heading == "LIMITATION OF LIABILITY"
    assert agent.rejected_clauses == 1


def test_locate_tolerates_line_breaks() -> None:
    """A clause wrapped across lines (as in a PDF) is still found."""
    assert _locate("shall not exceed the Charges", "Liability shall\nnot   exceed the Charges") == 10
    assert _locate("text that is not there", "Liability shall not exceed") == -1


def test_empty_contract_raises() -> None:
    """Too-short input stops the pipeline instead of producing an empty analysis."""
    agent = make_agent(FakeMessages())
    with pytest.raises(ContractParseError):
        agent.extract("Too short to be a contract.")


def test_unparseable_model_output_keeps_section() -> None:
    """AGENT DECISION 2: unparseable output keeps each section whole rather than losing it."""
    agent = make_agent(BrokenMessages())
    clauses = agent.extract(SMALL_CONTRACT)
    assert [clause.section_number for clause in clauses] == ["1", "2"]
    assert "reasonable skill and care" in clauses[0].text


@pytest.mark.live
@pytest.mark.skipif(os.environ.get("RUN_LIVE_TESTS") != "1",
                    reason="set RUN_LIVE_TESTS=1 to call the real gateway (costs about 2p)")
def test_clause_extractor_live() -> None:
    """The real agent, through the real gateway, extracts the sample contract."""
    agent = ClauseExtractorAgent()
    clauses = agent.extract(SAMPLE_CONTRACT)
    assert len(clauses) >= 15
    liability = [c for c in clauses if c.section_number == "9.1"]
    assert liability and "10,000" in liability[0].text