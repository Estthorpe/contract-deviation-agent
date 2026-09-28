"""
ClauseExtractorAgent - Agent 1 of the Contract Deviation Detection pipeline.

Primary technique: T11 Document Processing and Information Extraction.
Turns raw contract text into a list of validated ClauseObject records, one per
contractual clause. Python finds the document's visible structure; Claude Haiku
(called through the LiteLLM gateway) splits each section into clauses; Python
then checks every clause Haiku returns against the original contract.
"""

import json
import logging
import os
import re
from pathlib import Path

import anthropic
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(dotenv_path=PROJECT_ROOT / ".env")

DEFAULT_MODEL = "claude-haiku-4-5"
MAX_TOKENS = 2048
FALLBACK_CHUNK_WORDS = 700
MIN_CONTRACT_CHARS = 100

# Pricing used for the cost line (verified 28 Sep 2026: Haiku 4.5 = $1 / $5 per
# million input / output tokens). The exchange rate is an assumption - check it.
INPUT_USD_PER_MTOK = 1.0
OUTPUT_USD_PER_MTOK = 5.0
USD_TO_GBP = 0.79

SECTION_HEADING = re.compile(
    r"^[ \t]*(?:(?:ARTICLE|SECTION|CLAUSE)[ \t]+)?"
    r"(?P<number>\d+|[IVXLC]+)[.)]?[ \t]+"
    r"(?P<title>[A-Z][A-Z0-9 &,'()/-]{2,})[ \t]*$",
    re.MULTILINE,
)

SYSTEM_PROMPT = """You are a contract clause extraction engine. You receive one section of a commercial contract. Split it into its individual clauses and return them as a JSON array.

Rules:
1. Each numbered sub-clause (for example 9.1, 9.2, (a), (b)) is a separate clause.
2. A paragraph that continues the previous clause (for example one starting "provided that", "save that" or "except that") belongs to that previous clause. Do not split it off.
3. Copy each clause's text exactly as written, word for word. Do not paraphrase, correct, shorten or add anything. Do not include the clause number at the start of "text".
4. "heading" must be a heading that literally appears in the section text. If there is none, use null.
5. Ignore signature blocks, page numbers and execution lines such as "Signed by".
6. Return only the JSON array, with no commentary, in this exact form:
[{"section_number": "9.1", "heading": "LIMITATION OF LIABILITY", "text": "..."}]"""

logger = logging.getLogger(__name__)


class ContractParseError(ValueError):
    """Raised when no usable clauses can be extracted from a contract."""


class ClauseExtractionError(RuntimeError):
    """Raised when the model cannot be reached through the gateway."""


class ClauseObject(BaseModel):
    """One extracted contractual clause."""

    section_number: str
    heading: str | None = None
    text: str = Field(min_length=10)
    character_start: int = Field(ge=0)
    word_count: int = Field(ge=1)


def _locate(snippet, document):
    """
    Find where a clause's text appears in the contract, ignoring whitespace differences.

    Args:
        snippet (str): The clause text returned by the model.
        document (str): The full contract text.

    Returns:
        int: The character position where the clause starts, or -1 if the text
            does not appear in the contract.
    """
    words = snippet.split()
    if not words:
        return -1
    pattern = r"\s+".join(re.escape(word) for word in words)
    match = re.search(pattern, document)
    return match.start() if match else -1


def _parse_json_array(raw):
    """
    Pull a JSON array of objects out of a model response.

    Args:
        raw (str): The model's text response, possibly wrapped in ``` fences.

    Returns:
        list[dict]: The parsed objects.

    Raises:
        ValueError: If no valid JSON array can be found.
    """
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())
    start, end = cleaned.find("["), cleaned.rfind("]")
    if start == -1 or end == -1:
        raise ValueError("no JSON array in the model response")
    data = json.loads(cleaned[start : end + 1])
    if not isinstance(data, list):
        raise ValueError("model response is not a JSON array")
    return [item for item in data if isinstance(item, dict)]


class ClauseExtractorAgent:
    """Extracts every discrete clause from a contract as validated ClauseObjects."""

    def __init__(self, proxy_url=None, api_key=None, model=None):
        """
        Connect to Claude through the LiteLLM gateway.

        Args:
            proxy_url (str | None): Gateway URL. Defaults to LITELLM_PROXY_URL.
            api_key (str | None): Gateway master key. Defaults to LITELLM_MASTER_KEY.
            model (str | None): Model name on the gateway allow-list.

        Raises:
            ClauseExtractionError: If the gateway URL or key is missing.
        """
        self.proxy_url = proxy_url or os.environ.get("LITELLM_PROXY_URL")
        key = api_key or os.environ.get("LITELLM_MASTER_KEY")
        if not self.proxy_url or not key:
            raise ClauseExtractionError(
                "LITELLM_PROXY_URL and LITELLM_MASTER_KEY must be set (in .env "
                "locally, or Streamlit secrets in the cloud). All model calls go "
                "through the gateway - never directly to Anthropic."
            )
        self.model = model or DEFAULT_MODEL
        self._client = anthropic.Anthropic(
            base_url=self.proxy_url, api_key=key, timeout=90.0, max_retries=2
        )
        self.usage = {"calls": 0, "input_tokens": 0, "output_tokens": 0}
        self.rejected_clauses = 0

    def _split_sections(self, text):
        """
        Divide the contract into sections to send to the model one at a time.

        Args:
            text (str): The full contract text.

        Returns:
            list[dict]: Sections with keys "number", "title" and "text".
        """
        matches = list(SECTION_HEADING.finditer(text))

        # AGENT DECISION 1: Section boundary detection - numbered headings are
        # matched by pattern first because it is free, instant and deterministic.
        # If fewer than two are found the document does not follow a numbered
        # structure, so it is cut into paragraph chunks and Haiku finds the
        # clause structure itself.
        if len(matches) >= 2:
            sections = []
            for index, match in enumerate(matches):
                end = (
                    matches[index + 1].start()
                    if index + 1 < len(matches)
                    else len(text)
                )
                sections.append(
                    {
                        "number": match.group("number"),
                        "title": match.group("title").strip(),
                        "text": text[match.start() : end].strip(),
                    }
                )
            logger.info("Pattern detection found %s sections", len(sections))
            return sections

        logger.info(
            "No numbered structure found - using paragraph chunks with model fallback"
        )
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        chunks, current, words = [], [], 0
        for paragraph in paragraphs:
            size = len(paragraph.split())
            if current and words + size > FALLBACK_CHUNK_WORDS:
                chunks.append("\n\n".join(current))
                current, words = [], 0
            current.append(paragraph)
            words += size
        if current:
            chunks.append("\n\n".join(current))
        return [{"number": None, "title": None, "text": chunk} for chunk in chunks]

    def _extract_section(self, section):
        """
        Ask the model to split one section into clauses.

        Args:
            section (dict): A section from _split_sections().

        Returns:
            list[dict]: Raw clause dictionaries as returned by the model.

        Raises:
            ClauseExtractionError: If the gateway or model cannot be reached.
        """
        label = (
            f"{section['number']}. {section['title']}"
            if section["number"]
            else "unnumbered text"
        )
        try:
            response = self._client.messages.create(
                model=self.model,
                max_tokens=MAX_TOKENS,
                system=SYSTEM_PROMPT,
                messages=[
                    {
                        "role": "user",
                        "content": f"<section>\n{section['text']}\n</section>",
                    }
                ],
            )
        except anthropic.APIError as error:
            raise ClauseExtractionError(
                f"Model call for section '{label}' failed. Check the Render gateway "
                f"shows Live (it may be waking up), and that LITELLM_PROXY_URL and "
                f"LITELLM_MASTER_KEY in .env are correct. Underlying error: {error}"
            ) from error

        self.usage["calls"] += 1
        self.usage["input_tokens"] += response.usage.input_tokens
        self.usage["output_tokens"] += response.usage.output_tokens
        raw = "".join(block.text for block in response.content if block.type == "text")

        # AGENT DECISION 2: Clause completeness - the model decides where each
        # clause starts and ends. If its answer cannot be parsed (or was cut off),
        # the whole section is kept as a single clause rather than dropped, so a
        # parsing failure can never make a clause silently disappear.
        try:
            return _parse_json_array(raw)
        except (ValueError, json.JSONDecodeError) as error:
            logger.warning(
                "Section '%s': unparseable model output (%s, stop=%s) - "
                "keeping section as one clause",
                label,
                error,
                response.stop_reason,
            )
            return [
                {
                    "section_number": section["number"],
                    "heading": section["title"],
                    "text": section["text"],
                }
            ]

    def _validate_clauses(self, raw_clauses, section, contract_text):
        """
        Check each model-returned clause against the contract and build ClauseObjects.

        Args:
            raw_clauses (list[dict]): Clauses returned by the model for one section.
            section (dict): The section they came from.
            contract_text (str): The full contract text.

        Returns:
            list[ClauseObject]: The clauses that passed validation.
        """
        validated = []
        for raw in raw_clauses:
            clause_text = str(raw.get("text") or "").strip()
            heading = raw.get("heading")
            number = str(
                raw.get("section_number")
                or section["number"]
                or f"U{len(validated) + 1}"
            )

            # AGENT DECISION 2b: Hallucination guard - a clause is only accepted
            # if its text really exists in the contract. A model can return
            # plausible text that was never written; for legal use, an invented
            # clause is worse than a missing one.

            start = _locate(clause_text, contract_text)
            if start == -1:
                self.rejected_clauses += 1
                continue
            if not heading or heading not in contract_text:
                heading = section["title"]
            try:
                validated.append(
                    ClauseObject(
                        section_number=number,
                        heading=heading,
                        text=clause_text,
                        character_start=start,
                        word_count=len(clause_text.split()),
                    )
                )
            except ValidationError as error:
                self.rejected_clauses += 1
                logger.warning("Clause %s failed schema validation: %s", number, error)
        return validated

    def extract(self, contract_text):
        """
        Extract every clause from a contract.

        Args:
            contract_text (str): The full contract text.

        Returns:
            list[ClauseObject]: Validated clauses in document order.

        Raises:
            ContractParseError: If the text is too short or no clause survives validation.
            ClauseExtractionError: If the model cannot be reached.
        """
        if not contract_text or len(contract_text.strip()) < MIN_CONTRACT_CHARS:
            raise ContractParseError(
                f"Contract text is empty or shorter than {MIN_CONTRACT_CHARS} characters - "
                f"nothing to analyse. Check the upload or PDF extraction."
            )
        text = contract_text.replace("\r\n", "\n")
        self.rejected_clauses = 0
        clauses = []
        for section in self._split_sections(text):
            raw_clauses = self._extract_section(section)
            clauses.extend(self._validate_clauses(raw_clauses, section, text))

        if not clauses:
            raise ContractParseError(
                "No clauses could be extracted and validated from this contract. "
                "The pipeline has stopped rather than report an empty analysis."
            )
        clauses.sort(key=lambda clause: clause.character_start)
        logger.info(
            "Extracted %s clauses (%s rejected by validation)",
            len(clauses),
            self.rejected_clauses,
        )
        return clauses

    def estimated_cost_gbp(self):
        """
        Estimate the cost of the calls made so far.

        Returns:
            float: Estimated cost in GBP.
        """
        usd = (
            self.usage["input_tokens"] * INPUT_USD_PER_MTOK
            + self.usage["output_tokens"] * OUTPUT_USD_PER_MTOK
        ) / 1_000_000
        return usd * USD_TO_GBP


def main():
    """Self-test: extract clauses from the sample contract and report cost."""
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format="%(levelname)s %(name)s: %(message)s",
    )
    contract = (PROJECT_ROOT / "data" / "sample_contract.txt").read_text(
        encoding="utf-8"
    )
    agent = ClauseExtractorAgent()
    clauses = agent.extract(contract)

    print(f"\nExtracted {len(clauses)} clauses | rejected: {agent.rejected_clauses}")
    print("Section numbers:", ", ".join(clause.section_number for clause in clauses))
    first = clauses[0]
    print(
        f"First clause: [{first.section_number}] {first.heading} | "
        f"start {first.character_start} | {first.word_count} words"
    )
    liability = [c for c in clauses if c.section_number == "9.1"]
    print(
        "Clause 9.1 found:",
        bool(liability),
        "| contains £10,000:",
        bool(liability) and "10,000" in liability[0].text,
    )
    print(
        f"Calls: {agent.usage['calls']} | tokens in/out: {agent.usage['input_tokens']} / "
        f"{agent.usage['output_tokens']} | est. cost: £{agent.estimated_cost_gbp():.4f}"
    )


if __name__ == "__main__":
    main()
