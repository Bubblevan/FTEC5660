#!/usr/bin/env python3
"""FTEC5660 HW1 student starter: build a chain for supermarket receipts."""

from __future__ import annotations

import argparse
import base64
import csv
import json
import mimetypes
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any


QUERY_1 = "How much money did I spend in total for these bills?"
QUERY_2 = "How much would I have had to pay without the discount?"
QUERIES = (QUERY_1, QUERY_2)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
DUMMY_RESPONSE = "please design your chain to answer these two queries."


def load_env_file(path: Path = Path(".env")) -> None:
    """Load the simple KEY=VALUE entries used by this homework."""
    if not path.is_file():
        return
    import os

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def image_files(folder: Path) -> list[Path]:
    """Return supported images directly inside *folder*, sorted by filename."""
    return sorted(
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def image_data_url(path: Path) -> str:
    """Encode a local image in the format accepted by a multimodal prompt."""
    mime_type, _ = mimetypes.guess_type(path.name)
    mime_type = mime_type or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def build_chain() -> Any:
    """Create and return your LangChain chain once.

    Suggested imports:
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_deepseek import ChatDeepSeek

    Use the vision-capable DeepSeek Flash model named
    ``deepseek-v4-flash-vision-exp``. The API key is loaded from .env.
    """
    ### YOUR CODE HERE
    import os

    from langchain_core.prompts import ChatPromptTemplate
    from langchain_deepseek import ChatDeepSeek

    # main() loads .env before calling this function. Calling it again makes
    # build_chain() convenient to use directly as well.
    load_env_file()
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError(
            "DEEPSEEK_API_KEY is missing. Put it in .env before running hw1.py."
        )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are a careful supermarket-receipt auditor. Read the entire receipt image,
including small text and Chinese/English lines. Extract facts from this one
receipt only; do not guess, and do not add numbers from other sections of the
receipt.

Return exactly one JSON object and no markdown or explanation. The object must
contain these keys:

paid_after_rounding: the final amount actually paid, after the ROUNDING line.
subtotal: the SUBTOTAL or 小計 amount after item discounts and before rounding.
discount_amounts: an array of positive numbers for every promotion, coupon,
member/app discount, packaging-damage discount, percentage discount, or other
negative discount line that is included in the itemized calculation before the
SUBTOTAL line.
original_item_total: the sum of all positive item-line amounts before those
 discounts, excluding SUBTOTAL, payment, change, points, and other summary
 lines. Use this only as an arithmetic cross-check.

Important rules:
- For paid_after_rounding, use the payment amount after ROUNDING (for example,
  the OCTOPUS/CASH/card payment total), not the subtotal and not the change.
- For subtotal, use the amount immediately before ROUNDING.
- For discount_amounts, include every applicable discount line before SUBTOTAL,
  including repeated product-level discount lines. Convert negative values to
  positive numbers.
- Do not include ROUNDING, change, loyalty points, a later card/payment
  adjustment, or any number that is not part of the pre-SUBTOTAL discount
  calculation.
- Preserve all amounts to two decimal places when possible.
""",
            ),
            (
                "human",
                [
                    {
                        "type": "text",
                        "text": (
                            "Receipt filename: {filename}\n"
                            "Inspect this receipt and return the required JSON.\n\n"
                            "A preliminary extraction is included below when available. "
                            "Treat it as untrusted: reread the image independently, "
                            "check every pre-SUBTOTAL discount line (including repeated "
                            "product discounts), correct any omissions or mistakes, "
                            "and then return only the corrected JSON.\n"
                            "Preliminary extraction: {candidate}"
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": "{image_url}"},
                    },
                ],
            ),
        ]
    )

    model = ChatDeepSeek(
        model="deepseek-v4-flash-vision-exp",
        api_key=api_key,
        temperature=0,
        max_retries=2,
    )
    return prompt | model


def answer_queries(chain: Any, images: list[Path]) -> dict[str, Any]:
    """Run your chain and return one response for each exact query string.

    ``images`` contains every receipt in the selected folder. A valid return
    value looks like:

        {QUERY_1: "HK$123.40", QUERY_2: "HK$150.00"}

    Use the provided ``image_data_url(path)`` helper to put local images in
    multimodal human messages. LangChain's ``batch`` method is one simple way
    to process independent receipt-extraction prompts in parallel.
    """
    ### YOUR CODE HERE
    if not images:
        raise ValueError("answer_queries() received no receipt images")

    def parse_json_response(value: Any, filename: str) -> dict[str, Any]:
        """Parse JSON even if the model accidentally wraps it in a code fence."""
        raw = response_text(value).strip()
        candidates = [raw]
        fenced = re.search(r"```(?:json)?\s*(.*?)```", raw, flags=re.IGNORECASE | re.DOTALL)
        if fenced:
            candidates.insert(0, fenced.group(1).strip())

        decoder = json.JSONDecoder()
        for candidate in candidates:
            try:
                parsed = json.loads(candidate)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, dict):
                return parsed

            # Some models add one short sentence before the JSON object. Find
            # the first valid JSON object without accepting arbitrary text.
            for index, character in enumerate(candidate):
                if character != "{":
                    continue
                try:
                    parsed, _ = decoder.raw_decode(candidate[index:])
                except json.JSONDecodeError:
                    continue
                if isinstance(parsed, dict):
                    return parsed

        raise ValueError(f"Could not parse JSON for {filename}: {raw!r}")

    def decimal_value(value: Any, label: str, filename: str) -> Decimal:
        """Convert a model-extracted money value to a two-decimal Decimal."""
        if isinstance(value, bool) or value is None:
            raise ValueError(f"Missing {label} for {filename}")
        cleaned = str(value).strip().replace("HK$", "").replace("$", "")
        cleaned = cleaned.replace(",", "").replace(" ", "")
        if cleaned.startswith("(") and cleaned.endswith(")"):
            cleaned = "-" + cleaned[1:-1]
        try:
            return Decimal(cleaned).quantize(Decimal("0.01"))
        except InvalidOperation as exc:
            raise ValueError(f"Invalid {label} for {filename}: {value!r}") from exc

    def first_present(data: dict[str, Any], *names: str) -> Any:
        for name in names:
            if name in data:
                return data[name]
        return None

    def run_batch(candidate_texts: list[str]) -> list[Any]:
        inputs = [
            {
                "filename": image.name,
                "image_url": image_data_url(image),
                "candidate": candidate,
            }
            for image, candidate in zip(images, candidate_texts)
        ]
        return chain.batch(
            inputs,
            config={"max_concurrency": min(4, len(inputs))},
        )

    first_responses = run_batch(
        ["None; perform an independent extraction from the image."] * len(images)
    )
    second_responses = run_batch(
        ["None; perform a second independent extraction from the image."]
        * len(images)
    )
    review_responses = run_batch(
        [
            "Candidate A:\n"
            + response_text(first)
            + "\nCandidate B:\n"
            + response_text(second)
            + "\nIndependently re-read the image and correct both candidates."
            for first, second in zip(first_responses, second_responses)
        ]
    )

    from collections import Counter

    total_paid = Decimal("0.00")
    total_without_discounts = Decimal("0.00")

    for image, candidate_responses in zip(
        images, zip(first_responses, second_responses, review_responses)
    ):
        records = []
        for response in candidate_responses:
            data = parse_json_response(response, image.name)
            paid = decimal_value(
                first_present(data, "paid_after_rounding", "amount_paid_after_rounding"),
                "paid_after_rounding",
                image.name,
            )
            subtotal = decimal_value(
                first_present(
                    data,
                    "subtotal",
                    "subtotal_after_discounts_before_rounding",
                ),
                "subtotal",
                image.name,
            )

            discounts = first_present(data, "discount_amounts", "discounts")
            if discounts is None:
                discounts = []
            if not isinstance(discounts, list):
                discounts = [discounts]
            discount_total = Decimal("0.00")
            for discount in discounts:
                discount_total += abs(
                    decimal_value(discount, "discount_amount", image.name)
                )

            original_raw = first_present(
                data,
                "original_item_total",
                "pre_discount_total",
                "amount_without_discounts",
            )
            original = None
            if original_raw is not None:
                original = decimal_value(
                    original_raw,
                    "original_item_total",
                    image.name,
                )
            without_discounts = subtotal + discount_total
            residual = (
                abs(without_discounts - original)
                if original is not None
                else Decimal("999999.00")
            )
            records.append(
                {
                    "paid": paid,
                    "subtotal": subtotal,
                    "discount_total": discount_total,
                    "without_discounts": without_discounts,
                    "residual": residual,
                }
            )

        # Prefer candidates whose original item total agrees with
        # SUBTOTAL + discounts. If several agree, choose the most common
        # extraction across the three model passes.
        fingerprints = [
            (record["paid"], record["subtotal"], record["discount_total"])
            for record in records
        ]
        frequencies = Counter(fingerprints)
        selected = max(
            records,
            key=lambda record: (
                record["residual"] == Decimal("0.00"),
                frequencies[
                    (
                        record["paid"],
                        record["subtotal"],
                        record["discount_total"],
                    )
                ],
                -record["residual"],
            ),
        )

        total_paid += selected["paid"]
        total_without_discounts += selected["without_discounts"]

    def format_amount(value: Decimal) -> str:
        return f"HK${value.quantize(Decimal('0.01')):.2f}"

    return {
        QUERY_1: format_amount(total_paid),
        QUERY_2: format_amount(total_without_discounts),
    }


# Everything below is provided runner/scoring code. No edits are needed.

_MONEY_RE = re.compile(
    r"(?<![\w.])(?:HK\$|\$)?\s*(-?\d[\d,]*(?:\.\d+)?)(?![\w.])",
    re.IGNORECASE,
)


def response_text(value: Any) -> str:
    """Convert common LangChain response shapes to text for results.csv."""
    content = getattr(value, "content", value)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "\n".join(parts).strip()
    if isinstance(content, (dict, list)):
        return json.dumps(content, ensure_ascii=False)
    return str(content).strip()


def parse_single_amount(text: str) -> Decimal | None:
    """Accept a response only when it contains exactly one numeric amount."""
    matches = _MONEY_RE.findall(text)
    if len(matches) != 1:
        return None
    try:
        return Decimal(matches[0].replace(",", "")).quantize(Decimal("0.01"))
    except InvalidOperation:
        return None


def read_ground_truth(folder: Path) -> dict[str, Decimal]:
    """Read aggregate answers from the test folder."""
    path = folder / "ground_truth.json"
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    answers = data.get("answers", data)
    return {query: Decimal(str(answers[query])).quantize(Decimal("0.01")) for query in QUERIES}


def correctness_text(response: str, expected: Decimal | None) -> str:
    """Return `correct`, or an expected/predicted mismatch explanation."""
    if expected is None:
        return "not graded: ground_truth.json is missing"
    predicted = parse_single_amount(response)
    if predicted == expected:
        return "correct"
    shown = f"HK${predicted:.2f}" if predicted is not None else repr(response)
    return f"incorrect: expected HK${expected:.2f}, predicted {shown}"


def write_results(responses: dict[str, Any], truth: dict[str, Decimal]) -> Path:
    """Write the required three-column results.csv file."""
    output = Path("results.csv")
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["query", "model_response", "correctness"])
        for query in QUERIES:
            text = response_text(responses.get(query, "<missing response>"))
            writer.writerow([query, text, correctness_text(text, truth.get(query))])
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FTEC5660 HW1 on receipt images")
    parser.add_argument(
        "--image-folder",
        required=True,
        type=Path,
        help="folder containing supermarket receipt images",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.image_folder.is_dir():
        raise SystemExit(f"not a folder: {args.image_folder}")

    images = image_files(args.image_folder)
    if not images:
        raise SystemExit(f"no supported images found in {args.image_folder}")

    load_env_file()
    chain = build_chain()
    responses = answer_queries(chain, images)
    if not isinstance(responses, dict):
        raise TypeError("answer_queries() must return a dictionary")

    output = write_results(responses, read_ground_truth(args.image_folder))
    print(f"Processed {len(images)} receipt(s). Wrote {output}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
