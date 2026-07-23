"""test_targets/lcel_chain — a LangChain LCEL pipe-chain target (the `|` idiom).

`library_chain` is entirely langchain_core/langchain_openai objects — by design it must degrade
explicitly (is_library_object=True on every link), not vanish silently. `hybrid_chain` adds one
RunnableLambda-wrapped user function (`postprocess`) — real target code that must surface as a
harvestable component (is_library_object=False).

The factory-function idiom is exercised by a real cloned LangChain app in the scanner dogfood test,
not here. Nothing in this file is imported by AEH — the passes are pure ast.parse — so
langchain_core/langchain_openai need not be installed for the tests to run.
"""
from __future__ import annotations

from dataclasses import dataclass

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI

prompt = PromptTemplate.from_template("Answer the question: {question}")


def postprocess(text: str) -> str:
    """Real target code: trims and lowercases the model's raw answer."""
    return text.strip().lower()


library_chain = prompt | ChatOpenAI(model="gpt-4o-mini") | StrOutputParser()

hybrid_chain = (
    prompt | ChatOpenAI(model="gpt-4o-mini") | RunnableLambda(postprocess) | StrOutputParser()
)


@dataclass
class RetryPolicy:
    """A `str | None` annotation in the SAME file as a real LCEL chain — proves the PEP-604 guard
    still holds when both shapes coexist."""

    backoff_strategy: str | None = None


class LogLevel:
    DEBUG = 1
    INFO = 2


# Flag-style OR — must NOT be treated as an LCEL chain (the numpy/flag-enum guard).
_verbose_levels = LogLevel.DEBUG | LogLevel.INFO


def run(question: str) -> str:
    return hybrid_chain.invoke({"question": question})
