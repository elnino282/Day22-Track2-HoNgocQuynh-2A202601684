"""Canonical prompt versions shared by A/B routing and RAGAS evaluation."""

from langchain_core.prompts import ChatPromptTemplate


SYSTEM_V1 = (
    "You are a friendly AI assistant. Answer the question using only facts in "
    "the supplied context. Give a direct, concise answer in 1-3 sentences. "
    "Do not use outside knowledge or invent details. If the context does not "
    "support an answer, say exactly: 'I cannot find this information in the "
    "provided context.'\n\nContext:\n{context}"
)

SYSTEM_V2 = (
    "You are a rigorous information-analysis expert. Use only the supplied "
    "context and never infer unsupported facts. Respond in 3 clearly labeled "
    "parts: 'Answer' with the direct answer, 'Evidence' with the supporting "
    "fact from the context, and 'Confidence' with High, Medium, or Low. Keep "
    "the response to 3-5 sentences. If evidence is missing, state exactly: "
    "'I cannot find this information in the provided context.'\n\n"
    "Context:\n{context}"
)


def make_prompt(system_message: str) -> ChatPromptTemplate:
    """Build the common two-message prompt shape used by both versions."""
    return ChatPromptTemplate.from_messages([
        ("system", system_message),
        ("human", "{question}"),
    ])


PROMPT_V1 = make_prompt(SYSTEM_V1)
PROMPT_V2 = make_prompt(SYSTEM_V2)
PROMPTS = {"v1": PROMPT_V1, "v2": PROMPT_V2}
