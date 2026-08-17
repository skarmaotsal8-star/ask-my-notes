from __future__ import annotations

import os


def answer_question(question: str, sources: list[dict], provider: str, model: str) -> str:
    context = "\n\n".join(f"[Source {item['id']}: {item['label']}]\n{item['text']}" for item in sources)
    prompt = f"""Answer the user's question using only the supplied context.
If the context does not answer the question, say so plainly. Do not invent facts.
Cite every factual claim with the relevant source marker, such as [Source 1].

Context:
{context}

Question: {question}"""

    if provider == "openai":
        from openai import OpenAI

        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("Set OPENAI_API_KEY in your environment or .env file.")
        response = OpenAI().chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": "You are a precise research assistant."}, {"role": "user", "content": prompt}],
            temperature=0.2,
        )
        return response.choices[0].message.content or "No answer was generated."

    if provider == "anthropic":
        from anthropic import Anthropic

        if not os.getenv("ANTHROPIC_API_KEY"):
            raise RuntimeError("Set ANTHROPIC_API_KEY in your environment or .env file.")
        response = Anthropic().messages.create(
            model=model,
            max_tokens=900,
            temperature=0.2,
            system="You are a precise research assistant.",
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(block.text for block in response.content if hasattr(block, "text"))

    raise ValueError("Choose either OpenAI or Anthropic as the provider.")
