import logging

from app.config import Settings
from app.services.prompts import FRAUD_ANALYST_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


class RAGChainService:
    """Generation stage that uses OpenAI/Anthropic via LangChain with a deterministic fallback."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def generate_answer(self, question: str, chunks: list[dict]) -> tuple[str, int | None]:
        if not chunks:
            return (
                "No relevant evidence was found in the indexed documents for this question.",
                None,
            )

        context = self._build_context(chunks)
        llm = self._build_llm()
        if llm is None:
            return self._fallback_answer(question, chunks), None

        try:
            from langchain_core.prompts import ChatPromptTemplate

            prompt = ChatPromptTemplate.from_template(FRAUD_ANALYST_SYSTEM_PROMPT)
            messages = prompt.format_messages(context=context, question=question)
            response = llm.invoke(messages)
            answer = self._extract_content(response)
            tokens_used = self._extract_tokens(response)
            return answer, tokens_used
        except Exception:
            logger.exception("LLM invocation failed; returning fallback answer")
            return self._fallback_answer(question, chunks), None

    def _build_llm(self):
        provider = self.settings.llm_provider.lower().strip()

        if provider == "openai":
            if not self.settings.openai_api_key:
                return None
            try:
                from langchain_openai import ChatOpenAI

                return ChatOpenAI(
                    model=self.settings.llm_model,
                    api_key=self.settings.openai_api_key,
                    temperature=0,
                )
            except Exception:
                logger.warning("Unable to initialize OpenAI chat model; falling back to deterministic response")
                return None

        if provider == "anthropic":
            if not self.settings.anthropic_api_key:
                return None
            try:
                from langchain_anthropic import ChatAnthropic

                kwargs = {
                    "model": self.settings.llm_model,
                    "api_key": self.settings.anthropic_api_key,
                    "temperature": 0,
                }
                if self.settings.anthropic_base_url:
                    kwargs["base_url"] = self.settings.anthropic_base_url

                return ChatAnthropic(
                    **kwargs,
                )
            except Exception:
                logger.warning("Unable to initialize Anthropic chat model; falling back to deterministic response")
                return None

        return None

    def _build_context(self, chunks: list[dict]) -> str:
        formatted_chunks: list[str] = []
        for index, chunk in enumerate(chunks, start=1):
            metadata = chunk.get("metadata", {})
            source = metadata.get("source", "unknown-source")
            page = metadata.get("page")
            location = f"page {page}" if page is not None else "section unknown"
            content = chunk.get("content", "")
            formatted_chunks.append(f"[{index}] {source} ({location})\n{content}")

        return "\n\n".join(formatted_chunks)

    def _fallback_answer(self, question: str, chunks: list[dict]) -> str:
        top_chunk = chunks[0]
        top_meta = top_chunk.get("metadata", {})
        source = top_meta.get("source", "unknown-source")
        page = top_meta.get("page")
        location = f"page {page}" if page is not None else "section unknown"

        evidence_lines = []
        for chunk in chunks:
            metadata = chunk.get("metadata", {})
            citation_source = metadata.get("source", "unknown-source")
            citation_page = metadata.get("page")
            citation = f"{citation_source} p.{citation_page}" if citation_page is not None else citation_source
            excerpt = chunk.get("content", "").strip().replace("\n", " ")
            evidence_lines.append(f"- [{citation}] {excerpt[:220]}")

        evidence = "\n".join(evidence_lines)
        return (
            f"Based on the indexed evidence, the strongest match for '{question}' appears in "
            f"{source} ({location}).\n\n"
            f"Supporting evidence:\n{evidence}"
        )

    @staticmethod
    def _extract_content(response) -> str:
        content = getattr(response, "content", response)
        if isinstance(content, list):
            return " ".join(str(item) for item in content).strip()
        return str(content).strip()

    @staticmethod
    def _extract_tokens(response) -> int | None:
        metadata = getattr(response, "response_metadata", None)
        if not isinstance(metadata, dict):
            return None

        token_usage = metadata.get("token_usage") or metadata.get("usage")
        if not isinstance(token_usage, dict):
            return None

        total = token_usage.get("total_tokens")
        if total is None:
            input_tokens = token_usage.get("input_tokens", 0)
            output_tokens = token_usage.get("output_tokens", 0)
            total = input_tokens + output_tokens

        try:
            return int(total)
        except (TypeError, ValueError):
            return None
