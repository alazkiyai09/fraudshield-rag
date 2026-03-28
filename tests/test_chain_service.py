import sys
import types

from app.config import Settings
from app.services.chain import RAGChainService


def test_generate_answer_returns_no_context_message():
    service = RAGChainService(Settings())
    answer, tokens = service.generate_answer("What happened?", [])
    assert "No relevant evidence" in answer
    assert tokens is None


def test_generate_answer_fallback_without_llm(monkeypatch):
    service = RAGChainService(Settings())
    monkeypatch.setattr(service, "_build_llm", lambda: None)

    chunks = [
        {
            "content": "Repeated near-threshold transfers from AC-77821.",
            "metadata": {"source": "report.pdf", "page": 3},
            "score": 0.9,
        }
    ]
    answer, tokens = service.generate_answer("What typology exists?", chunks)

    assert "report.pdf" in answer
    assert "Supporting evidence" in answer
    assert tokens is None


def test_generate_answer_with_mocked_langchain_prompt(monkeypatch):
    service = RAGChainService(Settings())

    class FakePromptTemplate:
        def __init__(self, template: str) -> None:
            self.template = template

        @classmethod
        def from_template(cls, template: str):
            return cls(template)

        def format_messages(self, **kwargs):
            return [self.template.format(**kwargs)]

    class FakeResponse:
        content = "Model answer"
        response_metadata = {"token_usage": {"total_tokens": 123}}

    class FakeLLM:
        def invoke(self, messages):
            assert isinstance(messages, list)
            return FakeResponse()

    monkeypatch.setitem(sys.modules, "langchain_core.prompts", types.SimpleNamespace(ChatPromptTemplate=FakePromptTemplate))
    monkeypatch.setattr(service, "_build_llm", lambda: FakeLLM())

    chunks = [
        {
            "content": "Velocity-control gap observed.",
            "metadata": {"source": "case.txt", "page": 1},
            "score": 0.8,
        }
    ]
    answer, tokens = service.generate_answer("Which control failed?", chunks)

    assert answer == "Model answer"
    assert tokens == 123


def test_extract_tokens_from_usage_structure():
    response = types.SimpleNamespace(response_metadata={"usage": {"input_tokens": 10, "output_tokens": 7}})
    assert RAGChainService._extract_tokens(response) == 17


def test_extract_content_handles_list_payload():
    response = types.SimpleNamespace(content=["part-1", "part-2"])
    assert RAGChainService._extract_content(response) == "part-1 part-2"


def test_build_llm_openai_provider_with_mock_module(monkeypatch):
    captured = {}

    class FakeChatOpenAI:
        def __init__(self, model: str, api_key: str, temperature: int) -> None:
            captured["model"] = model
            captured["api_key"] = api_key
            captured["temperature"] = temperature

    monkeypatch.setitem(sys.modules, "langchain_openai", types.SimpleNamespace(ChatOpenAI=FakeChatOpenAI))

    settings = Settings(llm_provider="openai", llm_model="gpt-4o-mini", openai_api_key="key")
    service = RAGChainService(settings)
    llm = service._build_llm()

    assert isinstance(llm, FakeChatOpenAI)
    assert captured["model"] == "gpt-4o-mini"
    assert captured["api_key"] == "key"
    assert captured["temperature"] == 0


def test_build_llm_anthropic_provider_with_mock_module(monkeypatch):
    class FakeChatAnthropic:
        def __init__(self, model: str, api_key: str, temperature: int) -> None:
            self.model = model
            self.api_key = api_key
            self.temperature = temperature

    monkeypatch.setitem(
        sys.modules,
        "langchain_anthropic",
        types.SimpleNamespace(ChatAnthropic=FakeChatAnthropic),
    )

    settings = Settings(llm_provider="anthropic", llm_model="claude-sonnet", anthropic_api_key="ant-key")
    service = RAGChainService(settings)
    llm = service._build_llm()

    assert isinstance(llm, FakeChatAnthropic)
    assert llm.model == "claude-sonnet"
    assert llm.api_key == "ant-key"
    assert llm.temperature == 0


def test_generate_answer_falls_back_when_prompt_render_fails(monkeypatch):
    service = RAGChainService(Settings())

    class BrokenPromptTemplate:
        @classmethod
        def from_template(cls, template: str):
            raise RuntimeError("prompt failed")

    class FakeLLM:
        def invoke(self, messages):
            return types.SimpleNamespace(content="should-not-be-used")

    monkeypatch.setitem(
        sys.modules,
        "langchain_core.prompts",
        types.SimpleNamespace(ChatPromptTemplate=BrokenPromptTemplate),
    )
    monkeypatch.setattr(service, "_build_llm", lambda: FakeLLM())

    chunks = [
        {
            "content": "Suspicious ring behavior.",
            "metadata": {"source": "investigation.txt"},
            "score": 0.8,
        }
    ]
    answer, tokens = service.generate_answer("What happened?", chunks)

    assert "Supporting evidence" in answer
    assert tokens is None
