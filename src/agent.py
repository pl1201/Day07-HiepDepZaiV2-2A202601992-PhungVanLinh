from typing import Callable

from .store import EmbeddingStore


class KnowledgeBaseAgent:
    """
    An agent that answers questions using a vector knowledge base.

    Retrieval-augmented generation (RAG) pattern:
        1. Retrieve top-k relevant chunks from the store.
        2. Build a prompt with the chunks as context.
        3. Call the LLM to generate an answer.
    """

    def __init__(self, store: EmbeddingStore, llm_fn: Callable[[str], str]) -> None:
        self.store = store
        self.llm_fn = llm_fn

    def answer(self, question: str, top_k: int = 3) -> str:
        results = self.store.search(question, top_k=top_k)
        if results:
            context_blocks = []
            for index, result in enumerate(results, start=1):
                metadata = result.get("metadata", {})
                source = metadata.get("source_url") or metadata.get("source") or result.get("id", "unknown")
                context_blocks.append(f"[{index}] Nguồn: {source}\n{result['content']}")
            context = "\n\n".join(context_blocks)
        else:
            context = "Không tìm thấy ngữ cảnh liên quan trong cơ sở tri thức."

        prompt = (
            "Bạn là trợ lý tra cứu quy định đại học. Chỉ trả lời dựa trên NGỮ CẢNH; "
            "nếu dữ liệu không đủ, hãy nói rõ là chưa đủ thông tin. Trích dẫn nguồn bằng "
            "ký hiệu [1], [2] tương ứng và không tự suy đoán.\n\n"
            f"NGỮ CẢNH:\n{context}\n\nCÂU HỎI: {question}\n\nTRẢ LỜI:"
        )
        return self.llm_fn(prompt)
