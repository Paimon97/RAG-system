from src.services.generator.generator_service import GenerationService
from src.services.generator.promt_builder import PromptBuilder
# from src.services.retriever.retriever_service import RetrievalService
from src.orchestration.rag_orchestrator import RAGOrchestrator
from src.pipeline.chunker import TextChunker
from src.services.embedder import EmbeddingService
from src.services.retriever.retriever import HybridRetriever
from src.services.generator.llm_generator import SafeLLMGenerator
from src.services.validator import HallucinationGuard
from src.pipeline.processor import DocumentProcessor
from src.utils.cache import CacheService

from concurrent.futures import ThreadPoolExecutor
import spacy

thread_pool = ThreadPoolExecutor(max_workers=4)
nlp = spacy.load("ru_core_news_sm")

embedder = EmbeddingService()
chunker = TextChunker(nlp_model=nlp)
cache = CacheService()
validator = HallucinationGuard(embedder=embedder)

retriever = HybridRetriever(embedder=embedder)
generator = SafeLLMGenerator()
prompt_builder = PromptBuilder()

processor = DocumentProcessor(
    nlp_model=nlp,
    chunker=chunker,
    embedder=embedder,
    retriever=retriever
)

# retrieval_service = RetrievalService()

generation_service = GenerationService(
    llm_generator=generator, 
    prompt_builder=prompt_builder, 
    thread_pool=thread_pool
)

orchestrator = RAGOrchestrator(
    retriever=retriever,
    generation_service=generation_service,
    validator=validator,
    cache=cache
)
