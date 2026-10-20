"""ENGRAM repository documentation MTEB Retrieval evaluation task definition."""

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata

from dataloaders.engram_docs import EngramDocsRetrievalData


class EngramDocsRetrievalTask(AbsTaskRetrieval):
    # Keep the document-side fallback instruction empty for models that resolve prompts by task name.
    abstask_prompt = ""

    def __init__(self, dataset_loader: EngramDocsRetrievalData, **kwargs):
        self.dataset_loader = dataset_loader

        self.metadata = TaskMetadata(
            dataset={
                "path": "local",
                "revision": "main",
            },  # Arbitrary, we are overriding dataset loading.
            name=f"engram_docs_{dataset_loader.name}_retrieval",
            description=(
                f"Retrieval over the ENGRAM documentation records of the "
                f"{dataset_loader.name} repository with developer questions about it"
            ),
            type="Retrieval",
            category="t2t",
            eval_langs=["eng-Latn"],
            # Five graded relevant records per query, so nDCG rather than MRR.
            main_score="ndcg_at_10",
            prompt={
                "query": "Given a developer question about a software repository, "
                "retrieve the documentation records that answer it"
            },
        )
        super().__init__(**kwargs)

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
        if self.data_loaded:
            return

        split = "test"  # We are only testing models, so each split is a test
        corpus = {
            document_id: {"title": "", "text": text}
            for document_id, text in self.dataset_loader.corpus.items()
        }

        self.queries = {split: dict(self.dataset_loader.queries)}
        self.corpus = {split: corpus}
        self.relevant_docs = {split: dict(self.dataset_loader.relevant_docs)}
        self.data_loaded = True
