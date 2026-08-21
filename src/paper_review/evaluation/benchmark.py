"""Benchmark topics for the baseline-vs-treatment harness.

One arxiv_id per topic -- ReviewState only scopes retrieval to a single
paper_id today (agents/graph.py), so a multi-paper topic cluster would
silently only ever retrieve from whichever paper_id got passed in. Extending
retrieval to a paper_id list is a real future improvement, not done here.
"""

from dataclasses import dataclass


@dataclass
class BenchmarkTopic:
    name: str
    topic_prompt: str  # fed to ReviewState.topic
    arxiv_id: str


BENCHMARK_TOPICS: list[BenchmarkTopic] = [
    BenchmarkTopic(
        name="transformer_attention",
        topic_prompt=(
            "How does the attention mechanism work in the Transformer, and how "
            "does multi-head attention differ from single-head attention?"
        ),
        arxiv_id="1706.03762",  # Vaswani et al. -- Attention Is All You Need
    ),
    BenchmarkTopic(
        name="resnet_residual_learning",
        topic_prompt=(
            "How do residual connections in ResNet enable training of very deep "
            "neural networks, and what problem do they solve?"
        ),
        arxiv_id="1512.03385",  # He et al. -- Deep Residual Learning for Image Recognition
    ),
]
