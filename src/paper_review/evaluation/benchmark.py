"""Benchmark topics for the baseline-vs-treatment harness.

arxiv_ids is a list, not a single ID -- the pipeline now synthesizes across
a set of papers rather than reviewing one. Single-paper topics still work
identically (a one-element list), so the two original topics didn't need
to change in kind, only in shape.
"""

from dataclasses import dataclass


@dataclass
class BenchmarkTopic:
    name: str
    topic_prompt: str  # fed to ReviewState.topic
    arxiv_ids: list[str]


BENCHMARK_TOPICS: list[BenchmarkTopic] = [
    BenchmarkTopic(
        name="transformer_attention",
        topic_prompt=(
            "How does the attention mechanism work in the Transformer, and how "
            "does multi-head attention differ from single-head attention?"
        ),
        arxiv_ids=["1706.03762"],  # Vaswani et al. -- Attention Is All You Need
    ),
    BenchmarkTopic(
        name="resnet_residual_learning",
        topic_prompt=(
            "How do residual connections in ResNet enable training of very deep "
            "neural networks, and what problem do they solve?"
        ),
        arxiv_ids=["1512.03385"],  # He et al. -- Deep Residual Learning for Image Recognition
    ),
    BenchmarkTopic(
        name="architectural_innovations_for_depth_and_context",
        topic_prompt=(
            "Compare how the Transformer and ResNet each solve a depth/context "
            "problem in deep learning -- one lets every layer see the whole "
            "sequence directly via attention, the other lets gradients skip "
            "layers via residual connections. What problem does each solve, "
            "and how do the two approaches differ?"
        ),
        arxiv_ids=["1706.03762", "1512.03385"],  # genuinely multi-paper synthesis
    ),
]
