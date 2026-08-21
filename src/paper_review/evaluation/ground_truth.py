"""Hand-labeled claims against arXiv 1706.03762 (Attention Is All You Need),
used to measure the verifier's own accuracy rather than the pipeline's.

Deliberately balanced across all three verdicts, including CONTRADICTED --
that's the label supported/unsupported eval sets easily forget, since a
fabricated claim confidently asserting the wrong number of heads is a much
worse failure to miss than a claim the evidence merely doesn't cover.
"""

from dataclasses import dataclass
from typing import Literal

Verdict = Literal["supported", "contradicted", "unsupported"]


@dataclass
class LabeledClaim:
    claim: str
    paper_id: str
    expected: Verdict
    note: str  # why this label -- what fact it turns on


GROUND_TRUTH: list[LabeledClaim] = [
    # -- supported: true, central facts stated directly in the paper --
    LabeledClaim(
        "The base Transformer model uses 8 attention heads.",
        "1706.03762", "supported", "h=8 in the base model config",
    ),
    LabeledClaim(
        "Each attention head operates on 64-dimensional queries, keys, and values.",
        "1706.03762", "supported", "d_k = d_v = d_model/h = 512/8 = 64",
    ),
    LabeledClaim(
        "The encoder is composed of a stack of 6 identical layers.",
        "1706.03762", "supported", "N=6 for the encoder",
    ),
    LabeledClaim(
        "Attention weights are computed with a softmax over scaled dot products of queries and keys.",
        "1706.03762", "supported", "the scaled dot-product attention formula",
    ),
    LabeledClaim(
        "Positional encodings are added to the input embeddings at the bottom of the encoder and decoder stacks.",
        "1706.03762", "supported", "stated explicitly in section 3.5",
    ),
    LabeledClaim(
        "The model was trained using the Adam optimizer.",
        "1706.03762", "supported", "section 5.3, training regime",
    ),
    # -- contradicted: confident, plausible-sounding, but factually inverted --
    LabeledClaim(
        "The base Transformer model uses 16 attention heads.",
        "1706.03762", "contradicted", "actually 8 -- inverted number",
    ),
    LabeledClaim(
        "Each attention head operates on 128-dimensional queries and keys.",
        "1706.03762", "contradicted", "actually 64 -- inverted number",
    ),
    LabeledClaim(
        "The encoder is composed of a stack of 12 identical layers.",
        "1706.03762", "contradicted", "actually 6 -- inverted number",
    ),
    LabeledClaim(
        "Attention weights are computed using cosine similarity between queries and keys.",
        "1706.03762", "contradicted", "actually scaled dot-product, not cosine similarity",
    ),
    LabeledClaim(
        "The Transformer relies on convolutional layers to extract local features.",
        "1706.03762", "contradicted", "the paper explicitly eschews convolution entirely",
    ),
    LabeledClaim(
        "Positional information is learned only -- the sinusoidal encoding was never tried.",
        "1706.03762", "contradicted", "the paper tries and compares both variants",
    ),
    # -- unsupported: plausible, but this paper never addresses it --
    LabeledClaim(
        "The model was evaluated on the SQuAD question-answering benchmark.",
        "1706.03762", "unsupported", "paper evaluates WMT translation + constituency parsing only",
    ),
    LabeledClaim(
        "The model was fine-tuned using reinforcement learning from human feedback.",
        "1706.03762", "unsupported", "RLHF postdates this 2017 paper",
    ),
    LabeledClaim(
        "The model was pretrained on a large web-scale corpus before task-specific fine-tuning.",
        "1706.03762", "unsupported", "pretrain/fine-tune paradigm isn't this paper's setup",
    ),
    LabeledClaim(
        "The architecture supports multimodal image-and-text inputs.",
        "1706.03762", "unsupported", "paper is text-only machine translation",
    ),
]
