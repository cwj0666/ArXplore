import pytest

from arxplore.retrieval.signals import looks_outline_like_text, looks_reference_like_text

BODY = (
    "The policy is trained to prefer chosen responses over rejected ones using a simple "
    "classification loss on pairwise comparisons. This removes the need for a separate reward model."
)


class TestReferenceLikeText:
    @pytest.mark.parametrize(
        "text",
        [
            "[1] A. Smith. Scaling laws.\n[2] B. Jones. Sparse routing.\n[3] C. Lee. Long context.",
            "  [12] A. Smith. Title.\n\n[13] B. Jones. Title.\n[14] C. Lee. Title.\n[15] D. Kim. Title.",
            "prior work [4] in 2019 and [5] in 2021 studied this",
            "Proceedings of the Conference on Vision 2020, and ICLR 2021",
            "Smith, J., Doe, A. Scaling laws. CVPR 2020.",
        ],
        ids=[
            "three-numbered-entries",
            "numbered-entries-with-indent",
            "two-cites-two-years",
            "two-venues-two-years",
            "author-list-year-venue",
        ],
    )
    def test_detected(self, text: str) -> None:
        assert looks_reference_like_text(text) is True

    @pytest.mark.parametrize(
        "text",
        [
            BODY,
            "",
            "We follow the setup of [1] published in 2020.",
            "Results at NeurIPS 2022 show gains.",
            "as shown in [1], [2] and [3] the method scales",
            "Video generation has advanced rapidly [4, 32, 9]. Diffusion models [7] and [10] and [12] generate clips.",
            "Table 3. COCO comparison\nTAPe+ML v3 84.7 65.3\nRF-DETR-2XL [2]\n78.5 60.1 126.9 M\nRF-DETR-M [2]\n"
            "73.6 54.7 33.7 M\nYOLO11-M [1] 64.1 48.6 20.1 M",
            "[1] one entry only\nas discussed in [2] and later in [3]",
        ],
        ids=[
            "plain-body",
            "empty",
            "one-cite-one-year",
            "one-venue-one-year",
            "in-text-cites-only",
            "citation-heavy-intro",
            "table-rows-with-cites",
            "one-entry-two-in-text-cites",
        ],
    )
    def test_not_detected(self, text: str) -> None:
        assert looks_reference_like_text(text) is False

    def test_only_the_first_1200_characters_are_inspected(self) -> None:
        tail = "\n[1] a\n[2] b\n[3] c"
        assert looks_reference_like_text("x" * 1100 + tail) is True
        assert looks_reference_like_text("x" * 1200 + tail) is False
        assert looks_reference_like_text("x " * 550 + " [1] 2019 [2] 2020") is True
        assert looks_reference_like_text("x " * 600 + " [1] 2019 [2] 2020") is False


class TestOutlineLikeText:
    @pytest.mark.parametrize(
        "text",
        [
            "The remainder of this paper is structured as below.",
            "This paper is organized as follows.",
            "We conclude in Section 6.",
            "Section 4 describes the experiments.",
        ],
    )
    def test_detected(self, text: str) -> None:
        assert looks_outline_like_text(text) is True

    def test_detection_ignores_whitespace_and_case(self) -> None:
        assert looks_outline_like_text("THE REST\nOF THIS   PAPER is ...") is True

    @pytest.mark.parametrize("text", ["Section 6 presents results.", "", BODY])
    def test_not_detected(self, text: str) -> None:
        assert looks_outline_like_text(text) is False
