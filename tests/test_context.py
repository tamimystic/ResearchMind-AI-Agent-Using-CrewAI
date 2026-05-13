from researchmind.tools.context_loader import (
    ContextLoader
)


def test_context():
    loader = (
        ContextLoader()
    )

    query = """
    Explain the methodology.

    Include:
    dataset,
    preprocessing,
    augmentation,
    architecture,
    feature extraction,
    training process,
    optimizer,
    evaluation metrics,
    final workflow.
    """

    context = (
        loader
        .load_context(
            query
        )
    )

    print(context)


if __name__ == "__main__":
    test_context()