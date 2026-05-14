from researchmind.tools.context_loader import (
    ContextLoader
)


def test_context():

    loader = (
        ContextLoader()
    )

    query = """
    related
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