from researchmind.crew import (
    ResearchMindCrew
)


def run():
    print("\n" + "=" * 80)
    print("ResearchMind AI")
    print("Multi-Agent Research Intelligence System")
    print("=" * 80)

    print(
        "\nStarting ResearchMind workflow...\n"
    )

    try:
        result = (
            ResearchMindCrew()
            .crew()
            .kickoff(
                inputs={}
            )
        )

        print("\n" + "=" * 80)
        print(
            "ResearchMind Final Report"
        )
        print("=" * 80)

        print("\n")
        print(result)

    except Exception as error:
        print("\n" + "=" * 80)
        print("Workflow Failed")
        print("=" * 80)

        print(
            f"\nError: {str(error)}"
        )


if __name__ == "__main__":
    run()