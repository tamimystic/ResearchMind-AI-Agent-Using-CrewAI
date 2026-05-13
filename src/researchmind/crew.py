from crewai import (
    Agent,
    Crew,
    Process,
    Task,
    LLM
)

from crewai.project import (
    CrewBase,
    agent,
    crew,
    task
)

from dotenv import (
    load_dotenv
)

import os

from researchmind.tools.context_loader import (
    ContextLoader
)

load_dotenv()

groq_api_key = os.getenv(
    "GROQ_API_KEY"
)

openrouter_api_key = os.getenv(
    "OPENROUTER_API_KEY"
)

cerebras_api_key = os.getenv(
    "CEREBRAS_API_KEY"
)

required_keys = {
    "GROQ_API_KEY":
    groq_api_key,

    "OPENROUTER_API_KEY":
    openrouter_api_key,

    "CEREBRAS_API_KEY":
    cerebras_api_key
}

missing_keys = [
    key
    for key, value
    in required_keys.items()
    if not value
]

if missing_keys:
    raise ValueError(
        f"Missing API keys in .env: "
        f"{', '.join(missing_keys)}"
    )


@CrewBase
class ResearchMindCrew:

    agents_config = (
        "config/agents.yaml"
    )

    tasks_config = (
        "config/tasks.yaml"
    )

    context_loader = (
        ContextLoader()
    )

    general_llm = LLM(
        model="groq/llama-3.1-8b-instant",
        api_key=groq_api_key,
        temperature=0.2
    )

    reasoning_llm = LLM(
        model="cerebras/llama3.1-8b",
        api_key=cerebras_api_key,
        temperature=0.1
    )

    research_llm = LLM(
        model="openrouter/deepseek/deepseek-chat",
        api_key=openrouter_api_key,
        temperature=0.2
    )

    coding_llm = LLM(
        model="ollama/qwen2.5-coder:3b",
        base_url="http://localhost:11434",
        temperature=0.1
    )

    local_llm = LLM(
        model="ollama/qwen2.5:3b",
        base_url="http://localhost:11434",
        temperature=0.2
    )

    heavy_local_llm = LLM(
        model="ollama/llama3.1",
        base_url="http://localhost:11434",
        temperature=0.2
    )

    @agent
    def summary_agent(self):
        return Agent(
            config=self.agents_config[
                "summary_agent"
            ],
            llm=self.general_llm,
            verbose=False
        )

    @agent
    def methodology_agent(self):
        return Agent(
            config=self.agents_config[
                "methodology_agent"
            ],
            llm=self.research_llm,
            verbose=False
        )

    @agent
    def math_agent(self):
        return Agent(
            config=self.agents_config[
                "math_agent"
            ],
            llm=self.reasoning_llm,
            verbose=False
        )

    @agent
    def limitation_agent(self):
        return Agent(
            config=self.agents_config[
                "limitation_agent"
            ],
            llm=self.general_llm,
            verbose=False
        )

    @agent
    def related_papers_agent(self):
        return Agent(
            config=self.agents_config[
                "related_papers_agent"
            ],
            llm=self.heavy_local_llm,
            verbose=False
        )

    @agent
    def implementation_agent(self):
        return Agent(
            config=self.agents_config[
                "implementation_agent"
            ],
            llm=self.heavy_local_llm,
            verbose=False
        )

    @agent
    def code_agent(self):
        return Agent(
            config=self.agents_config[
                "code_agent"
            ],
            llm=self.coding_llm,
            verbose=False
        )

    @agent
    def quiz_agent(self):
        return Agent(
            config=self.agents_config[
                "quiz_agent"
            ],
            llm=self.local_llm,
            verbose=False
        )

    @agent
    def report_agent(self):
        return Agent(
            config=self.agents_config[
                "report_agent"
            ],
            llm=self.heavy_local_llm,
            verbose=False
        )

    @task
    def summary_task(self):

        context = (
            self.context_loader
            .load_context(
                "summary"
            )
        )

        return Task(
            description=f"""
            Summarize the uploaded
            research paper.

            Use ONLY this context:

            {context}
            """,
            expected_output="""
            Structured research summary
            with objective, methodology,
            findings, contribution and
            conclusion.
            """,
            agent=self.summary_agent()
        )

    @task
    def methodology_task(self):

        context = (
            self.context_loader
            .load_context(
                "methodology"
            )
        )

        return Task(
            description=f"""
            Explain methodology
            using ONLY this context.

            {context}
            """,
            expected_output="""
            Methodology explanation
            including dataset,
            preprocessing,
            architecture,
            training and evaluation.
            """,
            agent=self.methodology_agent()
        )

    @task
    def math_task(self):

        context = (
            self.context_loader
            .load_context(
                "math"
            )
        )

        return Task(
            description=f"""
            Explain mathematical
            concepts using ONLY
            this context.

            {context}
            """,
            expected_output="""
            Mathematical explanation
            including equations,
            formulas, variables,
            optimization and
            loss functions.
            """,
            agent=self.math_agent()
        )

    @task
    def limitation_task(self):

        context = (
            self.context_loader
            .load_context(
                "limitation"
            )
        )

        return Task(
            description=f"""
            Analyze limitations
            using ONLY this context.

            {context}
            """,
            expected_output="""
            Limitation analysis
            with weaknesses,
            risks, scalability,
            and future work.
            """,
            agent=self.limitation_agent()
        )

    @task
    def related_papers_task(self):

        context = (
            self.context_loader
            .load_context(
                "related"
            )
        )

        return Task(
            description=f"""
            Suggest related papers
            using ONLY this context.

            {context}
            """,
            expected_output="""
            Related papers with
            relevance explanation.
            """,
            agent=self.related_papers_agent()
        )

    @task
    def implementation_task(self):

        context = (
            self.context_loader
            .load_context(
                "implementation"
            )
        )

        return Task(
            description=f"""
            Create implementation
            roadmap using ONLY
            this context.

            {context}
            """,
            expected_output="""
            Step-by-step practical
            implementation roadmap.
            """,
            agent=self.implementation_agent()
        )

    @task
    def code_generation_task(self):

        context = (
            self.context_loader
            .load_context(
                "architecture"
            )
        )

        return Task(
            description=f"""
            Generate implementation
            code using ONLY
            this context.

            {context}
            """,
            expected_output="""
            Clean modular
            implementation code.
            """,
            agent=self.code_agent()
        )

    @task
    def quiz_task(self):

        context = (
            self.context_loader
            .load_context(
                "summary"
            )
        )

        return Task(
            description=f"""
            Create educational
            questions using ONLY
            this context.

            {context}
            """,
            expected_output="""
            MCQ, short and viva
            questions from paper.
            """,
            agent=self.quiz_agent()
        )

    @task
    def report_generation_task(self):

        context = (
            self.context_loader
            .load_context(
                "report"
            )
        )

        return Task(
            description=f"""
            Create a final
            research report
            using ONLY this
            context.

            {context}
            """,
            expected_output="""
            Final professional
            research report.
            """,
            agent=self.report_agent()
        )

    @crew
    def crew(self):
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
            memory=False
        )