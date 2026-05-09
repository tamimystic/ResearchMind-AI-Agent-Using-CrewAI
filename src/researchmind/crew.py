from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task


@CrewBase
class ResearchMindCrew:
    """ResearchMind AI Crew"""

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    # LLM Configuration
    general_llm = LLM(
        model="ollama/qwen3:4b",
        temperature=0.3
    )

    reasoning_llm = LLM(
        model="ollama/llama3.1",
        temperature=0.2
    )

    coding_llm = LLM(
        model="ollama/qwen2.5-coder:3b",
        temperature=0.1
    )

    # Agents
    @agent
    def orchestrator_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["orchestrator_agent"],
            llm=self.reasoning_llm,
            verbose=True,
            allow_delegation=False,
            memory=False
        )

    @agent
    def summary_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["summary_agent"],
            llm=self.general_llm,
            verbose=True,
            memory=False
        )

    @agent
    def methodology_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["methodology_agent"],
            llm=self.general_llm,
            verbose=True,
            memory=False
        )

    @agent
    def math_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["math_agent"],
            llm=self.reasoning_llm,
            verbose=True,
            memory=False
        )

    @agent
    def limitation_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["limitation_agent"],
            llm=self.general_llm,
            verbose=True,
            memory=False
        )

    @agent
    def related_papers_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["related_papers_agent"],
            llm=self.general_llm,
            verbose=True,
            memory=False
        )

    @agent
    def implementation_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["implementation_agent"],
            llm=self.general_llm,
            verbose=True,
            memory=False
        )

    @agent
    def code_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["code_agent"],
            llm=self.coding_llm,
            verbose=True,
            memory=False
        )

    @agent
    def quiz_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["quiz_agent"],
            llm=self.reasoning_llm,
            verbose=True,
            memory=False
        )

    @agent
    def report_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["report_agent"],
            llm=self.general_llm,
            verbose=True,
            memory=False
        )

    # Tasks
    @task
    def orchestrator_task(self) -> Task:
        return Task(
            config=self.tasks_config["orchestrator_task"]
        )

    @task
    def summary_task(self) -> Task:
        return Task(
            config=self.tasks_config["summary_task"]
        )

    @task
    def methodology_task(self) -> Task:
        return Task(
            config=self.tasks_config["methodology_task"]
        )

    @task
    def math_task(self) -> Task:
        return Task(
            config=self.tasks_config["math_task"]
        )

    @task
    def limitation_task(self) -> Task:
        return Task(
            config=self.tasks_config["limitation_task"]
        )

    @task
    def related_papers_task(self) -> Task:
        return Task(
            config=self.tasks_config["related_papers_task"]
        )

    @task
    def implementation_task(self) -> Task:
        return Task(
            config=self.tasks_config["implementation_task"]
        )

    @task
    def code_generation_task(self) -> Task:
        return Task(
            config=self.tasks_config["code_generation_task"]
        )

    @task
    def quiz_task(self) -> Task:
        return Task(
            config=self.tasks_config["quiz_task"]
        )

    @task
    def report_generation_task(self) -> Task:
        return Task(
            config=self.tasks_config["report_generation_task"]
        )

    # Crew
    @crew
    def crew(self) -> Crew:
        """Create ResearchMind Crew"""

        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
            memory=False
        )