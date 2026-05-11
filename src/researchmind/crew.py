from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from dotenv import load_dotenv
import os

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
cerebras_api_key = os.getenv("CEREBRAS_API_KEY")

required_keys = {
    "GROQ_API_KEY": groq_api_key,
    "OPENROUTER_API_KEY": openrouter_api_key,
    "CEREBRAS_API_KEY": cerebras_api_key
}

missing_keys = [
    key for key, value in required_keys.items()
    if not value
]

if missing_keys:
    raise ValueError(
        f"Missing API keys in .env: {', '.join(missing_keys)}"
    )


@CrewBase
class ResearchMindCrew:
    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

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

    report_llm = LLM(
        model="ollama/qwen3:4b",
        base_url="http://localhost:11434",
        temperature=0.2
    )

    @agent
    def summary_agent(self):
        return Agent(
            config=self.agents_config["summary_agent"],
            llm=self.general_llm,
            verbose=False,
            memory=False
        )

    @agent
    def methodology_agent(self):
        return Agent(
            config=self.agents_config["methodology_agent"],
            llm=self.research_llm,
            verbose=False,
            memory=False
        )

    @agent
    def math_agent(self):
        return Agent(
            config=self.agents_config["math_agent"],
            llm=self.reasoning_llm,
            verbose=False,
            memory=False
        )

    @agent
    def limitation_agent(self):
        return Agent(
            config=self.agents_config["limitation_agent"],
            llm=self.general_llm,
            verbose=False,
            memory=False
        )

    @agent
    def related_papers_agent(self):
        return Agent(
            config=self.agents_config["related_papers_agent"],
            llm=self.research_llm,
            verbose=False,
            memory=False
        )

    @agent
    def implementation_agent(self):
        return Agent(
            config=self.agents_config["implementation_agent"],
            llm=self.general_llm,
            verbose=False,
            memory=False
        )

    @agent
    def code_agent(self):
        return Agent(
            config=self.agents_config["code_agent"],
            llm=self.coding_llm,
            verbose=False,
            memory=False
        )

    @agent
    def quiz_agent(self):
        return Agent(
            config=self.agents_config["quiz_agent"],
            llm=self.local_llm,
            verbose=False,
            memory=False
        )

    @agent
    def report_agent(self):
        return Agent(
            config=self.agents_config["report_agent"],
            llm=self.report_llm,
            verbose=False,
            memory=False
        )

    @task
    def summary_task(self):
        return Task(
            config=self.tasks_config["summary_task"]
        )

    @task
    def methodology_task(self):
        return Task(
            config=self.tasks_config["methodology_task"]
        )

    @task
    def math_task(self):
        return Task(
            config=self.tasks_config["math_task"]
        )

    @task
    def limitation_task(self):
        return Task(
            config=self.tasks_config["limitation_task"]
        )

    @task
    def related_papers_task(self):
        return Task(
            config=self.tasks_config["related_papers_task"]
        )

    @task
    def implementation_task(self):
        return Task(
            config=self.tasks_config["implementation_task"]
        )

    @task
    def code_generation_task(self):
        return Task(
            config=self.tasks_config["code_generation_task"]
        )

    @task
    def quiz_task(self):
        return Task(
            config=self.tasks_config["quiz_task"]
        )

    @task
    def report_generation_task(self):
        return Task(
            config=self.tasks_config["report_generation_task"]
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