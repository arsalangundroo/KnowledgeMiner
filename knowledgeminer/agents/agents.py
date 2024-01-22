from langchain.agents import initialize_agent, AgentType
from .prompts.agent_prompts import MAIN_AGENT_PROMPT_PREFIX


def create_agent(llm, tools=[]):
    # TODO: create this in individual classes and put checks/validations on input parameter
    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True,
        agent_kwargs={
            'SYSTEM_MESSAGE_PREFIX': MAIN_AGENT_PROMPT_PREFIX,
        }
    )
    return agent
