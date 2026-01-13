// State of the agent, make sure this aligns with your agent's state.
export type AgentState = {
  agent_name: string;
  proverbs: string[];
  searches: {
    query: string;
    done: boolean;
  }[];
};
