// State of the agent, make sure this aligns with your agent's state.
export type AgentState = {
  proverbs: string[];
  searches: {
    query: string;
    done: boolean;
  }[];
};
