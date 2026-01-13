"use client";

import { MoonCard } from "@/components/moon";
import { ProverbsCard } from "@/components/proverbs";
import { WeatherCard } from "@/components/weather";
import { AgentState } from "@/lib/types";
import {
  useCoAgent,
  useFrontendTool,
  useHumanInTheLoop,
  useRenderToolCall,
} from "@copilotkit/react-core";
import { useAgent } from "@copilotkit/react-core/v2";
import { CopilotKitCSSProperties, CopilotSidebar } from "@copilotkit/react-ui";
import { useState } from "react";

export default function CopilotKitPage() {
  const [themeColor, setThemeColor] = useState("#6366f1");

  // 🪁 Frontend Actions: https://docs.copilotkit.ai/pydantic-ai/frontend-actions
  useFrontendTool({
    name: "setThemeColor",
    parameters: [
      {
        name: "themeColor",
        description: "The theme color to set. Make sure to pick nice colors.",
        required: true,
      },
    ],
    handler({ themeColor }) {
      setThemeColor(themeColor);
    },
  });

  return (
    <main
      style={
        { "--copilot-kit-primary-color": themeColor } as CopilotKitCSSProperties
      }
    >
      <CopilotSidebar
        defaultOpen={true}
        disableSystemMessage={true}
        clickOutsideToClose={false}
        labels={{
          title: "Popup Assistant",
          initial: "👋 Hi, there! You're chatting with an agent.",
        }}
        suggestions={[
          {
            title: "Generative UI",
            message: "Get the weather in San Francisco.",
          },
          {
            title: "Frontend Tools",
            message: "Set the theme to green.",
          },
          {
            title: "Human In the Loop",
            message: "Please go to the moon.",
          },
          {
            title: "Write Agent State",
            message: "Add a proverb about AI.",
          },
          {
            title: "Update Agent State",
            message:
              "Please remove 1 random proverb from the list if there are any.",
          },
          {
            title: "Read Agent State",
            message: "What are the proverbs?",
          },
        ]}
      >
        <YourMainContent themeColor={themeColor} />
      </CopilotSidebar>
    </main>
  );
}

function YourMainContent({ themeColor }: { themeColor: string }) {
  // 🪁 Shared State: https://docs.copilotkit.ai/pydantic-ai/shared-state
  const { state, setState } = useCoAgent<AgentState>({
    name: "sample_agent",
    initialState: {
      proverbs: [
        "CopilotKit may be new, but its the best thing since sliced bread.",
      ],
    },
  });

  // 🪁 Frontend Actions: https://docs.copilotkit.ai/coagents/frontend-actions
  useFrontendTool({
    name: "updateProverbs",
    parameters: [
      {
        name: "proverbs",
        description: "What the current list of proverbs should be updated to",
        type: "string[]",
        required: true,
      },
    ],
    handler: ({ proverbs }) => {
      setState({
        ...state,
        proverbs: proverbs,
      });
    },
  });

  //🪁 Generative UI: https://docs.copilotkit.ai/pydantic-ai/generative-ui
  useRenderToolCall(
    {
      name: "get_weather",
      description: "Get the weather for a given location.",
      parameters: [{ name: "location", type: "string", required: true }],
      render: ({ args }) => {
        return <WeatherCard location={args.location} themeColor={themeColor} />;
      },
    },
    [themeColor]
  );

  // 🪁 Human In the Loop: https://docs.copilotkit.ai/pydantic-ai/human-in-the-loop
  useHumanInTheLoop(
    {
      name: "go_to_moon",
      description: "Go to the moon on request.",
      render: ({ respond, status }) => {
        return (
          <MoonCard themeColor={themeColor} status={status} respond={respond} />
        );
      },
    },
    [themeColor]
  );

  return (
    <div
      style={{ backgroundColor: themeColor }}
      className="h-screen flex justify-center items-center flex-col transition-colors duration-300"
    >
      <ProverbsCard state={state} setState={setState} />
      <AgentDashboard />
    </div>
  );
}

function AgentDashboard() {
  const { agent } = useAgent({ agentId: "sample_agent" });
  return (
    <div className="p-8 max-w-4xl mx-auto space-y-6">
      {/* Status */}
      <div className="p-6 rounded-lg shadow">
        <h2 className="text-xl font-bold mb-4">Agent Status</h2>
        <div className="space-y-2">
          <div className="flex items-center gap-2">
            <div
              className={`w-3 h-3 rounded-full ${
                agent.isRunning ? "bg-yellow-500 animate-pulse" : "bg-green-500"
              }`}
            />
            <span>{agent.isRunning ? "Running" : "Idle"}</span>
          </div>
          <div>Thread: {agent.threadId}</div>
          <div>Messages: {agent.messages.length}</div>
        </div>
      </div>
      {/* State */}
      <div className="p-6 rounded-lg shadow">
        <h2 className="text-xl font-bold mb-4">Agent State</h2>
        <pre className="p-4 rounded text-sm overflow-auto">
          {JSON.stringify(agent.state, null, 2)}
        </pre>
      </div>
      {/* Messages */}
      <div className="p-6 rounded-lg shadow">
        <h2 className="text-xl font-bold mb-4">Conversation</h2>
        <div className="space-y-3">
          {agent.messages.map((msg) => (
            <div
              key={msg.id}
              className={`p-3 rounded-lg ${
                msg.role === "user" ? "ml-8" : "mr-8"
              }`}
            >
              <div className="font-semibold text-sm mb-1">
                {msg.role === "user" ? "You" : "Agent"}
              </div>
              <div>{JSON.stringify(msg.content)}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
