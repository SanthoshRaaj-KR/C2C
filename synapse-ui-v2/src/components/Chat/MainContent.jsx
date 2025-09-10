import React from "react";
import { useSessionState } from "../../context/SessionContext_dep.jsx";
import ChatWindow from "./ChatWindow";

export default function MainContent() {
  const { activeSessionId } = useSessionState();
  return (
    <main className="min-h-[85vh] bg-white rounded-lg shadow p-4">
      {activeSessionId ? <ChatWindow sessionId={activeSessionId} /> : <Welcome />}
    </main>
  );
}

function Welcome() {
  return (
    <div className="h-full flex flex-col items-center justify-center text-center text-slate-600">
      <h1 className="text-2xl font-semibold mb-2">Welcome to the Chat App</h1>
      <p className="max-w-lg">Create a new chat from a GitHub repo or open an existing session from the left. The assistant will analyze the repository and answer questions.</p>
    </div>
  );
}
