import React from "react";
import ChatHeader from "./ChatHeader";
import MessageList from "./MessageList";
import Composer from "./Composer";
import { useSessionState } from "../../context/SessionContext_dep.jsx";

export default function ChatWindow({ sessionId }) {
  const { sessions } = useSessionState();
  const session = sessions[sessionId];
  console.log(session);

  return (
    <div className="flex flex-col h-[78vh]">
      <ChatHeader session={session} />
      <div className="flex-1 overflow-y-auto mt-4">
        <MessageList messages={session.messages || []} />
      </div>
      <div className="mt-4">
        <Composer sessionId={sessionId} />
      </div>
    </div>
  );
}
