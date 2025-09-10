import React, { useEffect, useRef } from "react";
import MessageBubble from "./MessageBubble";

export default function MessageList({ messages = [] }) {
  const ref = useRef();
  useEffect(() => {
    if (ref.current) {
      ref.current.scrollTop = ref.current.scrollHeight;
    }
  }, [messages.length]);
  return (
    <div ref={ref} className="space-y-4 px-2">
      {messages.map((m) => (
        <MessageBubble key={m.id} message={m} />
      ))}
    </div>
  );
}
