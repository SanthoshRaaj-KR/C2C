import React from "react";
import ReactMarkdown from "react-markdown";
import rehypeHighlight from "rehype-highlight";
import "highlight.js/styles/github-dark.css";
import { timestampIsoToLocal } from "../../utils/format";

export default function MessageBubble({ message }) {
  const isUser = message.role === "user";
  return (
    <div className={`flex ${isUser ? "justify-end" : "justify-start"}`}>
      <div className={`${isUser ? "bg-indigo-600 text-white" : "bg-slate-100 text-slate-800"} max-w-[80%] p-4 rounded-lg shadow-sm`}>
        <div className="prose prose-sm max-w-none">
          <ReactMarkdown rehypePlugins={[rehypeHighlight]}>{message.content}</ReactMarkdown>
        </div>
        <div className={`mt-2 text-xs ${isUser ? "text-indigo-100" : "text-slate-500"}`}>
          {message.error ? "Failed • " : ""}{timestampIsoToLocal(message.timestamp || new Date().toISOString())}
        </div>
      </div>
    </div>
  );
}
