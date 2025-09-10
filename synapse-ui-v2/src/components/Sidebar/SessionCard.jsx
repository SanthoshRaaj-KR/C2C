import React from "react";
import { useSessionDispatch } from "../../context/SessionContext_dep.jsx";
import { snippet } from "../../utils/format";
import { Card } from "../ui/card";

export default function SessionCard({ session, active }) {
  const dispatch = useSessionDispatch();
  if (!session) return null;
  const last = (session.messages && session.messages[session.messages.length - 1]) || null;

  return (
    <div
      className={`cursor-pointer p-3 rounded-md hover:bg-slate-50 ${active ? "bg-slate-50 border" : ""}`}
      onClick={() => dispatch({ type: "OPEN_SESSION", payload: session.session_id })}
    >
      <div className="font-medium text-sm">{session.title}</div>
      <div className="text-xs text-slate-500 mt-1">{last ? snippet(last.content, 80) : "No messages yet"}</div>
    </div>
  );
}
