import React from "react";
import { Button } from "../ui/button";

export default function ChatHeader({ session }) {
  if (!session) return null;
  console.log(session);
  return (
    <header className="flex items-center justify-between border-b pb-3">
      <div>
        <div className="text-lg font-semibold">{session.title}</div>
        {session.repo_url && <div className="text-xs text-slate-500">{session.repo_url}</div>}
      </div>
      <div className="flex items-center gap-2">
        <Button variant="ghost" size="sm" onClick={() => window.open(session.repo_url || "#", "_blank")}>Open Repo</Button>
      </div>
    </header>
  );
}
