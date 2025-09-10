import React, { useState } from "react";
import { useSessionState, useSessionDispatch } from "../../context/SessionContext_dep.jsx";
import SessionCard from "./SessionCard";
import NewChatModal from "../Modals/NewChatModal";
import { Button } from "../ui/button";
import { Plus } from "lucide-react";

export default function Sidebar() {
  const { sessionOrder, sessions, activeSessionId } = useSessionState();
  const dispatch = useSessionDispatch();
  const [open, setOpen] = useState(false);

  return (
    <aside className="h-[85vh] bg-white rounded-lg shadow p-4 sticky top-6 overflow-y-auto">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-lg font-semibold">Chats</h2>
        <Button size="sm" onClick={() => setOpen(true)} variant="secondary" className="flex items-center gap-2">
          <Plus size={14} /> New
        </Button>
      </div>

      <div className="space-y-2">
        {sessionOrder.length === 0 && <div className="text-sm text-slate-500">No chats yet. Create one.</div>}
        {sessionOrder.map((id) => (
          <SessionCard key={id} session={sessions[id]} active={activeSessionId === id} />
        ))}
      </div>

      <NewChatModal isOpen={open} onClose={() => setOpen(false)} />
    </aside>
  );
}
