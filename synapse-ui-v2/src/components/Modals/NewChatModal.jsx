import React, { useState } from "react";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "../ui/dialog";
import { Input } from "../ui/input";
import { Button } from "../ui/button";
import { useSessionDispatch } from "../../context/SessionContext_dep.jsx";
import * as api from "../../lib/api";
import toast from "react-hot-toast";

export default function NewChatModal({ isOpen, onClose }) {
  const [repo, setRepo] = useState("");
  const [loading, setLoading] = useState(false);
  const dispatch = useSessionDispatch();

  async function handleCreate(e) {
    e.preventDefault();
    if (!repo.trim()) return toast.error("Provide a repo URL");
    setLoading(true);
    dispatch({ type: "CREATE_SESSION_INIT" });

    try {
        const data = await api.createSession(repo.trim());

        // Normalize backend response safely
        const payload = {
        session_id: data.session_id || data.sessionId || data.session?.session_id,
        repo_url: data.repo_url || repo.trim(),
        title: data.title || `Repo: ${repo.trim()}`,
        messages:
            Array.isArray(data.messages) && data.messages.length > 0
            ? data.messages
            : data.initial_message
            ? [data.initial_message]
            : [],
        };

        dispatch({
        type: "CREATE_SESSION_SUCCESS",
        payload,
        });

        toast.success("Session created");
        setRepo("");
        onClose();
    } catch (err) {
        console.error(err);
        dispatch({ type: "CREATE_SESSION_FAILURE", payload: err.message });
        toast.error("Failed to create session");
    } finally {
        setLoading(false);
    }
    }

  return (
    <Dialog open={isOpen} onOpenChange={(open) => !open && onClose()}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Start new chat from GitHub repo</DialogTitle>
        </DialogHeader>
        <form className="space-y-4" onSubmit={handleCreate}>
          <Input
            placeholder="https://github.com/user/repo"
            value={repo}
            onChange={(e) => setRepo(e.target.value)}
            disabled={loading}
          />
          <div className="flex justify-end gap-2">
            <Button variant="ghost" onClick={onClose}>Cancel</Button>
            <Button type="submit" disabled={loading}>
            {loading ? (
                <span className="flex items-center gap-2">
                <span className="animate-spin h-4 w-4 border-2 border-white border-t-transparent rounded-full"></span>
                Creating...
                </span>
            ) : (
                "Create"
            )}
      </Button>
          </div>
        </form>
      </DialogContent>
    </Dialog>
  );
}
