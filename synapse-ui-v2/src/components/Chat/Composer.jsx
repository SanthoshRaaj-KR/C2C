import React, { useState, useRef } from "react";
import { Button } from "../ui/button";
import { useSessionDispatch, useSessionState } from "../../context/SessionContext_dep.jsx";
import * as api from "../../lib/api";
import { v4 as uuidv4 } from "uuid";
import toast from "react-hot-toast";

export default function Composer({ sessionId }) {
  const [value, setValue] = useState("");
  const inputRef = useRef(null);   // 🔹 ref for textarea
  const dispatch = useSessionDispatch();
  const { sessions } = useSessionState();
  const session = sessions[sessionId];
  const isSending = session?.status === "sending";

  async function handleSend() {
    if (!value.trim()) return;
    const tempId = `temp_${uuidv4()}`;
    const optimistic = {
      id: tempId,
      role: "user",
      content: value,
      timestamp: new Date().toISOString(),
      pending: true,
    };

    dispatch({ type: "SEND_MESSAGE_INIT", payload: { sessionId, optimisticMessage: optimistic } });
    setValue(""); 

    // 🔹 refocus input after clearing
    if (inputRef.current) {
      inputRef.current.focus();
    }

    try {
      const data = await api.sendMessage(sessionId, value);
      const serverMessage = data.message || data;
      dispatch({ type: "SEND_MESSAGE_SUCCESS", payload: { sessionId, message: serverMessage, tempId } });
    } catch (err) {
      console.error(err);
      dispatch({ type: "SEND_MESSAGE_FAILURE", payload: { sessionId, tempId, error: err.message } });
      toast.error("Message failed");
    }
  }

  function handleKeyDown(e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  }

  return (
    <div className="flex items-center gap-2">
      <textarea
        ref={inputRef}   // 🔹 attach ref
        className="flex-1 p-3 rounded border resize-none h-14 focus:outline-none focus:ring"
        placeholder="Ask something..."
        value={value}
        onChange={(e) => setValue(e.target.value)}
        onKeyDown={handleKeyDown}
        disabled={isSending}
      />
      <Button onClick={handleSend} disabled={isSending || !value.trim()}>
        {isSending ? "Sending..." : "Send"}
      </Button>
    </div>
  );
}
