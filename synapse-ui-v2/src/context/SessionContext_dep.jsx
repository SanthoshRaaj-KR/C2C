import React, { createContext, useContext, useReducer, useEffect } from "react";
import * as api from "../lib/api";
import { snippet } from "../utils/format";

const SessionStateContext = createContext();
const SessionDispatchContext = createContext();

const initialState = {
  sessions: {},         // { sessionId: { session_id, title, messages: [], status } }
  sessionOrder: [],     // [sessionId, ...]
  activeSessionId: null,
  loading: false,
  error: null,
};

function reducer(state, action) {
  switch (action.type) {
    case "CREATE_SESSION_INIT":
      return { ...state, loading: true, error: null };
    case "CREATE_SESSION_SUCCESS": {
      const s = action.payload;
      return {
        ...state,
        loading: false,
        sessions: { ...state.sessions, [s.session_id]: { ...s, messages: s.messages || s.initial_messages || (s.initial_message ? [s.initial_message] : []) } },
        sessionOrder: [s.session_id, ...state.sessionOrder],
        activeSessionId: s.session_id,
      };
    }
    case "CREATE_SESSION_FAILURE":
      return { ...state, loading: false, error: action.payload };
    case "OPEN_SESSION":
        
      return { ...state, activeSessionId: action.payload };
    case "SEND_MESSAGE_INIT": {
      const { sessionId, optimisticMessage } = action.payload;
      const session = state.sessions[sessionId] || {};
      return {
        ...state,
        sessions: {
          ...state.sessions,
          [sessionId]: {
            ...session,
            messages: [...(session.messages || []), optimisticMessage],
            status: "sending",
          },
        },
      };
    }
    case "SEND_MESSAGE_SUCCESS": {
        const { sessionId, message, tempId } = action.payload;
        const session = state.sessions[sessionId] || {};

        // Keep all messages (including the user's optimistic one)
        const messages = (session.messages || []).map((m) =>
            m.id === tempId ? { ...m, pending: false } : m
        );

        // Append the assistant's message after user's
        messages.push(message);

        return {
            ...state,
            sessions: {
            ...state.sessions,
            [sessionId]: { ...session, messages, status: "ready" },
            },
        };
    }
    case "SEND_MESSAGE_FAILURE": {
      const { sessionId, tempId, error } = action.payload;
      const session = state.sessions[sessionId] || {};
      const messages = (session.messages || []).map((m) =>
        m.id === tempId ? { ...m, error: true, content: `${m.content}\n\n[Failed to send: ${error}]` } : m
      );
      return {
        ...state,
        sessions: {
          ...state.sessions,
          [sessionId]: { ...session, messages, status: "error" },
        },
      };
    }
    case "RECEIVE_SESSION_LIST": {
      const incoming = action.payload; // array of session summaries
      const sessions = {};
      const order = [];
      incoming.forEach((s) => {
        sessions[s.session_id] = { ...s, messages: s.messages || [] };
        order.push(s.session_id);
      });
      return { ...state, sessions, sessionOrder: order };
    }
    default:
      return state;
  }
}

export function SessionProvider({ children }) {
  const [state, dispatch] = useReducer(reducer, initialState);

  // optional: fetch sessions on mount
  useEffect(() => {
    let mounted = true;
    (async () => {
      try {
        const data = await api.fetchSessions();
        if (!mounted) return;
        dispatch({ type: "RECEIVE_SESSION_LIST", payload: data.sessions || [] });
      } catch (e) {
        // ignore silently for now
      }
    })();
    return () => (mounted = false);
  }, []);

  return (
    <SessionStateContext.Provider value={state}>
      <SessionDispatchContext.Provider value={dispatch}>{children}</SessionDispatchContext.Provider>
    </SessionStateContext.Provider>
  );
}

export function useSessionState() {
  return useContext(SessionStateContext);
}

export function useSessionDispatch() {
  return useContext(SessionDispatchContext);
}
