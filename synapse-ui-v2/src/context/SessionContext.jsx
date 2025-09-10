import React, { createContext, useContext, useReducer, useEffect } from "react";
import * as api from "../lib/api";

// -------------------------
// Contexts
// -------------------------
const SessionStateContext = createContext();
const SessionDispatchContext = createContext();

// -------------------------
// Initial State
// -------------------------
const initialState = {
  sessions: {},        // { sessionId: { session_id, repo_url, title, messages: [] } }
  sessionOrder: [],    // ordered list of session IDs
  activeSessionId: null,
  loading: false,
  error: null,
};

// -------------------------
// Reducer
// -------------------------
function reducer(state, action) {
  switch (action.type) {
    // -------------------------
    // Create session
    // -------------------------
    case "CREATE_SESSION_INIT":
      return { ...state, loading: true, error: null };

    case "CREATE_SESSION_SUCCESS": {
      const s = action.payload;

      // Normalize session data
      const normalizedSession = {
        session_id: s.session_id,
        repo_url: s.repo_url || null,
        title: s.title || `Repo: ${s.repo_url || "Untitled Repo"}`,
        messages:
          s.messages && Array.isArray(s.messages)
            ? s.messages
            : s.initial_message
            ? [s.initial_message]
            : [],
      };

      return {
        ...state,
        loading: false,
        sessions: {
          ...state.sessions,
          [normalizedSession.session_id]: normalizedSession,
        },
        sessionOrder: [normalizedSession.session_id, ...state.sessionOrder],
        activeSessionId: normalizedSession.session_id,
      };
    }

    case "CREATE_SESSION_FAILURE":
      return { ...state, loading: false, error: action.payload };

    // -------------------------
    // Open existing session
    // -------------------------
    case "OPEN_SESSION":
      return { ...state, activeSessionId: action.payload };

    // -------------------------
    // Sending message (optimistic user message)
    // -------------------------
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

    // -------------------------
    // Message sent successfully
    // -------------------------
    case "SEND_MESSAGE_SUCCESS": {
      const { sessionId, message, tempId } = action.payload;
      const session = state.sessions[sessionId] || {};

      // Mark optimistic user message as delivered
      const messages = (session.messages || []).map((m) =>
        m.id === tempId ? { ...m, pending: false } : m
      );

      // Append assistant reply
      messages.push(message);

      return {
        ...state,
        sessions: {
          ...state.sessions,
          [sessionId]: { ...session, messages, status: "ready" },
        },
      };
    }

    // -------------------------
    // Message failed to send
    // -------------------------
    case "SEND_MESSAGE_FAILURE": {
      const { sessionId, tempId, error } = action.payload;
      const session = state.sessions[sessionId] || {};

      const messages = (session.messages || []).map((m) =>
        m.id === tempId
          ? {
              ...m,
              error: true,
              pending: false,
              content: `${m.content}\n\n[Failed to send: ${error}]`,
            }
          : m
      );

      return {
        ...state,
        sessions: {
          ...state.sessions,
          [sessionId]: { ...session, messages, status: "error" },
        },
      };
    }

    // -------------------------
    // Load sessions list
    // -------------------------
    case "RECEIVE_SESSION_LIST": {
        const incoming = action.payload || [];
        const sessions = { ...state.sessions };
        const order = [...state.sessionOrder];

        incoming.forEach((s) => {
            sessions[s.session_id] = {
            session_id: s.session_id,
            repo_url: s.repo_url || null,
            title: s.title || `Repo: ${s.repo_url || "Untitled Repo"}`,
            messages: s.messages || [],
            };

            if (!order.includes(s.session_id)) {
            order.push(s.session_id);
            }
        });

        return { ...state, sessions, sessionOrder: order };
        }

    // -------------------------
    default:
      return state;
  }
}

// -------------------------
// Provider
// -------------------------
export function SessionProvider({ children }) {
  const [state, dispatch] = useReducer(reducer, initialState);

  // Auto-fetch session list on mount
  useEffect(() => {
    let mounted = true;
    (async () => {
      try {
        const data = await api.fetchSessions();
        if (!mounted) return;
        dispatch({
          type: "RECEIVE_SESSION_LIST",
          payload: data.sessions || [],
        });
      } catch (e) {
        // Fail silently (sidebar will just be empty)
        console.error("Failed to fetch sessions:", e);
      }
    })();
    return () => {
      mounted = false;
    };
  }, []);

  return (
    <SessionStateContext.Provider value={state}>
      <SessionDispatchContext.Provider value={dispatch}>
        {children}
      </SessionDispatchContext.Provider>
    </SessionStateContext.Provider>
  );
}

// -------------------------
// Hooks
// -------------------------
export function useSessionState() {
  return useContext(SessionStateContext);
}

export function useSessionDispatch() {
  return useContext(SessionDispatchContext);
}
