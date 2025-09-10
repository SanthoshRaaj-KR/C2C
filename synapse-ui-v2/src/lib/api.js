import axios from "axios";

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}
const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000";

export async function createSession(repo_url) {
  const res = await axios.post(`${API_BASE}/sessions`, { repo_url });
  await sleep(3000);
  return res.data;
}

export async function sendMessage(sessionId, content) {
  const res = await axios.post(`${API_BASE}/sessions/${sessionId}/messages`, { content });
  return res.data;
}

export async function fetchSessions() {
  const res = await axios.get(`${API_BASE}/sessions`);
  return res.data;
}

export async function fetchSession(sessionId) {
  const res = await axios.get(`${API_BASE}/sessions/${sessionId}`);
  return res.data;
}
