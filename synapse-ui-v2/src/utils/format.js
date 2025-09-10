export function snippet(text = "", length = 60) {
  if (!text) return "";
  return text.length > length ? text.slice(0, length).trim() + "…" : text;
}

export function timestampIsoToLocal(iso) {
  try {
    return new Date(iso).toLocaleString();
  } catch {
    return iso;
  }
}
