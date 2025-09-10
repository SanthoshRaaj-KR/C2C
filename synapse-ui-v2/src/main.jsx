import React from "react";
import { createRoot } from "react-dom/client";
import { SessionProvider } from "./context/SessionContext_dep.jsx";
import App from "./App";
import "./index.css";

createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <SessionProvider>
      <App />
    </SessionProvider>
  </React.StrictMode>
);
