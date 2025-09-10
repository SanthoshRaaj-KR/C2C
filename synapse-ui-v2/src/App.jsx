import React from "react";
import Sidebar from "./components/Sidebar/Sidebar";
import MainContent from "./components/Chat/MainContent";
import { Toaster } from "react-hot-toast";

/**
 * Root layout: left sidebar + right main area
 */
export default function App() {
  return (
    <div className="min-h-screen bg-slate-50 text-slate-900">
      <div className="max-w-[1400px] mx-auto grid grid-cols-[320px_1fr] gap-6 p-6">
        <Sidebar />
        <MainContent />
      </div>
      <Toaster position="top-right" />
    </div>
  );
}
