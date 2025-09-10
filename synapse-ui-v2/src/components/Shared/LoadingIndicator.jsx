import React from "react";

export default function LoadingIndicator({ size = 6 }) {
  return (
    <div className="flex items-center justify-center">
      <div className={`animate-spin rounded-full h-${size} w-${size} border-t-2 border-b-2 border-slate-400`} />
    </div>
  );
}
