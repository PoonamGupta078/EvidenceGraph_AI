"use client";
import React, { useState, useRef, useEffect, useCallback } from "react";
import { api, PersonaId } from "@/lib/api";

interface Message {
  role: "user" | "assistant";
  content: string;
  model?: string | null;
  llm_used?: boolean;
}

interface Props {
  investigationId: string;
  personaId: PersonaId;
  regionId: string;
  scenario: string;
}

const SUGGESTED: string[] = [
  "What is the root cause?",
  "Why did revenue drop?",
  "What action should I take?",
  "How confident is the system?",
  "Explain the causal chain",
];

/* ── Tiny icons ───────────────────────────────────────────────────────────── */
const IconChat = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>
  </svg>
);
const IconSend = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
    <path d="M22 2L11 13M22 2L15 22 11 13 2 9l20-7z"/>
  </svg>
);
const IconClose = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round">
    <line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/>
  </svg>
);
const IconMinus = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round">
    <line x1="5" y1="12" x2="19" y2="12"/>
  </svg>
);
const IconBot = () => (
  <svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor">
    <path d="M12 2a2 2 0 0 1 2 2c0 .74-.4 1.39-1 1.73V7h1a7 7 0 0 1 7 7H5a7 7 0 0 1 7-7h1V5.73c-.6-.34-1-.99-1-1.73a2 2 0 0 1 2-2zM5 14v5a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2v-5H5zM9 16a1 1 0 0 1 1 1v1a1 1 0 0 1-2 0v-1a1 1 0 0 1 1-1zm6 0a1 1 0 0 1 1 1v1a1 1 0 0 1-2 0v-1a1 1 0 0 1 1-1z"/>
  </svg>
);

/* ── Markdown-lite renderer ───────────────────────────────────────────────── */
function renderMarkdown(text: string): React.ReactNode {
  const parts = text.split(/(\*\*[^*]+\*\*|`[^`]+`)/g);
  return parts.map((part, i) => {
    if (part.startsWith("**") && part.endsWith("**")) {
      return <strong key={i} style={{ fontWeight: 700, color: "var(--text-primary)" }}>{part.slice(2, -2)}</strong>;
    }
    if (part.startsWith("`") && part.endsWith("`")) {
      return <code key={i} style={{ fontFamily: "var(--font-mono)", fontSize: "0.9em", background: "var(--bg-elevated)", padding: "1px 4px", borderRadius: 3 }}>{part.slice(1, -1)}</code>;
    }
    return part;
  });
}

/* ── Main Component ────────────────────────────────────────────────────────── */
export default function InvestigationChat({ investigationId, personaId, regionId, scenario }: Props) {
  const [open, setOpen] = useState(false);
  const [minimised, setMinimised] = useState(false);
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [sending, setSending] = useState(false);
  const [showSuggestions, setShowSuggestions] = useState(true);
  const bottomRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  // Scroll to bottom on new messages
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, open]);

  // Reset chat when region changes
  useEffect(() => {
    setMessages([]);
    setShowSuggestions(true);
    setInput("");
  }, [investigationId]);

  // Focus input when panel opens
  useEffect(() => {
    if (open && !minimised) {
      setTimeout(() => inputRef.current?.focus(), 100);
    }
  }, [open, minimised]);

  const send = useCallback(async (text: string) => {
    const trimmed = text.trim();
    if (!trimmed || sending) return;

    setShowSuggestions(false);
    const userMsg: Message = { role: "user", content: trimmed };
    const history = messages.map(m => ({ role: m.role, content: m.content }));

    setMessages(prev => [...prev, userMsg, { role: "assistant", content: "…" }]);
    setInput("");
    setSending(true);

    try {
      const res = await api.chat(investigationId, trimmed, personaId, history);
      const contentText = res.answer || res.reply || "No response text returned.";
      setMessages(prev => [
        ...prev.slice(0, -1), // remove placeholder
        { role: "assistant", content: contentText, model: res.model, llm_used: res.llm_used },
      ]);
    } catch (err) {
      setMessages(prev => [
        ...prev.slice(0, -1),
        { role: "assistant", content: "Sorry, I couldn't reach the backend. Make sure the investigation is still active.", llm_used: false },
      ]);
    } finally {
      setSending(false);
    }
  }, [investigationId, personaId, messages, sending]);

  const handleKey = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(input); }
  };

  /* ── Floating trigger button ── */
  if (!open) {
    return (
      <button
        id="chat-trigger-btn"
        onClick={() => setOpen(true)}
        style={{
          position: "fixed", bottom: 28, right: 28, zIndex: 1000,
          width: 52, height: 52, borderRadius: "50%",
          background: "linear-gradient(135deg, var(--accenture-purple), var(--accenture-purple-dark))",
          border: "none", cursor: "pointer",
          display: "flex", alignItems: "center", justifyContent: "center",
          color: "white", boxShadow: "0 4px 20px rgba(161,0,255,0.4), 0 0 0 0 rgba(161,0,255,0.3)",
          animation: "glow-pulse 2.5s ease infinite",
          transition: "transform 0.2s",
        }}
        onMouseEnter={e => (e.currentTarget.style.transform = "scale(1.1)")}
        onMouseLeave={e => (e.currentTarget.style.transform = "scale(1)")}
        title="Ask about this investigation"
      >
        <IconChat />
        {/* Unread indicator dot when investigation loaded */}
        <span style={{
          position: "absolute", top: 4, right: 4,
          width: 10, height: 10, borderRadius: "50%",
          background: "var(--act)", border: "2px solid white",
        }}/>
      </button>
    );
  }

  /* ── Chat panel ── */
  return (
    <div
      id="investigation-chat-panel"
      style={{
        position: "fixed", bottom: 28, right: 28, zIndex: 1000,
        width: 410,
        height: minimised ? "auto" : 560,
        background: "var(--bg-card)",
        border: "2px solid var(--accenture-purple)",
        borderRadius: "var(--radius-lg)",
        boxShadow: "0 8px 40px rgba(0,0,0,0.18), 0 0 0 1px rgba(161,0,255,0.1)",
        display: "flex", flexDirection: "column",
        overflow: "hidden",
        animation: "slide-up 0.3s cubic-bezier(0.16,1,0.3,1)",
      }}
    >
      {/* ── Header ── */}
      <div style={{
        padding: "12px 16px",
        background: "linear-gradient(135deg, var(--accenture-purple), var(--accenture-purple-dark))",
        display: "flex", alignItems: "center", gap: 10,
        flexShrink: 0,
      }}>
        <div style={{
          width: 28, height: 28, borderRadius: "50%",
          background: "rgba(255,255,255,0.2)",
          display: "flex", alignItems: "center", justifyContent: "center",
          color: "white",
        }}>
          <IconBot/>
        </div>
        <div style={{ flex: 1 }}>
          <div style={{ fontSize: "1.0rem", fontWeight: 700, color: "white", lineHeight: 1.2 }}>
            Investigation Assistant
          </div>
          <div style={{ fontSize: "0.78rem", color: "rgba(255,255,255,0.85)", lineHeight: 1.2, fontWeight: 500 }}>
            {regionId.replace(/_/g, " ").toUpperCase()} · {scenario.replace(/_/g, " ")}
          </div>
        </div>
        {/* Minimise / Close */}
        <button onClick={() => setMinimised(m => !m)} style={headerBtnStyle} title={minimised ? "Expand" : "Minimise"}><IconMinus/></button>
        <button onClick={() => { setOpen(false); setMinimised(false); }} style={headerBtnStyle} title="Close"><IconClose/></button>
      </div>

      {!minimised && (
        <>
          {/* ── Messages ── */}
          <div style={{
            flex: 1, overflowY: "auto", padding: "14px 16px",
            display: "flex", flexDirection: "column", gap: 12,
            background: "#fafafa",
          }}>

            {/* Welcome message */}
            {messages.length === 0 && (
              <div style={{
                padding: "14px 16px",
                background: "#ffffff",
                border: "1px solid #dcdde1",
                borderRadius: "var(--radius-md)",
                fontSize: "0.95rem", color: "#222222",
                lineHeight: 1.6,
              }}>
                <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 6 }}>
                  <div style={{ width: 20, height: 20, borderRadius: "50%", background: "var(--accenture-purple-glow)", display: "flex", alignItems: "center", justifyContent: "center" }}>
                    <IconBot/>
                  </div>
                  <span style={{ fontWeight: 700, color: "#111111", fontSize: "1.0rem" }}>Ready to help</span>
                </div>
                Ask me anything about this investigation — root cause, confidence, evidence, recommended actions, or what the data means.
              </div>
            )}

            {/* Suggestion chips */}
            {showSuggestions && messages.length === 0 && (
              <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
                {SUGGESTED.map(s => (
                  <button key={s} onClick={() => send(s)} style={{
                    padding: "6px 12px",
                    background: "#ffffff",
                    border: "1.5px solid var(--accenture-purple-glow)",
                    borderRadius: 100,
                    fontSize: "0.85rem", color: "#111111",
                    fontWeight: 600,
                    cursor: "pointer",
                    transition: "all 0.15s",
                    fontFamily: "var(--font-sans)",
                  }}
                  onMouseEnter={e => { e.currentTarget.style.borderColor = "var(--accenture-purple)"; e.currentTarget.style.background = "#f1e8ff"; }}
                  onMouseLeave={e => { e.currentTarget.style.borderColor = "var(--accenture-purple-glow)"; e.currentTarget.style.background = "#ffffff"; }}
                  >
                    {s}
                  </button>
                ))}
              </div>
            )}

            {/* Message bubbles */}
            {messages.map((msg, i) => (
              <div key={i} style={{
                display: "flex",
                flexDirection: msg.role === "user" ? "row-reverse" : "row",
                gap: 8, alignItems: "flex-end",
              }}>
                {msg.role === "assistant" && (
                  <div style={{
                    width: 24, height: 24, borderRadius: "50%", flexShrink: 0,
                    background: "linear-gradient(135deg, var(--accenture-purple), var(--accenture-purple-dark))",
                    display: "flex", alignItems: "center", justifyContent: "center",
                    color: "white", marginBottom: 2,
                  }}>
                    <IconBot/>
                  </div>
                )}
                <div style={{
                  maxWidth: "82%",
                  padding: "10px 14px",
                  borderRadius: msg.role === "user" ? "14px 14px 4px 14px" : "4px 14px 14px 14px",
                  background: msg.role === "user"
                    ? "linear-gradient(135deg, var(--accenture-purple), var(--accenture-purple-dark))"
                    : "#ffffff",
                  border: msg.role === "user" ? "none" : "1.5px solid #dcdde1",
                  color: msg.role === "user" ? "#ffffff" : "#111111",
                  fontWeight: msg.role === "user" ? 500 : 400,
                  fontSize: "1.0rem",
                  lineHeight: 1.6,
                  boxShadow: "0 2px 8px rgba(0,0,0,0.06)",
                }}>
                  {msg.content === "…" ? (
                    <span style={{ letterSpacing: 3, opacity: 0.5, animation: "pulse-ring 1s ease infinite" }}>●●●</span>
                  ) : (
                    <span>{renderMarkdown(msg.content)}</span>
                  )}
                  {msg.role === "assistant" && msg.model && msg.content !== "…" && (
                    <div style={{ marginTop: 6, fontSize: "0.8rem", color: "var(--accenture-purple)", fontWeight: 600, fontFamily: "var(--font-mono)" }}>
                      {msg.llm_used ? `⚡ ${msg.model}` : "📋 deterministic"}
                    </div>
                  )}
                </div>
              </div>
            ))}
            <div ref={bottomRef}/>
          </div>

          {/* ── Input Bar ── */}
          <div style={{
            padding: "12px 14px",
            borderTop: "1px solid #dcdde1",
            background: "#ffffff",
            display: "flex", gap: 8, alignItems: "center",
            flexShrink: 0,
          }}>
            <input
              ref={inputRef}
              id="chat-input"
              type="text"
              value={input}
              onChange={e => setInput(e.target.value)}
              onKeyDown={handleKey}
              placeholder="Ask about this investigation…"
              disabled={sending}
              style={{
                flex: 1,
                padding: "10px 16px",
                background: "#f5f6fa",
                border: "1.5px solid #dcdde1",
                borderRadius: 100,
                fontSize: "1.0rem",
                color: "#111111",
                fontFamily: "var(--font-sans)",
                outline: "none",
                transition: "all 0.15s",
              }}
              onFocus={e => { e.currentTarget.style.borderColor = "var(--accenture-purple)"; e.currentTarget.style.background = "#ffffff"; }}
              onBlur={e => { e.currentTarget.style.borderColor = "#dcdde1"; e.currentTarget.style.background = "#f5f6fa"; }}
            />
            <button
              id="chat-send-btn"
              onClick={() => send(input)}
              disabled={sending || !input.trim()}
              style={{
                width: 38, height: 38, borderRadius: "50%",
                background: input.trim() && !sending
                  ? "linear-gradient(135deg, var(--accenture-purple), var(--accenture-purple-dark))"
                  : "#f5f6fa",
                border: "none", cursor: input.trim() && !sending ? "pointer" : "default",
                display: "flex", alignItems: "center", justifyContent: "center",
                color: input.trim() && !sending ? "white" : "#7f8c8d",
                transition: "all 0.15s",
                flexShrink: 0,
              }}
            >
              <IconSend/>
            </button>
          </div>
        </>
      )}
    </div>
  );
}

const headerBtnStyle: React.CSSProperties = {
  background: "rgba(255,255,255,0.15)",
  border: "none",
  borderRadius: 6,
  width: 26, height: 26,
  display: "flex", alignItems: "center", justifyContent: "center",
  color: "white", cursor: "pointer",
  transition: "background 0.15s",
};
