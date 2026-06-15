"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import Link from "next/link";

/* ── Types ────────────────────────────────────────────────────── */

interface PredictResponse {
  answer: string;
  confidence: number;
  uncertainty_flag: boolean;
  heatmap_path: string | null;
  latency_ms: number;
  predictive_entropy: number | null;
  follow_up_questions: string[];
}

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  confidence?: number;
  latency_ms?: number;
  uncertainty_flag?: boolean;
  predictive_entropy?: number | null;
  follow_ups?: string[];
  heatmap_path?: string | null;
  findings?: string[];
  timestamp: Date;
}

interface PatientContext {
  age: string;
  sex: string;
  history: string;
  symptoms: string;
}

type Status =
  | { kind: "idle" }
  | { kind: "loading" }
  | { kind: "error"; message: string };

interface Toast {
  message: string;
  kind: "success" | "info";
}

/* ── Constants ──────────────────────────────────────────────── */

const STORAGE_KEY = "medvqa-conversation";
const MAX_MESSAGES = 100;
const FALLBACK_QUESTIONS = [
  "What abnormalities are visible?",
  "Is the anatomy normal?",
  "Are there signs of pathology?",
  "Describe the key findings.",
  "Is there evidence of infection?",
  "What is the most notable feature?",
];

const FINDING_PATTERNS = [
  { label: "Nodule", pattern: /\b(nodule|nodular)\b/i },
  { label: "Opacity", pattern: /\b(opacity|opacit|opaque)\b/i },
  { label: "Infiltrate", pattern: /\b(infiltrate|infiltration)\b/i },
  { label: "Effusion", pattern: /\b(effusion|pleural)\b/i },
  { label: "Consolidation", pattern: /\bconsolidat/i },
  { label: "Fracture", pattern: /\b(fracture|fx|break)\b/i },
  { label: "Edema", pattern: /\bedema/i },
  { label: "Pneumothorax", pattern: /\bpneumothorax/i },
  { label: "Atelectasis", pattern: /\batelectasis/i },
  { label: "Cardiomegaly", pattern: /\bcardiomegaly|enlarged heart\b/i },
  { label: "Calcification", pattern: /\bcalcif/i },
  { label: "Mass", pattern: /\bmass\b/i },
  { label: "Cyst", pattern: /\bcyst\b/i },
  { label: "Fibrosis", pattern: /\bfibrosis|fibrotic\b/i },
  { label: "Emphysema", pattern: /\bemphysema/i },
];

/* ── Helpers ──────────────────────────────────────────────────── */

function fmtTime(ms: number): string {
  if (ms < 1000) return `${Math.round(ms)}ms`;
  return `${(ms / 1000).toFixed(1)}s`;
}

function confGradient(c: number): string {
  if (c >= 0.8) return "from-emerald-400 to-emerald-500";
  if (c >= 0.6) return "from-emerald-400 to-teal-400";
  if (c >= 0.4) return "from-amber-400 to-orange-400";
  return "from-red-400 to-rose-500";
}

function confRingColor(c: number): string {
  if (c >= 0.6) return "#34d399";
  if (c >= 0.4) return "#fbbf24";
  return "#f87171";
}

function confLabel(c: number): string {
  if (c >= 0.8) return "High";
  if (c >= 0.6) return "Moderate";
  if (c >= 0.4) return "Low";
  return "Very low";
}

function extractFindings(text: string): string[] {
  const found = new Set<string>();
  for (const { label, pattern } of FINDING_PATTERNS) {
    if (pattern.test(text)) found.add(label);
  }
  return Array.from(found).sort();
}

function generateReport(messages: Message[]): string {
  const findings = new Set<string>();
  const pairs: { q: string; a: string }[] = [];

  for (let i = 0; i < messages.length; i++) {
    if (messages[i].role === "user" && messages[i + 1]?.role === "assistant") {
      pairs.push({ q: messages[i].content, a: messages[i + 1].content });
      messages[i + 1].findings?.forEach((f) => findings.add(f));
    }
  }

  const sections = [
    "# MedVQA Diagnostic Report",
    `_Generated ${new Date().toLocaleString()}_`,
    "",
    "## Conversation Summary",
    `${pairs.length} question${pairs.length !== 1 ? "s" : ""} asked`,
    "",
    "## Q&A Details",
  ];

  pairs.forEach((p, i) => {
    sections.push(`\n### Q${i + 1}: ${p.q}`);
    sections.push(`**Answer:** ${p.a}`);
  });

  if (findings.size > 0) {
    sections.push("\n## Findings Summary", Array.from(findings).map((f) => `- ${f}`).join("\n"));
  }

  return sections.join("\n");
}

function downloadMarkdown(content: string, filename: string) {
  const blob = new Blob([content], { type: "text/markdown" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

/* ── localStorage persistence ─────────────────────────────────── */

function loadMessages(): Message[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    const parsed: unknown[] = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    return parsed
      .filter((e): e is Record<string, unknown> => e != null && typeof e === "object")
      .map((e) => ({
        id: String(e.id ?? ""),
        role: (e.role === "user" || e.role === "assistant" ? e.role : "assistant") as "user" | "assistant",
        content: String(e.content ?? ""),
        confidence: e.confidence != null ? Number(e.confidence) : undefined,
        latency_ms: e.latency_ms != null ? Number(e.latency_ms) : undefined,
        uncertainty_flag: e.uncertainty_flag != null ? Boolean(e.uncertainty_flag) : undefined,
        follow_ups: Array.isArray(e.follow_ups) ? e.follow_ups.map(String) : undefined,
        findings: Array.isArray(e.findings) ? e.findings.map(String) : undefined,
        timestamp: new Date(e.timestamp as string | number),
      }))
      .filter((e) => e.id && e.content);
  } catch {
    return [];
  }
}

function saveMessages(entries: Message[]): void {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(entries.slice(-MAX_MESSAGES)));
  } catch {
    // private browsing or full
  }
}

function clearStoredMessages(): void {
  try {
    localStorage.removeItem(STORAGE_KEY);
  } catch {
    // ignore
  }
}

/* ── Confidence Ring Mini ─────────────────────────────────────── */

function MiniRing({ value, size = 28 }: { value: number; size?: number }) {
  const r = size * 0.35;
  const sw = size * 0.075;
  const circ = 2 * Math.PI * r;
  const off = circ - value * circ;
  return (
    <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} className="-rotate-90 shrink-0">
      <circle cx={size / 2} cy={size / 2} r={r} fill="none" strokeWidth={sw} className="stroke-slate-200 dark:stroke-white/10" />
      <circle cx={size / 2} cy={size / 2} r={r} fill="none" strokeWidth={sw} strokeLinecap="round" strokeDasharray={circ} strokeDashoffset={off} stroke={confRingColor(value)} />
    </svg>
  );
}

/* ── ROI Canvas Overlay ───────────────────────────────────────── */

function RoiOverlay({ imagePreview, onRoiChange }: { imagePreview: string; onRoiChange: (roi: { x: number; y: number; w: number; h: number } | null) => void }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [drawing, setDrawing] = useState(false);
  const [rect, setRect] = useState<{ x: number; y: number; w: number; h: number } | null>(null);
  const startRef = useRef({ x: 0, y: 0 });

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const img = new Image();
    img.onload = () => {
      canvas.width = img.naturalWidth;
      canvas.height = img.naturalHeight;
      ctx?.drawImage(img, 0, 0);
    };
    img.src = imagePreview;
  }, [imagePreview]);

  const redraw = useCallback((r: { x: number; y: number; w: number; h: number } | null, start?: { x: number; y: number }) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const img = new Image();
    img.onload = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(img, 0, 0);
      if (r) {
        ctx.strokeStyle = "#3b82f6";
        ctx.lineWidth = 3;
        ctx.setLineDash([]);
        ctx.strokeRect(r.x, r.y, r.w, r.h);
        ctx.fillStyle = "rgba(59, 130, 246, 0.15)";
        ctx.fillRect(r.x, r.y, r.w, r.h);
        // Label
        ctx.fillStyle = "#3b82f6";
        ctx.font = "12px sans-serif";
        ctx.fillText("ROI", r.x + 4, r.y - 6);
        // Remove button
        if (start) {
          ctx.fillStyle = "#ef4444";
          ctx.beginPath();
          ctx.arc(r.x + r.w - 10, r.y + 10, 8, 0, Math.PI * 2);
          ctx.fill();
          ctx.fillStyle = "#fff";
          ctx.font = "10px sans-serif";
          ctx.textAlign = "center";
          ctx.fillText("×", r.x + r.w - 10, r.y + 13);
          ctx.textAlign = "start";
        }
      }
    };
    img.src = imagePreview;
  }, [imagePreview]);

  useEffect(() => { redraw(rect); }, [rect, redraw]);

  return (
    <div className="relative">
      <canvas
        ref={canvasRef}
        className="w-full rounded-lg cursor-crosshair"
        style={{ maxHeight: "60vh", objectFit: "contain" }}
        onMouseDown={(e) => {
          const canvas = canvasRef.current;
          if (!canvas) return;
          const r = canvas.getBoundingClientRect();
          const scaleX = canvas.width / r.width;
          const scaleY = canvas.height / r.height;
          const x = (e.clientX - r.left) * scaleX;
          const y = (e.clientY - r.top) * scaleY;
          startRef.current = { x, y };
          setDrawing(true);
        }}
        onMouseMove={(e) => {
          if (!drawing) return;
          const canvas = canvasRef.current;
          if (!canvas) return;
          const r = canvas.getBoundingClientRect();
          const scaleX = canvas.width / r.width;
          const scaleY = canvas.height / r.height;
          const x = (e.clientX - r.left) * scaleX;
          const y = (e.clientY - r.top) * scaleY;
          const sx = startRef.current.x;
          const sy = startRef.current.y;
          const newRect = { x: Math.min(sx, x), y: Math.min(sy, y), w: Math.abs(x - sx), h: Math.abs(y - sy) };
          setRect(newRect);
        }}
        onMouseUp={() => {
          setDrawing(false);
          if (rect && rect.w > 5 && rect.h > 5) {
            // Normalize to 0-1 coordinates
            const canvas = canvasRef.current;
            if (canvas) {
              onRoiChange({ x: rect.x / canvas.width, y: rect.y / canvas.height, w: rect.w / canvas.width, h: rect.h / canvas.height });
            }
          } else {
            setRect(null);
            onRoiChange(null);
          }
        }}
      />
      {rect && rect.w > 5 && (
        <button
          onClick={() => { setRect(null); onRoiChange(null); }}
          className="absolute top-2 right-2 size-6 rounded-full bg-red-500 text-white text-xs flex items-center justify-center hover:bg-red-600 transition-colors shadow-sm"
        >
          ×
        </button>
      )}
    </div>
  );
}

/* ── Main Component ────────────────────────────────────────────── */

export default function DiagnosePage() {
  const [image, setImage] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState<Message[]>(() => loadMessages());
  const [status, setStatus] = useState<Status>({ kind: "idle" });
  const [dragOver, setDragOver] = useState(false);
  const [suggestedQuestions, setSuggestedQuestions] = useState<string[]>([]);
  const [questionsLoading, setQuestionsLoading] = useState(false);
  const [showPatientContext, setShowPatientContext] = useState(false);
  const [patientContext, setPatientContext] = useState<PatientContext>({ age: "", sex: "", history: "", symptoms: "" });
  const [roi, setRoi] = useState<{ x: number; y: number; w: number; h: number } | null>(null);
  const [showRoiCanvas, setShowRoiCanvas] = useState(false);
  const [toast, setToast] = useState<Toast | null>(null);

  /* ── Toast auto-dismiss ───────────────────────────────────── */
  useEffect(() => {
    if (!toast) return;
    const id = setTimeout(() => setToast(null), 2500);
    return () => clearTimeout(id);
  }, [toast]);

  const fileRef = useRef<HTMLInputElement>(null);
  const objUrlRef = useRef<string | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const chatEndRef = useRef<HTMLDivElement>(null);
  const chatContainerRef = useRef<HTMLDivElement>(null);

  /* ── Auto-scroll ──────────────────────────────────────────── */
  useEffect(() => {
    const el = chatContainerRef.current;
    if (!el) return;
    const threshold = 150;
    const isNearBottom = el.scrollHeight - el.scrollTop - el.clientHeight < threshold;
    if (isNearBottom) chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, status.kind]);

  /* ── Clear conversation ───────────────────────────────────── */
  const clearConversation = useCallback(() => {
    setMessages([]);
    setStatus({ kind: "idle" });
    clearStoredMessages();
    textareaRef.current?.focus();
  }, []);

  /* ── Keyboard shortcuts ───────────────────────────────────── */
  useEffect(() => {
    const fn = clearConversation;
    const handler = (e: KeyboardEvent) => {
      const isCtrl = e.ctrlKey || e.metaKey;
      if (isCtrl && e.key === "l") { e.preventDefault(); fn(); }
      if (isCtrl && e.key === "k") { e.preventDefault(); fileRef.current?.click(); }
      if (e.key === "Escape" && !imagePreview) { fileRef.current?.click(); }
    };
    document.addEventListener("keydown", handler);
    return () => document.removeEventListener("keydown", handler);
  }, [clearConversation, imagePreview]);

  /* ── Persist messages ─────────────────────────────────────── */
  useEffect(() => { saveMessages(messages); }, [messages]);

  /* ── Image handlers ────────────────────────────────────────── */
  const handleFile = useCallback((file: File) => {
    if (!file.type.startsWith("image/")) return;
    if (objUrlRef.current) URL.revokeObjectURL(objUrlRef.current);
    const url = URL.createObjectURL(file);
    objUrlRef.current = url;
    setImage(file);
    setImagePreview(url);
    setMessages([]);
    setStatus({ kind: "idle" });
    setSuggestedQuestions([]);
    setRoi(null);
    setQuestionsLoading(true);

    const form = new FormData();
    form.append("file", file);
    fetch("/api/suggest-questions", { method: "POST", body: form })
      .then(async (r) => { if (!r.ok) throw new Error("fail"); const d = await r.json(); setSuggestedQuestions(d.questions ?? []); })
      .catch(() => setSuggestedQuestions([]))
      .finally(() => setQuestionsLoading(false));

    setTimeout(() => textareaRef.current?.focus(), 100);
  }, []);

  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    const f = e.dataTransfer.files[0];
    if (f) handleFile(f);
  }, [handleFile]);

  const removeImage = useCallback(() => {
    if (objUrlRef.current) { URL.revokeObjectURL(objUrlRef.current); objUrlRef.current = null; }
    setImage(null);
    setImagePreview(null);
    setMessages([]);
    setStatus({ kind: "idle" });
    setSuggestedQuestions([]);
    setRoi(null);
    setQuestionsLoading(false);
    clearStoredMessages();
  }, []);

  /* ── Submit ─────────────────────────────────────────────────── */
  const handleSubmit = useCallback(async () => {
    if (!image) {
      setStatus({ kind: "error", message: "Please upload a medical image first." });
      return;
    }
    const q = input.trim();
    if (!q) {
      setStatus({ kind: "error", message: "Please enter a question." });
      textareaRef.current?.focus();
      return;
    }

    const userMsg: Message = { id: `user-${Date.now()}`, role: "user", content: q, timestamp: new Date() };
    setMessages((prev) => [...prev, userMsg]);
    setInput("");
    setStatus({ kind: "loading" });

    // Build conversation pairs
    const pairs: { question: string; answer: string }[] = [];
    for (let i = 0; i < messages.length; i++) {
      if (messages[i].role === "user" && messages[i + 1]?.role === "assistant") {
        pairs.push({ question: messages[i].content, answer: messages[i + 1].content });
      }
    }

    // Get accumulated findings for smart follow-ups
    const allFindings = new Set<string>();
    messages.forEach((m) => m.findings?.forEach((f) => allFindings.add(f)));

    const form = new FormData();
    form.append("file", image);
    form.append("question", q);
    form.append("conversation", JSON.stringify(pairs));

    // Patient context
    const ctx = patientContext;
    const hasContext = ctx.age || ctx.sex || ctx.history || ctx.symptoms;
    if (hasContext) {
      form.append("patient_context", JSON.stringify(ctx));
    }

    // ROI coordinates
    if (roi) {
      form.append("roi", JSON.stringify(roi));
    }

    try {
      const res = await fetch("/api/predict", { method: "POST", body: form });
      if (!res.ok) { const body = await res.json().catch(() => null); throw new Error(body?.detail ?? `Server error (${res.status})`); }
      const data: PredictResponse = await res.json();

      // Extract findings from answer
      const findings = extractFindings(data.answer);
      allFindings.forEach((f) => { if (!findings.includes(f)) findings.push(f); });

      // Build follow-ups, mixing API suggestions with finding-based suggestions
      let followUps = [...(data.follow_up_questions || [])];
      if (allFindings.size > 0 && followUps.length < 3) {
        followUps.push(...Array.from(allFindings).slice(0, 3).map((f) => `Tell me more about the ${f.toLowerCase()}`));
      }

      const assistantMsg: Message = {
        id: `assistant-${Date.now()}`, role: "assistant", content: data.answer,
        confidence: data.confidence, latency_ms: data.latency_ms,
        uncertainty_flag: data.uncertainty_flag, predictive_entropy: data.predictive_entropy,
        follow_ups: followUps.slice(0, 4), heatmap_path: data.heatmap_path,
        findings: findings.length > 0 ? findings : undefined,
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, assistantMsg]);
      setStatus({ kind: "idle" });
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Unknown error";
      setStatus({ kind: "error", message: msg });
    }
  }, [image, input, messages, patientContext, roi]);

  /* ── Export & Generate Report ──────────────────────────────── */
  const handleExport = useCallback(() => {
    if (messages.length === 0) return;
    const report = generateReport(messages);
    downloadMarkdown(report, `medvqa-report-${Date.now()}.md`);
    setToast({ message: "Conversation exported as Markdown", kind: "success" });
  }, [messages]);

  const handleGenerateReport = useCallback(() => {
    if (messages.length === 0) return;
    // Structured radiology-style report
    const findings = new Set<string>();
    const pairs: { q: string; a: string }[] = [];
    for (let i = 0; i < messages.length; i++) {
      if (messages[i].role === "user" && messages[i + 1]?.role === "assistant") {
        pairs.push({ q: messages[i].content, a: messages[i + 1].content });
        messages[i + 1].findings?.forEach((f) => findings.add(f));
      }
    }
    const lines = [
      "MEDVQA CLINICAL REPORT",
      "=".repeat(50),
      `Generated: ${new Date().toLocaleString()}`,
      `Questions asked: ${pairs.length}`,
      "",
      "---",
      "",
      "EXAMINATION & FINDINGS",
      "-".repeat(30),
    ];
    if (findings.size > 0) {
      lines.push("Observed findings:", Array.from(findings).map((f) => `  • ${f}`).join("\n"));
      lines.push("");
    }
    lines.push("DETAILED Q&A", "-".repeat(30));
    pairs.forEach((p, i) => {
      lines.push(`\nQ${i + 1}: ${p.q}`, `A: ${p.a}`, "");
    });
    lines.push("---", "", "DISCLAIMER: Research prototype. Not for clinical use.");
    downloadMarkdown(lines.join("\n"), `medvqa-clinical-report-${Date.now()}.md`);
    setToast({ message: "Clinical report downloaded", kind: "success" });
  }, [messages]);

  /* ── Derived ────────────────────────────────────────────────── */
  const isBusy = status.kind === "loading";
  const showUpload = !imagePreview && messages.length === 0 && status.kind !== "loading";
  const questions = suggestedQuestions.length > 0 ? suggestedQuestions : FALLBACK_QUESTIONS;
  const allFindings = new Set<string>();
  messages.forEach((m) => m.findings?.forEach((f) => allFindings.add(f)));

  /* ── Render ─────────────────────────────────────────────────── */
  return (
    <div className="min-h-screen bg-slate-50 dark:bg-slate-950 flex flex-col">
      {/* ── Nav ──────────────────────────────────────────────── */}
      <nav className="sticky top-0 z-30 border-b border-slate-200 dark:border-white/5 bg-white/80 dark:bg-slate-950/80 backdrop-blur-2xl shrink-0">
        <div className="max-w-5xl mx-auto px-4 sm:px-6 h-14 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Link href="/" className="text-slate-400 hover:text-slate-600 dark:text-white/30 dark:hover:text-white/60 text-sm transition-colors">← Home</Link>
            <span className="text-slate-200 dark:text-white/10">|</span>
            <span className="text-sm font-medium text-slate-700 dark:text-white/70">Diagnose</span>
            <kbd className="hidden sm:inline-flex text-[9px] px-1.5 py-0.5 rounded bg-slate-100 dark:bg-white/[0.04] border border-slate-200 dark:border-white/10 text-slate-300 dark:text-white/20 ml-1">Ctrl+L</kbd>
          </div>
          <div className="flex items-center gap-2">
            {messages.length > 0 && (
              <>
                <button onClick={handleExport} className="text-xs px-3 py-1.5 rounded-full border border-slate-200 dark:border-white/5 text-slate-400 dark:text-white/30 hover:text-slate-600 dark:hover:text-white/50 hover:border-slate-300 dark:hover:border-white/10 transition-all" title="Export as Markdown">
                  Export
                </button>
                <button onClick={handleGenerateReport} className="text-xs px-3 py-1.5 rounded-full bg-blue-50 dark:bg-blue-500/10 border border-blue-200 dark:border-blue-500/20 text-blue-600 dark:text-blue-300 hover:bg-blue-100 dark:hover:bg-blue-500/20 transition-all" title="Generate structured report">
                  Report
                </button>
              </>
            )}
            <span className="text-[11px] text-slate-300 dark:text-white/20">API mode</span>
          </div>
        </div>
      </nav>

      {/* ── Image banner ──────────────────────────────────────── */}
      {imagePreview && (
        <div className="border-b border-slate-200 dark:border-white/5 bg-white/50 dark:bg-white/[0.02] shrink-0">
          <div className="max-w-5xl mx-auto px-4 sm:px-6 py-2 flex items-center gap-3">
            <img src={imagePreview} alt="Uploaded" className="size-9 rounded-lg object-cover ring-1 ring-slate-200 dark:ring-white/10 shrink-0" />
            <div className="flex-1 min-w-0">
              <p className="text-xs font-medium text-slate-600 dark:text-white/60 truncate">{image?.name}</p>
              <p className="text-[10px] text-slate-400 dark:text-white/25">{image ? `${(image.size / 1024).toFixed(0)} KB` : ""}</p>
            </div>
            <div className="flex items-center gap-1.5">
              <button onClick={() => { clearConversation(); }} className="text-xs text-slate-400 dark:text-white/20 hover:text-slate-600 dark:hover:text-white/50 transition-colors px-2 py-1" title="Clear conversation (Ctrl+L)">New Question</button>
              <span className="text-slate-200 dark:text-white/10">|</span>
              <button onClick={() => setShowPatientContext(!showPatientContext)} className={`text-xs transition-colors px-2 py-1 ${showPatientContext ? "text-blue-500 dark:text-blue-400" : "text-slate-400 dark:text-white/20 hover:text-slate-600 dark:hover:text-white/50"}`}>Context</button>
              <span className="text-slate-200 dark:text-white/10">|</span>
              <button onClick={() => setShowRoiCanvas(!showRoiCanvas)} className={`text-xs transition-colors px-2 py-1 ${showRoiCanvas ? "text-blue-500 dark:text-blue-400" : "text-slate-400 dark:text-white/20 hover:text-slate-600 dark:hover:text-white/50"}`}>ROI</button>
              <span className="text-slate-200 dark:text-white/10">|</span>
              <button onClick={removeImage} className="text-xs text-slate-400 dark:text-white/20 hover:text-slate-600 dark:hover:text-white/50 transition-colors px-2 py-1">Change</button>
            </div>
          </div>
        </div>
      )}

      {/* ── Patient Context panel ─────────────────────────────── */}
      {imagePreview && showPatientContext && (
        <div className="border-b border-slate-200 dark:border-white/5 bg-white/50 dark:bg-slate-900/50 shrink-0 animate-slide-down">
          <div className="max-w-5xl mx-auto px-4 sm:px-6 py-3">
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
              <div>
                <label className="text-[10px] font-medium text-slate-400 dark:text-white/30 uppercase tracking-wider">Age</label>
                <input value={patientContext.age} onChange={(e) => setPatientContext((p) => ({ ...p, age: e.target.value }))} placeholder="e.g. 65" className="mt-1 w-full px-2.5 py-1.5 text-xs rounded-lg bg-white dark:bg-white/[0.05] border border-slate-200 dark:border-white/10 text-slate-700 dark:text-white/80 placeholder:text-slate-300 dark:placeholder:text-white/20 outline-none focus:border-blue-300 dark:focus:border-white/20 transition-colors" />
              </div>
              <div>
                <label className="text-[10px] font-medium text-slate-400 dark:text-white/30 uppercase tracking-wider">Sex</label>
                <select value={patientContext.sex} onChange={(e) => setPatientContext((p) => ({ ...p, sex: e.target.value }))} className="mt-1 w-full px-2.5 py-1.5 text-xs rounded-lg bg-white dark:bg-white/[0.05] border border-slate-200 dark:border-white/10 text-slate-700 dark:text-white/80 outline-none focus:border-blue-300 dark:focus:border-white/20 transition-colors">
                  <option value="">—</option>
                  <option value="Male">Male</option>
                  <option value="Female">Female</option>
                </select>
              </div>
              <div className="col-span-2">
                <label className="text-[10px] font-medium text-slate-400 dark:text-white/30 uppercase tracking-wider">History</label>
                <input value={patientContext.history} onChange={(e) => setPatientContext((p) => ({ ...p, history: e.target.value }))} placeholder="e.g. COPD, hypertension" className="mt-1 w-full px-2.5 py-1.5 text-xs rounded-lg bg-white dark:bg-white/[0.05] border border-slate-200 dark:border-white/10 text-slate-700 dark:text-white/80 placeholder:text-slate-300 dark:placeholder:text-white/20 outline-none focus:border-blue-300 dark:focus:border-white/20 transition-colors" />
              </div>
              <div className="col-span-2 sm:col-span-4">
                <label className="text-[10px] font-medium text-slate-400 dark:text-white/30 uppercase tracking-wider">Symptoms</label>
                <input value={patientContext.symptoms} onChange={(e) => setPatientContext((p) => ({ ...p, symptoms: e.target.value }))} placeholder="e.g. cough, shortness of breath, chest pain" className="mt-1 w-full px-2.5 py-1.5 text-xs rounded-lg bg-white dark:bg-white/[0.05] border border-slate-200 dark:border-white/10 text-slate-700 dark:text-white/80 placeholder:text-slate-300 dark:placeholder:text-white/20 outline-none focus:border-blue-300 dark:focus:border-white/20 transition-colors" />
              </div>
            </div>
            {patientContext.age || patientContext.sex || patientContext.history || patientContext.symptoms ? (
              <div className="mt-2 flex items-center gap-1.5 text-[10px] text-emerald-500 dark:text-emerald-400">
                <svg className="size-3" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M4.5 12.75l6 6 9-13.5" /></svg>
                Patient context will be included in all questions
                <button onClick={() => setPatientContext({ age: "", sex: "", history: "", symptoms: "" })} className="ml-auto text-slate-400 hover:text-slate-600 dark:text-white/20 dark:hover:text-white/50">Clear</button>
              </div>
            ) : null}
          </div>
        </div>
      )}

      {/* ── ROI Canvas ────────────────────────────────────────── */}
      {imagePreview && showRoiCanvas && (
        <div className="border-b border-slate-200 dark:border-white/5 bg-white/50 dark:bg-white/[0.02] shrink-0 animate-slide-down">
          <div className="max-w-3xl mx-auto px-4 sm:px-6 py-3">
            <div className="flex items-center justify-between mb-2">
              <span className="text-[10px] font-medium text-slate-400 dark:text-white/30 uppercase tracking-wider">Region of Interest — drag on the image to select an area</span>
              {roi && <span className="text-[10px] text-blue-500 dark:text-blue-400">ROI selected</span>}
            </div>
            <div className="rounded-xl overflow-hidden border border-slate-200 dark:border-white/10 bg-white dark:bg-black/20">
              <RoiOverlay imagePreview={imagePreview} onRoiChange={setRoi} />
            </div>
          </div>
        </div>
      )}

      {/* ── Upload hero ────────────────────────────────────────── */}
      {showUpload && (
        <div className="flex-1 flex flex-col items-center justify-center px-4 py-16 animate-fade-in">
          <div className="text-center mb-8">
            <h1 className="text-3xl sm:text-4xl font-bold tracking-tight mb-3">
              <span className="gradient-text">Medical Image Analysis</span>
            </h1>
            <p className="text-sm text-slate-400 dark:text-white/30 max-w-md mx-auto">
              Upload a medical image and ask clinical questions.
              <br />Get AI-powered analysis with confidence scoring.
            </p>
            <div className="mt-3 flex items-center justify-center gap-3 text-[10px] text-slate-300 dark:text-white/15">
              <span><kbd className="px-1 rounded bg-slate-100 dark:bg-white/[0.04] border border-slate-200 dark:border-white/10">Ctrl+K</kbd> upload</span>
              <span><kbd className="px-1 rounded bg-slate-100 dark:bg-white/[0.04] border border-slate-200 dark:border-white/10">Ctrl+L</kbd> clear</span>
              <span><kbd className="px-1 rounded bg-slate-100 dark:bg-white/[0.04] border border-slate-200 dark:border-white/10">Esc</kbd> focus</span>
            </div>
          </div>
          <div
            onDrop={onDrop} onDragOver={(e) => { e.preventDefault(); setDragOver(true); }} onDragLeave={() => setDragOver(false)}
            onClick={() => fileRef.current?.click()}
            className={`relative mx-auto max-w-xl w-full rounded-2xl border-2 border-dashed p-12 text-center cursor-pointer transition-all duration-300 group ${
              dragOver ? "border-blue-400 dark:border-blue-400 bg-blue-50/50 dark:bg-blue-500/5 scale-[1.01]" : "border-slate-200 dark:border-white/10 hover:border-slate-300 dark:hover:border-white/20 hover:bg-slate-50 dark:hover:bg-white/[0.02]"
            }`}
          >
            <div className="flex flex-col items-center gap-4">
              <div className="size-16 rounded-2xl bg-gradient-to-br from-blue-500/10 to-violet-500/10 flex items-center justify-center ring-1 ring-slate-200 dark:ring-white/10 group-hover:ring-blue-300 dark:group-hover:ring-white/20 transition-all group-hover:scale-105 duration-300">
                <svg className="size-7 text-blue-500 dark:text-blue-400/70" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5" />
                </svg>
              </div>
              <div>
                <p className="text-sm font-medium text-slate-600 dark:text-white/60">Drop your medical image here</p>
                <p className="text-xs text-slate-400 dark:text-white/25 mt-1">or <span className="text-blue-500 dark:text-blue-400/70 underline underline-offset-2">browse files</span></p>
              </div>
              <div className="flex items-center gap-2 text-[10px] text-slate-300 dark:text-white/20">
                <span>DICOM</span><span>·</span><span>PNG</span><span>·</span><span>JPEG</span><span>·</span><span>BMP</span>
              </div>
            </div>
            <input ref={fileRef} type="file" accept="image/*,.dcm" className="hidden" onChange={(e) => { const f = e.target.files?.[0]; if (f) handleFile(f); }} />
          </div>
        </div>
      )}

      {/* ── Conversation area ────────────────────────────────── */}
      {imagePreview && (
        <div ref={chatContainerRef} className="flex-1 overflow-y-auto">
          <div className="max-w-5xl mx-auto px-4 sm:px-6 py-6 space-y-5">

            {/* Accumulated findings board */}
            {allFindings.size > 0 && (
              <div className="rounded-xl bg-white dark:bg-white/[0.03] border border-slate-200 dark:border-white/5 p-3 animate-fade-in">
                <div className="flex items-center gap-2 mb-1.5">
                  <svg className="size-3.5 text-slate-400 dark:text-white/30" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
                  </svg>
                  <span className="text-[10px] font-medium text-slate-400 dark:text-white/30 uppercase tracking-wider">Findings</span>
                </div>
                <div className="flex flex-wrap gap-1.5">
                  {Array.from(allFindings).map((f) => (
                    <span key={f} className="text-[10px] px-2 py-0.5 rounded-full bg-amber-50 dark:bg-amber-500/10 border border-amber-200 dark:border-amber-500/20 text-amber-700 dark:text-amber-300 font-medium">
                      {f}
                    </span>
                  ))}
                </div>
              </div>
            )}

            {/* Initial suggestions */}
            {messages.length === 0 && status.kind !== "loading" && (
              <div className="text-center py-8 animate-fade-in">
                <p className="text-sm text-slate-400 dark:text-white/30 mb-4">Ask a question about this image to begin</p>
                <div className="flex flex-wrap items-center justify-center gap-2 max-w-xl mx-auto">
                  {questionsLoading ? (
                    <div className="flex items-center gap-2 text-xs text-slate-400 dark:text-white/20">
                      <svg className="size-3 animate-spin" viewBox="0 0 24 24" fill="none">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z" />
                      </svg>
                      Generating suggestions…
                    </div>
                  ) : (
                    questions.map((q) => (
                      <button key={q} onClick={() => { setInput(q); textareaRef.current?.focus(); }}
                        className="text-[11px] px-3 py-1.5 rounded-full border border-slate-200 dark:border-white/5 text-slate-400 dark:text-white/25 hover:text-slate-600 dark:hover:text-white/60 hover:border-slate-300 dark:hover:border-white/10 hover:bg-slate-50 dark:hover:bg-white/[0.03] transition-all"
                      >{q.length > 35 ? q.slice(0, 35) + "…" : q}</button>
                    ))
                  )}
                </div>
              </div>
            )}

            {/* Message thread */}
            {messages.map((msg) => (
              <div key={msg.id} className="animate-fade-in-up">
                {msg.role === "user" ? (
                  <div className="flex justify-end">
                    <div className="max-w-[75%] bg-blue-500 text-white rounded-2xl rounded-br-md px-4 py-2.5 text-sm leading-relaxed shadow-sm">{msg.content}</div>
                  </div>
                ) : (
                  <div className="space-y-2">
                    <div className="rounded-2xl bg-white dark:bg-white/[0.03] border border-slate-200 dark:border-white/5 overflow-hidden shadow-sm">
                      <div className="px-4 pt-3 pb-3">
                        <div className="flex items-start gap-3">
                          {msg.confidence != null && (
                            <div className="flex flex-col items-center gap-0.5 shrink-0 pt-1">
                              <MiniRing value={msg.confidence} />
                              <span className="text-[9px] text-slate-400 dark:text-white/30 font-medium">{confLabel(msg.confidence)}</span>
                            </div>
                          )}
                          <div className="flex-1 min-w-0">
                            <div className="flex items-center gap-2 mb-1 flex-wrap">
                              <span className="text-xs font-semibold text-slate-400 dark:text-white/40 uppercase tracking-wider">Answer</span>
                              {msg.latency_ms != null && <span className="text-[10px] text-slate-300 dark:text-white/15">{fmtTime(msg.latency_ms)}</span>}
                              {msg.uncertainty_flag && (
                                <span className="inline-flex items-center gap-1 text-[10px] text-amber-600 dark:text-amber-300 bg-amber-50 dark:bg-amber-500/10 px-1.5 py-0.5 rounded-full border border-amber-200 dark:border-amber-500/15">
                                  <svg className="size-2.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                                    <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v3.75m9-.75a9 9 0 11-18 0 9 9 0 0118 0zm-9 3.75h.008v.008H12v-.008z" />
                                  </svg>
                                  Low confidence
                                </span>
                              )}
                              {/* Copy button */}
                              <button
                                onClick={() => navigator.clipboard.writeText(msg.content)}
                                className="ml-auto text-slate-300 dark:text-white/15 hover:text-slate-500 dark:hover:text-white/40 transition-colors"
                                title="Copy answer"
                              >
                                <svg className="size-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                                  <path strokeLinecap="round" strokeLinejoin="round" d="M15.666 3.888A2.25 2.25 0 0013.5 2.25h-3c-1.03 0-1.9.693-2.166 1.638m7.332 0c.055.194.084.4.084.612v0a.75.75 0 01-.75.75H9a.75.75 0 01-.75-.75v0c0-.212.03-.418.084-.612m7.332 0c.646.049 1.288.11 1.927.184 1.1.128 1.907 1.077 1.907 2.185V19.5a2.25 2.25 0 01-2.25 2.25H6.75A2.25 2.25 0 014.5 19.5V6.257c0-1.108.806-2.057 1.907-2.185a48.208 48.208 0 011.927-.184" />
                                </svg>
                              </button>
                            </div>
                            <p className="text-sm text-slate-700 dark:text-white/80 leading-relaxed">{msg.content}</p>
                            {/* Findings tags */}
                            {msg.findings && msg.findings.length > 0 && (
                              <div className="flex flex-wrap gap-1 mt-2">
                                {msg.findings.map((f) => (
                                  <span key={f} className="text-[9px] px-1.5 py-0.5 rounded bg-slate-100 dark:bg-white/[0.05] text-slate-400 dark:text-white/30 border border-slate-200 dark:border-white/5">{f}</span>
                                ))}
                              </div>
                            )}
                          </div>
                        </div>
                      </div>
                      {msg.confidence != null && (
                        <div className="h-0.5 bg-slate-100 dark:bg-white/[0.03]">
                          <div className={`h-full bg-gradient-to-r ${confGradient(msg.confidence)} transition-all duration-1000 ease-out`} style={{ width: `${msg.confidence * 100}%` }} />
                        </div>
                      )}
                    </div>

                    {/* Follow-up suggestions */}
                    {msg.follow_ups && msg.follow_ups.length > 0 && (
                      <div className="flex flex-wrap items-center gap-1.5 pl-2">
                        <span className="text-[10px] text-slate-300 dark:text-white/15 mr-1">Follow-up:</span>
                        {msg.follow_ups.map((fu) => (
                          <button key={fu} onClick={() => { setInput(fu); textareaRef.current?.focus(); }}
                            className="text-[11px] px-2.5 py-1 rounded-full border border-slate-200 dark:border-white/5 text-slate-400 dark:text-white/25 hover:text-slate-600 dark:hover:text-white/60 hover:border-blue-300 dark:hover:border-blue-500/30 hover:bg-blue-50 dark:hover:bg-blue-500/5 transition-all"
                          >{fu.length > 35 ? fu.slice(0, 35) + "…" : fu}</button>
                        ))}
                      </div>
                    )}

                    {/* Grad-CAM */}
                    {msg.heatmap_path && (
                      <div className="rounded-xl bg-white dark:bg-white/[0.03] border border-slate-200 dark:border-white/5 overflow-hidden">
                        <div className="px-3 pt-2 pb-1 flex items-center gap-1.5">
                          <svg className="size-3 text-purple-500 dark:text-purple-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                            <path strokeLinecap="round" strokeLinejoin="round" d="M2.036 12.322a1.012 1.012 0 010-.639C3.423 7.51 7.36 4.5 12 4.5c4.638 0 8.573 3.007 9.963 7.178.07.207.07.431 0 .639C20.577 16.49 16.64 19.5 12 19.5c-4.638 0-8.573-3.007-9.963-7.178z" />
                            <path strokeLinecap="round" strokeLinejoin="round" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                          </svg>
                          <span className="text-[10px] font-medium text-slate-400 dark:text-white/30">Attention Heatmap</span>
                        </div>
                        <div className="px-3 pb-3"><img src={msg.heatmap_path} alt="Grad-CAM" className="w-full rounded-lg max-h-48 object-contain bg-slate-100 dark:bg-black/30" /></div>
                      </div>
                    )}
                  </div>
                )}
              </div>
            ))}

            {/* Loading */}
            {status.kind === "loading" && (
              <div className="flex justify-start animate-fade-in">
                <div className="bg-white dark:bg-white/[0.03] border border-slate-200 dark:border-white/5 rounded-2xl rounded-bl-md px-4 py-3 flex items-center gap-2.5">
                  <div className="flex gap-1">
                    <span className="size-1.5 rounded-full bg-slate-300 dark:bg-white/30 animate-bounce" style={{ animationDelay: "0ms" }} />
                    <span className="size-1.5 rounded-full bg-slate-300 dark:bg-white/30 animate-bounce" style={{ animationDelay: "150ms" }} />
                    <span className="size-1.5 rounded-full bg-slate-300 dark:bg-white/30 animate-bounce" style={{ animationDelay: "300ms" }} />
                  </div>
                  <span className="text-xs text-slate-400 dark:text-white/30">Analyzing…</span>
                </div>
              </div>
            )}

            {/* Error */}
            {status.kind === "error" && (
              <div className="rounded-2xl bg-red-50 dark:bg-red-500/5 border border-red-200 dark:border-red-500/15 px-4 py-3 flex items-center gap-2.5 animate-fade-in">
                <svg className="size-4 text-red-500 dark:text-red-400 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v3.75m9-.75a9 9 0 11-18 0 9 9 0 0118 0zm-9 3.75h.008v.008H12v-.008z" />
                </svg>
                <p className="text-xs text-red-600 dark:text-red-300/80">{status.message}</p>
              </div>
            )}

            <div ref={chatEndRef} />
          </div>
        </div>
      )}

      {/* ── Input area ────────────────────────────────────────── */}
      {imagePreview && (
        <div className="shrink-0 border-t border-slate-200 dark:border-white/5 bg-white/80 dark:bg-slate-950/80 backdrop-blur-xl pb-[env(safe-area-inset-bottom)]">
          <div className="max-w-5xl mx-auto px-4 sm:px-6 py-3">
            <div className="relative rounded-2xl bg-white dark:bg-white/[0.03] border border-slate-200 dark:border-white/10 transition-all focus-within:border-blue-300 dark:focus-within:border-white/20 focus-within:bg-slate-50 dark:focus-within:bg-white/[0.04]">
              <textarea
                ref={textareaRef}
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder={messages.length === 0 ? "Ask a clinical question about this image…" : "Ask a follow-up question…"}
                rows={1}
                className="w-full resize-none bg-transparent px-4 pt-3 pb-11 text-sm text-slate-700 dark:text-white/80 placeholder:text-slate-300 dark:placeholder:text-white/20 outline-none leading-relaxed"
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); handleSubmit(); }
                }}
                disabled={isBusy}
              />
              <div className="absolute bottom-2.5 right-2.5 flex items-center gap-2">
                <kbd className="hidden sm:inline-flex px-1.5 py-0.5 rounded text-[9px] bg-slate-100 dark:bg-white/[0.04] border border-slate-200 dark:border-white/10 text-slate-300 dark:text-white/20">↵</kbd>
                <button
                  onClick={handleSubmit}
                  disabled={isBusy || !input.trim()}
                  className="size-8 rounded-xl bg-blue-500 hover:bg-blue-600 text-white flex items-center justify-center disabled:opacity-30 disabled:cursor-not-allowed transition-all shadow-sm"
                >
                  {isBusy ? (
                    <svg className="size-4 animate-spin" viewBox="0 0 24 24" fill="none">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z" />
                    </svg>
                  ) : (
                    <svg className="size-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                      <path strokeLinecap="round" strokeLinejoin="round" d="M6 12L3.269 3.126A59.768 59.768 0 0121.485 12 59.77 59.77 0 013.27 20.876L5.999 12zm0 0h7.5" />
                    </svg>
                  )}
                </button>
              </div>
            </div>
            {messages.length > 0 && (
              <div className="mt-1.5 flex items-center justify-between text-[9px] text-slate-300 dark:text-white/10">
                <span>{messages.length} message{messages.length !== 1 ? "s" : ""} · {allFindings.size} finding{allFindings.size !== 1 ? "s" : ""}</span>
                <span>Ctrl+Enter to send</span>
              </div>
            )}
          </div>
        </div>
      )}

      {/* ── Toast notification ───────────────────────────────── */}
      {toast && (
        <div className="fixed bottom-20 left-1/2 -translate-x-1/2 z-50 animate-fade-in-up">
          <div className={`flex items-center gap-2 px-4 py-2.5 rounded-xl shadow-lg backdrop-blur-xl text-sm ${
            toast.kind === "success"
              ? "bg-emerald-50 dark:bg-emerald-500/10 border border-emerald-200 dark:border-emerald-500/20 text-emerald-700 dark:text-emerald-300"
              : "bg-blue-50 dark:bg-blue-500/10 border border-blue-200 dark:border-blue-500/20 text-blue-700 dark:text-blue-300"
          }`}>
            {toast.kind === "success" ? (
              <svg className="size-4 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M4.5 12.75l6 6 9-13.5" />
              </svg>
            ) : (
              <svg className="size-4 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
            )}
            <span>{toast.message}</span>
          </div>
        </div>
      )}

      {/* ── Footer ────────────────────────────────────────────── */}
      <div className="shrink-0 border-t border-slate-200 dark:border-white/5 bg-white/90 dark:bg-slate-950/90 backdrop-blur-xl py-2">
        <p className="text-[9px] text-slate-300 dark:text-white/10 text-center">Research prototype · Not for clinical use</p>
      </div>
    </div>
  );
}
