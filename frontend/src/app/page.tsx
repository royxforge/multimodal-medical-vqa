"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import Link from "next/link";

/* ── Types ────────────────────────────────────────────────────── */

interface FeatureCardProps {
  title: string;
  desc: string;
  icon: React.ReactNode;
  gradient: string;
  index: number;
}

interface StepCardProps {
  num: string;
  title: string;
  desc: string;
  color: string;
  bg: string;
  index: number;
}

/* ── Constants ────────────────────────────────────────────────── */

const FEATURES_CORE = [
  {
    title: "Conversational Workflow",
    desc: "Ask continuous follow-up questions in a threaded chat interface. Each answer comes with follow-up suggestions and the model remembers the full conversation history.",
    icon: (
      <svg className="size-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M20.25 8.511c.884.284 1.5 1.128 1.5 2.097v4.286c0 1.136-.847 2.1-1.98 2.193-.34.027-.68.052-1.02.072v3.091l-3-3c-1.354 0-2.694-.055-4.02-.163a2.115 2.115 0 01-.825-.242m9.345-8.334a2.126 2.126 0 00-.476-.095 48.64 48.64 0 00-8.048 0c-1.131.094-1.976 1.057-1.976 2.192v4.286c0 .837.46 1.58 1.155 1.951m9.345-8.334V6.637c0-1.621-1.152-3.026-2.76-3.235A48.455 48.455 0 0011.25 3c-2.115 0-4.198.137-6.24.402-1.608.209-2.76 1.614-2.76 3.235v6.226c0 1.621 1.152 3.026 2.76 3.235.577.075 1.157.14 1.74.194V21l4.155-4.155" />
      </svg>
    ),
    gradient: "from-blue-500 to-cyan-500",
  },
  {
    title: "Confidence Estimation",
    desc: "Every answer includes a calibrated confidence score displayed as an animated ring. Low-confidence answers are flagged so you know when to be cautious.",
    icon: (
      <svg className="size-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M3 13.125C3 12.504 3.504 12 4.125 12h2.25c.621 0 1.125.504 1.125 1.125v6.75C7.5 20.496 6.996 21 6.375 21h-2.25A1.125 1.125 0 013 19.875v-6.75zM9.75 8.625c0-.621.504-1.125 1.125-1.125h2.25c.621 0 1.125.504 1.125 1.125v11.25c0 .621-.504 1.125-1.125 1.125h-2.25a1.125 1.125 0 01-1.125-1.125V8.625zM16.5 4.125c0-.621.504-1.125 1.125-1.125h2.25C20.496 3 21 3.504 21 4.125v15.75c0 .621-.504 1.125-1.125 1.125h-2.25a1.125 1.125 0 01-1.125-1.125V4.125z" />
      </svg>
    ),
    gradient: "from-emerald-500 to-teal-500",
  },
  {
    title: "Grad-CAM Heatmaps",
    desc: "Attention heatmaps overlaid on input images highlight diagnostically relevant regions, showing exactly what regions the model focused on for each answer.",
    icon: (
      <svg className="size-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M2.036 12.322a1.012 1.012 0 010-.639C3.423 7.51 7.36 4.5 12 4.5c4.638 0 8.573 3.007 9.963 7.178.07.207.07.431 0 .639C20.577 16.49 16.64 19.5 12 19.5c-4.638 0-8.573-3.007-9.963-7.178z" />
        <path strokeLinecap="round" strokeLinejoin="round" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
      </svg>
    ),
    gradient: "from-violet-500 to-purple-500",
  },
  {
    title: "Finding Auto-Tagging",
    desc: "Medical findings (nodule, opacity, effusion, fracture, pneumothorax, etc.) are automatically extracted from each answer and displayed as tags with an accumulated findings board.",
    icon: (
      <svg className="size-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M9.568 3H5.25A2.25 2.25 0 003 5.25v4.318c0 .597.237 1.17.659 1.591l9.581 9.581c.699.699 1.78.872 2.607.33a18.095 18.095 0 005.223-5.223c.542-.827.369-1.908-.33-2.607L11.16 3.66A2.25 2.25 0 009.568 3z" />
        <path strokeLinecap="round" strokeLinejoin="round" d="M6 6h.008v.008H6V6z" />
      </svg>
    ),
    gradient: "from-pink-500 to-rose-500",
  },
];

const FEATURES_UX = [
  {
    title: "Smart Follow-ups",
    desc: "After each answer, the system suggests 3–4 relevant follow-up questions. If findings are detected, they're also injected as follow-up prompts.",
    icon: (
      <svg className="size-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M3.75 3.75v4.5m0-4.5h4.5m-4.5 0L9 9M3.75 20.25v-4.5m0 4.5h4.5m-4.5 0L9 15M20.25 3.75h-4.5m4.5 0v4.5m0-4.5L15 9m5.25 11.25h-4.5m4.5 0v-4.5m0 4.5L15 15" />
      </svg>
    ),
    gradient: "from-cyan-500 to-blue-500",
  },
  {
    title: "Patient Context",
    desc: "Toggle a panel to enter patient metadata (age, sex, history, symptoms). Context is injected into every question for more personalized answers.",
    icon: (
      <svg className="size-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M15.75 6a3.75 3.75 0 11-7.5 0 3.75 3.75 0 017.5 0zM4.501 20.118a7.5 7.5 0 0114.998 0A17.933 17.933 0 0112 21.75c-2.676 0-5.216-.584-7.499-1.632z" />
      </svg>
    ),
    gradient: "from-amber-500 to-yellow-500",
  },
  {
    title: "ROI Drawing",
    desc: "Toggle ROI mode to drag-select a specific region on the image. Coordinates are sent to the model, focusing analysis on your area of interest.",
    icon: (
      <svg className="size-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M3.75 6.75h16.5M3.75 12h16.5m-16.5 5.25h16.5" />
      </svg>
    ),
    gradient: "from-indigo-500 to-blue-500",
  },
  {
    title: "Report Generation",
    desc: "Export a structured clinical report (Findings, Detailed Q&A) or a raw Markdown transcript with one click. Toast notifications confirm every download.",
    icon: (
      <svg className="size-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M19.5 14.25v-2.625a3.375 3.375 0 00-3.375-3.375h-1.5A1.125 1.125 0 0113.5 7.125v-1.5a3.375 3.375 0 00-3.375-3.375H8.25m0 12.75h7.5m-7.5 3H12M10.5 2.25H5.625c-.621 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 00-9-9z" />
      </svg>
    ),
    gradient: "from-orange-500 to-red-500",
  },
  {
    title: "Keyboard Shortcuts",
    desc: "Ctrl+K to upload, Ctrl+L to clear conversation, Escape to focus the upload button. Designed for a fast, keyboard-driven workflow.",
    icon: (
      <svg className="size-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M9 9l10.5-3m0 6.553v3.75a2.25 2.25 0 01-1.632 2.163l-1.32.377a1.803 1.803 0 11-.99-3.467l2.31-.66a2.25 2.25 0 001.632-2.163zm0 0V2.25L9 5.25v10.303m0 0v3.75a2.25 2.25 0 01-1.632 2.163l-1.32.377a1.803 1.803 0 01-.99-3.467l2.31-.66A2.25 2.25 0 009 15.553z" />
      </svg>
    ),
    gradient: "from-teal-500 to-emerald-500",
  },
  {
    title: "Copy & Persist",
    desc: "One-click copy on any answer. Conversations survive page refreshes via localStorage — pick up right where you left off.",
    icon: (
      <svg className="size-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M15.666 3.888A2.25 2.25 0 0013.5 2.25h-3c-1.03 0-1.9.693-2.166 1.638m7.332 0c.055.194.084.4.084.612v0a.75.75 0 01-.75.75H9a.75.75 0 01-.75-.75v0c0-.212.03-.418.084-.612m7.332 0c.646.049 1.288.11 1.927.184 1.1.128 1.907 1.077 1.907 2.185V19.5a2.25 2.25 0 01-2.25 2.25H6.75A2.25 2.25 0 014.5 19.5V6.257c0-1.108.806-2.057 1.907-2.185a48.208 48.208 0 011.927-.184z" />
      </svg>
    ),
    gradient: "from-sky-500 to-blue-500",
  },
];

const STEPS = [
  {
    num: "01",
    title: "Upload Medical Image",
    desc: "Drag & drop or browse for an X-ray, MRI, CT scan, or pathology slide. Supported formats include PNG, JPEG, BMP, and DICOM.",
    color: "text-blue-600 dark:text-blue-400",
    bg: "bg-blue-50 dark:bg-blue-900/30",
  },
  {
    num: "02",
    title: "Ask & Follow Up",
    desc: "Type a clinical question, then continue the conversation with follow-ups suggested by the AI after each answer. Add patient context or draw a region of interest to refine results.",
    color: "text-violet-600 dark:text-violet-400",
    bg: "bg-violet-50 dark:bg-violet-900/30",
  },
  {
    num: "03",
    title: "Review & Export",
    desc: "Each answer includes a confidence ring, uncertainty flag, extracted finding tags, and optional Grad-CAM heatmap. Export as a structured clinical report or raw Markdown.",
    color: "text-emerald-600 dark:text-emerald-400",
    bg: "bg-emerald-50 dark:bg-emerald-900/30",
  },
];

const STATS = [
  { value: "7B", label: "LLM Parameters" },
  { value: "15", label: "Finding Patterns" },
  { value: "4-bit", label: "Quantization" },
  { value: "4/ea", label: "Follow-up Suggestions" },
];

/* ── Components ───────────────────────────────────────────────── */

function FeatureCard({ title, desc, icon, gradient, index }: FeatureCardProps) {
  const { ref, inView } = useInView();
  return (
    <div
      ref={ref}
      className={`group relative rounded-2xl border border-slate-200/80 dark:border-slate-700/80 bg-white/80 dark:bg-slate-800/80 p-6 shadow-sm hover:shadow-lg dark:hover:shadow-slate-900/60 transition-all duration-300 hover:-translate-y-0.5 ${
        inView ? "animate-fade-in-up" : "opacity-0 translate-y-5"
      }`}
      style={inView ? { animationDelay: `${index * 80}ms` } : undefined}
    >
      <div
        className={`size-11 rounded-xl bg-gradient-to-br ${gradient} flex items-center justify-center text-white shadow-sm mb-4 group-hover:scale-110 transition-transform duration-300`}
      >
        {icon}
      </div>
      <h3 className="text-base font-semibold text-slate-900 dark:text-white mb-2">{title}</h3>
      <p className="text-sm text-slate-500 dark:text-slate-400 leading-relaxed">{desc}</p>
    </div>
  );
}

function UxFeatureCard({ title, desc, icon, gradient, index }: FeatureCardProps) {
  const { ref, inView } = useInView();
  return (
    <div
      ref={ref}
      className={`group flex items-start gap-3 p-3.5 rounded-xl border border-slate-200/60 dark:border-slate-700/60 bg-white/60 dark:bg-slate-800/60 hover:bg-white dark:hover:bg-slate-700/80 hover:shadow-sm transition-all duration-200 ${
        inView ? "animate-fade-in-up" : "opacity-0 translate-y-5"
      }`}
      style={inView ? { animationDelay: `${index * 60}ms` } : undefined}
    >
      <div className={`size-9 shrink-0 rounded-lg bg-gradient-to-br ${gradient} flex items-center justify-center text-white shadow-sm`}>
        {icon}
      </div>
      <div>
        <h3 className="text-sm font-semibold text-slate-900 dark:text-white mb-0.5">{title}</h3>
        <p className="text-xs text-slate-500 dark:text-slate-400 leading-relaxed">{desc}</p>
      </div>
    </div>
  );
}

function StepCard({ num, title, desc, color, bg, index }: StepCardProps) {
  const { ref, inView } = useInView();
  return (
    <div
      ref={ref}
      className={`flex gap-5 ${
        inView ? "animate-fade-in-up" : "opacity-0 translate-y-5"
      }`}
      style={inView ? { animationDelay: `${index * 120}ms` } : undefined}
    >
      <div className="flex flex-col items-center">
        <div className={`size-10 rounded-xl ${bg} flex items-center justify-center ${color} text-sm font-bold`}>
          {num}
        </div>
        {index < 2 && (
          <div className="w-px flex-1 bg-gradient-to-b from-slate-200 dark:from-slate-700 to-transparent mt-2" />
        )}
      </div>
      <div className="pb-8">
        <h3 className="text-base font-semibold text-slate-900 dark:text-white mb-1.5">{title}</h3>
        <p className="text-sm text-slate-500 dark:text-slate-400 leading-relaxed max-w-md">{desc}</p>
      </div>
    </div>
  );
}

/* ── Intersection Observer hook ───────────────────────────────── */

function useInView(
  options?: IntersectionObserverInit
): { ref: (node: HTMLElement | null) => void; inView: boolean } {
  const [inView, setInView] = useState(false);
  const nodeRef = useRef<HTMLElement | null>(null);

  // Stable callback ref that stores the element without triggering re-renders
  const ref = useCallback((el: HTMLElement | null) => {
    nodeRef.current = el;
  }, []);

  // Set up observer on mount, clean up on unmount
  useEffect(() => {
    const el = nodeRef.current;
    if (!el) return;
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setInView(true);
          observer.disconnect();
        }
      },
      { threshold: 0.1, ...options }
    );
    observer.observe(el);
    return () => observer.disconnect();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  return { ref, inView };
}

/* ── Scroll-triggered section wrapper ─────────────────────────── */

function ScrollSection({
  children,
  className,
  childClassName,
  stagger,
}: {
  children: React.ReactNode;
  className?: string;
  childClassName?: string;
  stagger?: number;
}) {
  const { ref, inView } = useInView();
  const delay = stagger ? (stagger - 1) * 100 : 0;
  return (
    <div
      ref={ref}
      className={`transition-all duration-700 ease-out ${
        inView ? "opacity-100 translate-y-0" : "opacity-0 translate-y-6"
      } ${className ?? ""}`}
      style={{ transitionDelay: `${delay}ms` }}
    >
      <div className={childClassName ?? ""}>{children}</div>
    </div>
  );
}

/* ── CTA section with its own scroll animation ───────────────── */

function CtaSection() {
  const { ref, inView } = useInView();
  return (
    <section className="max-w-4xl mx-auto px-4 sm:px-6 pb-24">
      <div
        ref={ref}
        className={`relative overflow-hidden rounded-2xl bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 dark:from-slate-950 dark:via-slate-900 dark:to-slate-950 p-8 sm:p-12 text-center transition-all duration-700 ease-out ${
          inView ? "opacity-100 translate-y-0 scale-100" : "opacity-0 translate-y-8 scale-[0.97]"
        }`}
      >
        <div className="absolute top-0 left-1/2 -translate-x-1/2 w-96 h-96 bg-blue-500/10 rounded-full blur-3xl" />
        <div className="relative">
          <h2 className="text-2xl sm:text-3xl font-bold text-white mb-3">
            Ready to Try It?
          </h2>
          <p className="text-sm sm:text-base text-slate-300 dark:text-slate-400 max-w-lg mx-auto mb-8">
            No GPU required. Upload a medical image and start a conversation
            — the API handles the rest.
          </p>
          <div className="flex flex-col sm:flex-row items-center justify-center gap-3">
            <Link
              href="/diagnose"
              className="inline-flex items-center gap-2 h-12 px-8 rounded-xl bg-gradient-to-r from-blue-500 to-violet-500 text-white font-semibold text-sm shadow-lg hover:shadow-xl hover:from-blue-400 hover:to-violet-400 active:scale-[0.98] transition-all duration-200 animate-glow"
            >
              Open MedVQA Diagnose
              <svg className="size-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M13.5 4.5L21 12m0 0l-7.5 7.5M21 12H3" />
              </svg>
            </Link>
            <a
              href="#features"
              className="inline-flex items-center gap-2 h-12 px-6 rounded-xl border border-slate-600 bg-slate-800 text-slate-300 font-medium text-sm hover:bg-slate-700 hover:border-slate-500 active:scale-[0.98] transition-all duration-200"
            >
              View Features
              <svg className="size-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M19.5 8.25l-7.5 7.5-7.5-7.5" />
              </svg>
            </a>
          </div>
        </div>
      </div>
    </section>
  );
}

/* ── Keyboard Shortcut Badge ──────────────────────────────────── */

function Kbd({ children }: { children: string }) {
  return (
    <kbd className="px-1.5 py-0.5 rounded text-[10px] bg-slate-100 dark:bg-white/[0.06] border border-slate-200 dark:border-white/10 text-slate-400 dark:text-white/25 font-mono">
      {children}
    </kbd>
  );
}

/* ── Home Page ────────────────────────────────────────────────── */

export default function Home() {
  return (
    <div className="bg-grid">
      {/* ── Hero ───────────────────────────────────────────────── */}
      <section className="relative overflow-hidden px-4 sm:px-6 pt-16 sm:pt-24 pb-20 sm:pb-32">
        {/* Background blobs */}
        <div className="absolute top-0 -left-40 w-96 h-96 bg-blue-200/30 dark:bg-blue-500/10 rounded-full blur-3xl" />
        <div className="absolute top-20 -right-40 w-80 h-80 bg-violet-200/30 dark:bg-violet-500/10 rounded-full blur-3xl" />
        <div className="absolute bottom-0 left-1/3 w-64 h-64 bg-emerald-200/20 dark:bg-emerald-500/10 rounded-full blur-3xl" />

        <div className="relative max-w-4xl mx-auto text-center">
          {/* Badge */}
          <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-blue-50 dark:bg-blue-900/30 border border-blue-100 dark:border-blue-800 text-xs font-medium text-blue-600 dark:text-blue-400 mb-6 animate-fade-in-up">
            <span className="size-1.5 rounded-full bg-blue-500 animate-pulse" />
            Research Prototype &mdash; v0.1.0
          </div>

          {/* Title */}
          <h1 className="text-4xl sm:text-5xl lg:text-6xl font-bold tracking-tight text-slate-900 dark:text-white leading-[1.1] mb-6 animate-fade-in-up">
            Multimodal Medical
            <br />
            <span className="gradient-text">Visual Question Answering</span>
          </h1>

          {/* Subtitle */}
          <p className="text-lg sm:text-xl text-slate-500 dark:text-slate-400 max-w-2xl mx-auto leading-relaxed mb-8 animate-fade-in-up stagger-2">
            Upload a medical image and start a conversation. Ask clinical questions,
            get answers with confidence scores, follow up naturally, and export structured reports.
          </p>

          {/* Shortcut badges */}
          <div className="flex items-center justify-center gap-2.5 text-xs text-slate-400 dark:text-white/25 mb-10 animate-fade-in-up stagger-3">
            <span className="flex items-center gap-1"><Kbd>Ctrl+K</Kbd> upload</span>
            <span className="text-slate-200 dark:text-white/10">·</span>
            <span className="flex items-center gap-1"><Kbd>Ctrl+L</Kbd> clear</span>
            <span className="text-slate-200 dark:text-white/10">·</span>
            <span className="flex items-center gap-1"><Kbd>Esc</Kbd> focus</span>
            <span className="text-slate-200 dark:text-white/10">·</span>
            <span className="flex items-center gap-1"><Kbd>↵</Kbd> send</span>
          </div>

          {/* CTA */}
          <div className="flex flex-col sm:flex-row items-center justify-center gap-3 animate-fade-in-up stagger-4">
            <Link
              href="/diagnose"
              className="inline-flex items-center gap-2 h-12 px-6 rounded-xl bg-gradient-to-r from-blue-600 to-violet-600 text-white font-semibold text-sm shadow-md hover:shadow-lg hover:from-blue-500 hover:to-violet-500 active:scale-[0.98] transition-all duration-200"
            >
              Try the Demo
              <svg className="size-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M13.5 4.5L21 12m0 0l-7.5 7.5M21 12H3" />
              </svg>
            </Link>
            <a
              href="#features"
              className="inline-flex items-center gap-2 h-12 px-6 rounded-xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 text-slate-700 dark:text-slate-300 font-medium text-sm hover:bg-slate-50 dark:hover:bg-slate-700 hover:border-slate-300 dark:hover:border-slate-600 active:scale-[0.98] transition-all duration-200"
            >
              Learn More
              <svg className="size-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M19.5 8.25l-7.5 7.5-7.5-7.5" />
              </svg>
            </a>
          </div>

          {/* Floating visual */}
          <div className="hidden sm:block absolute -right-16 top-1/3 animate-float opacity-10 dark:opacity-5">
            <svg className="size-40 text-blue-500" fill="currentColor" viewBox="0 0 24 24">
              <path d="M9.75 3.104v5.714a2.25 2.25 0 01-.659 1.591L5 14.5M9.75 3.104c-.251.023-.501.05-.75.082m.75-.082a24.301 24.301 0 014.5 0m0 0v5.714c0 .597.237 1.17.659 1.591L19.8 15.3M14.25 3.104c.251.023.501.05.75.082M19.8 15.3l-1.57.393A9.065 9.065 0 0112 15a9.065 9.065 0 00-6.23.693L5 14.5m14.8.8l1.402 1.402c1.232 1.232.65 3.318-1.067 3.611A48.309 48.309 0 0112 21c-2.773 0-5.491-.235-8.135-.687-1.718-.293-2.3-2.379-1.067-3.61L5 14.5" />
            </svg>
          </div>
        </div>
      </section>

      {/* ── Stats ──────────────────────────────────────────────── */}
      <ScrollSection className="max-w-4xl mx-auto px-4 sm:px-6 -mt-8 mb-16"
        childClassName="glass rounded-2xl p-6 sm:p-8 grid grid-cols-2 sm:grid-cols-4 gap-6"
        stagger={3}
      >
        {STATS.map((stat) => (
          <div key={stat.label} className="text-center">
            <p className="text-2xl sm:text-3xl font-bold gradient-text">{stat.value}</p>
            <p className="text-xs text-slate-400 dark:text-slate-500 mt-1">{stat.label}</p>
          </div>
        ))}
      </ScrollSection>

      {/* ── Core Features ──────────────────────────────────────── */}
      <section id="features" className="max-w-6xl mx-auto px-4 sm:px-6 pb-16">
        <ScrollSection className="mb-12" childClassName="text-center" stagger={1}>
          <h2 className="text-2xl sm:text-3xl font-bold text-slate-900 dark:text-white mb-3">Core Intelligence</h2>
          <p className="text-sm text-slate-500 dark:text-slate-400 max-w-xl mx-auto">
            MedVQA combines state-of-the-art vision-language models with
            calibrated uncertainty, finding detection, and smart conversation features.
          </p>
        </ScrollSection>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-5">
          {FEATURES_CORE.map((f, i) => (
            <FeatureCard key={f.title} {...f} index={i} />
          ))}
        </div>
      </section>

      {/* ── UX & Productivity Features ─────────────────────────── */}
      <section className="max-w-6xl mx-auto px-4 sm:px-6 pb-16">
        <ScrollSection className="mb-10" childClassName="text-center" stagger={1}>
          <h2 className="text-2xl sm:text-3xl font-bold text-slate-900 dark:text-white mb-3">Interactive Capabilities</h2>
          <p className="text-sm text-slate-500 dark:text-slate-400 max-w-xl mx-auto">
            Designed for an efficient, keyboard-driven workflow with tools that
            put you in control of every analysis.
          </p>
        </ScrollSection>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
          {FEATURES_UX.map((f, i) => (
            <UxFeatureCard key={f.title} {...f} index={i} />
          ))}
        </div>
      </section>

      {/* ── How It Works ──────────────────────────────────────── */}
      <section className="max-w-4xl mx-auto px-4 sm:px-6 pb-20">
        <ScrollSection className="mb-12" childClassName="text-center" stagger={1}>
          <h2 className="text-2xl sm:text-3xl font-bold text-slate-900 dark:text-white mb-3">How It Works</h2>
          <p className="text-sm text-slate-500 dark:text-slate-400 max-w-xl mx-auto">
            From image upload to structured report in three steps.
          </p>
        </ScrollSection>

        <div className="max-w-lg mx-auto">
          {STEPS.map((s, i) => (
            <StepCard key={s.num} {...s} index={i} />
          ))}
        </div>
      </section>

      {/* ── Feature Showcase (callout row) ─────────────────────── */}
      <section className="max-w-5xl mx-auto px-4 sm:px-6 pb-20">
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
          {[
            { value: "15", label: "Finding patterns auto-detected in every answer", gradient: "from-blue-50 to-cyan-50 dark:from-blue-950/40 dark:to-cyan-950/40", border: "border-blue-100 dark:border-blue-900/50", textColor: "text-blue-600 dark:text-blue-400" },
            { value: "Unlimited", label: "Continuous follow-up questions in a single session", gradient: "from-violet-50 to-purple-50 dark:from-violet-950/40 dark:to-purple-950/40", border: "border-violet-100 dark:border-violet-900/50", textColor: "text-violet-600 dark:text-violet-400" },
            { value: "Persist", label: "Conversations survive page refreshes via localStorage", gradient: "from-emerald-50 to-teal-50 dark:from-emerald-950/40 dark:to-teal-950/40", border: "border-emerald-100 dark:border-emerald-900/50", textColor: "text-emerald-600 dark:text-emerald-400" },
          ].map((c, i) => {
            const { ref, inView } = useInView();
            return (
              <div key={c.value} ref={ref}
                className={`rounded-2xl bg-gradient-to-br ${c.gradient} border ${c.border} p-5 text-center transition-all duration-700 ease-out ${
                  inView ? "opacity-100 translate-y-0" : "opacity-0 translate-y-6"
                }`} style={{ transitionDelay: `${i * 120}ms` }}
              >
                <p className={`text-2xl font-bold ${c.textColor} mb-1`}>{c.value}</p>
                <p className="text-xs text-slate-500 dark:text-slate-400">{c.label}</p>
              </div>
            );
          })}
        </div>
      </section>

      {/* ── CTA Section ───────────────────────────────────────── */}
      <CtaSection />
    </div>
  );
}
