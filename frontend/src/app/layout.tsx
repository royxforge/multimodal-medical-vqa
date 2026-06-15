import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import Link from "next/link";
import { ThemeProvider } from "@/components/ThemeProvider";
import { ThemeToggle } from "@/components/ThemeToggle";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "MedVQA — Medical Visual Question Answering",
  description:
    "Upload a medical image and ask a clinical question. MedVQA answers with a confidence score and visual explanations.",
  icons: {
    icon: "/favicon.svg",
  },
};

function NavBar() {
  return (
    <nav className="fixed top-0 inset-x-0 z-50 glass border-b border-slate-200/50 dark:border-slate-700/50">
      <div className="max-w-6xl mx-auto px-4 sm:px-6 h-16 flex items-center justify-between">
        <Link href="/" className="flex items-center gap-2.5 group">
          <div className="size-9 rounded-xl bg-gradient-to-br from-blue-500 to-violet-600 flex items-center justify-center text-white text-sm font-bold shadow-sm group-hover:shadow-md group-hover:scale-105 transition-all duration-200">
            <svg className="size-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M9.75 3.104v5.714a2.25 2.25 0 01-.659 1.591L5 14.5M9.75 3.104c-.251.023-.501.05-.75.082m.75-.082a24.301 24.301 0 014.5 0m0 0v5.714c0 .597.237 1.17.659 1.591L19.8 15.3M14.25 3.104c.251.023.501.05.75.082M19.8 15.3l-1.57.393A9.065 9.065 0 0112 15a9.065 9.065 0 00-6.23.693L5 14.5m14.8.8l1.402 1.402c1.232 1.232.65 3.318-1.067 3.611A48.309 48.309 0 0112 21c-2.773 0-5.491-.235-8.135-.687-1.718-.293-2.3-2.379-1.067-3.61L5 14.5" />
            </svg>
          </div>
          <div>
            <h1 className="text-sm font-semibold tracking-tight text-slate-900 dark:text-white">MedVQA</h1>
            <p className="text-[11px] text-slate-400 dark:text-slate-500 -mt-0.5">
              Medical Visual Question Answering
            </p>
          </div>
        </Link>

        <div className="flex items-center gap-1 sm:gap-2">
          <Link
            href="/"
            className="px-3 py-1.5 text-sm text-slate-600 dark:text-slate-300 hover:text-slate-900 dark:hover:text-white hover:bg-slate-100 dark:hover:bg-slate-800 rounded-lg transition-colors"
          >
            Home
          </Link>
          <Link
            href="/diagnose"
            className="px-3 py-1.5 text-sm text-slate-600 dark:text-slate-300 hover:text-slate-900 dark:hover:text-white hover:bg-slate-100 dark:hover:bg-slate-800 rounded-lg transition-colors"
          >
            Diagnose
          </Link>
          <ThemeToggle />
          <div className="text-xs text-slate-400 dark:text-slate-500 bg-slate-100 dark:bg-slate-800 px-2.5 py-1 rounded-full font-medium">
            API mode
          </div>
        </div>
      </div>
    </nav>
  );
}

function Footer() {
  return (
    <footer className="border-t border-slate-200/60 dark:border-slate-800/60 bg-slate-50/50 dark:bg-slate-900/50 py-6">
      <div className="max-w-6xl mx-auto px-4 sm:px-6 text-center">
        <div className="flex items-center justify-center gap-1.5 mb-2">
          <div className="size-5 rounded-md bg-gradient-to-br from-blue-500 to-violet-600 flex items-center justify-center text-white text-[9px] font-bold">
            M
          </div>
          <span className="text-sm font-semibold text-slate-700 dark:text-slate-300">MedVQA</span>
        </div>
        <p className="text-xs text-slate-400 dark:text-slate-500">
          Research prototype &mdash;{" "}
          <span className="text-slate-300 dark:text-slate-600">Not for clinical use.</span>
        </p>
        <p className="text-[11px] text-slate-300 dark:text-slate-600 mt-1">
          Built with BioViL-T &middot; Mistral-7B &middot; QLoRA &middot; Grad-CAM
        </p>
      </div>
    </footer>
  );
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="en"
      className={`${geistSans.variable} ${geistMono.variable} h-full antialiased`}
      suppressHydrationWarning
    >
      <body className="min-h-full flex flex-col bg-gradient-to-br from-slate-50 via-white to-blue-50 dark:from-slate-950 dark:via-slate-900 dark:to-slate-950 font-[family-name:var(--font-geist-sans)]">
        {/* Inline script to prevent flash of light theme for dark-mode users */}
        <script
          dangerouslySetInnerHTML={{
            __html: `
              try {
                var t = localStorage.getItem('medvqa-theme');
                if (!t) t = window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
                if (t === 'dark') document.documentElement.classList.add('dark');
              } catch(e) {}
            `.trim(),
          }}
        />
        <ThemeProvider>
          <NavBar />
          <main className="flex-1 pt-16">{children}</main>
          <Footer />
        </ThemeProvider>
      </body>
    </html>
  );
}
