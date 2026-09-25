import { useState } from "react";
import { findPdfOnline, attachPdf, notifyRecommendationIndexStale } from "../api";
import type { Paper, PdfCandidate } from "../api";

interface FindPdfPanelProps {
  paper: Paper;
  onAttached: (paper: Paper) => void;
}

const sourceLabels: Record<string, string> = {
  unpaywall: "Unpaywall",
  semantic_scholar: "Semantic Scholar",
  arxiv: "arXiv",
  openalex: "OpenAlex",
};

// Errors from download_and_attach_pdf() that mean "the site blocked an
// automated download" rather than "nothing was found" -- these are the
// cases where suggesting the manual preview-then-upload path actually helps.
function isLikelyBlockedError(message: string): boolean {
  return (
    /HTTP \d{3}/.test(message) ||
    message.includes("did not return a PDF file") ||
    message.includes("Could not reach that link")
  );
}

export default function FindPdfPanel({ paper, onAttached }: FindPdfPanelProps) {
  const [searching, setSearching] = useState(false);
  const [searched, setSearched] = useState(false);
  const [candidates, setCandidates] = useState<PdfCandidate[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [attachingUrl, setAttachingUrl] = useState<string | null>(null);
  const [blockedUrl, setBlockedUrl] = useState<string | null>(null);

  async function handleSearch() {
    setSearching(true);
    setError(null);
    setBlockedUrl(null);

    try {
      const results = await findPdfOnline(paper.id);
      setCandidates(results);
      setSearched(true);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Search failed.");
    } finally {
      setSearching(false);
    }
  }

  async function handleConfirm(candidate: PdfCandidate) {
    setAttachingUrl(candidate.url);
    setError(null);
    setBlockedUrl(null);

    try {
      const updated = await attachPdf(paper.id, candidate.url);
      notifyRecommendationIndexStale();
      onAttached(updated);
    } catch (e) {
      const message =
        e instanceof Error ? e.message : "Could not attach that PDF.";
      setError(message);

      if (isLikelyBlockedError(message)) {
        setBlockedUrl(candidate.landing_page_url || candidate.url);
      }
    } finally {
      setAttachingUrl(null);
    }
  }

  return (
    <div className="mt-4 w-full max-w-md text-left">
      {!searched && (
        <button
          type="button"
          onClick={handleSearch}
          disabled={searching}
          className="rounded-md bg-gray-900 px-4 py-2 text-sm font-medium text-white hover:bg-gray-800 disabled:opacity-50"
        >
          {searching ? "Searching…" : "Find PDF Online"}
        </button>
      )}

      {error && (
        <div className="mt-3 rounded border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700">
          <p>{error}</p>

          {blockedUrl && (
            <p className="mt-2 text-xs leading-5 text-red-600">
              This source appears to be blocking automated downloads.
              Try{" "}
              <a
                href={blockedUrl}
                target="_blank"
                rel="noreferrer"
                className="underline"
              >
                opening it in your browser
              </a>
              , saving the PDF, then uploading it from the{" "}
              <strong>Upload</strong> page instead.
            </p>
          )}
        </div>
      )}

      {searched && !searching && candidates.length === 0 && !error && (
        <p className="mt-3 text-sm text-gray-500">
          No open-access PDF was found for this paper. You can still upload one manually.
        </p>
      )}

      {candidates.length > 0 && (
        <div className="mt-4 space-y-3">
          <p className="text-xs font-medium text-gray-500">
            Found {candidates.length} possible match{candidates.length > 1 ? "es" : ""} — confirm before attaching:
          </p>

          {candidates.map((candidate) => (
            <div key={candidate.url} className="rounded-md border border-gray-200 p-3">
              <p className="text-sm font-medium text-gray-900">
                {candidate.title || "Untitled result"}
              </p>

              <p className="mt-1 text-xs text-gray-500">
                Source: {sourceLabels[candidate.source] ?? candidate.source} ·{" "}
                {Math.round(candidate.confidence * 100)}% title match
                {candidate.license ? ` · ${candidate.license}` : ""}
              </p>

              <a
                href={candidate.landing_page_url || candidate.url}
                target="_blank"
                rel="noreferrer"
                className="mt-1 inline-block break-all text-xs text-blue-600 hover:underline"
              >
                {candidate.landing_page_url || candidate.url}
              </a>

              <div className="mt-2 flex gap-2">
                <button
                  type="button"
                  onClick={() => handleConfirm(candidate)}
                  disabled={attachingUrl !== null}
                  className="rounded-md bg-gray-900 px-3 py-1.5 text-xs font-medium text-white hover:bg-gray-800 disabled:opacity-50"
                >
                  {attachingUrl === candidate.url ? "Attaching…" : "Use this PDF"}
                </button>

                <a
                  href={candidate.url}
                  target="_blank"
                  rel="noreferrer"
                  className="rounded-md border border-gray-300 px-3 py-1.5 text-xs font-medium text-gray-700 hover:bg-gray-50"
                >
                  Preview
                </a>
              </div>
            </div>
          ))}

          <button
            type="button"
            onClick={handleSearch}
            disabled={searching}
            className="text-xs text-gray-500 hover:underline"
          >
            {searching ? "Searching…" : "Search again"}
          </button>
        </div>
      )}
    </div>
  );
}