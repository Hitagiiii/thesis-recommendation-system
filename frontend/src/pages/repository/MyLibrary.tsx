import { useEffect, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { getLibrary, removeFromLibrary, LibraryEntry, Paper } from "../../api";
import PaperViewerModal from "../../components/PaperViewerModal";
import { Button, EmptyState, PageHeader, PageShell } from "../../components/ui";

export default function MyLibrary() {
  const navigate = useNavigate();
  const [entries, setEntries] = useState<LibraryEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // PDF Viewer State
  const [selectedPaper, setSelectedPaper] = useState<Paper | null>(null);
  const [isViewerOpen, setIsViewerOpen] = useState(false);

  function load() {
    setLoading(true);
    setError(null);
    getLibrary()
      .then(setEntries)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }

  useEffect(load, []);

  async function handleRemove(paperId: number) {
    try {
      await removeFromLibrary(paperId);
      if (selectedPaper?.id === paperId) {
        handleClosePaper();
      }
      setEntries((prev) => prev.filter((entry) => entry.paper.id !== paperId));
    } catch (e) {
      setError(e instanceof Error ? e.message : "Couldn't remove paper from your library.");
    }
  }

  function handleOpenPaper(paper: Paper) {
    setSelectedPaper(paper);
    setIsViewerOpen(true);
  }

  function handleClosePaper() {
    setIsViewerOpen(false);
    setSelectedPaper(null);
  }

  function handleFindSimilar(paperId: number) {
    navigate("/recommendations", {
      state: {
        mode: "seed",
        seedPaperId: paperId,
        pipeline: "tfidf",
      },
    });
  }

  return (
    <PageShell>
      <PageHeader
        eyebrow="Saved papers"
        title="My Library"
        description={
          loading
            ? "Loading your saved papers…"
            : `${entries.length} saved paper${entries.length === 1 ? "" : "s"}.`
        }
        action={
          <Link to="/repository" className="text-sm text-gold hover:underline">
            + Browse Repository
          </Link>
        }
      />

      {error && (
        <div className="status-error mb-5 rounded border border-sbert/40 bg-sbert/10 px-3 py-2 text-sm text-sbert">
          Couldn't load your library: {error}. Is the backend running on port 8000?
        </div>
      )}

      {!loading && entries.length === 0 && !error ? (
        <EmptyState
          title="Your library is empty."
          description="Save papers from the repository or recommendation results to keep them here."
          action={
            <Link to="/repository" className="text-sm text-gold hover:underline">
              Browse repository
            </Link>
          }
        />
      ) : (
        <section className="overflow-hidden rounded-lg border border-line bg-panel">
          {entries.map(({ paper }) => {
            const subject = paper.subject_category?.split(":")[0]?.trim() ?? "";
            const isCS = subject.toLowerCase().includes("computer");

            const keywordList =
              paper.keywords
                ?.split(",")
                .map((k) => k.trim())
                .filter(Boolean) ?? [];

            return (
              <article key={paper.id} className="paper-row border-b border-line p-4 last:border-b-0">
                <div className="flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
                  <div className="min-w-0">
                    {/* Metadata Header with Tags */}
                    <div className="mb-2 flex items-center gap-2 text-xs">
                      <span
                        className={`rounded px-1.5 py-0.5 ${
                          isCS ? "bg-cs/20 text-cs" : "bg-math/20 text-math"
                        }`}
                      >
                        {isCS ? "CS" : "Math"}
                      </span>
                      <span className="text-muted">
                        {paper.publication_year ?? "—"}
                      </span>
                      <span className="text-muted">·</span>
                      <span className="text-muted">
                        {paper.citation_count ?? 0} cited
                      </span>
                    </div>

                    {/* Clickable Title */}
                    <button
                      type="button"
                      onClick={() => handleOpenPaper(paper)}
                      title="Open paper"
                      className="paper-title block text-left text-sm font-medium text-ink hover:text-gold hover:underline"
                    >
                      {paper.title}
                    </button>

                    <p className="mt-1 text-xs text-muted">
                      {paper.author ?? "Unknown author"}
                    </p>

                    {paper.abstract && (
                      <p className="mt-2 line-clamp-2 max-w-3xl text-xs leading-5 text-muted">
                        {paper.abstract}
                      </p>
                    )}

                    {/* Keywords List */}
                    <div className="mt-3 flex flex-wrap gap-1.5">
                      {keywordList.slice(0, 2).map((k) => (
                        <span
                          key={k}
                          className="rounded bg-panelAlt px-2 py-0.5 text-[11px] text-muted"
                        >
                          {k}
                        </span>
                      ))}
                      {keywordList.length > 2 && (
                        <span className="text-[11px] text-muted">
                          +{keywordList.length - 2} more
                        </span>
                      )}
                    </div>
                  </div>

                  {/* Actions - Vertically centered */}
                  <div className="paper-actions flex items-center gap-2 shrink-0 lg:mt-0 lg:justify-end">
                    <Button
                      variant="secondary"
                      type="button"
                      disabled={!paper.is_valid_for_recommendation}
                      title={
                        paper.is_valid_for_recommendation
                          ? undefined
                          : "This paper is missing required fields for recommendation"
                      }
                      onClick={() => handleFindSimilar(paper.id)}
                    >
                      Find Similar
                    </Button>

                    <Button
                      variant="quiet"
                      type="button"
                      onClick={() => handleOpenPaper(paper)}
                    >
                      View
                    </Button>

                    <button
                      type="button"
                      onClick={() => handleRemove(paper.id)}
                      className="inline-flex h-9 items-center justify-center rounded-md border border-sbert/40 px-3 text-xs font-medium text-sbert hover:border-sbert transition-colors"
                    >
                      Remove
                    </button>
                  </div>
                </div>
              </article>
            );
          })}
        </section>
      )}

      {/* PDF Viewer Modal */}
<PaperViewerModal
  paper={selectedPaper}
  open={isViewerOpen}
  onClose={handleClosePaper}
  onPaperUpdated={(updated) => {
    setEntries((prev) =>
      prev.map((entry) =>
        entry.paper.id === updated.id ? { ...entry, paper: updated } : entry
      )
    );
    setSelectedPaper(updated);
  }}
/>
    </PageShell>
  );
}