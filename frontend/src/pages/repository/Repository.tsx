import { useEffect, useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { listPapers, saveToLibrary, deletePaper, Paper } from "../../api";
import PaperViewerModal from "../../components/PaperViewerModal";
import { Button, EmptyState, PageHeader, PageShell, TextInput } from "../../components/ui";

const subjects = ["All Subjects", "Computer Science", "Mathematics"];
const documentTypes = ["All", "Journal Article", "Conference Paper", "Thesis", "Technical Report"];
const categories = [
  "All Categories", "Algorithms", "Distributed Systems", "Graph Theory",
  "Information Retrieval", "Linear Algebra", "Machine Learning",
  "Natural Language Processing", "Numerical Analysis", "Probability Theory", "Topology",
];

function categoryOf(paper: Paper) {
  const parts = paper.subject_category?.split(":", 2).map((p) => p.trim());
  return { subject: parts?.[0] ?? "", category: parts?.[1] ?? "" };
}

export default function Repository() {
  const [papers, setPapers] = useState<Paper[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [search, setSearch] = useState("");
  const [subject, setSubject] = useState(subjects[0]);
  const [category, setCategory] = useState(categories[0]);
  const [documentType, setDocumentType] = useState(documentTypes[0]);
  const [minYear, setMinYear] = useState(2008);
  const [maxYear, setMaxYear] = useState(2026);
  const [sortBy, setSortBy] = useState("date_added");
  const [savedIds, setSavedIds] = useState<Set<number>>(new Set());
  const [selectedPaper, setSelectedPaper] = useState<Paper | null>(null);

  const location = useLocation();
  const navigate = useNavigate();
  const navigationState = location.state as { selectSeed?: boolean; pipeline?: string } | null;
  const isSelectingSeed = navigationState?.selectSeed === true;
  const seedPipeline = navigationState?.pipeline ?? "tfidf";

  useEffect(() => {
    setLoading(true);
    setError(null);
    listPapers({
      search: search || undefined,
      subject,
      category,
      document_type: documentType,
      min_year: minYear,
      max_year: maxYear,
      sort_by: sortBy,
    })
      .then(setPapers)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, [search, subject, category, documentType, minYear, maxYear, sortBy]);

  async function handleSave(paperId: number) {
    try {
      await saveToLibrary(paperId);
      setSavedIds((prev) => new Set(prev).add(paperId));
    } catch (e) {
      setError(e instanceof Error ? `Couldn't save paper: ${e.message}` : "Couldn't save paper.");
    }
  }

  function handleSelectSeed(paperId: number) {
    navigate("/recommendations", { state: { mode: "seed", seedPaperId: paperId, pipeline: seedPipeline } });
  }

  async function handleDelete(paperId: number, title: string) {
    if (!window.confirm(`Delete "${title}" permanently? This removes the repository record, library entry, and stored file.`)) return;
    try {
      setError(null);
      await deletePaper(paperId);
      setPapers((prev) => prev.filter((paper) => paper.id !== paperId));
      if (selectedPaper?.id === paperId) setSelectedPaper(null);
    } catch (e) {
      setError(e instanceof Error ? `Couldn't delete "${title}": ${e.message}` : `Couldn't delete "${title}".`);
    }
  }

  return (
    <PageShell>
      <PageHeader
        eyebrow="Repository"
        title="Browse academic papers"
        description={`${loading ? "Loading papers…" : `${papers.length} papers shown`} · Filter by subject, category, document type, and publication year.`}
        action={
          <Button type="button" onClick={() => navigate("/upload")}>
            Upload paper
          </Button>
        }
      />

      {isSelectingSeed && (
        <div className="status-warning mb-5 flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
          <span>Select a paper to use as the seed document.</span>
          <button type="button" onClick={() => navigate("/repository", { replace: true })} className="text-sm underline">
            Cancel
          </button>
        </div>
      )}

      {/* Filters Section */}
      <section className="surface mb-6 p-4">
        <div className="grid gap-4 lg:grid-cols-[2fr_repeat(4,1fr)]">
          <div>
            <label className="filter-label" htmlFor="repo-search">Search</label>
            <TextInput id="repo-search" value={search} onChange={(e) => setSearch(e.target.value)} placeholder="Title, author, or keywords…" />
          </div>
          <FilterSelect label="Subject" value={subject} options={subjects} onChange={setSubject} />
          <FilterSelect label="Category" value={category} options={categories} onChange={setCategory} />
          <FilterSelect label="Document type" value={documentType} options={documentTypes} onChange={setDocumentType} />
          <div>
            <span className="filter-label">Year range</span>
            <div className="flex items-center gap-2">
              <TextInput type="number" value={minYear} onChange={(e) => setMinYear(Number(e.target.value))} aria-label="Minimum publication year" />
              <span className="text-muted">–</span>
              <TextInput type="number" value={maxYear} onChange={(e) => setMaxYear(Number(e.target.value))} aria-label="Maximum publication year" />
            </div>
          </div>
        </div>
        <div className="mt-4 flex items-center justify-between border-t border-line pt-3">
          <p className="text-xs text-muted">Filters apply as you change them.</p>
          <label className="flex items-center gap-2 text-xs text-muted">
            Sort
            <select value={sortBy} onChange={(e) => setSortBy(e.target.value)} className="rounded-md border border-line bg-navy px-2 py-1.5 text-ink focus:border-gold focus:outline-none">
              <option value="date_added">Date added</option>
              <option value="title">Title</option>
              <option value="publication_year">Publication year</option>
            </select>
          </label>
        </div>
      </section>

      {error && <div className="status-error mb-5">{error}</div>}

      {loading ? (
        <div className="empty-state"><p className="text-sm text-muted">Loading repository…</p></div>
      ) : papers.length === 0 ? (
        <EmptyState title="No papers match these filters." description="Try broadening the search or changing one of the filters." />
      ) : (
        <section className="overflow-hidden rounded-lg border border-line bg-panel" aria-label="Paper results">
          {papers.map((paper) => {
            const { subject: paperSubject, category: paperCategory } = categoryOf(paper);
            const isCS = paperSubject.toLowerCase().includes("computer");
            const valid = paper.is_valid_for_recommendation;

            return (
              <article key={paper.id} className="paper-row border-b border-line p-4 last:border-b-0">
                <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
                  <div className="min-w-0 flex-1">
                    {/* Metadata Badges */}
                    <div className="mb-1.5 flex items-center gap-2 text-xs">
                      {paperSubject && (
                        <span className={`rounded px-1.5 py-0.5 text-[11px] font-medium ${isCS ? "bg-cs/20 text-cs" : "bg-math/20 text-math"}`}>
                          {paperSubject}
                        </span>
                      )}
                      {paperCategory && (
                        <span className="text-muted">
                          {paperCategory}
                        </span>
                      )}
                      {(paperSubject || paperCategory) && <span className="text-muted">·</span>}
                      <span className="text-muted">{paper.publication_year ?? "Year unavailable"}</span>
                      <span className="text-muted">·</span>
                      <span className="text-muted">{paper.document_type ?? "Document"}</span>
                    </div>

                    {/* Paper Title */}
                    <button
                      type="button"
                      onClick={() => setSelectedPaper(paper)}
                      className="paper-title text-left text-sm font-medium text-ink hover:text-gold hover:underline"
                    >
                      {paper.title}
                    </button>

                    <p className="mt-1 text-xs text-muted">
                      {paper.author ?? "Unknown author"}
                    </p>
                  </div>

                  {/* Actions */}
                  <div className="paper-actions flex items-center gap-2 shrink-0 lg:justify-end">
                    {isSelectingSeed ? (
                      <Button variant="primary" type="button" disabled={!valid} onClick={() => handleSelectSeed(paper.id)}>
                        Use as seed
                      </Button>
                    ) : (
                      <>
                        <Button variant="secondary" type="button" onClick={() => handleSave(paper.id)} disabled={savedIds.has(paper.id)}>
                          {savedIds.has(paper.id) ? "Saved" : "Save paper"}
                        </Button>
                        
                        <Button variant="quiet" type="button" onClick={() => setSelectedPaper(paper)}>
                          View
                        </Button>

                        <button
                          type="button"
                          onClick={() => handleDelete(paper.id, paper.title)}
                          className="rounded border border-sbert/40 px-3 py-1.5 text-xs text-sbert hover:border-sbert transition-colors"
                        >
                          Delete
                        </button>
                      </>
                    )}
                  </div>
                </div>
              </article>
            );
          })}
        </section>
      )}

    <PaperViewerModal
      paper={selectedPaper}
      open={!!selectedPaper}
      onClose={() => setSelectedPaper(null)}
      canEdit
      onPaperUpdated={(updated) => {
        setPapers((prev) => prev.map((p) => (p.id === updated.id ? updated : p)));
        setSelectedPaper(updated);
    }}
/>
    </PageShell>
  );
}

function FilterSelect({ label, value, options, onChange }: { label: string; value: string; options: string[]; onChange: (value: string) => void }) {
  return (
    <div>
      <label className="filter-label">{label}</label>
      <select value={value} onChange={(e) => onChange(e.target.value)} className="min-h-10 w-full rounded-md border border-line bg-navy px-3 text-sm text-ink focus:border-gold focus:outline-none">
        {options.map((option) => <option key={option}>{option}</option>)}
      </select>
    </div>
  );
}