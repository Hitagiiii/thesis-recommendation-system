"""
One-time migration: adds the enrichment_notes column to the existing
papers table without touching any existing rows.

This column is purely a human-readable audit trail of what
app/services/metadata_enrichment.py changed and where it pulled the
value from, e.g.:

    "title <- crossref (confidence 0.81); abstract <- semantic_scholar
    (confidence 0.76)"

Nothing else in the system reads it -- it exists so you can say, in a
methodology write-up or a defense, exactly which fields were
auto-filled and from which source, for which papers. Safe to skip
entirely; app/services/metadata_enrichment.py checks with hasattr()
before writing to it, so enrichment still works without this column,
it just won't have anything to write the trace into.

Usage:
    python scripts/migrate_add_enrichment_notes.py

Safe to run more than once -- it checks whether the column already
exists first and does nothing if so.
"""

import os
import sqlite3

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))          # scripts/
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)                      # project root
DB_PATH = os.path.join(_PROJECT_ROOT, "app", "data", "academic_repository.db")
COLUMN_NAME = "enrichment_notes"


def column_exists(cursor, table: str, column: str) -> bool:
    cursor.execute(f"PRAGMA table_info({table})")
    existing_columns = [row[1] for row in cursor.fetchall()]
    return column in existing_columns


def main():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    if column_exists(cursor, "papers", COLUMN_NAME):
        print(f"'{COLUMN_NAME}' already exists on papers -- nothing to do.")
    else:
        cursor.execute(f"ALTER TABLE papers ADD COLUMN {COLUMN_NAME} TEXT")
        conn.commit()
        print(
            f"Added '{COLUMN_NAME}' column to papers. Existing rows are "
            f"untouched (new column defaults to NULL for them)."
        )

    conn.close()


if __name__ == "__main__":
    main()
