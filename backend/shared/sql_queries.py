"""Shared SQL query constants."""

SQL_SELECT_MANIFEST_FILES_BY_SNAPSHOT = """
SELECT rel_path, language, category
FROM manifest_files
WHERE snapshot_id=?
ORDER BY rel_path ASC
"""
