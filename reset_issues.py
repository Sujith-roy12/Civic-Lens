import sqlite3

DB_FILE = "civic.db"

conn = sqlite3.connect(DB_FILE)
cursor = conn.cursor()

# Delete all records
cursor.execute("DELETE FROM issues")

# Reset autoincrement counter
cursor.execute("DELETE FROM sqlite_sequence WHERE name='issues'")

conn.commit()
conn.close()

print("✅ Issues table cleared and ID reset to 1.")
