import sqlite3

conn = sqlite3.connect("civic.db")
cursor = conn.cursor()

print("Departments Table:")
for row in cursor.execute("SELECT * FROM departments"):
    print(row)

print("\nIssues Table:")
for row in cursor.execute("SELECT * FROM issues"):
    print(row)

conn.close()
