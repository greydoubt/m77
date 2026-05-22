import sqlite3
conn = sqlite3.connect("products.db")
cur = conn.cursor()
cur.execute("SELECT * FROM products WHERE Deleted = '1'")
records = cur.fetchmany(size=100)
for record in records:
    print(record)
 
cur.close()
conn.close()
 
