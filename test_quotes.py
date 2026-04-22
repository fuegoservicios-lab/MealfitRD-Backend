import sys
lines = open('cron_tasks.py', 'r', encoding='utf-8').readlines()
quotes = [i+1 for i, l in enumerate(lines) if '"""' in l]
print("TOTAL QUOTES:", len(quotes))
print(quotes)
