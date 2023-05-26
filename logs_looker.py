import json

with open('logs.json') as f:
    templates = json.load(f)
    
print(templates)

