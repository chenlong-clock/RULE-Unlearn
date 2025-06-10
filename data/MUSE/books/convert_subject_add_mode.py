import json



instances = json.load(open('data/MUSE/Target/books/reject_qa.json', 'r'))
for instance in instances:
    instance['subject'] = "books"
    instance['mode'] = "qa"
    instance["instruction"] = f"Question: {instance['instruction']}\nAnswer: "
    
with open('data/MUSE/Target/books/reject_qa.json', 'w') as f:
    json.dump(instances, f, indent=4)