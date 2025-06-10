import json
import random
random.seed(42)

def balance_dataset(completion_ratio=0.5, total_ratio=0.2):
    instances = json.load(open('data/MUSE/Target/books/reject_target.json', 'r'))
    
    qa_data = [instance for instance in instances if instance['mode'] == 'qa']
    completion_data = [instance for instance in instances if instance['mode'] == 'completion']
    completion_data = random.sample(completion_data, len(qa_data))
    with open(f'data/MUSE/Target/books/reject_target_{completion_ratio}.json', 'w') as f:
        json.dump(qa_data + completion_data, f, indent=4)
    total_data = qa_data + completion_data
    total_data = random.sample(total_data, int(len(total_data) * total_ratio))
    with open(f'data/MUSE/Target/books/reject_target_{completion_ratio}_{total_ratio}.json', 'w') as f:
        json.dump(total_data, f, indent=4)
    
    
    
if __name__ == '__main__':
    balance_dataset()