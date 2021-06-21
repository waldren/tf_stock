import os, random

def get_random_file(dir):
    n=0
    random.seed();
    for root, dirs, files in os.walk(dir):
        for name in files:
            n=n+1
            if random.uniform(0, n) < 1: rfile=os.path.join(root, name)
    return rfile

if __name__ == "__main__":
    for i in range(0,5):
        print(get_random_file('/app/data/training'))