import os
import time

def hello_world_function():
    print("Hello World")
    print("Hello World")
    print("Hello World")
    print("Hello World")
    print("Hello World")
    print("Hello World")
    print("Hello World")
    print("Hello World")
    print("Hello World")
    print("Hello World")
    print("Hello World")
    print("Hello World")
    print("Hello World")
    print("Hello World")
    print("Hello World")
    print("Hello World")
    print("Hello World")
    print("Hello World")
    print("Hello World")
    print("Hello World")
    print("Hello World")
    print("Hello World")
    print("Hello World")
    print("Hello World")
    print("Hello World")
    print("Hello World")
    print("Hello World")
    print("Hello World")
    print("Hello World")
    print("Hello World")
    print("Hello World")
    print("Hello World")
    print("Hello World")
    print("Hello World")
    print("Hello World")
    print("Hello World")

def monitor_directory(directory="."):
    print("Monitoring directory for new output videos...")
    existing_files = set(os.listdir(directory))

    while True:
        current_files = set(os.listdir(directory))
        new_files = current_files - existing_files

        for file in new_files:
            if file.startswith("output") and file.endswith(".mp4"):
                print(f"New video detected: {file}")
                hello_world_function()

        existing_files = current_files
        time.sleep(1)  # Check every second

if __name__ == "__main__":
    monitor_directory()