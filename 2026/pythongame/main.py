# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


import sys
import subprocess

def main():
    if len(sys.argv) < 2:
        print("Использование: python main.py [server|client]")
        return

    mode = sys.argv[1].lower()
    if mode == "server":
        subprocess.run([sys.executable, "server.py"])
    elif mode == "client":
        subprocess.run([sys.executable, "client.py"])
    else:
        print("Неизвестный режим. Используйте 'server' или 'client'.")

if __name__ == "__main__":
    main()
