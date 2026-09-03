import os

while True:

    print("\n" + "=" * 50)
    print("FEDERATED HEALTHCARE DIAGNOSIS SYSTEM")
    print("=" * 50)

    print("\n1. Start Server")
    print("2. Start Client")
    print("3. Exit")

    choice = input("\nEnter Choice: ")

    if choice == "1":

        os.system("python server_unified.py")

    elif choice == "2":

        os.system("python client_unified.py")

    elif choice == "3":

        print("\nExiting...")
        break

    else:

        print("\nInvalid Choice")