import tkinter as tk

from clapeg.usb_tools import (
    browse_button,
    detect_plugged_usbs,
    scan_usb_for_images
)


class USBImageScanner:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("USB Image Scanner")
        self.setup_gui()

    def check_entry_variable(self):
        return self.usb_path_entry.get()

    def setup_gui(self):
        # Variables
        self.usb_path_entry = tk.StringVar()
        self.progress_var = tk.IntVar()
        self.progress_var.set(0)

        # GUI elements
        tk.Label(self.root, text="Scan by folder:").grid(row=0, column=1, sticky="w", padx=10, pady=5)
        tk.Button(self.root, text="Path", command=self.browse_usb_path).grid(row=0, column=2, padx=10, pady=5)

        self.loading_label = tk.Label(self.root, text="  Start scanning for USBs: ")
        self.loading_label.grid(row=3, column=1, pady=10)

        tk.Button(
            self.root,
            text="Detect",
            command=self.start_detection,
        ).grid(row=3, column=2, pady=10, padx=10)

    def browse_usb_path(self):
        # Function to handle USB path browsing
        browse_button(self.usb_path_entry)
        csv_file_path = f"./metadata_from_usb.csv"
        scan_usb_for_images(self.check_entry_variable(), f"./images_from_usb/", csv_file_path)

    def start_detection(self):
        # Function to start USB image detection
        try:
            detect_plugged_usbs()
        except Exception as e:
            # Basic error handling for unexpected issues
            print(f"Error: {e}")
            # Consider showing an error message in the GUI

    def run(self):
        # Run the GUI
        self.root.mainloop()


if __name__ == "__main__":
    scanner = USBImageScanner()
    scanner.run()
