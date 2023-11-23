import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import subprocess  # Import the subprocess module

def check_website():
    url = website_entry.get()
    try:
        threshold = float(threshold_entry.get())
        # Replace this command with the correct command for your model
        command = f"python test_model.py --tokenizer_folder tokenizer --threshold {threshold} --model_dir saved_models --website_to_test {url}"
        result = subprocess.check_output(command, shell=True, text=True)
        messagebox.showinfo("Result", result)
    except ValueError:
        messagebox.showerror("Error", "Please enter a valid threshold (a number).")

# Create the main application window
app = tk.Tk()
app.title("Website Legitimacy Checker")

# Use a modern theme
style = ttk.Style()
style.theme_use("clam")

# Create a label and entry for the website URL
website_label = ttk.Label(app, text="Enter Website URL:")
website_label.pack()
website_entry = ttk.Entry(app, width=50)  # Increase the width
website_entry.pack()

# Create a label and entry for the threshold
threshold_label = ttk.Label(app, text="Threshold (e.g., 0.5):")
threshold_label.pack()
threshold_entry = ttk.Entry(app, width=10)  # Increase the width
threshold_entry.pack()

# Create a button to check the website
check_button = ttk.Button(app, text="Check Website", command=check_website)
check_button.pack()

# Start the GUI application
app.mainloop()
