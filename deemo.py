import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np

from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing import image


class DigitClassifierApp:

    def __init__(self, root):

        self.root = root
        self.root.title("AI Digit Classification System")
        self.root.geometry("850x650")
        self.root.resizable(False, False)

        self.file = None
        self.photo = None
        self.model = None

        # =========================
        # HEADER
        # =========================

        header = tk.Frame(
            root,
            bg="#1f2937",
            height=80
        )
        header.pack(fill="x")

        title = tk.Label(
            header,
            text="AI DIGIT CLASSIFICATION",
            font=("Arial", 24, "bold"),
            fg="white",
            bg="#1f2937"
        )
        title.pack(pady=(15, 0))

        subtitle = tk.Label(
            header,
            text="CNN Based Image Classification System",
            font=("Arial", 11),
            fg="#d1d5db",
            bg="#1f2937"
        )
        subtitle.pack()

        # =========================
        # MAIN AREA
        # =========================

        main = tk.Frame(root, bg="#f3f4f6")
        main.pack(fill="both", expand=True)

        # Image section

        image_frame = tk.Frame(
            main,
            bg="white",
            bd=1,
            relief="solid"
        )
        image_frame.place(
            x=40,
            y=30,
            width=370,
            height=430
        )

        image_title = tk.Label(
            image_frame,
            text="IMAGE PREVIEW",
            font=("Arial", 14, "bold"),
            bg="white"
        )
        image_title.pack(pady=15)

        self.image_label = tk.Label(
            image_frame,
            text="No Image Selected",
            font=("Arial", 13),
            bg="#e5e7eb",
            width=30,
            height=15
        )
        self.image_label.pack(pady=10)

        # =========================
        # RESULT SECTION
        # =========================

        result_frame = tk.Frame(
            main,
            bg="white",
            bd=1,
            relief="solid"
        )
        result_frame.place(
            x=440,
            y=30,
            width=370,
            height=430
        )

        result_title = tk.Label(
            result_frame,
            text="PREDICTION RESULT",
            font=("Arial", 14, "bold"),
            bg="white"
        )
        result_title.pack(pady=15)

        self.prediction_label = tk.Label(
            result_frame,
            text="--",
            font=("Arial", 60, "bold"),
            bg="white"
        )
        self.prediction_label.pack(pady=10)

        self.confidence_label = tk.Label(
            result_frame,
            text="Confidence: --",
            font=("Arial", 16),
            bg="white"
        )
        self.confidence_label.pack(pady=5)

        self.status_label = tk.Label(
            result_frame,
            text="Model loading...",
            font=("Arial", 11),
            bg="white"
        )
        self.status_label.pack(pady=15)

        # =========================
        # BUTTONS
        # =========================

        button_frame = tk.Frame(
            main,
            bg="#f3f4f6"
        )
        button_frame.place(
            x=40,
            y=490,
            width=770,
            height=90
        )

        browse_button = tk.Button(
            button_frame,
            text="Browse Image",
            font=("Arial", 12, "bold"),
            width=18,
            height=2,
            command=self.load_image
        )
        browse_button.grid(
            row=0,
            column=0,
            padx=20
        )

        classify_button = tk.Button(
            button_frame,
            text="Classify Image",
            font=("Arial", 12, "bold"),
            width=18,
            height=2,
            command=self.classify_image
        )
        classify_button.grid(
            row=0,
            column=1,
            padx=20
        )

        clear_button = tk.Button(
            button_frame,
            text="Clear",
            font=("Arial", 12, "bold"),
            width=18,
            height=2,
            command=self.clear
        )
        clear_button.grid(
            row=0,
            column=2,
            padx=20
        )

        # Load model
        self.load_model()

    # =========================
    # LOAD MODEL
    # =========================

    def load_model(self):

        try:

            with open("model.json", "r") as file:
                model_json = file.read()

            self.model = model_from_json(model_json)

            self.model.load_weights(
                "model.weights.h5"
            )

            self.status_label.config(
                text="✓ Model Ready"
            )

        except Exception as e:

            self.status_label.config(
                text="Model Loading Failed"
            )

            print("Model Error:", e)

    # =========================
    # LOAD IMAGE
    # =========================

    def load_image(self):

        file_name = filedialog.askopenfilename(
            title="Select Digit Image",
            filetypes=[
                (
                    "Image Files",
                    "*.jpg *.jpeg *.png *.bmp"
                )
            ]
        )

        if not file_name:
            return

        self.file = file_name

        img = Image.open(file_name)

        img.thumbnail((300, 300))

        self.photo = ImageTk.PhotoImage(img)

        self.image_label.config(
            image=self.photo,
            text=""
        )

        self.prediction_label.config(
            text="--"
        )

        self.confidence_label.config(
            text="Confidence: --"
        )

        self.status_label.config(
            text="Image selected"
        )

    # =========================
    # CLASSIFY
    # =========================

    def classify_image(self):

        if self.file is None:

            messagebox.showwarning(
                "No Image",
                "Please select an image first."
            )

            return

        if self.model is None:

            messagebox.showerror(
                "Model Error",
                "Model is not loaded."
            )

            return

        try:

            test_image = image.load_img(
                self.file,
                target_size=(128, 128),
                color_mode="grayscale"
            )

            test_image = image.img_to_array(
                test_image
            )

            test_image = test_image / 255.0

            test_image = np.expand_dims(
                test_image,
                axis=0
            )

            prediction = self.model.predict(
                test_image,
                verbose=0
            )

            # Top prediction
            predicted_digit = np.argmax(
                prediction[0]
            )

            confidence = (
                prediction[0][predicted_digit]
                * 100
            )

            # Display result
            self.prediction_label.config(
                text=str(predicted_digit)
            )

            self.confidence_label.config(
                text=f"Confidence: {confidence:.2f}%"
            )

            self.status_label.config(
                text="✓ Classification completed"
            )

        except Exception as e:

            messagebox.showerror(
                "Classification Error",
                str(e)
            )

    # =========================
    # CLEAR
    # =========================

    def clear(self):

        self.file = None
        self.photo = None

        self.image_label.config(
            image="",
            text="No Image Selected"
        )

        self.prediction_label.config(
            text="--"
        )

        self.confidence_label.config(
            text="Confidence: --"
        )

        self.status_label.config(
            text="Model Ready"
        )


# =========================
# START APPLICATION
# =========================

root = tk.Tk()

app = DigitClassifierApp(root)

root.mainloop()
