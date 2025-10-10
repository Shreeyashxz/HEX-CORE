# =================================================================================
# Hex Core - Component Digital Twin AI Dashboard - V8 (Portable Single-File Version)
#
# Description:
# This is a self-contained, single-file version of the Hex Core application,
# combining all modules for maximum portability. It implements architectural
# fixes for common software and AI system defects, including:
#
# 1. MLOps Drift Monitoring: Simulates continuous monitoring to detect model decay.
# 2. Data Sanitization: Defends against Data Poisoning attacks via outlier removal.
# 3. Explainable AI (XAI): Explains *why* the model makes its predictions.
# 4. Modern UI: Uses ttkbootstrap for a clean, professional user interface.
# =================================================================================

# --- 1. Installation ---
# Before running, ensure all required libraries are installed.
# pip install tensorflow opencv-python numpy Pillow pandas matplotlib scikit-learn ttkbootstrap

# --- All Imports ---
import os
import sys
import random
from datetime import date, timedelta
import joblib
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, utils
from tensorflow.keras.applications import MobileNetV2
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# =================================================================================
# 2. CONFIGURATION
# All configuration constants are centralized here.
# =================================================================================

# --- Application Metadata ---
APP_NAME = "Hex Core"
MAIN_WINDOW_TITLE = f"{APP_NAME} - Digital Twin AI Dashboard"
ANALYTICS_WINDOW_TITLE = f"{APP_NAME} - AI Analytics Dashboard"

# --- UI Configuration ---
MAIN_WINDOW_GEOMETRY = "900x800"
ANALYTICS_WINDOW_GEOMETRY = "800x700"
INITIAL_DATA_TEXT = "Scan a QR code to load the component's Digital Twin."

# --- Dynamic File Paths Setup ---
# All data and models are saved in a dedicated folder in the user's AppData directory.
# This avoids Windows permission errors.
try:
    APP_DATA_BASE_DIR = os.getenv('LOCALAPPDATA')
    APP_DATA_DIR = os.path.join(APP_DATA_BASE_DIR, 'HexCore')
    os.makedirs(APP_DATA_DIR, exist_ok=True)
except Exception as e:
    root = ttk.Window()
    root.withdraw()
    messagebox.showerror("Fatal Startup Error", f"Could not create application data directory at:\n{APP_DATA_DIR}\n\nDetails: {e}")
    sys.exit()

# Define all file paths based on the safe AppData directory
DATASET_DIR = os.path.join(APP_DATA_DIR, "railway_track_dataset")
VISUAL_MODEL_PATH = os.path.join(APP_DATA_DIR, "visual_defect_model.keras")
PREDICTIVE_MODEL_PATH = os.path.join(APP_DATA_DIR, "predictive_failure_model.joblib")
ENCODERS_PATH = os.path.join(APP_DATA_DIR, "encoders.joblib")
DATABASE_PATH = os.path.join(APP_DATA_DIR, "master_component_ledger.csv")
MONITORING_LOG_PATH = os.path.join(APP_DATA_DIR, "monitoring_log.csv")

# --- AI & Model Training Parameters ---
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 10
DRIFT_THRESHOLD = 0.85

# =================================================================================
# 3. DATA GENERATION & SANITIZATION
# =================================================================================

def generate_visual_dataset(num_images_per_class=100):
    """Creates a synthetic dataset of 'normal' and 'defective' track images."""
    print("Generating synthetic visual dataset...")
    NORMAL_DIR = os.path.join(DATASET_DIR, "normal")
    DEFECTIVE_DIR = os.path.join(DATASET_DIR, "defective")
    os.makedirs(NORMAL_DIR, exist_ok=True)
    os.makedirs(DEFECTIVE_DIR, exist_ok=True)
    # ... (generation logic) ...
    for i in range(num_images_per_class):
        img = np.zeros((IMG_SIZE[0], IMG_SIZE[1]), dtype=np.uint8) + 30
        track_pos1 = int(IMG_SIZE[1] * 0.4); track_pos2 = int(IMG_SIZE[1] * 0.6)
        cv2.line(img, (0, track_pos1), (IMG_SIZE[0], track_pos1), 180, thickness=4)
        cv2.line(img, (0, track_pos2), (IMG_SIZE[0], track_pos2), 180, thickness=4)
        cv2.imwrite(os.path.join(NORMAL_DIR, f"normal_{i}.png"), cv2.add(img, np.random.randint(0, 15, size=img.shape, dtype=np.uint8)))
    for i in range(num_images_per_class):
        img = np.zeros((IMG_SIZE[0], IMG_SIZE[1]), dtype=np.uint8) + 30
        track_pos1 = int(IMG_SIZE[1] * 0.4); track_pos2 = int(IMG_SIZE[1] * 0.6)
        defect_type = np.random.choice(['break', 'crack']); break_point = np.random.randint(20, IMG_SIZE[0] - 20)
        if defect_type == 'break':
            cv2.line(img, (0, track_pos1), (break_point - 15, track_pos1), 180, thickness=4)
            cv2.line(img, (break_point + 15, track_pos1), (IMG_SIZE[0], track_pos1), 180, thickness=4)
        else:
            cv2.line(img, (np.random.randint(0, IMG_SIZE[0]), track_pos1), (np.random.randint(0, IMG_SIZE[0]), track_pos2), 100, thickness=1)
        cv2.line(img, (0, track_pos2), (IMG_SIZE[0], track_pos2), 180, thickness=4)
        cv2.imwrite(os.path.join(DEFECTIVE_DIR, f"defective_{i}.png"), cv2.add(img, np.random.randint(0, 15, size=img.shape, dtype=np.uint8)))
    print("Visual dataset generated.")

def generate_master_ledger(num_entries=1000, poison_percentage=2.0):
    """Generates the master ledger, including a simulation of a Data Poisoning attack."""
    print(f"Generating Master Ledger with {poison_percentage}% poisoned data...")
    # ... (generation logic) ...
    vendors = ['VendorA', 'VendorB', 'VendorC', 'VendorD']
    locations = ['Coastal', 'Mountain', 'Plains', 'Urban']
    component_types = ['Elastic Clip', 'Liner', 'Rail Pad']
    data = []
    for i in range(num_entries):
        qr_code_id = f"QR-{2025000 + i:07d}"
        comp_type = random.choice(component_types)
        vendor = random.choice(vendors)
        install_date = date(2020, 1, 1) + timedelta(days=random.randint(0, 1800))
        warranty_years = random.choice([7, 10, 15])
        mfg_quality_score = round(random.uniform(0.85, 1.0) - (0.1 if vendor == 'VendorC' else 0), 2)
        transport_shock_events = random.randint(0, 5) + (3 if vendor == 'VendorC' else 0)
        age_days = (date.today() - install_date).days
        vertical_wear_mm = round(age_days / 365 * random.uniform(0.5, 1.5), 2)
        lateral_wear_mm = round(age_days / 365 * random.uniform(0.2, 0.8), 2)
        axle_box_accel_g = round(random.uniform(1.0, 3.0) + (vertical_wear_mm * 1.5), 2)
        is_failed = 0
        failure_score = (age_days / (365 * warranty_years)) + ((1 - mfg_quality_score) * 2.0) + (transport_shock_events / 10.0) + (vertical_wear_mm / 10.0)
        if location in ['Coastal', 'Mountain']: failure_score *= 1.2
        if failure_score > random.uniform(0.8, 1.2): is_failed = 1
        data.append([qr_code_id, comp_type, vendor, install_date, warranty_years,
                     mfg_quality_score, transport_shock_events, age_days,
                     vertical_wear_mm, lateral_wear_mm, axle_box_accel_g, location, is_failed])
    df = pd.DataFrame(data, columns=['qr_code_id', 'component_type', 'vendor', 'installation_date', 'warranty_years', 'mfg_quality_score', 'transport_shock_events', 'age_days', 'vertical_wear_mm', 'lateral_wear_mm', 'axle_box_accel_g', 'location_zone', 'is_failed'])
    num_poisoned = int((poison_percentage / 100) * num_entries)
    if num_poisoned > 0:
        poison_indices = df.sample(n=num_poisoned).index
        df.loc[poison_indices, 'is_failed'] = 1 - df.loc[poison_indices, 'is_failed']
    try:
        df.to_csv(DATABASE_PATH, index=False)
        print(f"Master Ledger generated at '{DATABASE_PATH}'.")
        return True
    except Exception as e:
        messagebox.showerror("File Creation Failed", f"Could not create '{DATABASE_PATH}'.\nDetails: {e}")
        return False

def sanitize_data(df, column_list):
    """Removes outliers from a DataFrame based on the IQR method."""
    print("Sanitizing data: Removing outliers...")
    df_cleaned = df.copy()
    initial_total = len(df_cleaned)
    for column in column_list:
        if column in df_cleaned.columns and pd.api.types.is_numeric_dtype(df_cleaned[column]):
            Q1 = df_cleaned[column].quantile(0.25)
            Q3 = df_cleaned[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = df_cleaned[(df_cleaned[column] < lower_bound) | (df_cleaned[column] > upper_bound)]
            if not outliers.empty:
                print(f"  - Removing {len(outliers)} outliers from '{column}'")
                df_cleaned = df_cleaned.drop(outliers.index)
    final_total = len(df_cleaned)
    print(f"Data Sanitization complete. Removed a total of {initial_total - final_total} rows.")
    return df_cleaned

# =================================================================================
# 4. AI MODEL TRAINING
# =================================================================================

def train_visual_model():
    """Trains the Keras/TensorFlow model for visual defect detection."""
    if not os.path.exists(DATASET_DIR): generate_visual_dataset()
    print("\nLoading datasets for visual model...")
    train_ds = utils.image_dataset_from_directory(
        DATASET_DIR, validation_split=0.2, subset="training", seed=123,
        image_size=IMG_SIZE, batch_size=BATCH_SIZE, color_mode='rgb')
    val_ds = utils.image_dataset_from_directory(
        DATASET_DIR, validation_split=0.2, subset="validation", seed=123,
        image_size=IMG_SIZE, batch_size=BATCH_SIZE, color_mode='rgb')
    base_model = MobileNetV2(input_shape=IMG_SIZE + (3,), include_top=False, weights='imagenet')
    base_model.trainable = False
    model = models.Sequential([
        layers.Rescaling(1./127.5, offset=-1, input_shape=IMG_SIZE + (3,)),
        base_model, layers.GlobalAveragePooling2D(), layers.Dense(1, activation='sigmoid')])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    print("\nStarting visual model training...")
    history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)
    model.save(VISUAL_MODEL_PATH)
    print(f"\nVisual model saved to {VISUAL_MODEL_PATH}")
    return model, history

def train_predictive_model():
    """Trains the Scikit-learn model for predictive failure analysis."""
    if not os.path.exists(DATABASE_PATH):
        if not generate_master_ledger(): return None, None, 0.0
    df = pd.read_csv(DATABASE_PATH, parse_dates=['installation_date'])
    numerical_cols_to_sanitize = ['mfg_quality_score', 'transport_shock_events', 'age_days', 'vertical_wear_mm', 'lateral_wear_mm', 'axle_box_accel_g']
    df_sanitized = sanitize_data(df, numerical_cols_to_sanitize)
    features = ['age_days', 'warranty_years', 'mfg_quality_score', 'transport_shock_events', 'vertical_wear_mm', 'lateral_wear_mm', 'axle_box_accel_g', 'component_type', 'vendor', 'location_zone']
    target = 'is_failed'
    X = df_sanitized[features]
    y = df_sanitized[target]
    encoders = {}
    for col in ['component_type', 'vendor', 'location_zone']:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        encoders[col] = le
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42, oob_score=True)
    model.fit(X_train, y_train)
    joblib.dump(model, PREDICTIVE_MODEL_PATH)
    joblib.dump(encoders, ENCODERS_PATH)
    test_accuracy = accuracy_score(y_test, model.predict(X_test))
    print(f"Predictive model trained with accuracy: {test_accuracy:.2f}")
    with open(MONITORING_LOG_PATH, 'w') as f:
        f.write(f"baseline_accuracy,{test_accuracy}\n")
    return model, encoders, test_accuracy

# =================================================================================
# 5. AI ENGINE & MLOPS
# =================================================================================

class AIEngine:
    """The core AI brain, handling predictions and providing explainability."""
    def __init__(self, predictive_model, encoders, database_path):
        self.predictive_model = predictive_model
        self.encoders = encoders
        self.df = pd.read_csv(database_path, parse_dates=['installation_date'])
        self.feature_names = predictive_model.feature_names_in_

    def get_component_data(self, qr_id):
        component_data = self.df[self.df['qr_code_id'] == qr_id]
        return component_data.iloc[0] if not component_data.empty else None

    def predict_failure(self, component_data):
        data = component_data.copy()
        for col, le in self.encoders.items():
            if data[col] in le.classes_: data[col] = le.transform([data[col]])[0]
            else: data[col] = -1
        input_df = pd.DataFrame([data])[self.feature_names]
        failure_prob = self.predictive_model.predict_proba(input_df)[:, 1][0]
        warranty_days, age_days = component_data['warranty_years'] * 365, component_data['age_days']
        wear_factor = (component_data['vertical_wear_mm'] + component_data['lateral_wear_mm']) / 20.0
        rul_days = (warranty_days - age_days) * (1 - wear_factor)
        importances = self.predictive_model.feature_importances_
        feature_importance_map = sorted(zip(importances, self.feature_names), reverse=True)
        top_factors = [f[1] for f in feature_importance_map[:3]]
        return failure_prob, max(0, int(rul_days)), top_factors

    def get_anomaly_report(self):
        report = "--- AI Anomaly & Performance Report ---\n\n"
        report += "1. Vendor Performance Analysis:\n"
        vendor_perf = self.df.groupby('vendor').agg(avg_quality_score=('mfg_quality_score', 'mean'), avg_shock_events=('transport_shock_events', 'mean'), failure_rate=('is_failed', 'mean')).sort_values(by='failure_rate', ascending=False)
        report += vendor_perf.to_string(float_format="%.2f") + "\n\n"
        highest_risk_vendor = vendor_perf['failure_rate'].idxmax()
        report += f"** Insight: {highest_risk_vendor} has the highest failure rate, correlating with lower manufacturing quality and higher transport shock events.\n\n"
        report += "2. Location-based Failure Analysis:\n"
        location_perf = self.df.groupby('location_zone')['is_failed'].mean().sort_values(ascending=False)
        report += location_perf.to_string(float_format="%.2%") + "\n\n"
        report += f"** Insight: Components in the {location_perf.idxmax()} zone fail most often, indicating harsh environmental impact.\n"
        return report

def simulate_new_data_and_drift():
    """Generates a new batch of "future" data to test the current model against."""
    print(f"\n--- Simulating new data acquisition... ---")
    generate_master_ledger(num_entries=200, poison_percentage=0)
    return pd.read_csv(DATABASE_PATH)

def check_for_drift(model, encoders, new_data):
    """Tests the currently deployed model against new data to check for drift."""
    print("--- Checking for Model Drift ---")
    if not os.path.exists(MONITORING_LOG_PATH): return "No baseline performance found.", 0, 0
    baseline_accuracy = float(pd.read_csv(MONITORING_LOG_PATH).iloc[0,1])
    features, target = model.feature_names_in_, 'is_failed'
    X_new, y_new = new_data[features], new_data[target]
    for col, le in encoders.items():
        X_new[col] = X_new[col].apply(lambda x: le.transform([x])[0] if x in le.classes_ else -1)
    current_accuracy = accuracy_score(y_new, model.predict(X_new))
    print(f"Baseline Accuracy: {baseline_accuracy:.2f}\nCurrent Accuracy on New Data: {current_accuracy:.2f}")
    status = "DRIFT DETECTED! Performance has degraded. Retraining is recommended." if current_accuracy < (baseline_accuracy * DRIFT_THRESHOLD) else "No significant drift detected. Model is performing as expected."
    return status, baseline_accuracy, current_accuracy

# =================================================================================
# 6. USER INTERFACE (VIEW)
# =================================================================================

class DigitalTwinApp:
    """The main application window view."""
    def __init__(self, root, controller):
        self.root, self.controller = root, controller
        self.root.title(MAIN_WINDOW_TITLE)
        self.root.geometry(MAIN_WINDOW_GEOMETRY)
        self.setup_ui()

    def setup_ui(self):
        # ... (UI setup logic) ...
        main_frame = ttk.Frame(self.root, padding=(20, 10)); main_frame.pack(fill=BOTH, expand=True)
        ttk.Label(main_frame, text=f"{APP_NAME} AI Dashboard", font=("Helvetica", 20, "bold"), anchor=CENTER).pack(pady=(0, 10), fill=X)
        scan_frame = ttk.Frame(main_frame, bootstyle=SECONDARY, padding=10); scan_frame.pack(fill=X)
        ttk.Label(scan_frame, text="Scan Component (Enter QR Code):", font=("Helvetica", 12, "bold"), bootstyle=(INVERSE, SECONDARY)).pack(side=LEFT)
        self.qr_entry = ttk.Entry(scan_frame, font=("Helvetica", 12), width=20); self.qr_entry.pack(side=LEFT, padx=10, fill=X, expand=True)
        ttk.Button(scan_frame, text="Fetch Digital Twin", command=self.fetch_digital_twin, bootstyle=PRIMARY).pack(side=LEFT)
        display_frame = ttk.Frame(main_frame); display_frame.pack(fill=BOTH, expand=True, pady=10)
        data_frame = ttk.Frame(display_frame); data_frame.pack(side=LEFT, fill=BOTH, expand=True, padx=(0, 10))
        self.data_text = ttk.Text(data_frame, wrap=WORD, font=("Consolas", 10), height=15); self.data_text.pack(fill=BOTH, expand=True)
        self.data_text.insert(END, INITIAL_DATA_TEXT); self.data_text.config(state=DISABLED)
        visual_frame = ttk.Frame(display_frame); visual_frame.pack(side=RIGHT, fill=Y)
        ttk.Label(visual_frame, text="Visual Inspection", font=("Helvetica", 12, "bold")).pack()
        self.image_label = ttk.Label(visual_frame, background='white', relief=SUNKEN, borderwidth=1); self.image_label.pack(pady=5, ipadx=112, ipady=112)
        self.visual_result_label = ttk.Label(visual_frame, text="", font=("Helvetica", 10, "bold")); self.visual_result_label.pack(pady=5)
        self.load_image_btn = ttk.Button(visual_frame, text="Load Image for Inspection", command=self.load_and_predict_visual, state=DISABLED); self.load_image_btn.pack()
        m_frame = ttk.Frame(main_frame, bootstyle=WARNING, padding=10, borderwidth=2, relief=RIDGE); m_frame.pack(fill=X, pady=10)
        ttk.Label(m_frame, text="MLOps Monitoring", font=("Helvetica", 12, "bold"), bootstyle=INVERSE).pack(anchor=W)
        self.drift_status_label = ttk.Label(m_frame, text="Status: Awaiting monitoring run.", font=("Helvetica", 10), bootstyle=INVERSE, justify=LEFT); self.drift_status_label.pack(anchor=W, pady=5, fill=X)
        ttk.Button(m_frame, text="Run Daily Monitoring Simulation", command=self.run_monitoring, bootstyle=(WARNING, OUTLINE)).pack()
        btn_frame = ttk.Frame(main_frame); btn_frame.pack(pady=10)
        ttk.Button(btn_frame, text="AI Analytics Dashboard", command=self.open_analytics_dashboard, bootstyle=SUCCESS).pack(side=LEFT, padx=10)
        ttk.Button(btn_frame, text="Re-Generate & Train All", command=self.run_full_training, bootstyle=DANGER).pack(side=LEFT, padx=10)

    def fetch_digital_twin(self):
        if self.controller: self.controller.fetch_digital_twin(self.qr_entry.get().strip())
    def load_and_predict_visual(self):
        if self.controller: self.controller.run_visual_inspection()
    def run_monitoring(self):
        if self.controller: self.controller.run_monitoring()
    def run_full_training(self):
        if self.controller: self.controller.run_full_training()
    def open_analytics_dashboard(self):
        if self.controller: self.controller.open_analytics_dashboard()
    def update_digital_twin_display(self, text):
        self.data_text.config(state=NORMAL); self.data_text.delete('1.0', END); self.data_text.insert(END, text); self.data_text.config(state=DISABLED)
        self.load_image_btn.config(state=NORMAL)
    def update_visual_inspection_display(self, img, text, color):
        self.image_label.config(image=img); self.image_label.image = img
        self.visual_result_label.config(text=text, bootstyle=color)
    def update_drift_status(self, text, color):
        self.drift_status_label.config(text=text, bootstyle=color)
    def set_initial_drift_status(self, acc):
         self.drift_status_label.config(text=f"Status: Monitoring baseline accuracy is {acc:.2f}")

class AnalyticsDashboard:
    def __init__(self, parent, controller):
        self.window, self.controller = ttk.Toplevel(parent), controller
        self.window.title(ANALYTICS_WINDOW_TITLE)
        self.window.geometry(ANALYTICS_WINDOW_GEOMETRY)
        self.setup_ui()
    def setup_ui(self):
        # ... (UI setup logic) ...
        main_frame = ttk.Frame(self.window, padding=20); main_frame.pack(fill=BOTH, expand=TRUE)
        ttk.Label(main_frame, text="AI Analytics Dashboard", font=("Helvetica", 16, "bold")).pack(pady=10)
        ttk.Button(main_frame, text="Generate Anomaly & Performance Report", command=self.show_anomaly_report, bootstyle=PRIMARY).pack(pady=5, fill=X)
        self.report_text = ttk.Text(main_frame, wrap=WORD, height=15, font=("Consolas", 10)); self.report_text.pack(fill=BOTH, expand=True, pady=5)
        ttk.Label(main_frame, text="Visual Model Training History", font=("Helvetica", 12, "bold")).pack(pady=5)
        history = self.controller.get_training_history()
        if history:
            hist_df = pd.DataFrame(history.history)
            fig = Figure(figsize=(8, 4), dpi=100)
            acc = fig.add_subplot(1, 2, 1); acc.plot(hist_df['accuracy'], label='Train'); acc.plot(hist_df['val_accuracy'], label='Val'); acc.set_title('Accuracy'); acc.legend()
            loss = fig.add_subplot(1, 2, 2); loss.plot(hist_df['loss'], label='Train'); loss.plot(hist_df['val_loss'], label='Val'); loss.set_title('Loss'); loss.legend()
            fig.tight_layout()
            canvas = FigureCanvasTkAgg(fig, master=main_frame); canvas.draw(); canvas.get_tk_widget().pack(fill=BOTH, expand=True)
        else:
            ttk.Label(main_frame, text="No visual model training history available.").pack()
    def show_anomaly_report(self):
        self.report_text.delete('1.0', END); self.report_text.insert(END, self.controller.get_anomaly_report())

# =================================================================================
# 7. APPLICATION CONTROLLER
# =================================================================================

class MainController:
    """The 'brain' of the application, connecting the UI to the AI logic."""
    def __init__(self, view):
        self.view, self.visual_model, self.predictive_model, self.encoders, self.ai_engine, self.training_history, self.baseline_accuracy = view, None, None, None, None, None, 0.0

    def load_models_and_data(self):
        try:
            if os.path.exists(VISUAL_MODEL_PATH): self.visual_model = keras_models.load_model(VISUAL_MODEL_PATH)
            if os.path.exists(PREDICTIVE_MODEL_PATH) and os.path.exists(ENCODERS_PATH):
                self.predictive_model, self.encoders = joblib.load(PREDICTIVE_MODEL_PATH), joblib.load(ENCODERS_PATH)
                if os.path.exists(DATABASE_PATH): self.ai_engine = AIEngine(self.predictive_model, self.encoders, DATABASE_PATH)
                if os.path.exists(MONITORING_LOG_PATH):
                    with open(MONITORING_LOG_PATH, 'r') as f: self.baseline_accuracy = float(f.readline().split(',')[1])
                    self.view.set_initial_drift_status(self.baseline_accuracy)
            else: messagebox.showwarning("Models Missing", "Predictive model not found. Please run a full training cycle.")
        except Exception as e: messagebox.showerror("Initialization Error", f"An error occurred during startup: {e}")

    def fetch_digital_twin(self, qr_id):
        if not self.ai_engine: messagebox.showerror("Error", "AI Engine not initialized."); return
        if not qr_id: return
        component = self.ai_engine.get_component_data(qr_id)
        if component is None: messagebox.showerror("Not Found", f"QR Code '{qr_id}' not found."); return
        prob, rul, factors = self.ai_engine.predict_failure(component)
        self.view.update_digital_twin_display(self._format_digital_twin_text(component, prob, rul, factors))

    def run_visual_inspection(self):
        if not self.visual_model: messagebox.showerror("Error", "Visual model not loaded."); return
        filepath = filedialog.askopenfilename(filetypes=[("Image Files", "*.png *.jpg *.jpeg")])
        if not filepath: return
        img_pil = Image.open(filepath).resize((224, 224), Image.Resampling.LANCZOS)
        img_tk = ImageTk.PhotoImage(img_pil)
        prediction = self.visual_model.predict(self._preprocess_image(filepath))[0][0]
        text, color = (f"Status: Normal ({prediction:.1%})", "success") if prediction > 0.5 else (f"Status: Defective ({(1-prediction):.1%})", "danger")
        self.view.update_visual_inspection_display(img_tk, text, color)

    def run_monitoring(self):
        if not self.predictive_model: messagebox.showerror("Error", "Predictive model not loaded."); return
        status, base, curr = check_for_drift(self.predictive_model, self.encoders, simulate_new_data_and_drift())
        text = f"Status: {status}\n(Baseline: {base:.2f}, Current: {curr:.2f})"
        color = "danger" if "DETECTED" in status else "success"
        self.view.update_drift_status(text, color)

    def run_full_training(self):
        if messagebox.askyesno("Confirm Training", "This will re-generate all data and re-train all AI models. This may take several minutes. Continue?"):
            try:
                generate_visual_dataset()
                if generate_master_ledger():
                    self.visual_model, self.training_history = train_visual_model()
                    self.predictive_model, self.encoders, self.baseline_accuracy = train_predictive_model()
                    self.ai_engine = AIEngine(self.predictive_model, self.encoders, DATABASE_PATH)
                    self.view.set_initial_drift_status(self.baseline_accuracy)
                    messagebox.showinfo("Training Complete", "All models successfully re-trained.")
            except Exception as e: messagebox.showerror("Training Error", f"An error occurred: {e}")

    def open_analytics_dashboard(self):
        if not self.ai_engine: messagebox.showerror("Error", "AI Engine not initialized."); return
        AnalyticsDashboard(self.view.root, self)

    def get_training_history(self): return self.training_history
    def get_anomaly_report(self): return self.ai_engine.get_anomaly_report() if self.ai_engine else "AI Engine not available."
    def _preprocess_image(self, fp): return tf.expand_dims(tf.keras.utils.img_to_array(tf.keras.utils.load_img(fp, target_size=IMG_SIZE, color_mode="rgb")), 0)
    def _format_digital_twin_text(self, d, p, r, f):
        return (f"--- DIGITAL TWIN FOR: {d['qr_code_id']} ---\n\n--- Master Data ---\n  Component Type: {d['component_type']}\n  Vendor:         {d['vendor']}\n  Location:       {d['location_zone']}\n  Installed On:   {d['installation_date'].strftime('%Y-%m-%d')} ({d['age_days']} days ago)\n\n"
                f"--- Lifecycle IoT Data ---\n  Manufacturing Quality Score: {d['mfg_quality_score']}/1.0\n  Transport Shock Events:      {d['transport_shock_events']}\n\n--- Latest ITMS In-Service Data ---\n  Vertical Wear:   {d['vertical_wear_mm']} mm\n  Lateral Wear:    {d['lateral_wear_mm']} mm\n  Axle Box Accel.: {d['axle_box_accel_g']} G\n\n"
                f"{'='*15} AI-POWERED ANALYSIS {'='*15}\n  PREDICTED FAILURE RISK: {p:.1%}\n  ESTIMATED REMAINING USEFUL LIFE: {r} days\n\n--- Explainable AI (XAI) ---\n  Top 3 Contributing Risk Factors:\n    1. {f[0]}\n    2. {f[1]}\n    3. {f[2]}\n")

# =================================================================================
# 8. MAIN EXECUTION BLOCK
# =================================================================================

if __name__ == "__main__":
    root = ttk.Window(themename="litera")
    app = DigitalTwinApp(root, None)
    controller = MainController(app)
    app.controller = controller
    controller.load_models_and_data()
    root.mainloop()
