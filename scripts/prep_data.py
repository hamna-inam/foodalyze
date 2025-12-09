import os
from pathlib import Path

BASE_PATH = Path(
    "/Users/bstar/Documents/Fall25/MLOps/MLFlow/IndianFoodDatasetFinalFiltered"
)
# Ensure the root path exists
if not os.path.exists(BASE_PATH):
    print(f"FATAL ERROR: Data directory not found at {BASE_PATH}")
    print(
        "Please adjust the BASE_PATH variable to point to your 'IndianFoodDatasetFinalFiltered' folder."
    )
    exit()

# --- Master Class List (MUST match the order in your data.yaml names: list) ---
CLASSES = [
    "aloo_gobi",
    "aloo_matar",
    "aloo_methi",
    "aloo_shimla_mirch",
    "aloo_tikki",
    "bhatura",
    "bhindi_masala",
    "biryani",
    "boondi",
    "butter_chicken",
    "chak_hao_kheer",
    "cham_cham",
    "chana_masala",
    "chapati",
    "chhena_kheeri",
    "chicken_razala",
    "chicken_tikka",
    "chicken_tikka_masala",
    "chikki",
    "daal_puri",
    "dal_makhani",
    "dal_tadka",
    "dharwad_pedha",
    "double_ka_meetha",
    "dum_aloo",
    "gajar_ka_halwa",
    "gulab_jamun",
    "imarti",
    "jalebi",
    "kachori",
    "kadai_paneer",
    "kajjikaya",
    "kalakand",
    "karela_bharta",
    "kofta",
    "lassi",
    "makki_di_roti_sarson_da_saag",
    "malapua",
    "misi_roti",
    "misti_doi",
    "mysore_pak",
    "naan",
    "navrattan_korma",
    "palak_paneer",
    "paneer_butter_masala",
    "poha",
    "poornalu",
    "pootharekulu",
    "qubani_ka_meetha",
    "rabri",
    "ras_malai",
    "rasgulla",
    "sandesh",
    "sheer_korma",
    "sheera",
    "sohan_halwa",
    "sohan_papdi",
    "unni_appam",
]
CLASS_TO_IDX = {cls_name: i for i, cls_name in enumerate(CLASSES)}


def convert_labels_to_yolo_ids():
    """
    Converts human-readable class names in YOLO label files to numeric class IDs.
    This modifies the label files in place.
    """
    print(f"Starting label conversion for {len(CLASSES)} classes...")

    # Iterate through training and validation sets
    for split in ["test"]:
        labels_dir = BASE_PATH / split / "labels"
        if not labels_dir.exists():
            print(
                f"⚠️ Warning: Labels directory not found for {split} at {labels_dir}. Skipping."
            )
            continue

        count = 0
        for file_path in labels_dir.glob("*.txt"):
            try:
                # Read the original content
                lines = file_path.read_text().strip().splitlines()
                new_lines = []

                # Process each bounding box line
                for line in lines:
                    parts = line.split()
                    if not parts:
                        continue

                    # The original format has the class name as the first part
                    class_name = parts[0]

                    if class_name in CLASS_TO_IDX:
                        cls_id = CLASS_TO_IDX[class_name]
                        # Reconstruct the line with the numeric ID first
                        new_line = " ".join([str(cls_id)] + parts[1:])
                        new_lines.append(new_line)
                    # Note: If the class is NOT in the list, it's silently removed/skipped

                # Write the converted content back to the same file
                file_path.write_text("\n".join(new_lines) + "\n")
                count += 1

            except Exception as e:
                print(f"❌ Error processing file {file_path.name}: {e}")

        print(
            f"✅ Conversion complete for {split} split. Processed {count} label files."
        )

    print("\nData preparation complete. Labels are now numeric and ready for training.")


if __name__ == "__main__":
    convert_labels_to_yolo_ids()
