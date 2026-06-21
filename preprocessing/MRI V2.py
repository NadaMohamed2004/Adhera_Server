import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Nilearn Imports
from nilearn.masking import compute_brain_mask
from nilearn.image import math_img, resample_img, smooth_img, clean_img
from nilearn import plotting, datasets
from nilearn.maskers import NiftiLabelsMasker # Crucial for Parcellation
from nilearn import image, masking # Added for fMRI steps

# Dipy imports for Registration
from dipy.align.imaffine import (transform_centers_of_mass, AffineRegistration)
from dipy.align.transforms import (TranslationTransform3D, RigidTransform3D, AffineTransform3D)
from dipy.align.imwarp import SymmetricDiffeomorphicRegistration
from dipy.align.metrics import CCMetric

# --- 1. Load Images Helper ---
def read_folders_structure(base_path, folder_names):
    results = {}
    for folder_name in folder_names:
        folder_path = os.path.join(base_path, folder_name)
        if not os.path.exists(folder_path):
            print(f"Warning: {folder_path} does not exist")
            continue

        smri_files = []
        fmri_files = []

        for root, dirs, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                if 'anat' in root.split(os.sep):
                    smri_files.append(file_path)
                elif 'func' in root.split(os.sep):
                    fmri_files.append(file_path)

        results[folder_name] = {'sMRI': sorted(smri_files), 'fMRI': sorted(fmri_files)}
        print(f"\n{folder_name}: sMRI files: {len(smri_files)} | fMRI files: {len(fmri_files)}")

    return results

# --- 2. New Processing Functions ---

def resample_nilearn(image, target_affine=None):
    # If no specific affine provided, we create a 3mm isotropic affine
    if target_affine is None:
        target_affine = np.diag([3, 3, 3])

    # Use built-in nilearn resample_img
    return resample_img(image, target_affine=target_affine)


def normalize_intensity_nilearn(image):
    # Using math_img to apply formula: (img - mean) / std
    return math_img('(img - np.mean(img)) / np.std(img)', img=image)


def smooth_nilearn(image, fwhm=6):
    return smooth_img(image, fwhm=fwhm)


# --- 3. Existing Functions (Skull Strip & Register) ---

def skull_strip_nilearn(image_path, view_result=True):
    print(f"\nProcessing: {os.path.basename(image_path)}...")
    try:
        raw_img = nib.load(image_path)
    except Exception as e:
        print(f"Error loading file: {e}")
        return None

    mask_img = compute_brain_mask(raw_img)
    masked_img = math_img('img * mask', img=raw_img, mask=mask_img)
    return masked_img


def register_images_dipy(moving_img, static_img):
    print("\n[Registration] Starting Linear & Non-Linear registration...")

    static_data = static_img.get_fdata()
    static_affine = static_img.affine
    moving_data = moving_img.get_fdata()
    moving_affine = moving_img.affine

    # 1. Center of Mass
    c_of_mass = transform_centers_of_mass(static_data, static_affine, moving_data, moving_affine)

    # 2. Linear Registration Pipeline
    affreg = AffineRegistration(metric=None, level_iters=[1000, 100, 10], sigmas=[3.0, 1.0, 0.0], ss_sigma_factor=3.0)

    transform = TranslationTransform3D()
    translation = affreg.optimize(static_data, moving_data, transform, None,
                                  static_affine, moving_affine, starting_affine=c_of_mass.affine)

    transform = RigidTransform3D()
    rigid = affreg.optimize(static_data, moving_data, transform, None,
                            static_affine, moving_affine, starting_affine=translation.affine)

    transform = AffineTransform3D()
    affine = affreg.optimize(static_data, moving_data, transform, None,
                             static_affine, moving_affine, starting_affine=rigid.affine)

    # 3. Non-Linear (SyN)
    print("  - Step 3: Non-Linear (SyN) Registration...")
    metric = CCMetric(3)
    sdr = SymmetricDiffeomorphicRegistration(metric, level_iters=[10, 10, 5])
    mapping = sdr.optimize(static_data, moving_data, static_affine, moving_affine, affine.affine)

    warped_moving = mapping.transform(moving_data)
    return nib.Nifti1Image(warped_moving, static_affine)


# --- CORRECTED: Parcellation Function ---

def parcellate_aal_nilearn(image):
    """
    Extracts mean intensity values for the 116 regions of the AAL atlas.
    """
    print("  - [Analysis] Parcellating Brain using AAL Atlas (116 Regions)...")

    # 1. Fetch AAL Atlas
    dataset_aal = datasets.fetch_atlas_aal(version='SPM12')
    labels = dataset_aal.labels
    maps_img = dataset_aal.maps

    # 2. Setup Masker
    masker = NiftiLabelsMasker(labels_img=maps_img, standardize=False,
                               resampling_target='data', verbose=0)

    # 3. Extract Signals
    # This calculates the mean value of the image within each of the 116 regions
    roi_values = masker.fit_transform(image)

    # --- ERROR FIX SECTION ---

    # FIX: Ensure roi_values is 2D (shape (1, N)) for consistent indexing.
    # If the array is 1D (shape (N,)), np.atleast_2d makes it (1, N).
    roi_values = np.atleast_2d(roi_values)

    # Now accessing shape[1] is safe
    n_regions_extracted = roi_values.shape[1]

    # Fix 1: Decode labels if they are bytes (e.g., b'Precentral_L' -> 'Precentral_L')
    cleaned_labels = []
    for label in labels:
        if hasattr(label, 'decode'):
            cleaned_labels.append(label.decode())
        else:
            cleaned_labels.append(str(label))

    # Fix 2: Handle Shape Mismatch (117 labels vs 116 extracted regions)
    if len(cleaned_labels) > n_regions_extracted:
        cleaned_labels = cleaned_labels[:n_regions_extracted]

    return roi_values, cleaned_labels

###########################################################################

#                     Run Preprocessing


# --- 5. Main Execution Pipeline (Single Subject with Visualization) ---
output_dir = "D:\\MRI_Model\\outputs"
os.makedirs(output_dir, exist_ok=True)

base_path = "D:\\MRI_Model\\"
folder_names = ['KKI']

# Load Data
data = read_folders_structure(base_path, folder_names)
all_smri = []
for folder_data in data.values():
    all_smri.extend(folder_data['sMRI'])

# Load Template
print("Loading MNI152 template...")
mni_template = datasets.load_mni152_template(resolution=2)

if len(all_smri) > 0:
    # اختيار أول سابجكت فقط للعمل عليه
    smri_file = all_smri[0]
    subject_name = f"Subject_1_{os.path.basename(smri_file).split('.')[0]}"

    print("\n" + "=" * 30)
    print(f"VISUALIZING PIPELINE FOR: {subject_name}")
    print("=" * 30)

    # --- الخطوة 0: الصورة الأصلية Raw Image ---
    raw_img = nib.load(smri_file)
    plotting.plot_anat(raw_img, title="1. Original sMRI", display_mode='ortho', colorbar=True)

    # --- الخطوة 1: Skull Strip ---
    img_stripped = skull_strip_nilearn(smri_file)
    plotting.plot_anat(img_stripped, title="2. After Skull Stripping", display_mode='ortho', colorbar=True)

    # --- الخطوة 2: Registration (Dipy) ---
    img_registered = register_images_dipy(moving_img=img_stripped, static_img=mni_template)
    plotting.plot_anat(img_registered, title="3. After Registration to MNI", display_mode='ortho', colorbar=True)

    # --- الخطوة 3: Resample ---
    img_resampled = resample_nilearn(img_registered, target_affine=np.diag([2, 2, 2]))
    plotting.plot_anat(img_resampled, title="4. After Resampling (2mm)", display_mode='ortho', colorbar=True)

    # --- الخطوة 4: Normalize ---
    img_normalized = normalize_intensity_nilearn(img_resampled)
    plotting.plot_anat(img_normalized, title="5. After Intensity Normalization", display_mode='ortho', colorbar=True)

    # --- الخطوة 5: Smoothing ---
    final_smri_img = smooth_nilearn(img_normalized, fwhm=6)
    plotting.plot_anat(final_smri_img, title="6. Final Smoothed Image (FWHM=6)", display_mode='ortho', colorbar=True)

    # --- الخطوة 6: Parcellation & Plotting ---
    smri_roi_values, smri_labels = parcellate_aal_nilearn(final_smri_img)

    # عرض الرسم البياني للقيم المستخرجة
    plt.figure(figsize=(12, 5))
    plt.bar(range(len(smri_labels)), smri_roi_values[0])
    plt.title(f"Final Step: AAL Parcellation Intensities - {subject_name}")
    plt.xlabel("Region ID (AAL Atlas)")
    plt.ylabel("Mean Intensity")

    # إظهار كل الصور التي تم إنشاؤها
    plt.show()

    print("\n✅ Processing and Visualization complete for the first subject.")
else:
    print("No sMRI files found to process.")