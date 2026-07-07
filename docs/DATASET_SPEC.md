# DATASET_SPEC.md

Version: 1.0

Status: Living Technical Specification

Last Updated: July 2026

Project Name

Adaptive Multi-Agent Framework for Non-Invasive Hemoglobin Estimation

---

# Chapter 1

# Purpose

This document defines the standard dataset specification used throughout the Adaptive Multi-Agent Framework.

Its purpose is to ensure every dataset follows a consistent structure so that the software framework remains independent of any specific dataset.

The Dataset Specification defines

• Folder Structure

• Naming Conventions

• Metadata Format

• Image Format

• Segmentation Masks

• Dataset Validation

• Dataset Splitting

• Metadata Handling

• Dataset Outputs

Every dataset supported by this framework must conform to this specification.

---

# Chapter 2

# Dataset Philosophy

The framework should never assume knowledge of a specific dataset.

Instead,

every dataset should satisfy a common interface.

The DatasetManager should therefore load any compatible dataset without requiring source-code modifications.

Datasets should remain

• Modular

• Self-describing

• Versioned

• Validated

• Reproducible

Future datasets should require only configuration changes.

---

# Chapter 3

# Root Directory Structure

Every dataset should follow the same directory layout.

Dataset/

│

├── metadata/

│      patients.csv

│

├── images/

│      eye/

│      tongue/

│      palm/

│      nail/

│

├── masks/

│      eye/

│      tongue/

│      palm/

│      nail/

│

├── splits/

│      train.csv

│      validation.csv

│      test.csv

│

├── statistics/

│

└── README.md

Additional tissues may be added without changing the framework.

---

# Chapter 4

# Tissue Organization

Each tissue should be stored independently.

Example

images/

↓

eye/

↓

tongue/

↓

palm/

↓

nail/

The framework should support any number of tissues.

The currently supported tissues include

• Left Eye

• Right Eye

• Left Palm

• Right Palm

• Tongue

• Left Nail

• Right Nail

Future tissues may include

• Lip Mucosa

• Retinal Fundus

• Facial Images

without modifying the DatasetManager.

---

# Chapter 5

# Image Naming Convention

Every image should have a unique filename.

Recommended format

PatientID_Tissue_Instance.extension

Examples

000001_LE_01.jpg

000001_RE_01.jpg

000001_LT_01.jpg

000001_RT_01.jpg

000001_TG_01.jpg

000001_LN_01.jpg

000001_RN_01.jpg

Patient IDs should remain consistent across all tissues.

No duplicate filenames should exist.

---

# Chapter 6

# Supported Image Formats

The framework should support

JPEG

PNG

TIFF

BMP

Future formats may be added.

The DatasetManager should automatically detect the format.

Unsupported formats should generate validation errors.

---

# Chapter 7

# Image Requirements

Images should satisfy

RGB Colour Space

Three Channels

Non-corrupted

Readable

Correct Orientation

Images should remain in their original resolution.

Resizing should occur during preprocessing rather than permanently modifying the dataset.

---

# Chapter 8

# Segmentation Masks

Masks should be stored separately.

Each mask should correspond to exactly one image.

Recommended naming

000001_LE_01_mask.png

Mask dimensions should exactly match the corresponding image.

Supported mask formats

Binary

Multi-Class

Grayscale

PNG is recommended.

---

# Chapter 9

# Metadata File

Every dataset should contain one primary metadata file.

Recommended filename

patients.csv

The metadata file acts as the central reference for the entire dataset.

Every image should correspond to one patient record.

No duplicate Patient IDs should exist.

---

# Chapter 10

# Mandatory Metadata Fields

The following fields are required.

Patient_ID

Hemoglobin

Age

Gender

Tissue Availability

Image Availability

Dataset Split

These columns are mandatory.

Framework execution should stop if any mandatory column is missing.

---

# Chapter 11

# Optional Metadata Fields

The framework should also support optional metadata.

Examples

Height

Weight

BMI

Socioeconomic Status

Smoking Status

Medical History

SpO₂

Blood Pressure

Pulse Rate

Ethnicity

Location

Collection Device

Camera Type

Lighting Condition

Additional laboratory measurements

Missing optional columns should never prevent execution.

The framework should simply ignore unavailable metadata.

---

# Chapter 12

# Metadata Validation

The DatasetManager should automatically verify

Duplicate Patient IDs

Missing Hb Values

Missing Images

Missing Masks

Duplicate Images

Invalid Paths

Missing Metadata

Invalid Numeric Values

Unsupported File Formats

Corrupted Files

At the end of validation,

a Dataset Validation Report should be generated automatically.

The report should summarize

• Number of Patients

• Number of Images

• Number of Masks

• Missing Images

• Missing Masks

• Missing Metadata

• Corrupted Files

• Duplicate Records

• Invalid Labels

The framework should refuse training if critical validation errors exist.
# Chapter 13

# Dataset Splitting

## Purpose

The framework should generate reproducible dataset splits while preventing information leakage.

Splitting should always occur at the patient level rather than the image level.

All images belonging to a single patient must belong to only one split.

No patient should appear in multiple splits.

---

## Default Split

Training

80%

Validation

10%

Testing

10%

These values should remain configurable through the configuration files.

Alternative split strategies should also be supported.

Examples

- 70 / 15 / 15
- 5-Fold Cross Validation
- Leave-One-Center-Out
- External Validation

---

## Split Requirements

Every split should satisfy

• No duplicate Patient IDs

• Similar Hb distribution

• Balanced demographic distribution where possible

• Similar tissue availability

The framework should automatically generate summary statistics for every split.

---

# Chapter 14

# Dataset Versioning

Datasets should be version controlled.

Each dataset should contain

Dataset Name

Dataset Version

Creation Date

Source

Number of Patients

Number of Images

Number of Masks

Metadata Version

Annotation Version

Every experiment should record the exact dataset version used.

---

# Chapter 15

# Dataset Statistics

The DatasetManager should automatically compute

Number of Patients

Number of Images

Images per Tissue

Patients per Tissue

Hb Distribution

Age Distribution

Gender Distribution

Height Distribution

Weight Distribution

BMI Distribution

SES Distribution

Missing Images

Missing Masks

Missing Metadata

Corrupted Images

Duplicate Images

Dataset Size

Average Image Resolution

Average File Size

These statistics should be automatically exported.

---

## Generated Outputs

dataset_statistics.csv

dataset_summary.json

dataset_report.pdf

distribution_plots/

histograms/

pie_charts/

---

# Chapter 16

# Dataset Preprocessing

The DatasetManager should support configurable preprocessing.

Supported operations

Resize

Normalization

Intensity Scaling

Color Space Conversion

CLAHE

Histogram Equalization

Gamma Correction

Noise Reduction

Background Removal

Padding

Cropping

Preprocessing should never modify the original dataset.

Processed images should remain inside cache directories.

---

# Chapter 17

# Data Augmentation

Augmentation should only be applied during training.

Validation and testing data must never be augmented.

Supported augmentations

Horizontal Flip

Vertical Flip

Rotation

Random Crop

Scaling

Brightness

Contrast

Hue

Saturation

Gaussian Noise

Gaussian Blur

Motion Blur

Elastic Transform

Random Erasing

Cutout

MixUp

CutMix

Augmentation probabilities should remain configurable.

---

# Chapter 18

# Multi-Tissue Support

The framework should support any combination of tissues.

Examples

Eye

Palm

Tongue

Nail

Eye + Palm

Eye + Tongue

Eye + Palm + Tongue

All Available Tissues

The DatasetManager should automatically determine available tissues for every patient.

Missing tissues should not prevent inference.

---

# Chapter 19

# Dataset Cache

Repeated preprocessing should be avoided.

The framework should maintain a cache.

Examples

Resized Images

Normalized Images

Processed Masks

Feature Files

Intermediate Predictions

Cache should be automatically invalidated whenever preprocessing settings change.

---

# Chapter 20

# Dataset Outputs

Every sample returned by the DatasetManager should follow one common structure.

Example

Sample

↓

Patient_ID

↓

Image

↓

Mask

↓

Tissue

↓

Metadata

↓

Hb

↓

Dataset Split

↓

Image Path

↓

Mask Path

↓

Quality Label (Optional)

↓

Additional Labels

Every module in the framework should consume this standardized sample.

---

# Chapter 21

# Dataset Quality Control

Before training begins,

the DatasetManager should automatically perform

Image Integrity Check

Mask Integrity Check

Resolution Consistency

Metadata Consistency

Hb Range Validation

Duplicate Detection

Class Distribution Analysis

Missing Tissue Analysis

Outlier Detection

Any critical issues should stop execution.

Minor issues should generate warnings.

---

# Chapter 22

# Dataset Reports

The DatasetManager should automatically generate

Dataset Summary

Quality Report

Validation Report

Missing Data Report

Hb Distribution Report

Patient Statistics

Demographic Statistics

Image Statistics

Mask Statistics

Split Statistics

These reports should be stored inside the experiment directory.

---

# Chapter 23

# Dataset Interface

The DatasetManager should expose a clean public interface.

initialize()

validate()

load()

split()

preprocess()

augment()

statistics()

cache()

visualize()

summary()

export()

shutdown()

No other module should access dataset files directly.

Every interaction with the dataset must occur through the DatasetManager.

---

# Chapter 24

# Acceptance Criteria

The Dataset Specification is complete when

✓ Any compatible dataset can be loaded.

✓ Patient-level splitting prevents leakage.

✓ Validation detects errors automatically.

✓ Statistics are generated automatically.

✓ Reports are exported automatically.

✓ Preprocessing is configurable.

✓ Multi-tissue datasets are supported.

✓ Metadata is optional where appropriate.

✓ The DatasetManager provides a single standardized interface.

---

# End of Dataset Specification

This document defines the standard dataset format used by the Adaptive Multi-Agent Framework.

Any future dataset integrated into the framework should conform to this specification to ensure compatibility, reproducibility, and extensibility.