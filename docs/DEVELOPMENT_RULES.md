# DEVELOPMENT_RULES.md

Version: 1.0

Status: Permanent Coding Standard

Last Updated: July 2026

Project

Adaptive Multi-Agent Framework for Non-Invasive Hemoglobin Estimation

---

# 1. Purpose

This document defines the mandatory software engineering standards used throughout the project.

Its objectives are

• Maintain code quality

• Maintain consistency

• Reduce technical debt

• Simplify maintenance

• Improve readability

• Improve reproducibility

Every source file must follow these rules.

---

# 2. General Philosophy

The project should resemble a professional software framework rather than a collection of research scripts.

The repository should remain

Modular

Reusable

Documented

Testable

Extensible

Reproducible

Readable

Every implementation decision should prioritize long-term maintainability over short-term convenience.

---

# 3. Design Principles

Every module should have one responsibility.

Avoid monolithic files.

Avoid duplicate implementations.

Avoid circular dependencies.

Avoid hardcoded values.

Prefer composition over inheritance.

Prefer configuration over hardcoding.

Prefer interfaces over implementation-specific code.

---

# 4. File Size

Recommended limits

Python File

<300 lines

Ideal

150–250 lines

Absolute Maximum

500 lines

Large modules should be divided into smaller files.

---

# 5. Function Size

Recommended

10–30 lines

Preferred Maximum

50 lines

Functions exceeding 75 lines should normally be refactored.

Every function should perform one logical task.

---

# 6. Class Size

Classes should remain focused.

Avoid "God Classes."

Each class should encapsulate one responsibility.

Large systems should be implemented through managers coordinating smaller components.

---

# 7. Naming Convention

Classes

PascalCase

Example

DatasetManager

SegmentationTrainer

HbPipeline

Functions

snake_case

Variables

snake_case

Constants

UPPER_CASE

Private members

_prefix

Configuration files

lowercase_with_underscores.yaml

---

# 8. Documentation

Every public class must include

Purpose

Parameters

Returns

Raises

Example usage (where appropriate)

Every public function should contain a concise docstring.

Comments should explain *why*, not *what*.

---

# 9. Type Hints

All public functions must use Python type hints.

Avoid untyped APIs unless unavoidable.

Static analysis should be supported.

---

# 10. Logging

Never use print() for operational logging.

Use the project's logging system.

Every major operation should log

Start

Completion

Warnings

Errors

Execution time

Log levels should be used consistently.
