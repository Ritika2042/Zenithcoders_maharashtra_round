# ReLearn — ML Project Contract

**Project:** ReLearn — Adaptive Multimodal Learning Environment  
**Purpose:** Shared technical contract for the entire ML team and all AI coding assistants.

> This document is the single source of truth for the ML development of ReLearn.
> Before modifying code, datasets, models, or architecture, read this file first.

---

# 1. PROJECT OBJECTIVE

## Problem

Traditional Learning Management Systems (LMS) generally identify whether a student's answer is correct or incorrect, but they do not reliably identify the underlying misconception responsible for the student's mistake.

ReLearn aims to build an adaptive learning system that:

1. Understands a student's response.
2. Identifies the underlying misconception.
3. Distinguishes between different misconceptions.
4. Provides a targeted intervention for that specific misconception.
5. Generates a new question testing the same concept.
6. Reassesses the student.
7. Determines whether the misconception has actually been resolved.
8. Updates the learner's progress/history.

## Core Learning Loop

```text
Student Question
       ↓
Student Response
       ↓
Misconception Detection
       ↓
Misconception ID + Confidence
       ↓
Targeted Intervention
       ↓
Follow-up Question
       ↓
Student's New Response
       ↓
Resolution Assessment
       ↓
┌─────────────────────┐
│                     │
RESOLVED         NOT RESOLVED
│                     │
↓                     ↓
Continue          Further
Learning        Intervention
       ↓
Learner Model Update