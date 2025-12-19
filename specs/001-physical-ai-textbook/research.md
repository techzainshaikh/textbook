# Research: Physical AI and Humanoid Robotics Textbook

## Overview
This document captures research findings and technical decisions for implementing the Physical AI and Humanoid Robotics textbook using Docusaurus v3.

## Docusaurus Setup Research

### Decision: Docusaurus v3 Installation and Configuration
**Rationale**: Following the user's requirement to use Docusaurus v3 (latest ~3.9.x as of Dec 2025) with static deployment to GitHub Pages. This provides a modern documentation platform with built-in search, versioning, and responsive design.

**Alternatives considered**:
- GitBook: Less flexible for complex technical documentation
- MkDocs: Limited plugin ecosystem compared to Docusaurus
- Custom solution: Higher maintenance overhead

### Decision: Static Site Architecture
**Rationale**: The requirement specifies all content must be local/static with no backend services. Docusaurus generates static HTML files that can be deployed to GitHub Pages efficiently.

## Technology Stack Research

### Decision: ROS 2 Kilted Kaiju
**Rationale**: As the latest ROS 2 distribution (Kilted Kaiju) released in 2025, it provides the most current features and security updates for robotics development.

**Alternatives considered**:
- ROS 2 Humble Hawksbill (LTS): More stable but older
- ROS 2 Jazzy Jalisco: Previous version

### Decision: NVIDIA Isaac Sim 5.0
**Rationale**: Latest version providing photorealistic simulation capabilities and synthetic data generation as required by the specification.

### Decision: Modern Gazebo (Jetty/Gz)
**Rationale**: Latest version of Gazebo for physics simulation, sensor simulation, and environment modeling as required.

## Docusaurus Configuration Research

### Decision: Plugin Selection
**Rationale**: Need to configure specific plugins to meet requirements:
- @docusaurus/plugin-sitemap for SEO
- @docusaurus/plugin-client-redirects for navigation
- @docusaurus/plugin-google-gtag for analytics (optional)

### Decision: Frontmatter Requirements
**Rationale**: Every document needs consistent frontmatter with title, description, keywords, sidebar_position as per requirements.

## Content Structure Research

### Decision: Module Organization
**Rationale**: Following the 4-module structure specified in requirements:
1. The Robotic Nervous System (ROS 2)
2. The Digital Twin (Gazebo & Unity)
3. The AI-Robot Brain (NVIDIA Isaac™)
4. Vision-Language-Action (VLA)

### Decision: Chapter Template Structure
**Rationale**: Each chapter must follow the required pattern:
1. Learning Objectives (measurable)
2. Prerequisites
3. Core Concepts → Implementation → Examples
4. Summary
5. Exercises (minimum 3: conceptual, logical, implementation)

## Technical Implementation Research

### Decision: Mermaid Diagrams for Visual Learning
**Rationale**: As required by constitution, diagrams are needed for architectures, processes, and spatial concepts. Mermaid provides integration with Docusaurus for creating these diagrams.

### Decision: Citation Management (APA Style)
**Rationale**: Required by constitution for academic rigor. Will implement citation format guidelines and maintain bibliography.

### Decision: Code Example Standards
**Rationale**: As required by constitution, all code examples must be:
- Complete
- Executable
- Commented (WHAT & WHY)
- Version-pinned (Dec 2025 stack)

## Deployment Research

### Decision: GitHub Actions Workflow
**Rationale**: For automated deployment to GitHub Pages as specified in requirements. Will implement workflow that builds and deploys on push to main branch.

### Decision: Build Validation Steps
**Rationale**: Required to meet zero build warnings/errors constraint. Will implement spell check and broken link validation in the CI/CD pipeline.

## Context7 MCP Research

### Decision: Use Context7 for All Setup Tasks
**Rationale**: As required by constitution, the exact phrase "use context7" must be used verbatim for all setup/config tasks, including Docusaurus config, sidebars.js, plugins, build commands, deployment YAML, ROS 2 examples, URDF, Gazebo configs, Isaac Sim setups, etc.

## Compliance Research

### Decision: MIT License Implementation
**Rationale**: As specified in requirements, the textbook will be released under MIT License for open-source distribution.

### Decision: Glossary and Notation Files
**Rationale**: Required by constitution to maintain docs/glossary.md and docs/notation.md for academic clarity and consistency.