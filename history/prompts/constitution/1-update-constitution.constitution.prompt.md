---
id: 1
title: Update Constitution
stage: constitution
date: 2025-12-19
surface: agent
model: claude-sonnet-4-5-20250929
feature: none
branch: master
user: Claude
command: sp.constitution
labels: [constitution,documentation,setup]
links:
  spec: null
  ticket: null
  adr: null
  pr: null
files:
 - .specify/memory/constitution.md
tests:
 - none
---

## Prompt

```text
Project Name: Physical AI and Humanoid Robotics Book
Project Type: Docusaurus-based educational book (MDX docs) for Panaversity's Physical AI & Humanoid Robotics course.
Tech Stack: Docusaurus v3 (latest ~3.9.x as of Dec 2025), deployed to GitHub Pages via GitHub Actions. Static site only – all resources local (no server-side).
License: MIT open-source.

Core Principles (STRICTLY ENFORCED):
- Create and maintain a simple docs/glossary.md for all key terms/references.
- All mathematical equations/principles MUST be validated with derivation or citation.
- Code examples MUST be complete, functional, tested (no pseudocode unless marked), with explanatory comments (WHAT & WHY), versioned dependencies.
- Citations REQUIRED (APA style) for research, algorithms, external concepts. Technical claims need citation, derivation, or validation.
- No speculative/unverified claims (esp. hardware/safety/performance).
- Logical progression: fundamentals → advanced. Each chapter declares explicit Prerequisites.
- Each chapter starts with measurable Learning Objectives.
- 4 Modules: 1. The Robotic Nervous System (ROS 2), 2. The Digital Twin (Gazebo & Unity), 3. The AI-Robot Brain (NVIDIA Isaac™), 4. Vision-Language-Action (VLA).
- Module Pattern: 1. Introduction, 2. Chapters list.
- Chapter Pattern: 1. Learning Objectives, 2. Prerequisites, 3. Content (Core Concepts → Implementation → Examples), 4. Summary, 5. Exercises (min 3: logical, conceptual, implementation).
- Diagrams REQUIRED (Mermaid) for architectures, processes, spatial concepts.
- Consistent notation defined in docs/notation.md.
- 0% plagiarism – all facts traceable/cited (APA).
- Versions: Specify current (Dec 2025) – ROS 2 Kilted Kaiju, NVIDIA Isaac Sim 5.0, modern Gazebo (Jetty/Gz), etc.

Docusaurus Standards:
- Fully navigable/searchable (Algolia DocSearch if applicable).
- Every .mdx/.md: YAML frontmatter with title, description, keywords, sidebar_position.
- SEO: Keywords in headings, metadata, first paragraph. Sitemap auto-generated (use context7 for @docusaurus/plugin-sitemap config). robots.txt in static/ (use context7 for setup).
- All content local/static for serverless GitHub Pages.

Code Quality:
- Examples: Complete logic, comments, versioned deps (e.g., ROS 2 Kilted, Isaac Sim 5.0).
- Use standard/widely-adopted libraries only.

CRITICAL Context7 MCP Usage:
- The phrase "use context7" must never be modified, replaced, or rephrased. This exact wording (lowercase) is required for MCP tool invocation.
- use context7 MCP for EVERY code/setup task (Docusaurus config, sidebars.js, plugins, build commands, deployment YAML, ROS 2 examples, URDF, Gazebo configs, Isaac Sim setups, etc.).
- use context7 to fetch official/latest docs for Docusaurus, ROS 2, NVIDIA Isaac, Gazebo, etc. – never rely on internal knowledge.
- All infrastructure (init, config, deployment) requires use context7 verification.

Deployment & Build Gates:
- Static site for GitHub Pages.
- Automated GitHub Actions workflow.
- Build MUST succeed with ZERO errors/warnings (use context7 for docusaurus build commands).
- Check broken links (internal/external).
- Strict spell check (pass before deploy).
- SEO: Sitemap (use context7 plugin), robots.txt (static folder, use context7).

Proceed step-by-step:
```

## Response snapshot

Created a comprehensive constitution for the Physical AI and Humanoid Robotics Book project. The constitution establishes core principles for educational integrity, code examples, content progression, documentation structure, visual learning, and zero plagiarism. It also defines standards for Docusaurus implementation, technology stack management, and the four-module structure. The development workflow emphasizes MCP Context7 verification and quality assurance gates. The governance section establishes how the constitution is maintained and enforced.

## Outcome

- ✅ Impact: Created project constitution with detailed principles and standards
- 🧪 Tests: none
- 📁 Files: Updated .specify/memory/constitution.md with comprehensive project constitution
- 🔁 Next prompts: Continue with project setup and documentation creation
- 🧠 Reflection: The constitution provides a strong foundation for maintaining quality and consistency across the educational content

## Evaluation notes (flywheel)

- Failure modes observed: None
- Graders run and results (PASS/FAIL): PASS
- Prompt variant (if applicable): N/A
- Next experiment (smallest change to try): Begin creating initial documentation structure based on constitution