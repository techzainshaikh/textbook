<!-- SYNC IMPACT REPORT
Version change: 1.0.0 → 1.0.0 (initial constitution)
Modified principles: None (new constitution)
Added sections: All principles and sections as specified
Removed sections: None (new constitution)
Templates requiring updates:
  - .specify/templates/plan-template.md ✅ updated
  - .specify/templates/spec-template.md ✅ updated
  - .specify/templates/tasks-template.md ✅ updated
  - .specify/templates/commands/*.md ⚠ pending
Runtime docs requiring updates:
  - README.md ⚠ pending
  - docs/quickstart.md ⚠ pending
Follow-up TODOs: None
-->

# Physical AI and Humanoid Robotics Book Constitution

## Core Principles

### Educational Integrity and Academic Rigor
All mathematical equations/principles MUST be validated with derivation or citation. Citations REQUIRED (APA style) for research, algorithms, external concepts. Technical claims need citation, derivation, or validation. No speculative/unverified claims (esp. hardware/safety/performance). All mathematical content must be rigorously validated through either formal derivation or authoritative citation to established sources.

### Complete and Functional Code Examples
Code examples MUST be complete, functional, tested (no pseudocode unless marked), with explanatory comments (WHAT & WHY), versioned dependencies. Every code example must be fully executable with proper documentation of dependencies and expected outcomes. This ensures students can reproduce and understand all implementations.

### Logical Content Progression and Learning Objectives
Logical progression: fundamentals → advanced. Each chapter starts with measurable Learning Objectives and declares explicit Prerequisites. Content must follow a clear pedagogical sequence that builds upon previous concepts. This ensures students develop a solid foundation before advancing to complex topics.

### Comprehensive Documentation Structure
Each chapter follows the established pattern: 1. Learning Objectives, 2. Prerequisites, 3. Content (Core Concepts → Implementation → Examples), 4. Summary, 5. Exercises (min 3: logical, conceptual, implementation). Modules follow the pattern: 1. Introduction, 2. Chapters list. This creates consistent learning experiences throughout the book.

### Visual Learning and Technical Clarity
Diagrams REQUIRED (Mermaid) for architectures, processes, spatial concepts. Consistent notation defined in docs/notation.md. All technical concepts must be supported by appropriate visual aids to enhance comprehension. This accommodates different learning styles and clarifies complex technical relationships.

### Zero Plagiarism and Citation Compliance
0% plagiarism – all facts traceable/cited (APA). All content must be original or properly attributed with full citations following APA format. This maintains academic integrity and allows readers to verify and expand upon the information provided.

## Additional Standards

### Docusaurus Implementation Requirements
Fully navigable/searchable (Algolia DocSearch if applicable). Every .mdx/.md: YAML frontmatter with title, description, keywords, sidebar_position. SEO: Keywords in headings, metadata, first paragraph. Sitemap auto-generated (use context7 for @docusaurus/plugin-sitemap config). robots.txt in static/ (use context7 for setup). All content local/static for serverless GitHub Pages.

### Technology Stack and Version Management
Use current technology versions as of Dec 2025: ROS 2 Kilted Kaiju, NVIDIA Isaac Sim 5.0, modern Gazebo (Jetty/Gz), Docusaurus v3 (~3.9.x). All dependencies must be versioned and compatible with the specified tech stack. This ensures content remains relevant and functional with current tools.

### Four-Module Structure Adherence
Content organized into 4 Modules: 1. The Robotic Nervous System (ROS 2), 2. The Digital Twin (Gazebo & Unity), 3. The AI-Robot Brain (NVIDIA Isaac™), 4. Vision-Language-Action (VLA). Each module must maintain thematic coherence and build upon prerequisite knowledge appropriately.

## Development Workflow

### MCP Context7 Verification Requirement
The phrase "use context7" must never be modified, replaced, or rephrased. This exact wording (lowercase) is required for MCP tool invocation. use context7 MCP for EVERY code/setup task (Docusaurus config, sidebars.js, plugins, build commands, deployment YAML, ROS 2 examples, URDF, Gazebo configs, Isaac Sim setups, etc.). use context7 to fetch official/latest docs for Docusaurus, ROS 2, NVIDIA Isaac, Gazebo, etc. – never rely on internal knowledge. All infrastructure (init, config, deployment) requires use context7 verification.

### Quality Assurance Gates
Static site for GitHub Pages with automated GitHub Actions workflow. Build MUST succeed with ZERO errors/warnings (use context7 for docusaurus build commands). Check broken links (internal/external). Strict spell check (pass before deploy). SEO: Sitemap (use context7 plugin), robots.txt (static folder, use context7).

### Glossary and Reference Maintenance
Create and maintain a simple docs/glossary.md for all key terms/references. This provides centralized access to definitions and terminology used throughout the book, supporting student comprehension and retention.

## Governance

This constitution governs all aspects of the Physical AI and Humanoid Robotics Book project. All contributors must adhere to these principles and standards. Amendments to this constitution require explicit documentation of changes, approval from project maintainers, and a migration plan for existing content. All pull requests and reviews must verify compliance with these principles. Complexity must be justified with clear pedagogical or technical benefits. Use docs/quickstart.md for runtime development guidance.

**Version**: 1.0.0 | **Ratified**: 2025-12-19 | **Last Amended**: 2025-12-19