# Data Model: Physical AI and Humanoid Robotics Textbook

## Overview
This document defines the data model for the Physical AI and Humanoid Robotics textbook, representing the content structure and relationships.

## Content Entities

### Module
**Description**: A major section of the textbook containing related chapters
**Fields**:
- id: String (unique identifier, e.g., "module-1-ros2")
- title: String (display title)
- description: String (brief overview)
- chapters: Array<Chapter> (ordered list of chapters)
- prerequisites: Array<String> (knowledge required before starting)
- learningObjectives: Array<String> (measurable outcomes)

**Validation Rules**:
- Must contain 1-6 chapters (as per specification)
- Must have 1-3 learning objectives
- Title and description are required

### Chapter
**Description**: A unit of educational content within a module
**Fields**:
- id: String (unique identifier, e.g., "chapter-1-nodes-topics-services")
- title: String (display title)
- moduleId: String (reference to parent module)
- learningObjectives: Array<String> (measurable outcomes)
- prerequisites: Array<String> (specific to this chapter)
- coreConcepts: String (theoretical foundations)
- implementation: String (practical application)
- examples: Array<Example> (worked examples)
- summary: String (key takeaways)
- exercises: Array<Exercise> (minimum 3: conceptual, logical, implementation)
- frontmatter: Frontmatter (metadata for Docusaurus)

**Validation Rules**:
- Must have 1-5 learning objectives
- Must include at least 3 exercises (conceptual, logical, implementation)
- Core concepts, implementation, and examples are required
- Frontmatter must include title, description, keywords, sidebar_position

### Example
**Description**: A worked example demonstrating concepts
**Fields**:
- id: String (unique identifier)
- title: String (brief description)
- description: String (explanation of what the example demonstrates)
- code: String (executable code with comments explaining WHAT and WHY)
- dependencies: Array<String> (version-pinned requirements)
- expectedOutput: String (what student should see when running)

**Validation Rules**:
- Code must be complete and executable
- Dependencies must specify exact versions
- Must include both WHAT and WHY comments

### Exercise
**Description**: An assessment item for student learning
**Fields**:
- id: String (unique identifier)
- type: Enum (conceptual | logical | implementation)
- title: String (brief description)
- description: String (the problem statement)
- difficulty: Enum (beginner | intermediate | advanced)
- solution: String (for instructor reference)
- hints: Array<String> (optional guidance)

**Validation Rules**:
- Type must be one of the three specified
- Each chapter must have at least one of each type

### Frontmatter
**Description**: Metadata for Docusaurus documentation system
**Fields**:
- title: String (display title)
- description: String (SEO description)
- keywords: Array<String> (for search optimization)
- sidebar_position: Number (ordering in navigation)
- tags: Array<String> (optional categorization)

**Validation Rules**:
- Title and description are required
- Keywords should be relevant to content

### MathematicalConcept
**Description**: A mathematical equation or principle
**Fields**:
- id: String (unique identifier)
- expression: String (the mathematical expression)
- derivation: String (step-by-step derivation OR citation)
- citation: Citation (APA-formatted reference if no derivation)
- application: String (how it's used in robotics)

**Validation Rules**:
- Must have either derivation OR citation (not both)
- If citation, must follow APA format

### Citation
**Description**: A reference to external source
**Fields**:
- id: String (unique identifier)
- apaFormat: String (full citation in APA format)
- url: String (optional - link to source)
- accessedDate: Date (when source was accessed)
- sourceType: Enum (book | journal | website | documentation | other)

**Validation Rules**:
- apaFormat must follow current APA guidelines
- All citations must be accessible and verifiable

### Diagram
**Description**: A visual representation of concepts
**Fields**:
- id: String (unique identifier)
- type: Enum (architecture | process | spatial | other)
- title: String (brief description)
- description: String (what the diagram illustrates)
- mermaidCode: String (Mermaid syntax for the diagram)
- altText: String (accessibility description)

**Validation Rules**:
- Must be created using Mermaid syntax
- Alt text required for accessibility

## Relationships

```
Module 1---* Chapter
Chapter 1---* Example
Chapter 1---* Exercise
Chapter *---1 Frontmatter
Example *---1 MathematicalConcept (optional)
Chapter *---* MathematicalConcept (optional)
Chapter *---* Diagram (optional)
```

## State Transitions

### Chapter States
- **DRAFT**: Initial state, content being created
- **REVIEW**: Content complete, awaiting review
- **APPROVED**: Content reviewed and approved
- **PUBLISHED**: Content published in textbook

## Validation Rules Summary

1. **Academic Rigor**: All mathematical concepts must have derivation or citation
2. **Code Quality**: All examples must be complete, executable with WHAT/WHY comments
3. **Content Structure**: Each chapter must follow Learning Objectives → Prerequisites → Core Concepts → Implementation → Examples → Summary → Exercises
4. **Visual Learning**: Complex concepts should have corresponding diagrams
5. **Assessment**: Each chapter must have minimum 3 exercises (conceptual, logical, implementation)
6. **Citation Compliance**: All external information must be properly cited in APA format
7. **Technology Compliance**: All code examples must use version-pinned dependencies