// @ts-check

/** @type {import('@docusaurus/plugin-content-docs').SidebarsConfig */
const sidebars = {
  tutorialSidebar: [
    {
      type: 'category',
      label: 'Introduction',
      items: [
        'intro',
        'glossary',
        'notation'
      ],
    },
    {
      type: 'category',
      label: 'Module 1: The Robotic Nervous System (ROS 2)',
      items: [
        'module-1-ros2/intro',
        'module-1-ros2/chapter-1-nodes-topics-services',
        'module-1-ros2/chapter-1-exercises',
        'module-1-ros2/chapter-2-rclpy-agents',
        'module-1-ros2/chapter-2-exercises',
        'module-1-ros2/chapter-3-urdf-modeling',
        'module-1-ros2/chapter-3-exercises',
        'module-1-ros2/chapter-4-control-architecture',
        'module-1-ros2/chapter-4-exercises',
        'module-1-ros2/chapter-5-summary-exercises',
        'module-1-ros2/implementation-plan',
        'module-1-ros2/module-summary'
      ],
    },
    {
      type: 'category',
      label: 'Module 2: The Digital Twin (Gazebo & Unity)',
      items: [
        'module-2-digital-twin/intro',
        'module-2-digital-twin/chapter-1-physics-simulation',
        'module-2-digital-twin/chapter-2-sensor-simulation',
        'module-2-digital-twin/chapter-3-environment-modeling',
        'module-2-digital-twin/chapter-4-unity-visualization',
        'module-2-digital-twin/chapter-4-capstone-integration',
        'module-2-digital-twin/module-summary'
      ],
    },
    {
      type: 'category',
      label: 'Module 3: The AI-Robot Brain (NVIDIA Isaac™)',
      items: [
        'module-3-ai-brain/intro',
        'module-3-ai-brain/chapter-1-isaac-platform',
        'module-3-ai-brain/chapter-2-synthetic-data',
        'module-3-ai-brain/chapter-3-perception-pipelines',
        'module-3-ai-brain/chapter-4-nav2-planning',
        'module-3-ai-brain/chapter-5-reinforcement-learning',
        'module-3-ai-brain/chapter-6-sim-to-real'
      ],
    },
    {
      type: 'category',
      label: 'Module 4: Vision-Language-Action (VLA)',
      items: [
        'module-4-vla/intro',
        'module-4-vla/chapter-1-vla-overview',
        'module-4-vla/chapter-2-speech-recognition',
        'module-4-vla/chapter-3-llm-planning',
        'module-4-vla/chapter-4-ros2-actions',
        'module-4-vla/chapter-5-multimodal-perception',
        'module-4-vla/module-summary'
      ],
    },
    {
      type: 'doc',
      id: 'capstone-project',
      label: 'Capstone Project',
    },
  ],
};

module.exports = sidebars;