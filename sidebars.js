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
  ],
};

module.exports = sidebars;