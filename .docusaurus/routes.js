import React from 'react';
import ComponentCreator from '@docusaurus/ComponentCreator';

export default [
  {
    path: '/textbook/docs',
    component: ComponentCreator('/textbook/docs', '8a2'),
    routes: [
      {
        path: '/textbook/docs',
        component: ComponentCreator('/textbook/docs', '471'),
        routes: [
          {
            path: '/textbook/docs',
            component: ComponentCreator('/textbook/docs', '3f6'),
            routes: [
              {
                path: '/textbook/docs/capstone-project',
                component: ComponentCreator('/textbook/docs/capstone-project', '82c'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/textbook/docs/examples/ros2-publisher',
                component: ComponentCreator('/textbook/docs/examples/ros2-publisher', '338'),
                exact: true
              },
              {
                path: '/textbook/docs/examples/ros2-service',
                component: ComponentCreator('/textbook/docs/examples/ros2-service', 'c5b'),
                exact: true
              },
              {
                path: '/textbook/docs/examples/ros2-subscriber',
                component: ComponentCreator('/textbook/docs/examples/ros2-subscriber', '707'),
                exact: true
              },
              {
                path: '/textbook/docs/examples/urdf-example',
                component: ComponentCreator('/textbook/docs/examples/urdf-example', 'ecc'),
                exact: true
              },
              {
                path: '/textbook/docs/glossary',
                component: ComponentCreator('/textbook/docs/glossary', '2c2'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/textbook/docs/intro',
                component: ComponentCreator('/textbook/docs/intro', '94a'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/textbook/docs/module-1-ros2/chapter-1-exercises',
                component: ComponentCreator('/textbook/docs/module-1-ros2/chapter-1-exercises', '619'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/textbook/docs/module-1-ros2/chapter-1-nodes-topics-services',
                component: ComponentCreator('/textbook/docs/module-1-ros2/chapter-1-nodes-topics-services', 'f7c'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/textbook/docs/module-1-ros2/chapter-2-exercises',
                component: ComponentCreator('/textbook/docs/module-1-ros2/chapter-2-exercises', '051'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/textbook/docs/module-1-ros2/chapter-2-rclpy-agents',
                component: ComponentCreator('/textbook/docs/module-1-ros2/chapter-2-rclpy-agents', '2eb'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/textbook/docs/module-1-ros2/chapter-3-exercises',
                component: ComponentCreator('/textbook/docs/module-1-ros2/chapter-3-exercises', '721'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/textbook/docs/module-1-ros2/chapter-3-urdf-modeling',
                component: ComponentCreator('/textbook/docs/module-1-ros2/chapter-3-urdf-modeling', 'd89'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/textbook/docs/module-1-ros2/chapter-4-control-architecture',
                component: ComponentCreator('/textbook/docs/module-1-ros2/chapter-4-control-architecture', '0c5'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/textbook/docs/module-1-ros2/chapter-4-exercises',
                component: ComponentCreator('/textbook/docs/module-1-ros2/chapter-4-exercises', 'a44'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/textbook/docs/module-1-ros2/chapter-5-summary-exercises',
                component: ComponentCreator('/textbook/docs/module-1-ros2/chapter-5-summary-exercises', '602'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/textbook/docs/module-1-ros2/implementation-plan',
                component: ComponentCreator('/textbook/docs/module-1-ros2/implementation-plan', '54d'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/textbook/docs/module-1-ros2/intro',
                component: ComponentCreator('/textbook/docs/module-1-ros2/intro', '9fb'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/textbook/docs/module-1-ros2/module-summary',
                component: ComponentCreator('/textbook/docs/module-1-ros2/module-summary', '96d'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/textbook/docs/module-2-digital-twin/chapter-1-physics-simulation',
                component: ComponentCreator('/textbook/docs/module-2-digital-twin/chapter-1-physics-simulation', '4f9'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/textbook/docs/module-2-digital-twin/chapter-2-sensor-simulation',
                component: ComponentCreator('/textbook/docs/module-2-digital-twin/chapter-2-sensor-simulation', '132'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/textbook/docs/module-2-digital-twin/chapter-3-environment-modeling',
                component: ComponentCreator('/textbook/docs/module-2-digital-twin/chapter-3-environment-modeling', '350'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/textbook/docs/module-2-digital-twin/chapter-4-capstone-integration',
                component: ComponentCreator('/textbook/docs/module-2-digital-twin/chapter-4-capstone-integration', 'e2a'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/textbook/docs/module-2-digital-twin/chapter-4-unity-visualization',
                component: ComponentCreator('/textbook/docs/module-2-digital-twin/chapter-4-unity-visualization', '2e4'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/textbook/docs/module-2-digital-twin/intro',
                component: ComponentCreator('/textbook/docs/module-2-digital-twin/intro', 'a9a'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/textbook/docs/module-2-digital-twin/module-summary',
                component: ComponentCreator('/textbook/docs/module-2-digital-twin/module-summary', 'f87'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/textbook/docs/module-3-ai-brain/chapter-1-isaac-platform',
                component: ComponentCreator('/textbook/docs/module-3-ai-brain/chapter-1-isaac-platform', 'ad6'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/textbook/docs/module-3-ai-brain/chapter-2-synthetic-data',
                component: ComponentCreator('/textbook/docs/module-3-ai-brain/chapter-2-synthetic-data', '366'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/textbook/docs/module-3-ai-brain/chapter-3-perception-pipelines',
                component: ComponentCreator('/textbook/docs/module-3-ai-brain/chapter-3-perception-pipelines', '0ba'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/textbook/docs/module-3-ai-brain/chapter-4-nav2-planning',
                component: ComponentCreator('/textbook/docs/module-3-ai-brain/chapter-4-nav2-planning', 'f29'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/textbook/docs/module-3-ai-brain/chapter-5-reinforcement-learning',
                component: ComponentCreator('/textbook/docs/module-3-ai-brain/chapter-5-reinforcement-learning', '7bd'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/textbook/docs/module-3-ai-brain/chapter-6-sim-to-real',
                component: ComponentCreator('/textbook/docs/module-3-ai-brain/chapter-6-sim-to-real', '61a'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/textbook/docs/module-3-ai-brain/intro',
                component: ComponentCreator('/textbook/docs/module-3-ai-brain/intro', 'f2a'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/textbook/docs/module-4-vla/chapter-1-vla-overview',
                component: ComponentCreator('/textbook/docs/module-4-vla/chapter-1-vla-overview', '2c6'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/textbook/docs/module-4-vla/chapter-2-speech-recognition',
                component: ComponentCreator('/textbook/docs/module-4-vla/chapter-2-speech-recognition', 'dc4'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/textbook/docs/module-4-vla/chapter-3-llm-planning',
                component: ComponentCreator('/textbook/docs/module-4-vla/chapter-3-llm-planning', '243'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/textbook/docs/module-4-vla/chapter-4-ros2-actions',
                component: ComponentCreator('/textbook/docs/module-4-vla/chapter-4-ros2-actions', '1f6'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/textbook/docs/module-4-vla/chapter-5-multimodal-perception',
                component: ComponentCreator('/textbook/docs/module-4-vla/chapter-5-multimodal-perception', '1b3'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/textbook/docs/module-4-vla/intro',
                component: ComponentCreator('/textbook/docs/module-4-vla/intro', '325'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/textbook/docs/module-4-vla/module-summary',
                component: ComponentCreator('/textbook/docs/module-4-vla/module-summary', '2eb'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/textbook/docs/notation',
                component: ComponentCreator('/textbook/docs/notation', 'acd'),
                exact: true,
                sidebar: "tutorialSidebar"
              }
            ]
          }
        ]
      }
    ]
  },
  {
    path: '/textbook/',
    component: ComponentCreator('/textbook/', 'd4f'),
    exact: true
  },
  {
    path: '*',
    component: ComponentCreator('*'),
  },
];
