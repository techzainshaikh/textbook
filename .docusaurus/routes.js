import React from 'react';
import ComponentCreator from '@docusaurus/ComponentCreator';

export default [
  {
    path: '/__docusaurus/debug',
    component: ComponentCreator('/__docusaurus/debug', '5ff'),
    exact: true
  },
  {
    path: '/__docusaurus/debug/config',
    component: ComponentCreator('/__docusaurus/debug/config', '5ba'),
    exact: true
  },
  {
    path: '/__docusaurus/debug/content',
    component: ComponentCreator('/__docusaurus/debug/content', 'a2b'),
    exact: true
  },
  {
    path: '/__docusaurus/debug/globalData',
    component: ComponentCreator('/__docusaurus/debug/globalData', 'c3c'),
    exact: true
  },
  {
    path: '/__docusaurus/debug/metadata',
    component: ComponentCreator('/__docusaurus/debug/metadata', '156'),
    exact: true
  },
  {
    path: '/__docusaurus/debug/registry',
    component: ComponentCreator('/__docusaurus/debug/registry', '88c'),
    exact: true
  },
  {
    path: '/__docusaurus/debug/routes',
    component: ComponentCreator('/__docusaurus/debug/routes', '000'),
    exact: true
  },
  {
    path: '/docs',
    component: ComponentCreator('/docs', '5c5'),
    routes: [
      {
        path: '/docs',
        component: ComponentCreator('/docs', 'a68'),
        routes: [
          {
            path: '/docs',
            component: ComponentCreator('/docs', 'e44'),
            routes: [
              {
                path: '/docs/examples/ros2-publisher',
                component: ComponentCreator('/docs/examples/ros2-publisher', 'fd8'),
                exact: true
              },
              {
                path: '/docs/examples/ros2-service',
                component: ComponentCreator('/docs/examples/ros2-service', 'e16'),
                exact: true
              },
              {
                path: '/docs/examples/ros2-subscriber',
                component: ComponentCreator('/docs/examples/ros2-subscriber', '08a'),
                exact: true
              },
              {
                path: '/docs/examples/urdf-example',
                component: ComponentCreator('/docs/examples/urdf-example', '886'),
                exact: true
              },
              {
                path: '/docs/glossary',
                component: ComponentCreator('/docs/glossary', 'a11'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/intro',
                component: ComponentCreator('/docs/intro', '61d'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/module-1-ros2/chapter-1-exercises',
                component: ComponentCreator('/docs/module-1-ros2/chapter-1-exercises', 'da1'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/module-1-ros2/chapter-1-nodes-topics-services',
                component: ComponentCreator('/docs/module-1-ros2/chapter-1-nodes-topics-services', 'b9a'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/module-1-ros2/chapter-2-exercises',
                component: ComponentCreator('/docs/module-1-ros2/chapter-2-exercises', '08d'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/module-1-ros2/chapter-2-rclpy-agents',
                component: ComponentCreator('/docs/module-1-ros2/chapter-2-rclpy-agents', 'd4f'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/module-1-ros2/chapter-3-exercises',
                component: ComponentCreator('/docs/module-1-ros2/chapter-3-exercises', '40f'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/module-1-ros2/chapter-3-urdf-modeling',
                component: ComponentCreator('/docs/module-1-ros2/chapter-3-urdf-modeling', '249'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/module-1-ros2/chapter-4-control-architecture',
                component: ComponentCreator('/docs/module-1-ros2/chapter-4-control-architecture', 'ec2'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/module-1-ros2/chapter-4-exercises',
                component: ComponentCreator('/docs/module-1-ros2/chapter-4-exercises', 'dca'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/module-1-ros2/chapter-5-summary-exercises',
                component: ComponentCreator('/docs/module-1-ros2/chapter-5-summary-exercises', '5eb'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/module-1-ros2/implementation-plan',
                component: ComponentCreator('/docs/module-1-ros2/implementation-plan', 'cac'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/module-1-ros2/intro',
                component: ComponentCreator('/docs/module-1-ros2/intro', '313'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/module-1-ros2/module-summary',
                component: ComponentCreator('/docs/module-1-ros2/module-summary', '406'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/notation',
                component: ComponentCreator('/docs/notation', 'd34'),
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
    path: '/',
    component: ComponentCreator('/', '2e1'),
    exact: true
  },
  {
    path: '*',
    component: ComponentCreator('*'),
  },
];
