
import React from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';

import styles from './index.module.css';

function HomepageHeader() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <header className={clsx('hero hero--primary', styles.heroBanner)}>
      <div className="container">
        <div className={styles.heroContent}>
          <h1 className="hero__title">{siteConfig.title}</h1>
          <p className="hero__subtitle">{siteConfig.tagline}</p>
          <div className={styles.buttons}>
            <Link
              className="button button--secondary button--lg"
              to="/docs/intro">
              Start Reading
            </Link>
            <Link
              className="button button--primary button--lg"
              to="/docs/module-1-ros2/intro">
              Begin Module 1
            </Link>
          </div>
        </div>
      </div>
    </header>
  );
}

export default function Home() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout
      title={`Physical AI & Humanoid Robotics Textbook`}
      description="Comprehensive textbook bridging AI software intelligence with physical robotic embodiment">
      <HomepageHeader />
      <main>
        <section className={styles.features}>
          <div className="container padding-vert--lg">
            <div className="row">
              <div className="col col--10 col--offset-1">
                <h2 className={styles.sectionTitle}>About this Textbook</h2>
                <p className={styles.sectionDescription}>
                  This comprehensive textbook covers the fundamentals of Physical AI and Humanoid Robotics,
                  bridging the gap between artificial intelligence software and physical robotic embodiment.
                  Designed for students, researchers, and engineers, it provides both theoretical foundations
                  and practical implementations.
                </p>
              </div>
            </div>

            <div className="row padding-vert--lg">
              <div className="col col--4">
               
              </div>
              <div className="col col--4">
               
              </div>
              <div className="col col--4">
               
              </div>
            </div>

            <div className="row padding-vert--lg">
              <div className="col col--10 col--offset-1">
                <h2 className={styles.sectionTitle}>Modules</h2>
                <p className={styles.sectionDescription}>
                  Explore our comprehensive curriculum covering all aspects of Physical AI and Humanoid Robotics
                </p>
              </div>
            </div>

            <div className="row padding-vert--lg">
              <div className="col col--3">
                <div className={styles.moduleCard}>
                  <div className={styles.moduleIcon}>🤖</div>
                  <h3 className={styles.moduleTitle}>Module 1</h3>
                  <h4 className={styles.moduleSubtitle}>The Robotic Nervous System</h4>
                  <p className={styles.moduleDescription}>
                    Foundation concepts of ROS 2 including nodes, topics, services, and agents
                  </p>
                  <div className={styles.moduleStatus}>
                    <span className={`${styles.statusBadge} ${styles.statusCompleted}`}>Completed</span>
                  </div>
                  <Link
                    className={styles.moduleButton}
                    to="/docs/module-1-ros2/intro">
                    Start Learning
                  </Link>
                </div>
              </div>

              <div className="col col--3">
                <div className={styles.moduleCard}>
                  <div className={styles.moduleIcon}>🎮</div>
                  <h3 className={styles.moduleTitle}>Module 2</h3>
                  <h4 className={styles.moduleSubtitle}>The Digital Twin</h4>
                  <p className={styles.moduleDescription}>
                    Physics simulation, sensor simulation, environment modeling, and Unity-based visualization
                  </p>
                  <div className={styles.moduleStatus}>
                    <span className={`${styles.statusBadge} ${styles.statusCompleted}`}>Completed</span>
                  </div>
                  <Link
                    className={styles.moduleButton}
                    to="/docs/module-2-digital-twin/intro">
                    Start Learning
                  </Link>
                </div>
              </div>

              <div className="col col--3">
                <div className={styles.moduleCard}>
                  <div className={styles.moduleIcon}>🧠</div>
                  <h3 className={styles.moduleTitle}>Module 3</h3>
                  <h4 className={styles.moduleSubtitle}>The AI-Robot Brain</h4>
                  <p className={styles.moduleDescription}>
                    NVIDIA Isaac Sim, perception pipelines, Nav2 navigation, and reinforcement learning
                  </p>
                  <div className={styles.moduleStatus}>
                    <span className={`${styles.statusBadge} ${styles.statusCompleted}`}>Completed</span>
                  </div>
                  <Link
                    className={styles.moduleButton}
                    to="/docs/module-3-ai-brain/intro">
                    Start Learning
                  </Link>
                </div>
              </div>

              <div className="col col--3">
                <div className={styles.moduleCard}>
                  <div className={styles.moduleIcon}>👁️</div>
                  <h3 className={styles.moduleTitle}>Module 4</h3>
                  <h4 className={styles.moduleSubtitle}>Vision-Language-Action</h4>
                  <p className={styles.moduleDescription}>
                    Speech recognition, LLM-based planning, ROS 2 actions, and multimodal perception
                  </p>
                  <div className={styles.moduleStatus}>
                    <span className={`${styles.statusBadge} ${styles.statusCompleted}`}>Completed</span>
                  </div>
                  <Link
                    className={styles.moduleButton}
                    to="/docs/module-4-vla/intro">
                    Start Learning
                  </Link>
                </div>
              </div>
            </div>

            <div className="row padding-vert--lg">
              <div className="col col--6 col--offset-3">
                <div className={styles.capstoneCard}>
                  <div className={styles.capstoneIcon}>🏆</div>
                  <h3 className={styles.capstoneTitle}>Capstone Project</h3>
                  <p className={styles.capstoneDescription}>
                    Integration of all modules: A simulated humanoid robot receiving spoken commands, converting to text, using LLM for task planning, navigating with ROS 2, perceiving objects, and manipulating them
                  </p>
                  <Link
                    className={styles.capstoneButton}
                    to="/docs/capstone-project">
                    View Project
                  </Link>
                </div>
              </div>
            </div>
          </div>
        </section>
      </main>
    </Layout>
  );
}