import React from 'react';
import Layout from '@theme/Layout';

export default function LayoutWrapper(props) {
  return (
    <Layout {...props}>
      {props.children}
    </Layout>
  );
}