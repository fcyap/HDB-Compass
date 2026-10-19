import React from 'react';

interface SplitScreenLayoutProps {
  leftContent: React.ReactNode;
  rightContent: React.ReactNode;
}

export const SplitScreenLayout: React.FC<SplitScreenLayoutProps> = ({
  leftContent,
  rightContent,
}) => {
  return (
    <main className="app-container">
      {leftContent}
      <section className="right-panel">
        {rightContent}
      </section>
    </main>
  );
};
