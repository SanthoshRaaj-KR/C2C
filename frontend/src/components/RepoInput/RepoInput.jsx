import React, { useState } from 'react';
import styles from './RepoInput.module.css';

const RepoInput = ({ onSubmit }) => {
  const [url, setUrl] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
    if (url.trim()) {
      onSubmit(url);
    }
  };

  return (
    <div className={styles.container}>
      <div className={styles.content}>
        <h2 className={styles.title}>Ask questions about your code</h2>
        <p className={styles.subtitle}>
          Paste a link to a public repository on GitHub to get started.
        </p>
        <form onSubmit={handleSubmit} className={styles.form}>
          <input
            type="text"
            value={url}
            onChange={(e) => setUrl(e.target.value)}
            className={styles.input}
            placeholder="https://github.com/user/repository"
          />
          <button type="submit" className={styles.button}>
            Submit
          </button>
        </form>
      </div>
    </div>
  );
};

export default RepoInput;