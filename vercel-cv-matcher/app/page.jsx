'use client';

import { useState } from 'react';
import styles from './page.module.css';

export default function Home() {
  const [file, setFile] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [results, setResults] = useState(null);
  const [searchMethod, setSearchMethod] = useState(null);

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files.length > 0) {
      setFile(e.target.files[0]);
      setError(null);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!file) {
      setError('Please select a CV file to upload');
      return;
    }
    
    setIsLoading(true);
    setError(null);
    setResults(null);
    
    try {
      const formData = new FormData();
      formData.append('cv', file);
      
      const response = await fetch('/api/match-cv', {
        method: 'POST',
        body: formData,
      });
      
      const data = await response.json();
      
      if (!response.ok) {
        throw new Error(data.error || 'Failed to process CV');
      }
      
      setResults(data.results.matches);
      setSearchMethod(data.results.method);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An unknown error occurred');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <main className={styles.main}>
      <div className={styles.container}>
        <h1 className={styles.title}>CV Job Matcher</h1>
        <p className={styles.description}>
          Upload your CV to find matching job opportunities
        </p>
        
        <form onSubmit={handleSubmit} className={styles.form}>
          <div className={styles.fileInput}>
            <label htmlFor="cv-upload">
              {file ? file.name : 'Select your CV file (.txt, .pdf, .docx)'}
            </label>
            <input
              id="cv-upload"
              type="file"
              accept=".txt,.pdf,.docx"
              onChange={handleFileChange}
            />
          </div>
          
          <button 
            type="submit" 
            className={styles.button}
            disabled={isLoading || !file}
          >
            {isLoading ? 'Processing...' : 'Find Matching Jobs'}
          </button>
        </form>
        
        {error && (
          <div className={styles.error}>
            <p>{error}</p>
          </div>
        )}
        
        {results && (
          <div className={styles.results}>
            <h2>Job Matches</h2>
            <p>Search method: {searchMethod}</p>
            <p>Found {results.length} potential matches:</p>
            
            <div className={styles.jobList}>
              {results.map((job, index) => (
                <div key={index} className={styles.jobCard}>
                  <h3>{job.Title}</h3>
                  <div className={styles.jobDetails}>
                    <p><strong>Company:</strong> {job.Company}</p>
                    <p><strong>Location:</strong> {job.Area}</p>
                    <p><strong>Posted:</strong> {job.posting_date}</p>
                    <p><strong>Match score:</strong> {job.score.toFixed(2)}</p>
                  </div>
                  <div className={styles.jobActions}>
                    <a 
                      href={job.url} 
                      target="_blank" 
                      rel="noopener noreferrer"
                      className={styles.applyButton}
                    >
                      Apply Now
                    </a>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </main>
  );
}