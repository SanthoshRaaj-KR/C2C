import React from 'react';
import styles from './Header.module.css';
import { FaCode, FaCog, FaSun, FaMoon } from 'react-icons/fa';
import { CiUser } from "react-icons/ci";


const Header = ({ theme, toggleTheme }) => {
  return (
    <header className={styles.header}>
      <div className={styles.logoContainer}>
        <FaCode size={24} className={styles.logoIcon}/>
        <h1 className={styles.logoTitle}>CodeQuery</h1>
      </div>
      <div className={styles.controls}>
        <button onClick={toggleTheme} className={styles.iconButton}>
          {theme === 'light' ? <FaMoon size={20} /> : <FaSun size={20} />}
        </button>
        <FaCog size={20} className={styles.icon}/>
        <CiUser size={28} className={styles.icon} />
      </div>
    </header>
  );
};

export default Header;