import React from 'react';
import styles from './Message.module.css';
import { FaCode } from 'react-icons/fa';
import { CiUser } from 'react-icons/ci';

const Message = ({ sender, text }) => {
  const isUser = sender === 'user';

  return (
    <div className={`${styles.messageContainer} ${isUser ? styles.user : styles.system}`}>
      <div className={styles.avatar}>
        {isUser ? <CiUser size={24} /> : <FaCode size={24} />}
      </div>
      <div className={styles.messageContent}>
        <span className={styles.senderName}>{isUser ? 'User' : 'System'}</span>
        <div className={styles.messageBubble}>{text}</div>
      </div>
    </div>
  );
};

export default Message;