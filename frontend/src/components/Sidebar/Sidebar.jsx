import React from 'react';
import styles from './Sidebar.module.css';
import { FaPlus } from 'react-icons/fa';

const Sidebar = ({ chats, currentChatId, onSelectChat, onCreateNewChat }) => {
  return (
    <aside className={styles.sidebar}>
      <button className={styles.newChatButton} onClick={onCreateNewChat}>
        <FaPlus size={14} />
        <span>New Chat</span>
      </button>
      <nav className={styles.chatList}>
        {chats.map((chat) => (
          <a
            key={chat.id}
            href="#"
            className={`${styles.chatItem} ${chat.id === currentChatId ? styles.active : ''}`}
            onClick={(e) => {
              e.preventDefault();
              onSelectChat(chat.id);
            }}
          >
            {chat.title}
          </a>
        ))}
      </nav>
    </aside>
  );
};

export default Sidebar;