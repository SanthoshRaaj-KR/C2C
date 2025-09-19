import React, { useState, useEffect, useRef } from 'react';
import styles from './ChatWindow.module.css';
import Message from '../Message/Message';
import { IoSend } from 'react-icons/io5';

const initialMessages = [
  {
    sender: 'system',
    text: "Hello! I'm here to help you understand your GitHub repository. What would you like to know?",
  },
  {
    sender: 'user',
    text: 'What is the purpose of this repository?',
  },
];

const ChatWindow = () => {
  const [messages, setMessages] = useState(initialMessages);
  const [input, setInput] = useState('');
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSendMessage = (e) => {
    e.preventDefault();
    if (input.trim() === '') return;

    const userMessage = { sender: 'user', text: input };
    setMessages((prev) => [...prev, userMessage]);
    setInput('');

    // Simulate system response
    setTimeout(() => {
      const systemResponse = {
        sender: 'system',
        text: 'This repository is designed to demonstrate a full-stack React application with a Node.js backend.',
      };
      setMessages((prev) => [...prev, systemResponse]);
    }, 1500);
  };

  return (
    <main className={styles.chatWindow}>
      <div className={styles.messageList}>
        {messages.map((msg, index) => (
          <Message key={index} sender={msg.sender} text={msg.text} />
        ))}
        <div ref={messagesEndRef} />
      </div>
      <form className={styles.inputArea} onSubmit={handleSendMessage}>
        <input
          type="text"
          className={styles.input}
          placeholder="Ask a question about the repository..."
          value={input}
          onChange={(e) => setInput(e.target.value)}
        />
        <button type="submit" className={styles.sendButton}>
          <IoSend size={20} />
        </button>
      </form>
    </main>
  );
};

export default ChatWindow;