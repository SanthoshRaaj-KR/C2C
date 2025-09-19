import { useState, useEffect } from 'react';
import React from 'react';
import Header from './components/header/header';
import RepoInput from './components/repoInput/RepoInput';
import Sidebar from './components/Sidebar/Sidebar';
import ChatWindow from './components/ChatWindow/ChatWindow';
import './index.css';

function App() {
  const [theme, setTheme] = useState('dark');
  const [repoSubmitted, setRepoSubmitted] = useState(false);
  const [chats, setChats] = useState([
    { id: 1, title: 'Purpose of this repository' },
    { id: 2, title: 'How to run the tests' },
    { id: 3, title: 'Explain the auth flow' },
  ]);
  const [currentChatId, setCurrentChatId] = useState(1);

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
  }, [theme]);

  const toggleTheme = () => {
    setTheme((prevTheme) => (prevTheme === 'light' ? 'dark' : 'light'));
  };

  const handleRepoSubmit = (url) => {
    console.log('Repository URL submitted:', url);
    // In a real app, you would process the URL here
    setRepoSubmitted(true);
  };

  const handleCreateNewChat = () => {
      // Logic to create a new chat
      const newChat = { id: Date.now(), title: "New Chat" };
      setChats([newChat, ...chats]);
      setCurrentChatId(newChat.id);
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100vh' }}>
      <Header theme={theme} toggleTheme={toggleTheme} />
      <div style={{ display: 'flex', flexGrow: 1, overflow: 'hidden' }}>
        {repoSubmitted ? (
          <>
            <Sidebar 
                chats={chats} 
                currentChatId={currentChatId} 
                onSelectChat={setCurrentChatId}
                onCreateNewChat={handleCreateNewChat}
            />
            <ChatWindow />
          </>
        ) : (
          <RepoInput onSubmit={handleRepoSubmit} />
        )}
      </div>
    </div>
  );
}

export default App;