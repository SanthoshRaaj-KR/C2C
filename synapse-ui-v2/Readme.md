# 📖 Frontend Chat App Documentation

## 1. Overview

### Description

* **Vite** – fast bundler & dev server
* **React** – UI framework
* **TailwindCSS** – utility-first CSS
* **Shadcn UI components** – styled components with accessibility & design built-in
* **Axios** – API client
* **React Context + Reducer** – state management
* **React Markdown + Highlight.js** – message rendering with syntax highlighting
* **react-hot-toast** – user notifications

The app interacts with a backend API for:

* Creating chat sessions (linked to GitHub repos)
* Sending & receiving messages
* Fetching session lists and details

---

## 2. Folder Structure

```
src/
├── App.jsx                     # Root layout (Sidebar + MainContent)
├── main.jsx                    # React entry point
├── styles/globals.css          # Tailwind base styles
│
├── lib/
│   └── api.js                  # API client (createSession, sendMessage, etc.)
│
├── utils/
│   └── format.js               # Helpers for text & timestamps
│
├── context/
│   └── SessionContext.jsx      # State management with reducer
│
├── components/
│   ├── Sidebar/
│   │   ├── Sidebar.jsx         # Sidebar layout with session list + new chat
│   │   └── SessionCard.jsx     # Preview of individual session
│   │
│   ├── Modals/
│   │   └── NewChatModal.jsx    # Modal for creating a new session
│   │
│   ├── Chat/
│   │   ├── MainContent.jsx     # Chat panel (either Welcome or ChatWindow)
│   │   ├── ChatWindow.jsx      # Chat header + messages + composer
│   │   ├── ChatHeader.jsx      # Displays session info
│   │   ├── MessageList.jsx     # Scrollable message container
│   │   ├── MessageBubble.jsx   # User/assistant message rendering (markdown)
│   │   └── Composer.jsx        # Textarea & send button
│   │
│   └── Shared/
│       └── LoadingIndicator.jsx # Reusable spinner
│
└── ...
```

---

## 3. State Management

### Context: `SessionContext`

* Holds global app state:

  ```js
  {
    sessions: { [id]: { session_id, title, repo_url, messages: [] } },
    sessionOrder: [id, id, ...],
    activeSessionId: string | null,
    loading: boolean,
    error: string | null
  }
  ```
* Actions:

  * `CREATE_SESSION_INIT / SUCCESS / FAILURE`
  * `OPEN_SESSION`
  * `SEND_MESSAGE_INIT / SUCCESS / FAILURE`
  * `RECEIVE_SESSION_LIST`
* Provides **`useSessionState()`** and **`useSessionDispatch()`** hooks.

### Data flow

1. User creates session → dispatch `CREATE_SESSION_INIT` → call API → `CREATE_SESSION_SUCCESS`.
2. User sends message → optimistic UI adds message → API → assistant reply → dispatch `SEND_MESSAGE_SUCCESS`.

---

## 4. API Layer: `lib/api.js`

Centralizes backend communication (uses Axios).
Endpoints:

* `createSession(repo_url)`
* `sendMessage(sessionId, content)`
* `fetchSessions()`
* `fetchSession(sessionId)`

> 🔧 Change only here if API routes change.

---

## 5. Components

### **App.jsx**

* Root layout: sidebar on left, main content on right.
* Uses `Sidebar` + `MainContent`.

---

### **Sidebar/**

* `Sidebar.jsx` → Displays list of chat sessions + "New Chat" button.
* `SessionCard.jsx` → Individual clickable chat preview (title + last message).
* Uses Shadcn `Button` & `Card`.

---

### **Modals/**

* `NewChatModal.jsx` → Dialog for starting a new session.

  * Input: GitHub repo URL
  * Calls `api.createSession()`
  * Dispatches reducer actions
  * Closes on success / error

---

### **Chat/**

* `MainContent.jsx`

  * Shows either:

    * Welcome screen (if no active session)
    * `ChatWindow` (if a session is active)
* `ChatWindow.jsx`

  * Container with:

    * `ChatHeader` (title + repo link)
    * `MessageList` (all messages)
    * `Composer` (input + send button)
* `ChatHeader.jsx`

  * Displays repo info + "Open Repo" button
* `MessageList.jsx`

  * Scrollable list of `MessageBubble`s
  * Auto-scrolls to bottom on new messages
* `MessageBubble.jsx`

  * Renders messages:

    * Markdown rendering
    * Syntax highlighting
    * User = right side (blue), Assistant = left side (gray)
    * Timestamp + error state
* `Composer.jsx`

  * Textarea input + send button
  * Press **Enter** → send (Shift+Enter → newline)
  * Optimistic message sending
  * Disabled while sending

---

### **Shared/**

* `LoadingIndicator.jsx` → Reusable spinner (e.g. loading sessions).

---

## 6. UI/UX Patterns

* **Shadcn UI**:

  * Used for inputs, dialogs, buttons
  * Tailwind + shadcn ensures consistent UI
* **Markdown rendering**:

  * Assistant replies support text formatting, code blocks
* **Optimistic UI**:

  * Messages appear instantly before server confirms
* **Error handling**:

  * Failed sends show inline error
  * Toast notifications for failures

---

## 7. Environment Variables

* `VITE_API_BASE` → API base URL (default: `http://localhost:8000`)

  ```bash
  VITE_API_BASE=http://localhost:8000
  ```

---

## 8. Extending the App

### 🔧 Adding Features

* **Add session rename**:

  * Add reducer action `RENAME_SESSION`
  * API call `PATCH /sessions/:id`
  * Update `ChatHeader` to show editable title.

* **Add authentication**:

  * Add `auth.js` in `lib/`
  * Use Axios interceptors to attach JWT tokens
  * Wrap `SessionProvider` with `AuthProvider`

* **Add routing (`/sessions/:id`)**:

  * Install React Router
  * Map session ID from URL to `activeSessionId`
  * Update `Sidebar` to use `<Link>`

### 🖌️ Theming

* Tailwind config (`tailwind.config.js`) can be extended
* Shadcn components accept `variant` props for dark/light modes

---

## 9. How to Modify Safely

* If changing **API endpoints** → only edit `lib/api.js`
* If changing **state behavior** → edit `SessionContext.jsx`
* If changing **UI** → modify respective component in `components/`
* If adding **new views** → create a new folder under `components/`

---

## 10. Developer Guide

### Start Dev Server

```bash
npm install
npm run dev
```

### Build for Production

```bash
npm run build
npm run preview
```

### Lint & Format

```bash
npm run lint
```

---

## 11. Testing with Mock Server

Run the mock backend (from `server.js` example earlier):

```bash
node server.js
```

Then in your frontend `.env`:

```
VITE_API_BASE=http://localhost:8000
```